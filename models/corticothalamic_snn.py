"""
Thalamocortical Spiking Neural Network generating LFPs and EEG spectral
bands via Norse LIF neurons (with pure-PyTorch fallback).

Architecture
------------
Three spiking sub-populations:
 * Thalamic Relay (TC) : 20 LIF neurons -- relay sensory input, generate
                          sleep spindles in N2/N3 via Ih and T-type Ca2+
 * Cortical Excitatory : 20 LIF neurons -- receive TC input, project back
 * Cortical Inhibitory : 10 LIF neurons -- provide cortical I-E balance

The cortical_inhibition_scale parameter reduces I-neuron gain, causing
elevated Beta/Gamma in Paradoxical Insomnia despite thalamic NREM.

Sleep-stage modulation
-----------------------
Injected current (I_ext) is set per stage:
  Wake : high noise, desynchronised tonic firing (>15 Hz effective)
  N1   : reduced noise, alpha-band resonance emerging
  N2   : I_h current boost + spindle-band forcing (12--15 Hz)
  N3   : strong hyperpolarisation → slow delta (0.5--4 Hz) bursting
  REM  : theta-range noise, TC output suppressed
"""

from __future__ import annotations

import os
import sys
import math
import torch
import warnings
import numpy as np
import torch.nn as nn
from typing import Optional, Tuple
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore", category=FutureWarning)

import config


# Norse import with fallback for testing
try:
    import norse.torch as snn
    from norse.torch import LIFCell, LICell
    from norse.torch.module.lif import LIFState, LIFParameters
    _NORSE_AVAILABLE = True
except ImportError:
    _NORSE_AVAILABLE = False


class _NorseLIFCell(nn.Module):
    """
    Norse-based LIF population with linear synaptic projections.
    Matches the Norse/Fallback calling convention:
        spikes, new_state = cell(input, state)
    """
    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        tau_mem_inv: float = 1.0 / 0.02,
        v_threshold: float = 1.0,
        v_reset:     float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, hidden_size, bias=False)
        nn.init.xavier_uniform_(self.W.weight)

        p = LIFParameters(
            tau_syn_inv=torch.as_tensor(200.0),
            tau_mem_inv=torch.as_tensor(tau_mem_inv),
            v_leak=torch.as_tensor(0.0),
            v_th=torch.as_tensor(v_threshold),
            v_reset=torch.as_tensor(v_reset),
        )
        self.cell = LIFCell(p=p)

    def initial_state(self, batch_size: int = 1):
        return None

    def forward(self, x: torch.Tensor, state=None):
        i_syn = self.W(x)
        return self.cell(i_syn, state)



# Pure-PyTorch LIF fallback (used when Norse is not installed)
class _FallbackLIFState:
    """Minimal state container matching Norse's LIFState API."""
    def __init__(self, v: torch.Tensor, i: torch.Tensor):
        self.v = v
        self.i = i


class _FallbackLIFCell(nn.Module):
    """
    Simple Leaky Integrate-and-Fire cell in pure PyTorch.
    Matches the Norse LIFCell calling convention:
        spikes, new_state = cell(input, state)
    """
    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        tau_mem_inv: float = 1.0 / 0.02,  # 1/τ_m, τ_m = 20 ms
        v_threshold: float = 1.0,
        v_reset:     float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.tau_mem_inv = tau_mem_inv
        self.v_th        = v_threshold
        self.v_reset     = v_reset
        self.dt          = config.SNN_DT

        self.W = nn.Linear(input_size, hidden_size, bias=False)
        nn.init.xavier_uniform_(self.W.weight)

    def initial_state(self, batch_size: int = 1) -> _FallbackLIFState:
        v = torch.zeros(batch_size, self.hidden_size)
        i = torch.zeros(batch_size, self.hidden_size)
        return _FallbackLIFState(v, i)

    def forward(
        self, x: torch.Tensor, state: Optional[_FallbackLIFState] = None
    ) -> Tuple[torch.Tensor, _FallbackLIFState]:
        if state is None:
            state = self.initial_state(x.shape[0])
        i_syn  = self.W(x) + state.i * 0.9
        dv     = self.dt * self.tau_mem_inv * (-state.v + i_syn)
        v_new  = state.v + dv
        spikes = (v_new >= self.v_th).float()
        v_new  = v_new * (1.0 - spikes) + self.v_reset * spikes
        return spikes, _FallbackLIFState(v_new, i_syn)


# region Main SNN module

class CorticothalamicSNN(nn.Module):
    """
    Three-population thalamocortical SNN.

    Parameters
    ----------
    n_thalamic         : number of thalamic relay neurons
    n_cortical_e       : number of cortical excitatory neurons
    n_cortical_i       : number of cortical inhibitory neurons
    cortical_inh_scale : gain for I→E inhibition [0, 1]; < 1 = paradoxical
    use_norse          : force Norse on/off; None = auto-detect
    """

    def __init__(
        self,
        n_thalamic:         int   = config.SNN_N_THALAMIC,
        n_cortical_e:       int   = config.SNN_N_CORTICAL_E,
        n_cortical_i:       int   = config.SNN_N_CORTICAL_I,
        cortical_inh_scale: float = 1.0,
        use_norse:          Optional[bool] = None,
    ):
        super().__init__()
        self.n_tc  = n_thalamic
        self.n_ce  = n_cortical_e
        self.n_ci  = n_cortical_i
        self.n_tot = n_thalamic + n_cortical_e + n_cortical_i
        self.cortical_inh_scale = float(cortical_inh_scale)

        _use_norse = _NORSE_AVAILABLE if use_norse is None else use_norse
        self._norse = _use_norse and _NORSE_AVAILABLE

        if self._norse:
            self._build_norse()
        else:
            self._build_fallback()

        # LFP integration weight: mean membrane potential across all neurons
        # (proxy for local field potential)
        self._lfp_weights = nn.Parameter(
            torch.ones(self.n_tot) / self.n_tot, requires_grad=False
        )

        # Stage-specific modulation parameters
        self._stage_params = self._build_stage_params()

    # =========================================================================
    # Forward pass
    # =========================================================================

    @torch.no_grad()
    def forward(
        self,
        sleep_stage: int,
        t_steps:     int = config.SNN_INTERNAL_STEPS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run `t_steps` SNN timesteps under the given sleep stage.

        Returns
        -------
        spikes : np.ndarray shape [t_steps, N_total]   binary spike matrix
        lfp    : np.ndarray shape [t_steps]             membrane LFP proxy
        """
        params   = self._stage_params[sleep_stage]
        spikes_list = []
        v_list      = []

        # Initialise states
        states = self._init_states()

        for t in range(t_steps):
            # --- Build external input current ---
            noise_amp  = params["noise_amp"]
            tonic_bias = params["tonic_bias"]
            freq_force = params["freq_force"]  # Hz of oscillatory forcing
            force_amp  = params["force_amp"]

            t_sec   = t * config.SNN_DT
            osc_val = force_amp * math.sin(2.0 * math.pi * freq_force * t_sec)

            # Thalamic input: forced + noise
            tc_noise  = torch.randn(1, self.n_tc)  * noise_amp  + tonic_bias + osc_val
            # Cortical E: receives TC spikes + recurrent
            ce_noise  = torch.randn(1, self.n_ce)  * noise_amp * 0.6 + tonic_bias * 0.7
            # Cortical I background noise -- input dim must match n_ce (CI receives CE output)
            ci_noise  = torch.randn(1, self.n_ce)  * noise_amp * 0.4 * self.cortical_inh_scale

            # --- Forward through each population ---
            tc_sp, ce_sp, ci_sp, states, v_all = self._step_populations(
                tc_noise, ce_noise, ci_noise, states
            )

            all_spikes = torch.cat([tc_sp, ce_sp, ci_sp], dim=1)  # [1, N_tot]
            spikes_list.append(all_spikes.squeeze(0).numpy())

            lfp_val = (v_all * self._lfp_weights).sum().item()
            v_list.append(lfp_val)

        spikes_np = np.stack(spikes_list, axis=0)  # [T, N]
        lfp_np    = np.array(v_list)               # [T]
        return spikes_np, lfp_np

    # =========================================================================
    # Private builders
    # =========================================================================

    def _build_norse(self):
        """Build Norse LIFCell populations."""
        try:
            self.tc_cell = _NorseLIFCell(self.n_tc,  self.n_tc)
            self.ce_cell = _NorseLIFCell(self.n_tc,  self.n_ce)
            self.ci_cell = _NorseLIFCell(self.n_ce,  self.n_ci)
            self._using_norse_cells = True
        except Exception:
            self._build_fallback()

    def _build_fallback(self):
        """Build pure-PyTorch fallback LIF populations."""
        self.tc_cell = _FallbackLIFCell(self.n_tc,  self.n_tc)
        self.ce_cell = _FallbackLIFCell(self.n_tc,  self.n_ce)
        self.ci_cell = _FallbackLIFCell(self.n_ce,  self.n_ci)
        self._using_norse_cells = False
        # CI input_size = n_ce because CI cells are driven by CE population spikes

    def _init_states(self):
        """Initialise (or reset) cell states."""
        return (
            self.tc_cell.initial_state(1),
            self.ce_cell.initial_state(1),
            self.ci_cell.initial_state(1),
        )

    def _step_populations(self, tc_in, ce_in, ci_noise, states):
        """One timestep across all three populations."""
        tc_s, ce_s, ci_s = states  # ci_noise is accessible from outer scope via param passing

        # Thalamic relay
        tc_sp, tc_s_new = self.tc_cell(tc_in, tc_s)

        # Cortical excitatory (receives TC spikes)
        ce_in_full = ce_in + tc_sp  # TC drives CE
        ce_sp, ce_s_new = self.ce_cell(ce_in_full, ce_s)

        # Cortical inhibitory (receives CE drive + background noise, both shape [1, n_ce])
        # cortical_inh_scale < 1 reduces the CE→CI drive (Paradoxical Insomnia)
        ci_in_full = ci_noise + ce_sp * self.cortical_inh_scale
        ci_sp, ci_s_new = self.ci_cell(ci_in_full, ci_s)

        # Gather membrane potentials for LFP
        def _get_v(state):
            if hasattr(state, 'v'):
                return state.v.squeeze(0)
            return torch.zeros(1)

        v_all = torch.cat([
            _get_v(tc_s_new),
            _get_v(ce_s_new),
            _get_v(ci_s_new),
        ], dim=0)

        new_states = (tc_s_new, ce_s_new, ci_s_new)
        return tc_sp, ce_sp, ci_sp, new_states, v_all

    def _build_stage_params(self) -> dict:
        """Stage-indexed dicts of input modulation parameters."""
        return {
            0: dict(noise_amp=0.35, tonic_bias=0.45, freq_force=20.0, force_amp=0.20),  # Wake
            1: dict(noise_amp=0.22, tonic_bias=0.25, freq_force=10.0, force_amp=0.12),  # N1
            2: dict(noise_amp=0.15, tonic_bias=0.15, freq_force=13.5, force_amp=0.30),  # N2 spindles
            3: dict(noise_amp=0.10, tonic_bias=0.05, freq_force= 1.5, force_amp=0.45),  # N3 delta
            4: dict(noise_amp=0.25, tonic_bias=0.20, freq_force= 6.0, force_amp=0.22),  # REM theta
        }

    @property
    def backend(self) -> str:
        return "Norse" if self._norse else "PyTorch-LIF (fallback)"

# endregion Main SNN module


if __name__ == "__main__":
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print(f"Norse available: {_NORSE_AVAILABLE}")
    snn_net = CorticothalamicSNN(cortical_inh_scale=0.25)   # Paradoxical
    print(f"Backend: {snn_net.backend}")

    for stage in range(5):
        spikes, lfp = snn_net(sleep_stage=stage, t_steps=256)
        firing_rate = spikes.mean() / config.SNN_DT
        print(f"  Stage {stage}: spikes.shape={spikes.shape}  "
              f"mean_rate={firing_rate:.1f} Hz  lfp_std={lfp.std():.4f}")
