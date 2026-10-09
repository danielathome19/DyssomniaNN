# DyssomniaNN: Multi-Scale Spiking Neural Model of Sleep Disorders

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MNE-Python](https://img.shields.io/badge/MNE-EDF%20Ground%20Truth-brightgreen.svg)](https://mne.tools/)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-Realtime%20Telemetry-orange.svg)](https://www.tensorflow.org/tensorboard)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**DyssomniaNN** is a multi-scale biophysical computational model simulating 72-hour sleep-wake dynamics and cortical electrophysiology across five concurrent clinical cohorts. The framework bridges Borbély's two-process homeostatic/circadian regulation, Saper's hypothalamic bistable flip-flop switch (VLPO/LHA), and a corticothalamic leaky integrate-and-fire (LIF) spiking neural network validated against empirical human polysomnography from the PhysioNet Sleep-EDF database.

Full mathematical formulations, biophysical derivations, and clinical findings are detailed in the provided research paper:
 * "DyssomniaNN: Multi-Scale Spiking Neural Modeling of Human Sleep Disorders and Perceived Restedness" (DOI: [TBA](https://doi.org/...)) 

---

## Simulated Clinical Cohorts

DyssomniaNN simulates five distinct patient phenotypes under an identical 72-hour photoperiod (16h light, 8h dark):

| Cohort | Phenotype | Primary Biophysical Mechanism | Rest Status |
|---|---|---|---|
| **0** | **Normal Sleep** | Balanced homeostatic buildup ($\tau_r=18.2\text{h}$) and decay ($\tau_d=4.2\text{h}$), stable SCN entrainment | Optimal (86.4%) |
| **1** | **CRSWD (Delayed Phase)** | $+4.0\text{h}$ circadian phase delay ($\Delta\phi$), blunted photic sensitivity ($\gamma_{\text{light}}=0.40$) | Phase-Shifted (47.9%) |
| **2** | **Chronic Insomnia** | Tonic LHA hyperarousal drive ($B_{\text{arousal}}=0.60$), impaired VLPO GABAergic gain ($g_{\text{vlpo}}=0.45$) | Fragmented (43.3%) |
| **3** | **Paradoxical Insomnia** | Intact subcortical sleep staging, impaired cortical GABAergic interneuron gain ($\gamma_{\text{inh}}=0.25$) yielding persistent beta power | Non-Restorative (37.6%) |
| **4** | **Narcolepsy Type 1** | Severe hypocretin/orexin depletion ($O_{\text{tone}}=0.05$), boundary instability ($\eta=0.75$), diagnostic SOREMPs | Fragmented (48.2%) |

---

## Project Structure

```
DyssomniaNN/
├── config.py                 # Global parameters, cohort configurations, and simulation constants
├── main.py                   # Master parallel simulator and CLI entrypoint
├── requirements.txt          # Dependencies (PyTorch, MNE, NumPy, SciPy, Matplotlib, TensorBoard)
├── serve_dashboard.py        # Local HTTP server for the interactive web dashboard
├── cohorts/                  # Clinical parameter profiles for the 5 patient cohorts
├── data/                     # PhysioNet Sleep-EDF data fetcher and preprocessing
├── environment/              # 24h solar photoperiod and circadian zeitgeber clock
├── evaluation/               # Spectral decomposition (Welch PSD) and Well-Rested Index metrics
├── models/                   # Biophysical modules (Process S/C, Flip-Flop switch, Corticothalamic SNN)
└── visualization/            # PyTorch TensorBoard logger and live dashboard
```

---

## Installation & Setup

Python 3.10+ is required.

### 1. Create and Setup Virtual Environment

```bash
python -m venv venv
```

Followed by:

```bash
python -m pip install -r requirements.txt
```

---

## Running Simulations

### Quick Pipeline Check (Dry Run)
Validates data ingestion, SNN compilation, and telemetry logging in ~10 seconds (20 simulation steps):
```bash
python main.py --dry-run
```

### Accelerated 24-Hour Cycle (Demo Run)
Simulates a full 24-hour cycle with daylight transitions, sleep onset, and ultradian cycling in ~30 seconds (144 steps, $dt=10\,\text{min}$):
```bash
python main.py --demo-run
```

### Full 72-Hour Research Simulation
Executes the complete multi-day parallel simulation across all 5 cohorts (4,320 steps, $dt=1.0\,\text{min}$):
```bash
python -u main.py --no-gui --log-file simulation_72h.log
```
*Tip: To monitor real-time execution in PowerShell:*
```powershell
Get-Content -Path .\simulation_72h.log -Wait
```

---

## Telemetry & Visualization

### Interactive Web Dashboard
Launches a standalone dark-mode dashboard featuring 72-hour hypnograms, homeostatic/circadian curves, spectral power trends, and clinical diagnostic tables:
```bash
python serve_dashboard.py
```
Open your browser to: **`http://localhost:8000/dashboard.html`**

### TensorBoard Telemetry
Streams real-time multi-cohort scalar overlays, 24-hour spiking raster plots, and hypnograms:
```bash
tensorboard --logdir="runs/sleep_simulation"
```
Open your browser to: **`http://localhost:6006`**


## License

DyssomniaNN is licensed under the BSD-3 License. See the [LICENSE](LICENSE.md) file for more information.

<!-- Project development began September 16th, 2026. -->

---

## Citation

If you use this code for your research, please cite this project as:

```bibtex
@misc{Szelogowski_DyssomniaNN_2026,
 author = {Szelogowski, Daniel},
 doi = {TBD},
 month = {oct},
 title = {{DyssomniaNN: Multi-Scale Spiking Neural Modeling of Human Sleep Disorders and Perceived Restedness}},
 url = {https://github.com/danielathome19/DyssomniaNN/},
 year = {2026}
}
```