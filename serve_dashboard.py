"""
Lightweight local HTTP server for the DyssomniaNN Interactive Web Dashboard.
Usage:
    python serve_dashboard.py
"""

import os
import webbrowser
import http.server
import socketserver


PORT = 8000
DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

if __name__ == "__main__":
    os.chdir(DIRECTORY)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/dashboard.html"
        print(f"============================================================")
        print(f"  DyssomniaNN Web Dashboard running at: {url}")
        print(f"  Press Ctrl+C to stop.")
        print(f"============================================================")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard server stopped.")
