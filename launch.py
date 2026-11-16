import argparse
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def wait_until_ready(backend: subprocess.Popen, url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline and backend.poll() is None:  # stop waiting if the backend died
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(description="Launch the MTG Reverse Image Search backend and open the frontend in your browser.")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--backend-host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    backend_url = f"http://{args.backend_host}:{args.backend_port}"
    print(f"Starting backend at {backend_url}")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", args.backend_host, "--port", str(args.backend_port)],
        cwd=ROOT_DIR)  # uvicorn imports main.py from here

    try:
        print("Waiting for backend to become available...")
        # First run downloads the index (~110 MB) and CLIP model before it's ready.
        if not wait_until_ready(backend, f"{backend_url}/health", timeout=600):
            sys.exit("Backend did not start. See the messages above.")

        print(f"Opening frontend in browser at {backend_url}...")
        webbrowser.open(backend_url)

        print("Press Ctrl+C to stop.")
        backend.wait()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        backend.terminate()
        backend.wait()


if __name__ == "__main__":
    main()
