import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

BACKEND_PORT = 8000
BACKEND_HOST = "127.0.0.1"
ROOT_DIR = Path(__file__).resolve().parent


def launch_backend(port: int, host: str, env: dict) -> subprocess.Popen:
    command = [sys.executable, "-m", "uvicorn", "main:app", "--host", host, "--port", str(port)]
    return subprocess.Popen(command, env=env, cwd=ROOT_DIR)  # uvicorn imports main.py from here


def wait_until_ready(url: str, timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=2):
                return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(description="Launch the MTG Reverse Image Search backend and open the frontend in your browser.")
    parser.add_argument("--backend-port", type=int, default=BACKEND_PORT)
    parser.add_argument("--backend-host", type=str, default=BACKEND_HOST)
    args = parser.parse_args()

    env = os.environ.copy()
    allowed_origins = env.get(
        "ALLOWED_ORIGINS",
        f"http://{args.backend_host}:{args.backend_port}"
    )
    env["ALLOWED_ORIGINS"] = allowed_origins

    backend_url = f"http://{args.backend_host}:{args.backend_port}"

    print(f"Starting backend at {backend_url}")
    backend_process = launch_backend(args.backend_port, args.backend_host, env)

    try:
        print("Waiting for backend to become available...")
        # First run downloads the index (~110 MB) and CLIP model before it's ready.
        if not wait_until_ready(f"{backend_url}/health", timeout=600):
            raise RuntimeError("Backend did not start in time.")

        print(f"Opening frontend in browser at {backend_url}...")
        webbrowser.open(backend_url)

        print("Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        backend_process.terminate()
        backend_process.wait()


if __name__ == "__main__":
    main()
