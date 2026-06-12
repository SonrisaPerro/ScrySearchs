import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_PORT = 8000
BACKEND_HOST = "127.0.0.1"


def launch_backend(port: int, host: str, env: dict) -> subprocess.Popen:
    command = [sys.executable, "-m", "uvicorn", "main:app", "--host", host, "--port", str(port)]
    return subprocess.Popen(command, env=env)


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


def assets_ready() -> bool:
    model_dir = ROOT_DIR / "clip-model"
    index_file = ROOT_DIR / "scryfall_index.faiss"
    mapping_file = ROOT_DIR / "id_mapping.json"

    missing = []
    if not model_dir.exists():
        missing.append("clip-model/")
    if not index_file.exists():
        missing.append("scryfall_index.faiss")
    if not mapping_file.exists():
        missing.append("id_mapping.json")

    if missing:
        print("Missing required assets:")
        for item in missing:
            print(f"  - {item}")
        return False

    return True


def prompt_yes_no(question: str) -> bool:
    while True:
        answer = input(f"{question} [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter 'y' or 'n'.")


def build_assets() -> bool:
    print("Building missing assets with base.py. This may take a while...")
    command = [sys.executable, "base.py"]
    result = subprocess.run(command)
    return result.returncode == 0


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

    if not assets_ready():
        if prompt_yes_no("Required assets are missing. Would you like to build them now?"):
            if not build_assets():
                print("Asset build failed. Exiting.")
                sys.exit(1)
            print("Asset build complete.")
        else:
            print("Cannot start the app without the required assets.")
            sys.exit(1)

    backend_url = f"http://{args.backend_host}:{args.backend_port}"

    print(f"Starting backend at {backend_url}")
    backend_process = launch_backend(args.backend_port, args.backend_host, env)

    try:
        print("Waiting for backend to become available...")
        if not wait_until_ready(f"{backend_url}/health"):
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
