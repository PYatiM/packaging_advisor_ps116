"""Turnkey full-stack launcher for PackOptima Enterprise (FastAPI + Streamlit).

Usage:
    python run_app.py            # Starts API (:8000) and Control Tower UI (:8501)
    python run_app.py --check    # Boots both services, verifies HTTP 200 health, exits cleanly
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def _wait_for_http_200(url: str, timeout_s: float = 45.0) -> tuple[bool, str]:
    deadline = time.time() + timeout_s
    last_err = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True, resp.read().decode("utf-8", errors="ignore")[:300]
        except Exception as exc:
            last_err = str(exc)
        time.sleep(0.6)
    return False, last_err


def _terminate_process(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)
    except Exception:
        pass
    print(f"[STOP] {name} stopped.")


def main() -> int:
    parser = argparse.ArgumentParser(description="PackOptima Enterprise Stack Launcher")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=8501)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Start both services, verify HTTP 200 health endpoints, then shut down cleanly.",
    )
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["API_URL"] = f"http://127.0.0.1:{args.api_port}/api/v1"

    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.api_port),
    ]
    ui_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "frontend/app.py",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(args.ui_port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    api_proc: subprocess.Popen | None = None
    ui_proc: subprocess.Popen | None = None

    try:
        print(f"[BOOT] Starting FastAPI backend on http://127.0.0.1:{args.api_port} ...")
        api_proc = subprocess.Popen(api_cmd, cwd=root_dir, env=env)

        api_health_url = f"http://127.0.0.1:{args.api_port}/api/v1/health"
        ok_api, api_info = _wait_for_http_200(api_health_url, timeout_s=45.0)
        if not ok_api:
            print(f"[FAIL] API failed health check at {api_health_url}: {api_info}")
            return 1
        print(f"[OK]   API healthy -> {api_info}")

        print(f"[BOOT] Starting Streamlit Control Tower on http://127.0.0.1:{args.ui_port} ...")
        ui_proc = subprocess.Popen(ui_cmd, cwd=root_dir, env=env)

        ui_health_url = f"http://127.0.0.1:{args.ui_port}/_stcore/health"
        ok_ui, ui_info = _wait_for_http_200(ui_health_url, timeout_s=45.0)
        if not ok_ui:
            print(f"[FAIL] UI failed health check at {ui_health_url}: {ui_info}")
            return 1
        print(f"[OK]   UI healthy -> {ui_info}")

        print("\n" + "=" * 64)
        print("  PackOptima Enterprise Control Tower is LIVE")
        print(f"  Dashboard UI : http://localhost:{args.ui_port}")
        print(f"  FastAPI Docs : http://localhost:{args.api_port}/docs")
        print("=" * 64 + "\n")

        if args.check:
            print("[CHECK] Full-stack startup verification passed. Shutting down.")
            return 0

        while True:
            if api_proc.poll() is not None:
                print(f"[EXIT] API process exited with code {api_proc.returncode}")
                return api_proc.returncode or 1
            if ui_proc.poll() is not None:
                print(f"[EXIT] UI process exited with code {ui_proc.returncode}")
                return ui_proc.returncode or 1
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[INT] Shutdown requested...")
        return 0
    finally:
        _terminate_process(ui_proc, "Streamlit UI")
        _terminate_process(api_proc, "FastAPI API")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    sys.exit(main())
