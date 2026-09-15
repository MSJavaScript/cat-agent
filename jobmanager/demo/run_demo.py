"""One-command demo launcher.

Starts the simulated front end and the demo job manager as child processes,
then runs the interactive client in the foreground. Stop the client (Ctrl+C or
by declining to submit another job) and both servers are shut down.

Bundled offline demo (stub agent + local provider, ports from demo/config.json):

    python -m jobmanager.demo.run_demo

Use your real settings (LLM from jobmanager/.env, providers from
jobmanager_config.json):

    python -m jobmanager.demo.run_demo \
        --config jobmanager/jobmanager_config.json --real-agent

The frontend binds the callback/stream ports read from the config file, so the
demo frontend receives exactly the POSTs the job manager sends.

If you prefer separate terminals, run these three instead:

    python -m jobmanager.demo.frontend --callback-port 5600 --stream-port 5601
    python -m jobmanager.demo.server --config jobmanager/jobmanager_config.json \
        --port 8000 --real-agent
    python -m jobmanager.demo.client --url http://127.0.0.1:8000
"""

import argparse
import json
import os
import subprocess
import sys
import time
from urllib.parse import urlparse

import httpx

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PARENT = os.path.dirname(os.path.dirname(_HERE))


def wait_ready(url: str, timeout: float = 90) -> bool:
    """The server imports langchain/langgraph, which can take a few seconds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _port_of(url: str, default: int) -> int:
    try:
        parsed = urlparse(url)
        if parsed.port:
            return parsed.port
        return 80 if parsed.scheme == "http" else 443
    except Exception:
        return default


def ports_from_config(config_path: str) -> tuple[int, int]:
    """Read callback/stream ports from the config the job manager will use."""
    try:
        with open(config_path) as f:
            setting = json.load(f)
        callback_port = _port_of(setting.get("callback_url", ""), 9001)
        stream_port = _port_of(setting.get("front_end_url", ""), callback_port)
        return callback_port, stream_port
    except Exception:
        return 9001, 9001


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the full jobmanager demo.")
    parser.add_argument("--config", default=os.path.join(_HERE, "config.json"))
    parser.add_argument("--jobmanager-port", type=int, default=8000)
    parser.add_argument("--real-agent", action="store_true",
                        help="Use the production LLM agent (.env) and the providers from --config.")
    parser.add_argument("--no-fresh", action="store_true",
                        help="Keep the existing demo database instead of starting clean "
                             "(stub mode only).")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        print(f"[launcher] config not found: {config_path}", flush=True)
        return 1
    callback_port, stream_port = ports_from_config(config_path)

    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO_PARENT + os.pathsep + env.get("PYTHONPATH", "")

    def spawn(module: str, *extra: str) -> subprocess.Popen:
        cmd = [sys.executable, "-m", module, *extra]
        print(f"[launcher] {' '.join(cmd)}", flush=True)
        return subprocess.Popen(cmd, cwd=_REPO_PARENT, env=env)

    procs = []
    try:
        fe_args = ["--callback-port", str(callback_port)]
        if stream_port != callback_port:
            fe_args += ["--stream-port", str(stream_port)]
        procs.append(spawn("jobmanager.demo.frontend", *fe_args))

        server_args = ["--config", config_path, "--port", str(args.jobmanager_port)]
        if args.real_agent:
            server_args.append("--real-agent")
        elif not args.no_fresh:
            server_args.append("--fresh")
        server = spawn("jobmanager.demo.server", *server_args)

        jm_url = f"http://127.0.0.1:{args.jobmanager_port}/providers"
        if not wait_ready(jm_url):
            if server.poll() is not None:
                print(f"[launcher] job manager exited early (rc={server.returncode}).", flush=True)
            else:
                print("[launcher] timed out waiting for the job manager to become ready.", flush=True)
            return 1
        print("\n[launcher] servers ready, handing over to the interactive client.\n", flush=True)
        client = spawn("jobmanager.demo.client", "--url", f"http://127.0.0.1:{args.jobmanager_port}")
        client.wait()
        return client.returncode or 0
    except KeyboardInterrupt:
        return 0
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())