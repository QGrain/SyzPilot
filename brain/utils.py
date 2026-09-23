import os
import requests
import signal
import socket
import subprocess
import time
from typing import List, Set
from fastapi import HTTPException

def bind_test_port(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        s.bind(("localhost", port))
        s.close()
        return True
    except socket.error:
        s.close()
        return False

def alloc_port(pool: Set[int]) -> int:
    while pool:
        p = pool.pop()
        if bind_test_port(p):
            return p
    raise HTTPException(status_code=503, detail="No available ports")

def release_port(pool: Set[int], port: int):
    pool.add(port)

def run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)

def kill_process(proc: subprocess.Popen, *, process_group: bool = False) -> bool:
    """Terminate a subprocess and report whether its owned processes are gone."""
    if proc is None:
        return True
    if not process_group:
        if proc.poll() is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                # The process exited between poll() and signal delivery.
                return True
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    raise RuntimeError("process did not exit after SIGKILL")
        return True

    process_group_id = proc.pid
    try:
        actual_group_id = os.getpgid(proc.pid)
    except ProcessLookupError:
        # The leader may already be reaped while its children still occupy the
        # session. In that case, continue only if the original group exists.
        proc.poll()
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
    except OSError:
        return False
    else:
        if actual_group_id != process_group_id:
            return False

    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        proc.poll()
        return True
    except OSError:
        return False

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        # poll() also reaps the group leader, avoiding a zombie leader that
        # would otherwise make the process group appear alive indefinitely.
        proc.poll()
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        time.sleep(0.05)

    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        proc.poll()
        return True
    except OSError:
        return False

    kill_deadline = time.monotonic() + 1
    while time.monotonic() < kill_deadline:
        proc.poll()
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        time.sleep(0.05)
    return False

def generate_task_id(fuzzer_id: str, task_name: str, run_id: int) -> str:
    # use @ as separator for easy splitting
    return f"{fuzzer_id}@{task_name}@{run_id}"

def api_request(base_url, port, route, params=None, headers=None, method="POST",
                timeout=10):
    """
    A function to make an API request to a specified endpoint.

    :param base_url: The base URL (e.g., "http://localhost")
    :param port: The port number (e.g., 37031)
    :param route: The API route (e.g., "/token")
    :param params: Dictionary of URL parameters to send in the query string (e.g., {"type": "management"})
    :param headers: Dictionary of HTTP headers to send with the request (e.g., {"Authorization": "Bearer m8c6oxp6"})
    :param method: The HTTP method to use for the request ("GET", "POST", etc.)
    :return: Response object from the request.
    """
    url = f"{base_url}:{port}{route}"
    print(f"[DEBUG] URL: {url} params={params}")

    if method == "GET":
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
    elif method == "POST":
        response = requests.post(url, params=params, headers=headers, timeout=timeout)
    elif method == "PUT":
        response = requests.put(url, params=params, headers=headers, timeout=timeout)
    elif method == "DELETE":
        response = requests.delete(url, params=params, headers=headers, timeout=timeout)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

    response.raise_for_status()
    return response
