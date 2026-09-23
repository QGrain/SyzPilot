import os
import json
import shutil
import signal
import subprocess
import tempfile
from time import monotonic, sleep, time

import psutil
import requests
try:
    from .utils import api_request
    from .gpu_admission import validate_existing_torchserve_config
except ImportError:  # Script execution keeps brain/ as the import root.
    from utils import api_request
    from gpu_admission import validate_existing_torchserve_config

TERM_GRACE_SECONDS = 10
KILL_GRACE_SECONDS = 5


class ServeOperator:
    def __init__(self, model_dir, start_port=37030, token_expiration=432000,
                 disable_auth=False, inference_gpu_id=None):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.cwd = os.getcwd()
        self.config_path = None
        self.key_path = os.path.join(self.cwd, 'key_file.json')
        self.start_port = start_port
        self.inference_port, self.management_port, self.metrics_port = self.start_port, self.start_port+1, self.start_port+2
        self.grpc_inference_port, self.grpc_management_port = self.start_port+3, self.start_port+4
        self.token_expiration = token_expiration
        self.token_last_update = 0
        self.ts = 'torchserve'
        self.disable_auth = disable_auth
        self.inference_gpu_id = inference_gpu_id
        self._owned_process = None
        self._owned_java_process = None
        self._runtime_dir = None
        print(
            f'[ServeOperator][DEBUG] cwd={self.cwd}, '
            f'disable_auth={self.disable_auth}, '
            f'inference_gpu_id={self.inference_gpu_id}'
        )

    @property
    def has_owned_service(self):
        """Return whether this operator still owns cleanup responsibility."""
        return self._owned_process is not None

    def is_service_ready(self):
        """Probe the owned listeners and TorchServe management API once."""
        process = self._owned_process
        return process is not None and self._management_api_is_ready(process)

    def _service_env(self):
        """Return a child environment restricted to the serving GPU."""
        environment = os.environ.copy()
        if self.inference_gpu_id is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(self.inference_gpu_id)
        return environment

    ### Deploy-related functions
    def create_index2name(self, num_labels, task_dir, stage, force=True):
        t0 = time()
        # Can I change the filename of index_to_name.json? To be tested
        os.makedirs(task_dir, exist_ok=True)
        index2name_path = os.path.join(task_dir, 'index_to_name.json')
        if os.path.isfile(index2name_path) and force == False:
            print(f'[ServeOperator][INFO] {index2name_path} already exist and force==False')
            return index2name_path
        if stage not in (1, 2):
            raise ValueError(f"invalid training stage: {stage}")
        if num_labels < 2:
            raise ValueError("num_labels must be at least 2")
        if stage == 1:
            index2name = {"0": "Unreachable", "1": "Reachable"}
        else:
            index2name = {"0": "Unreachable"}
            for i in range(1, num_labels):
                index2name[str(i)] = "Reach_Func%s" % str(i)

        with open(index2name_path, 'w') as f:
            json.dump(index2name, f, indent=4)
        print(f'[ServeOperator][INFO] Create {index2name_path}, cost {time()-t0:.4f}s')
        return index2name_path

    def create_config(self, num_gpus=1, batch_size=16, force=False):
        self.config_path = os.path.join(self.cwd, 'config.properties')
        if os.path.isfile(self.config_path) and force == False:
            print(f'[INFO] {self.config_path} already exist and force==False, will not re-generate it.')
            return self.config_path

        config_content = f"""
        inference_address=http://0.0.0.0:{self.inference_port}
        management_address=http://0.0.0.0:{self.management_port}
        metrics_address=http://0.0.0.0:{self.metrics_port}
        grpc_inference_address=0.0.0.0
        grpc_inference_port={self.grpc_inference_port}
        grpc_management_address=0.0.0.0
        grpc_management_port={self.grpc_management_port}
        number_of_gpu={num_gpus}
        batch_size={batch_size}
        token_expiration_min={self.token_expiration}
        """.strip()
        with open(self.config_path, 'w') as f:
            for line in config_content.split('\n'):
                f.write(f'{line.strip()}\n')
        print(f'[ServeOperator][INFO] Create config file at {self.config_path}')
        return self.config_path

    def pack_model(self, model_name, index2name_path, torch_script_path, handler_path, version='1.0', force=False):
        t0 = time()
        mar_name = f'{model_name}.mar'
        mar_path = os.path.join(self.model_dir, mar_name)
        if os.path.isfile(mar_path):
            if not force:
                print(f'[ServeOperator][INFO] marfile {mar_path} already exists, return')
                return
            # Remove old .mar for re-packing (hot-swap)
            print(f'[ServeOperator][INFO] Removing old {mar_path} for re-packing')
            os.remove(mar_path)
        print(f'[ServeOperator][INFO] Start to pack marfile {mar_path}')
        command = [
            "torch-model-archiver",
            "--model-name", model_name,
            "--version", version,
            "--serialized-file", torch_script_path,
            "--handler", handler_path,
            "--extra-files", index2name_path
        ]
        try:
            subprocess.run(command, check=True)
            print(f'[ServeOperator][INFO] pack_model done in {time()-t0:.4f}s')
        except subprocess.CalledProcessError as e:
            print(f'[ServeOperator][ERROR] Failed to pack model: {e}')
            raise RuntimeError(f"Failed to pack model: {e}")
        # now mv mar file from cwd to model_dir
        shutil.move(os.path.join(self.cwd, mar_name), mar_path)
        return mar_path

    def start_service(self, ts_config_path, models=None):
        t0 = time()
        process = None
        if self._owned_process is not None:
            raise RuntimeError("TorchServe is already owned by this operator")
        if not os.path.isfile(ts_config_path):
            raise RuntimeError(f"TorchServe config is missing: {ts_config_path}")
        command = [
            self.ts, "--start", "--foreground",
            "--model-store", f"{self.model_dir}",
            "--enable-model-api"
        ]
        if models is not None:
            command.extend(("--models", models))
        if self.disable_auth:
            command.append("--disable-token-auth")

        try:
            self._runtime_dir = tempfile.mkdtemp(prefix="syzpilot-torchserve-")
            # Validate the exact bytes TorchServe will read, not a mutable
            # shared config that can change between preflight and launch.
            private_config = os.path.join(
                self._runtime_dir, "validated.properties"
            )
            shutil.copyfile(ts_config_path, private_config)
            validate_existing_torchserve_config(
                self.start_port, private_config
            )
            command.extend(("--ts-config", private_config))
            environment = self._service_env()
            # TorchServe writes .model_server.pid under tempfile.gettempdir().
            # Isolate it so neither startup nor shutdown touches another user.
            environment["TMPDIR"] = self._runtime_dir
            environment["TEMP"] = self._runtime_dir
            # TorchServe gives this variable precedence over --ts-config.
            environment.pop("TS_CONFIG_FILE", None)
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=environment,
            )
            self._owned_process = process
            if not self._wait_for_management_api(timeout=30, process=process):
                raise RuntimeError(
                    "TorchServe management API and owned listeners did not "
                    "become ready within 30 seconds"
                )
            self._discover_owned_java(process)
            print(
                f'[ServeOperator][INFO] Owned TorchServe process started and '
                f'management API is ready in {time()-t0:.4f}s'
            )
            self.token_last_update = time()
            return process
        except Exception as e:
            print(f'[ServeOperator][ERROR] Start torchserve failed: {e}')
            if self._owned_process is not None:
                self.stop_service()
            else:
                self._cleanup_runtime_dir()
                raise
            return None

    def _discover_owned_java(self, process):
        """Record Java identity before readiness, including failure paths."""
        if self._owned_java_process is not None:
            return self._owned_java_process
        if self._runtime_dir is None:
            return None
        pid_file = os.path.join(self._runtime_dir, ".model_server.pid")
        try:
            with open(pid_file, encoding="ascii") as handle:
                pid = int(handle.readline().strip())
            server = psutil.Process(pid)
            if ("org.pytorch.serve.ModelServer" in server.cmdline() and
                    os.getpgid(pid) == process.pid):
                server.create_time()
                self._owned_java_process = server
                return server
        except (OSError, ValueError, psutil.Error):
            pass
        return None

    def _owned_server_pid(self, process):
        """Require the owned Java process to listen on all configured ports."""
        if process.poll() is not None:
            return None
        server = self._discover_owned_java(process)
        if server is None:
            return None
        try:
            expected_ports = {
                self.inference_port, self.management_port, self.metrics_port,
                self.grpc_inference_port, self.grpc_management_port,
            }
            listening_ports = {
                connection.laddr.port
                for connection in server.net_connections(kind="tcp")
                if connection.status == psutil.CONN_LISTEN
            }
            if (server.is_running() and
                    os.getpgid(server.pid) == process.pid and
                    expected_ports <= listening_ports):
                return server.pid
        except (OSError, psutil.Error):
            pass
        return None

    def _wait_for_management_api(self, timeout=30, process=None):
        """Wait for a response that identifies the TorchServe management API."""
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            if self._management_api_is_ready(process):
                return True
            if process is not None and process.poll() is not None:
                return False
            sleep(0.2)
        return False

    def _management_api_is_ready(self, process=None):
        """Require an owned listener set and a valid management response."""
        if process is not None and process.poll() is not None:
            return False
        try:
            headers = None
            if not self.disable_auth:
                management_key, _, _ = self.__read_key_file()
                if not management_key:
                    return False
                headers = {"Authorization": f"Bearer {management_key}"}
            response = requests.get(
                f"http://127.0.0.1:{self.management_port}/models",
                headers=headers,
                timeout=0.5,
            )
            payload = response.json() if response.status_code == 200 else None
            if not (
                response.status_code == 200 and
                isinstance(payload, dict) and
                isinstance(payload.get("models"), list)
            ):
                return False
            return process is None or self._owned_server_pid(process) is not None
        except (
            requests.RequestException,
            OSError,
            KeyError,
            TypeError,
            ValueError,
        ):
            return False

    def _cleanup_runtime_dir(self):
        if self._runtime_dir is not None:
            shutil.rmtree(self._runtime_dir)
            self._runtime_dir = None

    @staticmethod
    def _live_group_members(group_id):
        """List live members of our session, excluding harmless zombies."""
        members = []
        for candidate in psutil.process_iter(attrs=("pid", "status")):
            try:
                if (candidate.info["status"] != psutil.STATUS_ZOMBIE and
                        os.getpgid(candidate.pid) == group_id):
                    members.append(candidate)
            except (OSError, psutil.Error):
                continue
        return members

    def _has_owned_group_anchor(self, process):
        if process.poll() is None:
            return True
        server = self._discover_owned_java(process)
        try:
            return (server is not None and server.is_running() and
                    server.status() != psutil.STATUS_ZOMBIE and
                    os.getpgid(server.pid) == process.pid)
        except (OSError, psutil.Error):
            return False

    def stop_service(self):
        """Stop only the foreground process group started by this operator."""
        t0 = time()
        process = self._owned_process
        if process is None:
            return
        try:
            if self._has_owned_group_anchor(process):
                os.killpg(process.pid, signal.SIGTERM)
            deadline = monotonic() + TERM_GRACE_SECONDS
            while self._live_group_members(process.pid):
                if monotonic() >= deadline:
                    if not self._has_owned_group_anchor(process):
                        raise RuntimeError(
                            "TorchServe group still has members but its "
                            "owner identity cannot be confirmed"
                        )
                    os.killpg(process.pid, signal.SIGKILL)
                    deadline = monotonic() + KILL_GRACE_SECONDS
                    while self._live_group_members(process.pid):
                        if monotonic() >= deadline:
                            raise RuntimeError(
                                "TorchServe process group did not exit; "
                                "ownership has been retained"
                            )
                        sleep(0.1)
                    break
                sleep(0.1)
            process.wait(timeout=1)
            print(f'[ServeOperator][INFO] stop_service done in {time()-t0:.4f}s')
        except ProcessLookupError:
            process.wait(timeout=5)
        self._owned_process = None
        self._owned_java_process = None
        self._cleanup_runtime_dir()

    def register_model(self, model_name, init_worker=2, batch_size=16,
                       max_batch_delay=10, sync=False):
        t0 = time()
        if self.disable_auth:
            headers = None
        else:
            mgr_key, _, _ = self.__read_key_file()
            headers = {"Authorization": f"Bearer {mgr_key}"}
        params = {
            "url": f"{model_name}.mar",
            "initial_workers": init_worker,
            "batch_size": batch_size,
            "max_batch_delay": max_batch_delay,
            "synchronous": sync
        }
        response = api_request('http://localhost', self.management_port, '/models',
                               params, headers, "POST")
        print(f'[ServeOperator][INFO] register {model_name} done in {time()-t0:.4f}s')
        return response

    def scale_worker(self, model_name, min_worker=1, sync=False):
        t0 = time()
        if self.disable_auth:
            headers = None
        else:
            mgr_key, _, _ = self.__read_key_file()
            headers = {"Authorization": f"Bearer {mgr_key}"}
        response = api_request('http://localhost', self.management_port,
                               f'/models/{model_name}',
                               {"min_worker": min_worker, "synchronous": sync},
                               headers, "PUT")
        print(f'[ServeOperator][INFO] scale_worker min_worker={min_worker} done for {model_name} in {time()-t0:.4f}s')
        return response

    def get_model_info(self, model_name):
        """Get model info including worker count."""
        if self.disable_auth:
            headers = None
        else:
            mgr_key, _, _ = self.__read_key_file()
            headers = {"Authorization": f"Bearer {mgr_key}"}
        resp = api_request('http://localhost', self.management_port,
                           f'/models/{model_name}', None, headers, "GET")
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return data

    def unregister_model(self, model_name, version='1.0'):
        t0 = time()
        if self.disable_auth:
            headers = None
        else:
            mgr_key, _, _ = self.__read_key_file()
            headers = {"Authorization": f"Bearer {mgr_key}"}
        response = api_request('http://localhost', self.management_port,
                               f'/models/{model_name}/{version}',
                               None, headers, "DELETE")
        print(f'[ServeOperator][INFO] unregister {model_name} done in {time()-t0:.4f}s')
        return response

    def serve_model(self, model_name, mar_name, ts_config_path):
        """Start a model with the same owned foreground lifecycle."""
        full_mar_name = f'{mar_name}.mar'
        self.mar_name = full_mar_name
        self.model_name = model_name
        process = self.start_service(
            ts_config_path, models=f"{model_name}={full_mar_name}"
        )
        if process is None:
            raise RuntimeError(f"Failed to serve model {model_name}")
        return process

    def __read_key_file(self):
        if not os.path.isfile(self.key_path):
            print('[ServeOperator][WARNING] key_file.json does not exist, torchserve start first')
            return None, None, None
        with open(self.key_path, 'r') as f:
            keys = json.load(f)
        management_key = keys["management"]["key"]
        inference_key = keys["inference"]["key"]
        api_key = keys["API"]["key"]
        return management_key, inference_key, api_key

    def ck_read_key_file(self):
        management_key, inference_key, api_key = self.__read_key_file()
        # fresh the key_file when pass 80% of the expiration period
        if (time()-self.token_last_update) > (0.8 * (self.token_expiration*60)):
            self.mgr_fresh_key()
        management_key, inference_key, api_key = self.__read_key_file()
        return management_key, inference_key, api_key

    ### Manager operation with curl
    def mgr_fresh_key(self):
        _, _, api_key = self.__read_key_file()
        api_request('http://localhost', self.management_port, '/token',
                    {"type":"management"}, {"Authorization": f"Bearer {api_key}"},
                    "POST")
        self.token_last_update = time()

    def mgr_get_models(self):
        if self.disable_auth:
            headers = None
        else:
            mgr_key, _, _ = self.__read_key_file()
            headers = {"Authorization": f"Bearer {mgr_key}"}
        models = api_request('http://localhost', self.management_port, '/models',
                    None, headers,
                    "GET")
        return models

    def mgr_get_config(self):
        if self.disable_auth:
            headers = None
        else:
            mgr_key, _, _ = self.__read_key_file()
            headers = {"Authorization": f"Bearer {mgr_key}"}
        model_config = api_request('http://localhost', self.management_port, f'/models/{self.model_name}',
                    None, headers,
                    "GET")
        return model_config

    def mgr_update_config(self, params):
        if self.disable_auth:
            headers = None
        else:
            mgr_key, _, _ = self.__read_key_file()
            headers = {"Authorization": f"Bearer {mgr_key}"}
        api_request('http://localhost', self.management_port, f'/models/{self.model_name}',
                    params, headers,
                    "PUT")

class TrainOperator:
    def __init__(self):
        pass
