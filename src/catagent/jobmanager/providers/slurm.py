from catagent.jobmanager.base import BaseProvider
from catagent.jobmanager.utils import run_shell_command
from catagent.jobmanager.models import JobStatus, JobSetting

from pydantic import BaseModel
from typing import List, Dict, Literal, Optional, Set
import os, logging, shlex, shutil, tempfile, threading, contextlib, re
import asyncio,paramiko, atexit, scp 

logger = logging.getLogger("jobmanager")

class SlurmJobEnv(BaseModel):
    type:Literal["bohr"] = "slurm"
    name: str 
    softwares: List[str]
    initialize_command: List[str]
    # dataset_path: Optional[str]
    # datasets:Optional[List[str]]

class SlurmResource(BaseModel):
    type:Literal["bohr"] = "slurm"
    partition_name: str
    partition_type: Literal["cpu", "gpu", "cpu,gpu"]
    n_nodes:int
    n_cpu_core:Optional[int]
    n_gpu_core:Optional[int]
    memeory_size: Optional[float]
    video_memeory_size: Optional[float]
    price:Optional[float]


def remote_shell_path(path: str) -> str:
    """Quote a remote path for a shell command without breaking a leading ``~``.

    ``shlex.quote("~/jobs/x")`` produces ``'~/jobs/x'``, and the quotes stop the
    remote shell from expanding the tilde.  Slurm's ``remote_root`` is
    ``~/jobs`` by default, so that turned every download into
    ``cd: ~/jobs/<uuid>: No such file or directory``.  Keep the tilde
    expandable by emitting ``"$HOME"/...`` instead.
    """
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        rest = path[2:]
        return '"$HOME"/' + shlex.quote(rest) if rest else '"$HOME"'
    return shlex.quote(path)


def run_ssh_remote_command(client:paramiko.SSHClient, command:str) -> List[List[str]]|None:
    """Run a command and return ``[stdin_lines, stdout_lines, stderr_lines,
    [exit_status]]``.

    The exit status is appended as a fourth element so callers can tell a
    failed command from an empty successful one.  Existing callers index
    ``[1]``/``[2]`` and are unaffected.
    """
    try:
        channelFiles = client.exec_command(command)
        out = []
        for cf in channelFiles:
            try:
                lines = cf.readlines()
            except (IOError, OSError):
                lines = []
            out.append(lines)
        try:
            exit_status = channelFiles[1].channel.recv_exit_status()
        except Exception:
            exit_status = None
        for cf in channelFiles:
            try:
                cf.close()
            except Exception:
                pass
        out.append(["" if exit_status is None else str(exit_status)])
        return out
    except (paramiko.SSHException, EOFError, OSError) as e:
        logger.error(f"Command {command} failed, {e}.")
        return None  


_SBATCH_JOB_ID_RE = re.compile(r"Submitted\s+batch\s+job\s+(\d+)",
                               re.IGNORECASE)
_VALID_SBATCH_JOB_ID_RE = re.compile(r"[1-9]\d*")


def _parse_submit_job_id(stdout: str) -> Optional[str]:
    """Return the numeric job id printed by ``sbatch``.

    The id is read from the ``Submitted batch job <id>`` message rather than
    from a fixed line/token.  Login shells on some clusters print a storage or
    MOTD banner asynchronously, so a decorative line such as
    ``--------------------------------------------------`` can arrive
    *after* sbatch's message.  The previous implementation took the last
    stdout line's last token and persisted that banner as the remote id; every
    subsequent status query then failed with ``Invalid job id``.

    ``None`` means no usable id.  The caller must fail the submission instead
    of storing a value that can never be polled.  Multiple distinct ids are
    ambiguous because one group maps to exactly one remote job.
    """
    matches = _SBATCH_JOB_ID_RE.findall(stdout or "")
    if not matches:
        return None
    job_ids = {str(int(value)) for value in matches}
    if len(job_ids) > 1:
        raise RuntimeError(
            f"`sbatch` reported multiple job ids: {sorted(job_ids)}")
    job_id = job_ids.pop()
    if not _VALID_SBATCH_JOB_ID_RE.fullmatch(job_id):
        return None
    return job_id


_SLURM_STATE_IN_SCONTROL_RE = re.compile(r"\bJobState=(\S+)")

_SLURM_RUNNING_STATES = frozenset(
    {"RUNNING", "COMPLETING", "CONFIGURING", "RESUMING"})
_SLURM_PENDING_STATES = frozenset({"PENDING", "SUSPENDED"})
_SLURM_COMPLETED_STATES = frozenset({"COMPLETED"})
_SLURM_CANCELLED_STATES = frozenset({"CANCELLED"})
_SLURM_FAILED_STATES = frozenset(
    {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
     "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"})


def _normalize_slurm_state(raw: str) -> str:
    """Normalize scontrol/sacct state text.

    ``sacct`` writes terminal states as e.g. ``CANCELLED by 1234`` and may
    append ``+`` to a state; ``scontrol`` writes just the state name.
    """
    state = (raw or "").strip()
    if not state:
        return ""
    return state.split()[0].rstrip("+").upper()


def _job_status_from_slurm_state(raw: str) -> Optional[JobStatus]:
    """Map a Slurm state string to a JobStatus, or None if unrecognized."""
    state = _normalize_slurm_state(raw)
    if state in _SLURM_RUNNING_STATES:
        return JobStatus.RUNNING
    if state in _SLURM_PENDING_STATES:
        return JobStatus.SUBMITTED
    if state in _SLURM_COMPLETED_STATES:
        return JobStatus.COMPLETED
    if state in _SLURM_CANCELLED_STATES:
        return JobStatus.CANCELLED
    if state in _SLURM_FAILED_STATES:
        return JobStatus.FAILED
    return None


def _parse_scontrol_state(stdout: str) -> Optional[str]:
    """Extract JobState from ``scontrol show jobid -o`` output."""
    match = _SLURM_STATE_IN_SCONTROL_RE.search(stdout or "")
    return match.group(1) if match else None


def _parse_sacct_state(stdout: str, job_id: str) -> Optional[str]:
    """Extract the state for ``job_id`` from ``sacct -P --format=JobID,State``.
    """
    for line in (stdout or "").splitlines():
        fields = [field.strip() for field in line.strip().split("|")]
        if len(fields) >= 2 and fields[0] == job_id:
            return fields[1]
    return None


class SlurmProvider(BaseProvider):
    type: str = "slurm"
    def __init__(self, 
                 name,
                 processing_units: List[str],
                 job_envs_setting: dict,
                 remote_files_setting:List[dict],
                 host: str,
                 user: str,
                 port: int,
                 password: str,
                 remote_root: str = "~/jobs",
                 keyfile: str = "",
                 resources_setting: list = None):
        
        super().__init__(name, processing_units)
        self.job_envs:Dict[str, SlurmJobEnv] = dict()
        for env_key in job_envs_setting.keys():
            self.job_envs[env_key] = SlurmJobEnv(name = env_key,
                                                 softwares = job_envs_setting[env_key]["softwares"],
                                                 initialize_command = job_envs_setting[env_key]["initialize_command"],
                                                 dataset_path=job_envs_setting[env_key].get("dataset_path"),
                                                 datasets=job_envs_setting[env_key].get("datasets"))

        self.remote_files = remote_files_setting
        self.remote_root = remote_root
        self._host = host
        self._user = user
        self._port = port
        self._password = password
        self._keyfile = keyfile

        # `_ssh_lock` serializes connect/exec and protects `ssh_client`, which
        # is not safe for concurrent use.  `_scp_lock` serializes transfers:
        # `scp.SCPClient` keeps the channel and in-progress state on the
        # instance, so one client must never be shared by concurrent transfers.
        self._ssh_lock = threading.RLock()
        self._scp_lock = threading.Lock()
        self.ssh_client: Optional[paramiko.SSHClient] = None
        try:
            with self._ssh_lock:
                self._connect_locked()
        except Exception as e:
            raise RuntimeError(f"Cannot connect to {user}@{host}. Error: {e}")

        atexit.register(self.disconnect)

        self.resources:List[SlurmResource] = []
        if resources_setting is None:
            command="sinfo -O 'partition,cpus,nodes,gres' --noheader | sed 's/\\*//g'"
            result = self._run_ssh(command)
            if result:
                for line in result[1]:
                    a,b,c,d = line.strip().split()
                    partition_type = "cpu,gpu" if "gpu" in d else "cpu"
                    self.resources.append(SlurmResource(partition_name=a,
                                                        partition_type=partition_type,
                                                        n_cpu_core=int(b),
                                                        n_nodes=int(c)))

        else:
            for item in resources_setting:
                self.resources.append(SlurmResource(partition_name=item['partition_name'],
                                                    partition_type=item['partition_type'],
                                                    n_cpu_core=item['n_cpu_core'],
                                                    n_nodes=item['n_nodes'],
                                                    n_gpu_core=item.get('n_gpu_core'),
                                                    memeory_size=item.get('memeory_size'),
                                                    video_memeory_size=item.get('video_memeory_size'),
                                                    price=item.get('price')))

    def _connect_locked(self) -> None:
        """Open a new SSH connection.  The caller must hold ``_ssh_lock``."""
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if len(self._keyfile) > 0:
            client.connect(
                hostname=self._host,
                username=self._user,
                key_filename=self._keyfile,
                port=self._port,
                timeout=30,
                banner_timeout=30,
                auth_timeout=30,
            )
        else:
            client.connect(
                hostname=self._host,
                username=self._user,
                password=self._password,
                port=self._port,
                timeout=30,
                banner_timeout=30,
                auth_timeout=30,
            )
        transport = client.get_transport()
        if transport is not None:
            # The jobmanager polls once a minute, but firewalls/NAT between
            # runs are idle for much longer and silently drop the TCP session.
            # Keepalive makes the death visible and prevents most drops.
            transport.set_keepalive(30)

        old = self.ssh_client
        self.ssh_client = client
        if old is not None:
            with contextlib.suppress(Exception):
                old.close()
        logger.info(f"Connect {self._user}@{self._host} successfully.")

    def _ensure_connected(self) -> None:
        """Reconnect when the cached SSH session is dead.

        A long-lived ``paramiko.SSHClient`` can outlive its transport: after a
        network hiccup every later ``exec_command`` raises
        ``SSHException("SSH session not active")``.  Previously the provider
        kept using that dead client forever, so every status poll failed and
        running/completed jobs stayed ``submitted`` in the database.
        """
        transport = (self.ssh_client.get_transport()
                     if self.ssh_client is not None else None)
        if transport is not None and transport.is_active():
            return
        logger.warning(
            f"SSH connection to {self._user}@{self._host} is not active; "
            f"reconnecting.")
        self._connect_locked()

    def _drop_connection_locked(self) -> None:
        """Close and forget the cached client.  Caller holds ``_ssh_lock``."""
        if self.ssh_client is not None:
            with contextlib.suppress(Exception):
                self.ssh_client.close()
            self.ssh_client = None

    def _run_ssh(self, command: str,
                 retry_on_disconnect: bool = False) -> Optional[List[List[str]]]:
        """Run one command on a live connection, reconnecting if necessary.

        Storage/tar/submit commands are not retried by default: repeating a
        command whose result was lost could submit a Slurm job twice.  When
        such a command fails, the client is dropped so the next call starts
        from a fresh connection instead of reusing a broken one.  Read-only
        state queries pass ``retry_on_disconnect=True`` and are retried once
        immediately.
        """
        with self._ssh_lock:
            try:
                self._ensure_connected()
            except Exception as e:
                logger.error(
                    f"Cannot reconnect to {self._user}@{self._host}: {e}")
                return None
            result = run_ssh_remote_command(self.ssh_client, command)
            if result is not None:
                return result
            if not retry_on_disconnect:
                self._drop_connection_locked()
                return None

            logger.warning(
                f"SSH connection to {self._user}@{self._host} failed while "
                f"running a status query; reconnecting and retrying.")
            try:
                self._connect_locked()
            except Exception as e:
                logger.error(
                    f"Cannot reconnect to {self._user}@{self._host}: {e}")
                self._drop_connection_locked()
                return None
            retry_result = run_ssh_remote_command(self.ssh_client, command)
            if retry_result is None:
                self._drop_connection_locked()
            return retry_result

    @staticmethod
    def _command_succeeded(result: Optional[List[List[str]]]) -> bool:
        """Return whether a remote command reported exit status 0."""
        if not result:
            return False
        status = (result[3][0].strip()
                  if len(result) > 3 and result[3] else "")
        if status:
            return status == "0"
        # Older SSH helpers may not expose the status; fall back to stderr.
        return not "".join(result[2]).strip()

    @staticmethod
    def _command_error(result: Optional[List[List[str]]]) -> str:
        if not result:
            return "SSH error"
        stderr = "".join(result[2]).strip()
        status = (result[3][0].strip()
                  if len(result) > 3 and result[3] else "")
        if stderr:
            return stderr
        return f"exit status {status or 'unknown'}"

    def disconnect(self):
        with self._ssh_lock:
            self._drop_connection_locked()
        logger.info("SSH connection closed.")

    def _scp_get(self, remote_path: str, local_path: str) -> None:
        """Receive one path with a dedicated SCP channel.

        A new ``SCPClient`` per call is required because the client instance
        stores the active channel and transfer state; concurrent calls on a
        shared instance cross the two streams.
        """
        with self._scp_lock:
            with self._ssh_lock:
                self._ensure_connected()
                transport = self.ssh_client.get_transport()
            client = scp.SCPClient(transport)
            try:
                client.get(remote_path, local_path)
            finally:
                client.close()

    def _scp_put(self, local_path: str, remote_path: str) -> None:
        with self._scp_lock:
            with self._ssh_lock:
                self._ensure_connected()
                transport = self.ssh_client.get_transport()
            client = scp.SCPClient(transport)
            try:
                client.put(local_path, remote_path)
            finally:
                client.close()
    
    def get_info(self) -> dict:
        return {
            "name": self.name, 
            "processing_units": self.processing_units,
            "job_envs": {key:self.job_envs[key].softwares for key in self.job_envs.keys()},
            "remote_files": [ {"description":item["description"], 
                            "files":item["files"]} for item in self.remote_files ]
        }
    
    def get_remote_files_list(self) -> List[dict]:
            return self.remote_files
    
    def get_job_env(self, job_env_name:str)->dict:
        return self.job_envs[job_env_name].model_dump(mode="python")
    
    def get_resources(self) -> List[dict]:
        return [item.model_dump(mode="python") for item in self.resources]

    async def get_job_states(self, slurm_job_ids:Set[str]) -> Dict[str, JobStatus]:
        result: Dict[str, JobStatus] = {}
        for slurm_job_id in slurm_job_ids:
            result[slurm_job_id] = await self._get_one_job_state(slurm_job_id)
        return result

    async def _get_one_job_state(self, slurm_job_id: str) -> JobStatus:
        # `scontrol` is authoritative for jobs still in the controller's job
        # table (PENDING/RUNNING/...).  It only prints one line with `-o`, so
        # the JobState can be parsed regardless of the user's shell setup.
        output = await asyncio.to_thread(
            self._run_ssh,
            f"scontrol show jobid -o {shlex.quote(slurm_job_id)}",
            retry_on_disconnect=True)
        if output is None:
            logger.error(f"Fail to get the job state of job {slurm_job_id} "
                         f"on {self.name}, SSH error.")
            return JobStatus.NONE

        stdout = "".join(output[1])
        stderr = "".join(output[2])
        raw_state = _parse_scontrol_state(stdout)
        status = _job_status_from_slurm_state(raw_state) if raw_state else None
        if status is not None:
            return status

        # Slurm purges completed jobs from scontrol after MinJobAge (5 minutes
        # by default), so a finished group looks like a nonexistent job while
        # its results are still waiting to be downloaded.  Accounting keeps
        # the history: ask sacct for the final state instead of giving up.
        acct_output = await asyncio.to_thread(
            self._run_ssh,
            f"sacct -n -X -P -j {shlex.quote(slurm_job_id)} "
            f"--format=JobID,State",
            retry_on_disconnect=True)
        if acct_output is None:
            logger.error(f"Fail to get the accounting state of job "
                         f"{slurm_job_id} on {self.name}, SSH error.")
            return JobStatus.NONE

        acct_stdout = "".join(acct_output[1])
        acct_stderr = "".join(acct_output[2])
        acct_state = _parse_sacct_state(acct_stdout, slurm_job_id)
        acct_status = (_job_status_from_slurm_state(acct_state)
                       if acct_state else None)
        if acct_status is not None:
            logger.info(f"Job {slurm_job_id} on {self.name} is {acct_state} "
                        f"according to accounting.")
            return acct_status

        if acct_state:
            # Do not treat an unrecognized transient state as terminal.
            logger.warning(f"Unrecognized slurm state {acct_state!r} for job "
                           f"{slurm_job_id}; will query it again.")
            return JobStatus.NONE
        if raw_state:
            logger.warning(f"Unrecognized slurm state {raw_state!r} for job "
                           f"{slurm_job_id}; will query it again.")
            return JobStatus.NONE

        if ("invalid job id" in stderr.lower()
                or "invalid job id" in acct_stderr.lower()):
            logger.error(f"Fail to get the job state of job {slurm_job_id} on "
                         f"{self.name}. Job may not exist.")
            return JobStatus.UNKNOWN

        logger.error(
            f"Fail to get the job state of job {slurm_job_id} on {self.name}. "
            f"scontrol stdout={stdout!r} stderr={stderr!r}; "
            f"sacct stdout={acct_stdout!r} stderr={acct_stderr!r}")
        return JobStatus.NONE

    async def cancel(self, slurm_job_id: str):
        result = await asyncio.to_thread(
            self._run_ssh, f"scancel -v {shlex.quote(slurm_job_id)}")
        if result is None:
            logger.error(f"Fail to cancel job {slurm_job_id} on {self.name}, SSH error.")
            return
        # scancel 的输出无论成功与否都在标准错误result[2]中
        if len(result[2]) == 2 and "Invalid job id" in result[2][1]:
            logger.info(f"Job {slurm_job_id} does not exist on {self.name}.")
        else:
            logger.info(f"Job {slurm_job_id} on slurm {self.name} is cancelled.")

    async def download(self, slurm_job_id:str, reomte_dir:str, local_dir:str)->bool:
        if not os.path.isdir(local_dir):
            raise NotADirectoryError(f"{local_dir} is not a directory.")

        # Do not reject the archive just because tar printed a warning (for
        # example about a file that changed while it was being read): tar
        # writes warnings to stderr even when the archive is valid.  Ask tar
        # for an explicit marker and require a non-empty archive instead.
        command = (f"cd {remote_shell_path(reomte_dir)} && "
                   f"tar -czf out.tar.gz --exclude='out.tar.gz' * && "
                   f"test -s out.tar.gz && echo __SLURM_TAR_OK__")
        result = await asyncio.to_thread(self._run_ssh, command)
        if result is None:
            logger.error(f"Cannot compress files in {reomte_dir}, SSH error.")
            return False
        if not any("__SLURM_TAR_OK__" in line for line in result[1]):
            logger.error(f"Cannot compress files in {reomte_dir}. "
                         f"Error: {''.join(result[2]).strip()}")
            return False

        # Download into a fresh temp directory.  Writing straight into
        # local_dir leaves a truncated out.tar.gz behind when SCP fails; the
        # next retry would then extract stale/mixed data.
        tmpdir = tempfile.mkdtemp(prefix="slurm_download_")
        try:
            await asyncio.to_thread(
                self._scp_get,
                os.path.join(reomte_dir, "out.tar.gz"),
                tmpdir)
            archive = os.path.join(tmpdir, "out.tar.gz")
            if not os.path.isfile(archive) or os.path.getsize(archive) == 0:
                logger.error(f"Downloaded archive for {slurm_job_id} is missing or empty.")
                return False

            # Validate the complete gzip/tar stream before touching local_dir.
            # A truncated transfer must never overwrite the previous result.
            output = await run_shell_command(
                f"tar -tzf {shlex.quote(archive)} >/dev/null", cwd=tmpdir, timeout=120)
            if output["return_code"] != 0:
                logger.error(f"Downloaded archive for {slurm_job_id} is corrupt: "
                             f"{output['stderr'].strip()}")
                return False

            # The archive is complete and verified, so result files may
            # overwrite older files (including files left by a failed
            # download).  Inputs are excluded to keep the local workdir
            # authoritative for them.
            output = await run_shell_command(
                f"tar -xzf {shlex.quote(archive)} -C {shlex.quote(local_dir)} "
                f"--exclude='*.py' --exclude='*.json' --exclude='*.md' --exclude='*.sh' "
                f"--exclude=INCAR --exclude=POSCAR --exclude='POTCAR*'",
                cwd=tmpdir, timeout=120)
            if output["return_code"] != 0:
                logger.error(f"Fail to unpack the result archive in {local_dir}. "
                             f"{output['stderr'].strip()}")
                return False
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        # A failed older download may have left this file in the workdir.
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(local_dir, "out.tar.gz"))

        logger.info(f"The result files of {slurm_job_id} is downloaded in to {local_dir}. ")
        # 清理远程的压缩包文件
        command = f"cd {remote_shell_path(reomte_dir)} && rm -f out.tar.gz"
        await asyncio.to_thread(self._run_ssh, command)
        return True 

    async def submit(self, setting:JobSetting) -> str:
        # 运行这个函数之前假设agent已经把所有的额外内容在run.sh 中修改好
        # 例如要激活哪些环境，export什么变量，NEB计算时的核数。多个任务并行时核数的分配等。
        if not os.path.isdir(setting.local_dir):
            raise NotADirectoryError(f"{setting.local_dir} is not a directory.")
        local_run_sh = os.path.join(setting.local_dir, "run.sh")
        if not os.path.isfile(local_run_sh):
            raise FileNotFoundError(f"There should be a run.sh file in directory {setting.local_dir}. ")

        # 这里要做的只是把目录上传到远端然后提交。
        remote_dir = remote_shell_path(setting.remote_dir)
        command = f"if [ -d {remote_dir} ]; then rm -f -r {remote_dir}; fi && mkdir -p {remote_dir}"
        result = await asyncio.to_thread(self._run_ssh, command)
        if result is None or not self._command_succeeded(result):
            raise RuntimeError(
                f"Fail to prepare remote directory {setting.remote_dir} on "
                f"{self.name}. Error: {self._command_error(result)}")
        # 把当前目录下的内容打包
        await run_shell_command(command="tar -czf input.tar.gz --exclude='input.tar.gz' * ",
                                cwd=setting.local_dir,
                                timeout=120)
        # 上传到远端。每次传输使用独立的 SCP 通道，避免并发上传/下载互相踩踏。
        await asyncio.to_thread(self._scp_put,
                                os.path.join(setting.local_dir, 'input.tar.gz'),
                                setting.remote_dir)
        # 在远端解压然后提交任务
        command = f"cd {remote_dir} && tar -xf input.tar.gz && rm -f input.tar.gz "
        result = await asyncio.to_thread(self._run_ssh, command)
        if not self._command_succeeded(result):
            raise RuntimeError(
                f"Fail to unpack inputs in {setting.remote_dir} on {self.name}. "
                f"Error: {self._command_error(result)}")
        # 删除本地的Input.tar.gz 
        await run_shell_command(command="rm -f input.tar.gz",
                                cwd=setting.local_dir,
                                timeout=120)

        command = f"bash -l -c 'cd {remote_dir} && sbatch run.sh'"
        result = await asyncio.to_thread(self._run_ssh, command)
        if result is None:
            raise RuntimeError(f"Fail to submit job {setting.job_name} to {self.name}. SSH error.")
        stdout = "".join(result[1])
        stderr = "".join(result[2])
        remote_id = _parse_submit_job_id(stdout)
        if remote_id is None:
            logger.error(
                f"Fail to parse the slurm job id of {setting.job_name} on "
                f"{self.name}. stdout: {stdout!r}, stderr: {stderr!r}")
            raise RuntimeError(
                f"Fail to submit job {setting.job_name} to {self.name}. "
                f"`sbatch` printed no valid job id. Stdout: {stdout!r}, "
                f"Stderr: {stderr!r}")
        logger.info(f"Job {setting.job_name} is submitted to {self.name}, "
                    f"remote id {remote_id}.")
        return remote_id
        
    async def cleanup(self, remote_dir:str):
        # 删除远端的目录
        await asyncio.to_thread(
            self._run_ssh,
            f"rm -f -r {remote_shell_path(remote_dir)}.")


    
    




