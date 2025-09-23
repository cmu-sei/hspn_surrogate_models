#!/usr/bin/env python3
"""Primary driver for starting ray on slurm.

# Bare metal
$ salloc --qos=standard --account=<account> --time=01:00:00 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
# Now on assigned node
$ cd "$SLURM_SUBMIT_DIR"
$ uv run python -u cluster/ray_head.py --dry-run
$ uv run python -u cluster/ray_head.py
# New shell (can be same node or any host with route to node)
$ ssh <assigned node>
$ uv run --extra hpo -m hspn.train --config-name=ray_optuna_hpo.yaml

# Apptainer/Singularity
$ salloc --time=01:00:00 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --account=<account> --qos=standard
# Now on assigned node
$ cd "$SLURM_SUBMIT_DIR"

# Start Ray head inside a SIF.
# --container=auto activates container mode when --sif is provided.
$ python -u cluster/ray_head.py \
    --container auto \
    --sif /path/to/hspn.sif \
    --project-root "$SLURM_SUBMIT_DIR" \
    --ray-tmpdir "$SLURM_SUBMIT_DIR/.ray_tmp" \
    --dry-run

$ python -u cluster/ray_head.py \
    --container auto \
    --sif /path/to/hspn.sif \
    --project-root "$SLURM_SUBMIT_DIR" \
    --ray-tmpdir "$SLURM_SUBMIT_DIR/.ray_tmp"

# New shell
$ ssh <assigned node>
$ # TODO: explain this
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import logging
import os
import shlex
import signal
import socket
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING, Any, Dict, Final, Iterable, List, Literal, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from types import FrameType

JSON_DEFAULT_PATH: Final[Path] = Path(".ray-head.json")
PIDFILE_DEFAULT_PATH: Final[Path] = Path(".ray-head.pid")
DEFAULT_IFACE_ORDER: Final[Tuple[str, ...]] = ("ib0", "enp1s0f0", "enp1s0f1", "eno1")

_SIOCGIFADDR: Final[int] = 0x8915  # <linux/sockios.h>
_IFNAMSIZ: Final[int] = 16

ContainerKind = Literal["none", "apptainer", "singularity", "auto"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr),
    ],
)

logging.getLogger().handlers[0].setLevel(logging.DEBUG)
logging.getLogger().handlers[1].setLevel(logging.WARNING)


@dataclass(frozen=True)
class SlurmInfo:
    """Job metadata."""

    job_id: Optional[str]
    node_name: Optional[str]
    submit_dir: Optional[str]

    @classmethod
    def detect(cls) -> SlurmInfo:
        return cls(
            job_id=os.environ.get("SLURM_JOB_ID"),
            node_name=os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST"),
            submit_dir=os.environ.get("SLURM_SUBMIT_DIR"),
        )


@dataclass(frozen=True)
class RayHeadRecord:
    """Ray head metadata."""

    hostname: str
    ip: str
    port: int
    ray_address: str
    created_at: str
    pid: int
    slurm: SlurmInfo
    container: Dict[str, Any]


def infer_num_cpus() -> Optional[int]:
    """Prefer SLURM_CPUS_PER_TASK, fallback to os.cpu_count()."""
    s_cpt = os.environ.get("SLURM_CPUS_PER_TASK")
    if s_cpt:
        try:
            n = int(s_cpt)
            if n > 0:
                return n
        except ValueError:
            pass
    return os.cpu_count()


def infer_num_gpus() -> Optional[int]:
    """Check CUDA_VISIBLE_DEVICES.

    - unset => None (we dont assume anything, let something else discover devices)
    - empty / unset => 0
    - '-1' => 0
    - otherwise length of the found comma-separated entries
    """
    cvd_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd_env is None:
        return None
    cvd = (cvd_env or "").strip()
    if not cvd or cvd == "-1":
        return 0
    toks = [t for t in cvd.split(",") if t.strip() != ""]
    return len(toks) or None


def _ioctl_ipv4_for_iface(sock: socket.socket, iface: str) -> Optional[str]:
    """Use SIOCGIFADDR to obtain IPv4 for a specific interface (assumes Linux)."""
    if len(iface) >= _IFNAMSIZ:
        return None
    ifreq = struct.pack("16sH14s", iface.encode("ascii"), socket.AF_INET, b"\x00" * 14)
    try:
        res = fcntl.ioctl(sock.fileno(), _SIOCGIFADDR, ifreq)
        sa = res[16 : 16 + 16]
        _family, _port, inaddr = struct.unpack("!HH4s", sa[:8])
        del _family, _port
        return socket.inet_ntoa(inaddr)
    except OSError:
        return None


def _list_sysfs_ifaces() -> List[str]:
    sysfs = Path("/sys/class/net")
    if not sysfs.is_dir():
        return []
    ifaces = [p.name for p in sysfs.iterdir() if p.is_dir()]
    return [i for i in ifaces if i != "lo"]


def _first_addr_in_getaddrinfo(hostname: str) -> Optional[str]:
    try:
        for fam, _, _, _, sockaddr in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            del fam
            ip = sockaddr[0]
            assert isinstance(ip, str)
            if not ip.startswith(("127.", "0.0.0.0")):
                return ip
    except socket.gaierror:
        return None
    return None


def resolve_head_ip(preferred: Iterable[str]) -> str:
    """Try preferred interfaces first via ioctl, then any sysfs iface, then getaddrinfo(hostname)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for iface in preferred:
            ip = _ioctl_ipv4_for_iface(s, iface)
            if ip and not ip.startswith(("127.", "0.0.0.0")):
                return ip
        for iface in _list_sysfs_ifaces():
            ip = _ioctl_ipv4_for_iface(s, iface)
            if ip and not ip.startswith(("127.", "0.0.0.0")):
                return ip
    hn = socket.gethostname()
    ip = _first_addr_in_getaddrinfo(hn)
    if ip:
        return ip
    raise RuntimeError("Could not resolve a non-loopback IPv4 address.")


def _parse_env_nulls(env_blob: bytes) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for entry in env_blob.split(b"\x00"):
        if not entry:
            continue
        # Split on first '=' only. value can contain '='
        k, _, v = entry.partition(b"=")
        env[k.decode("utf-8")] = v.decode("utf-8")
    return env


def _apply_env_delta(new_env: Dict[str, str]) -> None:
    # Simple replace: bring current env in sync with new_env
    # (Avoid deleting existing keys not present in new_env to be conservative.)
    for k, v in new_env.items():
        os.environ[k] = v


def maybe_load_module(name: str) -> bool:
    """Best-effort: try to `module load <name>` in a login shell and import resulting env."""
    bash = which("bash")
    if not bash:
        return False
    # Use a login shell (-l) to pick up /etc/profile.d/modules.sh then emit env as NUL-separated.
    cmd = [bash, "-lc", f"module load {shlex.quote(name)} >/dev/null 2>&1 && env -0 || env -0"]
    try:
        out = subprocess.check_output(cmd)
        new_env = _parse_env_nulls(out)
        _apply_env_delta(new_env)
    except Exception:
        return False
    else:
        return True


def resolve_container_exe(preferred: Sequence[str]) -> Optional[str]:
    """Find apptainer/singularity, attempt module load if missing."""
    for exe in preferred:
        path = which(exe)
        if path:
            return exe
    # Try module load for each, then check again.
    for exe in preferred:
        if maybe_load_module(exe):
            path = which(exe)
            if path:
                logging.info(f"Loaded module and found '{exe}' at {path}")
                return exe
    return None


def ensure_cuda() -> None:
    """If GPUs requested and nvidia-smi isn't visible, try to module-load CUDA/NVIDIA."""
    if which("nvidia-smi"):
        return
    for mod in ("cuda", "nvidia"):
        if maybe_load_module(mod) and which("nvidia-smi"):
            logging.info(f"Enabled GPU tools via 'module load {mod}'")
            return
    logging.warning(
        "nvidia-smi not found, proceeding without it (Apptainer --nv may still bind drivers).",
    )


def build_container_prefix(
    kind: ContainerKind,
    sif: Optional[Path],
    project_root: Path,
    ray_tmpdir: Path,
    gpu: bool,
) -> Tuple[List[str], Dict[str, Any]]:
    """Return (argv_prefix, meta) for container execution."""
    meta: Dict[str, Any] = {"enabled": False}

    if kind == "none":
        return [], meta

    # 'auto' => enable only if SIF provided
    if kind == "auto":
        if sif is None:
            return [], {"enabled": False, "reason": "no SIF provided"}
        resolved = resolve_container_exe(("apptainer", "singularity"))
        if not resolved:
            logging.warning("Container 'auto' requested but no apptainer/singularity found, running bare.")
            return [], {"enabled": False, "reason": "container runtime not found"}
        kind = "apptainer" if resolved == "apptainer" else "singularity"  # narrow type

    if kind in ("apptainer", "singularity"):
        exe = resolve_container_exe((kind,))
        if not exe:
            logging.warning(f"Requested container '{kind}' but not found on PATH, trying module load.")
            exe = resolve_container_exe((kind,))
        if not exe:
            logging.warning(f"Could not find '{kind}'. Running bare.")
            return [], {"enabled": False, "reason": f"{kind} missing"}

        if sif is None:
            raise SystemExit(f"--sif is required when --container {kind} is used.")

        # Ensure paths exist
        project_root = project_root.resolve()
        ray_tmpdir = ray_tmpdir.resolve()
        ray_tmpdir.mkdir(parents=True, exist_ok=True)

        nv_flag: List[str] = ["--nv"] if gpu else []
        prefix: List[str] = [
            exe,
            "exec",
            *nv_flag,
            "--bind",
            f"{project_root!s}",
            "--pwd",
            f"{project_root!s}",
            "--no-home",
            "--writable-tmpfs",
            "--bind",
            f"{ray_tmpdir!s}:{ray_tmpdir!s}",
            str(sif),
        ]
        meta = {
            "enabled": True,
            "runtime": exe,
            "sif": str(sif),
            "binds": [str(project_root), f"{ray_tmpdir!s}:{ray_tmpdir!s}"],
            "nv": gpu,
        }
        return prefix, meta

    # Fallback
    return [], meta


def start_ray(
    ip: str,
    port: int,
    container_prefix: Optional[Sequence[str]] = None,
    env: Optional[Dict[str, str]] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
    head: bool = True,
) -> subprocess.Popen[bytes]:
    """Launch Ray subprocess (optionally inside container) inheriting tty."""
    cmd: List[str] = list(container_prefix or [])
    cmd = [
        "uv",
        "run",
        "ray",
        "start",
        f"--node-ip-address={ip}",
        f"--port={port}",
        "--block",
    ]
    if num_cpus:
        cmd.append(f"--num-cpus={num_cpus}")
    if num_gpus:
        cmd.append(f"--num-gpus={num_gpus}")
    if head:
        cmd.extend(("--head", "--blocK"))
    logging.info("Exec: " + " ".join(shlex.quote(c) for c in cmd))
    return subprocess.Popen(cmd, env=env)


def ray_stop(ray_address: str, container_prefix: Optional[Sequence[str]], env: Dict[str, str]) -> None:
    """Best-effort cleanup. Errors are ignored."""
    with contextlib.suppress(Exception):
        prefix = container_prefix or []
        subprocess.run(
            [*prefix, "uv", "run", "ray", "stop", f"--address={ray_address}", "--force"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )


def forward_signals(
    proc: subprocess.Popen[bytes],
    sigs: Iterable[signal.Signals] = (signal.SIGINT, signal.SIGTERM),
    to_process_group: bool = False,
) -> None:
    """Forward selected signals to the child (or its process group).

    If `to_process_group=True`, launch the child with `preexec_fn=os.setsid`
    and we forward to the whole group via `os.killpg`.
    """

    def _sig_handler(signum: int, _: Optional[FrameType]) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            if to_process_group:
                # Requires Popen(..., preexec_fn=os.setsid)
                os.killpg(proc.pid, signum)
            else:
                proc.send_signal(signum)

    for s in sigs:
        with contextlib.suppress(ValueError, OSError, RuntimeError):
            # ValueError: invalid signal on this platform, or not catchable
            # OSError/RuntimeError: certain signals cannot be set
            signal.signal(s, _sig_handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ray_ctrl",
        description="Launch a Ray cluster in a SLURM job, metadata is stored in ray-head.json.",
    )
    parser.add_argument("--port", type=int, default=6379, help="Ray GCS port (default: 6379).")
    parser.add_argument(
        "--iface-order",
        type=str,
        default=",".join(DEFAULT_IFACE_ORDER),
        help=f"Comma-separated interface preference (default: {','.join(DEFAULT_IFACE_ORDER)}).",
    )
    parser.add_argument("--outfile", type=Path, default=JSON_DEFAULT_PATH, help="JSON output path.")
    parser.add_argument("--pidfile", type=Path, default=PIDFILE_DEFAULT_PATH, help="PID file path.")
    parser.add_argument("--cpus", type=int, required=False, default=None, help="Override --num-cpus.")
    parser.add_argument("--gpus", type=int, required=False, default=None, help="Override --num-gpus.")

    parser.add_argument(
        "--container",
        type=str,
        choices=["none", "apptainer", "singularity", "auto"],
        default="none",
        help="Container runtime (default: none). Use 'auto' to use container only if --sif is provided.",
    )
    parser.add_argument(
        "--container-cmd",
        type=str,
        default=None,
        help="Use a custom container command. Will be used as a prefix.",
    )
    parser.add_argument(
        "--sif",
        type=Path,
        default=None,
        help="Path to SIF image (required if --container is apptainer/singularity, used if --container=auto).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root to bind and use as working dir inside container. Defaults to $PROJECT_ROOT or "
        "$SLURM_SUBMIT_DIR or CWD.",
    )
    parser.add_argument(
        "--ray-tmpdir",
        type=Path,
        default=None,
        help="Host path to use for RAY_TMPDIR (bind-mounted if containerized). "
        "Defaults to $RAY_TMPDIR or <project-root>/.ray_tmp.",
        # "Can be <src> or <src>:<dest>. "
    )

    parser.add_argument("--dry-run", action="store_true", help="Compute values, write JSON, exit.")
    args = parser.parse_args(argv)

    if args.container in ("none", "auto") and not which("ray"):
        logging.warning(
            "'ray' not found on PATH, if you intend to run inside container, "
            "pass --container and --sif or --container-cmd.",
        )
    if args.container_cmd and (args.container != "none" or args.sif):
        raise ValueError("--container-cmd option is mututally exclusive with --container/--sif")

    preferred = [x.strip() for x in args.iface_order.split(",") if x.strip()]
    ip = resolve_head_ip(preferred)
    port = args.port
    hostname = socket.gethostname()

    num_cpus = args.cpus if args.cpus is not None else infer_num_cpus()
    num_gpus = args.gpus if args.gpus is not None else infer_num_gpus()
    enable_gpu = num_gpus is None or num_gpus > 0

    if enable_gpu:
        ensure_cuda()

    project_root = (
        args.project_root or Path(os.environ.get("PROJECT_ROOT") or os.environ.get("SLURM_SUBMIT_DIR") or os.getcwd())
    ).resolve()
    ray_tmpdir = (args.ray_tmpdir or Path(os.environ.get("RAY_TMPDIR") or (project_root / ".ray_tmp"))).resolve()
    ray_tmpdir.mkdir(parents=True, exist_ok=True)

    container_prefix: Optional[List[str]] = None
    cmeta = {}
    if args.container_cmd is not None:
        container_prefix = shlex.split(args.container_cmd)
    if not args.container_cmd:
        container_prefix, cmeta = build_container_prefix(
            kind=args.container,
            sif=args.sif,
            project_root=project_root,
            ray_tmpdir=ray_tmpdir,
            gpu=enable_gpu,
        )

    # env for spawned process
    env: Dict[str, str] = dict(os.environ)
    env["RAY_TMPDIR"] = str(ray_tmpdir)

    ray_address = f"{ip}:{port}"
    slurm_info = SlurmInfo.detect()
    rec = RayHeadRecord(
        hostname=hostname,
        ip=ip,
        port=port,
        ray_address=ray_address,
        created_at=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(),
        slurm=slurm_info,
        container=cmeta,
    )

    code = 0
    try:
        logging.info(f"JSON: {args.outfile}")
        logging.info(f"PID : {args.pidfile}")
        logging.info(f"RAY_ADDRESS={ray_address}")
        logging.info(f"CPUs={num_cpus}  GPUs={num_gpus}")
        if args.container_cmd:
            logging.info(f"Using provided container cmd: {args.container_cmd}")
        elif cmeta.get("enabled"):
            logging.info(f"Container: {cmeta['runtime']}  SIF={cmeta.get('sif')}  --nv={cmeta.get('nv')}")

        # If running bare and ray is missing, fail early.
        if not cmeta.get("enabled") and not which("ray"):
            logging.error("'ray' not found and no container runtime configured.")
            return 127

        if args.dry_run:
            logging.info("Container prefix: %s", container_prefix)
            return 0

        args.outfile.parent.mkdir(parents=True, exist_ok=True)
        args.outfile.write_text(json.dumps(asdict(rec), indent=2) + "\n", encoding="utf-8")
        args.pidfile.write_text(str(os.getpid()), encoding="utf-8")

        try:
            proc = start_ray(
                ip=ip,
                port=port,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                container_prefix=container_prefix,
                env=env,
                head=True,
            )
            # propagate signals, not using pg here as we will defer to ray (if you see immortals consider pg)
            forward_signals(proc, sigs=signal.valid_signals(), to_process_group=False)
            code = proc.wait()
        finally:
            with contextlib.suppress(Exception):
                time.sleep(1.0)
                ray_stop(ray_address, container_prefix=container_prefix, env=env)
    finally:
        with contextlib.suppress(Exception):
            args.pidfile.unlink(missing_ok=True)
            args.outfile.unlink(missing_ok=True)

    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
