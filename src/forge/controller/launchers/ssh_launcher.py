# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SSHLauncher """

# from .launcher import get_meshes_from_config
import atexit
import collections
import os
import re
import sys

from dataclasses import dataclass, field
from typing import List

from forge.types import LauncherConfig
from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
from monarch._rust_bindings.monarch_hyperactor.config import configure
from monarch._src.job.job import SSHJob
from monarch.actor import HostMesh, ProcMesh

from .base import BaseLauncher


def fetch_hostfile(hostfile_path):
    # e.g., worker-0 slots=16
    with open(hostfile_path, "r") as fd:
        hostfile_text = fd.readlines()

    return parse_hostfile(hostfile_text)


def parse_hostfile(hostfile_lines):
    # Regex matches one or more non-whitespace characters (\S+) at the start of
    # the line, followed by one or more whitespace characters (\s+), followed
    # by the string "slots=", followed by one or more digits (\d+).
    pattern = r"^(\S+)\s+slots=(\d+)"

    resource_pool = collections.OrderedDict()

    for line in hostfile_lines:
        line = line.strip()
        match = re.search(pattern, line)
        if line.startswith("#") or line == "":
            # hostfile comment or empty line, ignore
            continue
        elif match:
            host = match.group(1)
            num_slots = int(match.group(2))
            if host in resource_pool:
                raise ValueError(
                    f"Hostfile contains multiple entries for {host}, unable to proceed with launching"
                )
            resource_pool[host] = num_slots
        else:
            raise ValueError(
                f"Hostfile contains a bad entry: {line}, unable to proceed with launching"
            )

    if len(resource_pool) == 0:
        raise ValueError(
            "Hostfile is empty or not formatted correctly, unable to proceed with launching."
        )

    return resource_pool


GENERATOR_KEY = "generator"
REFERENCE_KEY = "ref_model"
TRAINER_KEY = "trainer"
ProcMesh_KEYS = [GENERATOR_KEY, REFERENCE_KEY, TRAINER_KEY]


@dataclass
class SSHForgeProc:
    is_actor: bool = True
    num_hosts: int = 0


@dataclass
class SSHActor:
    name: str | None = None
    host_mesh: HostMesh | None = None

    def get_host_mesh(self):
        return self.host_mesh


@dataclass
class SSHService:
    name: str | None = None
    host_mesh_list: List[HostMesh] = field(default_factory=list)
    current_idx: int = 0

    def get_host_mesh(self):
        host_mesh = self.host_mesh_list[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.host_mesh_list)
        return host_mesh


def get_meshes_from_config(cfg: LauncherConfig) -> dict[str, int]:
    """Extract mesh requirements from launcher config.

    Args:
        cfg: The launcher configuration

    Returns:
        Dictionary mapping mesh names to number of hosts required
    """
    meshes: dict[str, int] = {}

    # Add services that need remote hosts
    for service_name, service_cfg in cfg.services.items():
        hosts = getattr(service_cfg, "hosts", None)
        if hosts and hosts > 0:
            mesh_name = service_cfg.mesh_name or service_name
            meshes[mesh_name] = hosts

    # Add actors that need remote hosts
    for actor_name, actor_cfg in cfg.actors.items():
        hosts = getattr(actor_cfg, "hosts", None)
        if hosts and hosts > 0:
            mesh_name = actor_cfg.mesh_name or actor_name
            meshes[mesh_name] = hosts

    return meshes


class SSHLauncher(BaseLauncher):
    def __init__(self, cfg: LauncherConfig):
        self.cfg = cfg
        self.job = None
        self.state = None
        self.ssh_hostfile = cfg.ssh_hostfile
        self.host_pool = fetch_hostfile(self.ssh_hostfile)
        self.host_ips = list(self.host_pool.keys())

        self.meshes = get_meshes_from_config(self.cfg)
        print(f"sshlauncher: {self.meshes=}")

        self.forge_proc_map = self._create_forge_proc_map(cfg)
        n_generator_slice_end = self.forge_proc_map[GENERATOR_KEY].num_hosts
        n_trainer_slice_end = (
            n_generator_slice_end + self.forge_proc_map[TRAINER_KEY].num_hosts
        )
        n_ref_model_slice_start = (
            n_generator_slice_end
            if cfg.colocate_ref_and_trainer
            else n_trainer_slice_end
        )
        n_ref_model_slice_end = (
            n_ref_model_slice_start + self.forge_proc_map[REFERENCE_KEY].num_hosts
        )

        self.host_ip_map = {
            GENERATOR_KEY: self.host_ips[:n_generator_slice_end],
            TRAINER_KEY: self.host_ips[n_generator_slice_end:n_trainer_slice_end],
            REFERENCE_KEY: self.host_ips[n_ref_model_slice_start:n_ref_model_slice_end],
        }

        self.host_mesh_map = dict()

    def _create_forge_proc_map(self, cfg: LauncherConfig) -> dict:
        proc_map = {}
        for key in ProcMesh_KEYS:  # self.meshes:
            if key in cfg.services:
                proc_map[key] = SSHForgeProc(
                    is_actor=False, num_hosts=cfg.services[key].hosts
                )
            else:
                proc_map[key] = SSHForgeProc(
                    is_actor=True, num_hosts=cfg.actors[key].hosts
                )

        return proc_map

    async def _setup_job(self):
        job = SSHJob(
            python_exe=os.getenv("PYTHON_EXE", sys.executable),
            ssh_args=[
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "BatchMode=yes",
            ],
            monarch_port=int(
                os.getenv("MONARCH_PORT", "22222")
            ),  # make sure this is open
        )

        # for proc_key in self.meshes:
        for proc_key in ProcMesh_KEYS:
            # job.add_mesh(f"{proc_key}_mesh", self.host_ip_map[proc_key])
            job.add_mesh(f"{proc_key}", self.host_ip_map[proc_key])

        return job

    async def get_host_mesh(self, name: str, num_hosts: int) -> tuple[HostMesh, str]:
        for key in self.host_mesh_map.keys():
            if key in name:
                host_mesh = self.host_mesh_map[key].get_host_mesh()
                return host_mesh, None

        return None, None

    def _create_actor_or_service(self, mesh_name, job_host_mesh):
        if self.forge_proc_map[mesh_name].is_actor:
            return SSHActor(name=mesh_name, host_mesh=job_host_mesh)

        return SSHService(
            name=mesh_name,
            host_mesh_list=[
                job_host_mesh.slice(hosts=slice(i, i + 1))
                for i in range(len(self.host_ip_map[mesh_name]))
            ],
        )

    async def initialize(self) -> None:
        # HostMesh currently requires explicit configuration
        # of the underlying transport from client to mesh.
        # This can be removed in the future once this has been removed.
        configure(default_transport=ChannelTransport.TcpWithHostname)

        self.job = await self._setup_job()
        self.state = self.job.state()

        self.host_mesh_map[TRAINER_KEY] = self._create_actor_or_service(
            mesh_name=TRAINER_KEY,
            job_host_mesh=self.state.trainer,
            # job_host_mesh=self.state.trainer_mesh,
        )
        self.host_mesh_map[GENERATOR_KEY] = self._create_actor_or_service(
            mesh_name=GENERATOR_KEY,
            job_host_mesh=self.state.generator,
            # job_host_mesh=self.state.generator_mesh,
        )
        self.host_mesh_map[REFERENCE_KEY] = self._create_actor_or_service(
            mesh_name=REFERENCE_KEY,
            job_host_mesh=self.state.ref_model,
            # job_host_mesh=self.state.ref_model_mesh,
        )

        # Register cleanup handler
        atexit.register(self.job.kill)

        print(f"ssh_launcher[init]: {self.ssh_hostfile=} {self.host_ips=}")
        print(f"ssh_launcher[init]: {self.host_ip_map=}")
        return self.job, self.state

    async def remote_setup(self, procs: ProcMesh) -> None:
        return

    @classmethod
    def validate_configuration(cls, cfg: LauncherConfig) -> tuple[bool, str]:

        error_msg_prefix = "SSH Launcher configuration error:"
        # Check that a valid hostfile is specified.
        if cfg.ssh_hostfile is None or not os.path.isfile(cfg.ssh_hostfile):
            error_msg = f"{error_msg_prefix} Invalid hostfile path {cfg.ssh_hostfile}."
            return False, error_msg

        # Check that all services/actors which use gpus also specify host count
        gpu_host_count = {
            name: service.hosts
            for name, service in cfg.services.items()
            if service.with_gpus
        }
        gpu_host_count.update(
            {name: actor.hosts for name, actor in cfg.actors.items() if actor.with_gpus}
        )
        missing_hosts = [
            name for name, count in gpu_host_count.items() if count is None or count < 1
        ]
        if len(missing_hosts) > 0:
            error_msg = f"{error_msg_prefix} missing host counts for services/actors that need gpus {missing_hosts}"
            return False, error_msg

        # Check that sufficient hosts available
        available_host_count = len(fetch_hostfile(cfg.ssh_hostfile))
        required_host_count = sum(list(gpu_host_count.values()))

        if (
            cfg.colocate_ref_and_trainer
            and TRAINER_KEY in gpu_host_count
            and REFERENCE_KEY in gpu_host_count
        ):
            excess_gpu_count = min(
                gpu_host_count[TRAINER_KEY], gpu_host_count[REFERENCE_KEY]
            )
            required_host_count -= excess_gpu_count

        if required_host_count > available_host_count:
            error_msg = f"{error_msg_prefix} {required_host_count=} exceeds {available_host_count=}"
            return False, error_msg

        return True, None
