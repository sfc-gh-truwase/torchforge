# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launcher factory"""


from forge.types import Launcher, LauncherConfig

from .base import BaseLauncher
from .slurm_launcher import Slurmlauncher
from .ssh_launcher import SSHLauncher

JOB_NAME_KEY = "job_name"
LAUNCHER_KEY = "launcher"


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


def get_launcher(cfg: LauncherConfig | None = None) -> BaseLauncher | None:
    if not cfg:
        return None
    if cfg.launcher == Launcher.SLURM:
        return Slurmlauncher(cfg)
    elif cfg.launcher == Launcher.SSH:
        success, error_msg = SSHLauncher.validate_configuration(cfg)
        if not success:
            raise ValueError(error_msg)
        return SSHLauncher(cfg)
    elif cfg.launcher == Launcher.MAST:
        try:
            from forge.fb.mast_launcher import MastLauncher

            return MastLauncher(cfg, detached=False)
        except ImportError as err:
            raise ValueError("MAST is not available, cannot launch MAST jobs.") from err

    else:
        raise ValueError(f"Unsupported config provided, got {cfg}")
