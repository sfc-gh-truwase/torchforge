# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base Launcher class"""

from typing import Any
import monarch
from monarch.actor import ProcMesh, HostMesh


class BaseLauncher:
    async def initialize(self) -> None:
        pass

    async def get_allocator(self, name: str, num_hosts: int) -> tuple[Any, Any, str]:
        pass

    async def get_host_mesh(self, name: str, num_hosts: int) -> tuple[HostMesh, str]:
        alloc, alloc_constraints, server_name = await self.get_allocator(
            name, num_hosts
        )

        # We are asking Monarch to allocate a single process on
        # every host, reflected in the Extent we provide below.

        # Technically, this is ["hosts", "procs"] but to reduce
        # confusion on its relationship with procs elsewhere,
        # we call it "no_dim".

        # TODO - remove this once Monarch supports HostMesh without it.
        host_mesh = HostMesh.allocate_nonblocking(
            name=name,
            extent=Extent(["hosts", "no_dim"], [num_hosts, 1]),
            allocator=alloc,
            alloc_constraints=alloc_constraints,
        )
        return host_mesh, server_name        

    async def remote_setup(self, procs: ProcMesh) -> None:
        pass

