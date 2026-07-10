# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Localized launch across SM partitions (cuda.core green contexts) on one GPU."""

import numpy as np

import warp as wp

BLOCKSIZE = wp.constant(2)


@wp.kernel
def scale_tiles(A: wp.array2d(dtype=float), C: wp.array2d(dtype=float)):
    x, y = wp.tid()
    a = wp.tile_load(A, shape=(BLOCKSIZE, BLOCKSIZE), offset=(BLOCKSIZE * x, BLOCKSIZE * y))
    b = a * 2.0
    wp.tile_store(C, b, offset=(BLOCKSIZE * x, BLOCKSIZE * y))


nx = 12
ny = 8

A_host = np.arange(nx * ny, dtype=np.float32).reshape(nx, ny)
A = wp.array(A_host, dtype=float, device="cuda:0")
C = wp.zeros((nx, ny), dtype=float, device="cuda:0")

# One stream per SM partition of cuda:0
streams = wp.green_places(n_places=4, sms_per_place=8)
print("places:", [str(s.device) for s in streams])
assert all(s.green_context.is_green for s in streams)

policy = wp.blocked(dim=(nx // BLOCKSIZE, ny // BLOCKSIZE), places=len(streams))

wp.launch_tiled_localized(
    scale_tiles,
    dim=(nx // BLOCKSIZE, ny // BLOCKSIZE),
    inputs=[A],
    outputs=[C],
    block_dim=32,
    mapping=policy,
    streams=streams,
)

wp.synchronize()
np.testing.assert_allclose(C.numpy(), 2.0 * A_host)
print("green places localized launch: OK")
