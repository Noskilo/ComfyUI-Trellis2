import math

import torch
import torch.nn as nn

from .. import SparseTensor


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    stride_tuple = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
    if any(s != 1 for s in stride_tuple) or padding is not None:
        raise NotImplementedError("torch_native sparse conv fallback only supports submanifold stride=1, padding=None convolutions.")

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
    self.stride = stride_tuple
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3
    self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)


def _linearize_coords(coords: torch.Tensor, spatial_shape: torch.Size) -> torch.Tensor:
    dims = torch.tensor(
        [int(spatial_shape[0]), int(spatial_shape[1]), int(spatial_shape[2])],
        dtype=torch.long,
        device=coords.device,
    )
    coords = coords.long()
    return (((coords[:, 0] * dims[0] + coords[:, 1]) * dims[1] + coords[:, 2]) * dims[2] + coords[:, 3])


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    coords = x.coords
    feats = x.feats
    spatial_shape = x.spatial_shape
    keys = _linearize_coords(coords, spatial_shape)
    sorted_keys, order = torch.sort(keys)

    out = feats.new_zeros((feats.shape[0], self.out_channels))
    center = [k // 2 for k in self.kernel_size]

    for kz in range(self.kernel_size[0]):
        for ky in range(self.kernel_size[1]):
            for kx in range(self.kernel_size[2]):
                offset = torch.tensor(
                    [
                        0,
                        (kz - center[0]) * self.dilation[0],
                        (ky - center[1]) * self.dilation[1],
                        (kx - center[2]) * self.dilation[2],
                    ],
                    dtype=coords.dtype,
                    device=coords.device,
                )
                neighbor_coords = coords + offset
                valid = (
                    (neighbor_coords[:, 1] >= 0)
                    & (neighbor_coords[:, 2] >= 0)
                    & (neighbor_coords[:, 3] >= 0)
                    & (neighbor_coords[:, 1] < spatial_shape[0])
                    & (neighbor_coords[:, 2] < spatial_shape[1])
                    & (neighbor_coords[:, 3] < spatial_shape[2])
                )
                if not torch.any(valid):
                    continue

                neighbor_keys = _linearize_coords(neighbor_coords[valid], spatial_shape)
                pos = torch.searchsorted(sorted_keys, neighbor_keys)
                found = (pos < sorted_keys.numel()) & (sorted_keys[pos.clamp_max(sorted_keys.numel() - 1)] == neighbor_keys)
                if not torch.any(found):
                    continue

                out_indices = torch.nonzero(valid, as_tuple=False).flatten()[found]
                in_indices = order[pos[found]]
                kernel = self.weight[:, :, kz, ky, kx]
                out[out_indices] += feats[in_indices].matmul(kernel.t())

    if self.bias is not None:
        out += self.bias

    return x.replace(out)


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError("SparseInverseConv3d is not implemented for torch_native fallback.")


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError("SparseInverseConv3d is not implemented for torch_native fallback.")
