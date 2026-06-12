from __future__ import annotations

from contextlib import nullcontext
from importlib import import_module
from typing import Optional, Union

import torch


def _has_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def resolve_device(device: Union[str, torch.device, None] = "auto") -> torch.device:
    if device is None:
        device = "auto"
    if isinstance(device, torch.device):
        return device

    name = str(device).lower()
    if name == "auto":
        if _has_xpu():
            return torch.device("xpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "xpu" and not _has_xpu():
        raise RuntimeError("Intel XPU device requested, but torch.xpu.is_available() is false.")
    if name == "cuda" and not torch.cuda.is_available():
        if _has_xpu():
            print("[Trellis2] CUDA was requested by the workflow, but CUDA is unavailable. Falling back to Intel XPU.")
            return torch.device("xpu")
        print("[Trellis2] CUDA was requested by the workflow, but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def device_type(device: Union[str, torch.device, None]) -> str:
    return resolve_device(device).type


def is_xpu(device: Union[str, torch.device, None]) -> bool:
    return device_type(device) == "xpu"


def is_cuda(device: Union[str, torch.device, None]) -> bool:
    return device_type(device) == "cuda"


def synchronize(device: Union[str, torch.device, None]) -> None:
    dtype = device_type(device)
    if dtype == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif dtype == "xpu" and _has_xpu():
        torch.xpu.synchronize()


def empty_cache(device: Union[str, torch.device, None]) -> None:
    dtype = device_type(device)
    if dtype == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif dtype == "xpu" and _has_xpu() and hasattr(torch.xpu, "empty_cache"):
        torch.xpu.empty_cache()


def cleanup(device: Union[str, torch.device, None]) -> None:
    synchronize(device)
    empty_cache(device)


def manual_seed_all(seed: int, device: Union[str, torch.device, None] = "auto") -> None:
    resolved = resolve_device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif resolved.type == "xpu" and _has_xpu() and hasattr(torch.xpu, "manual_seed_all"):
        torch.xpu.manual_seed_all(seed)


def autocast(device: Union[str, torch.device, None], dtype: Optional[torch.dtype] = None, enabled: bool = True):
    resolved = resolve_device(device)
    if resolved.type == "cpu" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type=resolved.type, dtype=dtype, enabled=enabled)


def optional_import(module_name: str, feature: str):
    try:
        return import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"{feature} requires optional dependency '{module_name}', which is not available in this environment."
        ) from exc


def require_cuda_optional(module_name: str, feature: str):
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{feature} currently requires the CUDA-only optional dependency '{module_name}'. "
            "Use a CPU fallback node/option on Intel Arc."
        )
    return optional_import(module_name, feature)
