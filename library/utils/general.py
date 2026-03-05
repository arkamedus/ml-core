# utils/general.py
"""
Utility helpers for PyTorch models.

- No environment variable mutation unless explicitly necessary.
- No implicit device switching beyond what's requested.
- Clear, consistent docstrings and type hints.
"""

from __future__ import annotations

import gc
import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

# Matplotlib is an optional dependency for plotting utilities.
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


DeviceLike = Union[None, str, torch.device]
FigSize = Tuple[float, float]


# ────────────────────────────── Model inspection ──────────────────────────────
def print_model_params_count(model: torch.nn.Module) -> None:
    """
    Print parameter counts for a PyTorch model.

    Args:
        model: A torch.nn.Module instance.

    Prints:
        - Total parameters (trainable + non-trainable)
        - Trainable parameters (requires_grad=True)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


# ────────────────────────────────── Cleanup ───────────────────────────────────
def cleanup() -> None:
    """
    Best-effort cleanup for long-running training/inference sessions.

    - Clears CUDA cache if available.
    - Clears MPS cache if available.
    - Runs Python GC.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # On some PyTorch versions, torch.backends.mps.is_available exists even when
    # torch.mps doesn't. Guard both for robustness.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    gc.collect()


# ────────────────────────────────── Device ────────────────────────────────────
def get_device(device: DeviceLike = None) -> torch.device:
    """
    Resolve a torch.device from a user preference.

    Fixes issues in the original code:
    - The MPS logic was unreachable/incorrect (nested contradictory checks).
    - Environment variables were being set unconditionally and incorrectly
      (e.g., "Limit to GPU 1" while setting CUDA_VISIBLE_DEVICES="0").
    - Returned device could ignore the input 'device'.

    Behavior:
    - If `device` is a torch.device: returned as-is.
    - If `device` is a string: it is honored (e.g., "cuda:1", "cpu", "mps"),
      but validated against availability with sensible fallbacks.
    - If `device` is None: prefer CUDA, then MPS, then CPU.

    Args:
        device: None | "cpu" | "cuda" | "cuda:1" | "mps" | torch.device

    Returns:
        A resolved torch.device.
    """
    # Honor explicit torch.device
    if isinstance(device, torch.device):
        return device

    # Normalize string device hints
    requested = str(device).lower() if isinstance(device, str) else None

    # Helper: choose best available if no explicit request
    def _auto_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # If not specified, pick best available
    if requested is None or requested.strip() == "":
        dev = _auto_device()
    else:
        # Explicit requests with validation + fallback
        if requested.startswith("cuda"):
            if torch.cuda.is_available():
                # Allow "cuda" or "cuda:N"
                try:
                    dev = torch.device(requested)
                except Exception:
                    dev = torch.device("cuda")
            else:
                dev = torch.device("cpu")
        elif requested == "mps":
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                dev = torch.device("mps")
            else:
                dev = torch.device("cpu")
        elif requested == "cpu":
            dev = torch.device("cpu")
        else:
            # Unknown string -> try torch.device parsing, else fallback
            try:
                dev = torch.device(requested)
            except Exception:
                dev = _auto_device()

    # Optional: set matmul precision where supported (safe, no device forcing)
    # This API exists in newer torch; keep it best-effort.
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # If using MPS, you may optionally set the fallback env var (safe default: allow fallback).
    # Do not aggressively disable fallback here; that can break runs unexpectedly.
    if dev.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    return dev


# ───────────────────────────────── Plotting ───────────────────────────────────
import matplotlib.pyplot as plt
import torch

# helper: flatten anything into 2D for visualization
def _to_2d(t: torch.Tensor):
    if t.dim() == 0:              # scalar → make a (1,1) matrix
        return t.view(1, 1)
    elif t.dim() == 1:            # vector → make a single row
        return t.unsqueeze(0)     # shape (1, N)
    elif t.dim() == 2:            # already a matrix
        return t
    else:                         # 3D+ → flatten last dims
        return t.view(t.size(0), -1)

def plot_model_matrices(model,
                        *,
                        figsize=(12, 22),
                        cmap="viridis",
                        max_plots=400):
    mats = []

    # grab *all* parameters (including 0D/1D/2D+)
    for name, p in model.named_parameters():
        if (".bias" not in name):
            mats.append((name, _to_2d(p.detach().cpu())))

    if not mats:
        print("no parameters found")
        return

    # layout
    mats = mats[:max_plots]
    n    = len(mats)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for ax, (title, mat) in zip(axes, mats):
        im = ax.imshow(mat, aspect='auto', cmap=cmap)
        ax.set_title(f"{title}\n{list(mat.shape)}", fontsize=7)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01)

    for ax in axes[len(mats):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


import torch

def load_partial_state_dict(model: torch.nn.Module,
                            checkpoint_path: str,
                            map_location=None) -> None:
    """
    Load as many parameters as possible from `checkpoint_path` into `model`.
    Any keys in the checkpoint that either
      • don’t exist in the model, or
      • have a mismatched shape
    will be skipped, leaving those parameters at their current value.
    Prints a summary of loaded / missing / unexpected keys.
    """

    # 1) load the checkpoint
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    # allow both bare state_dicts and full checkpoints
    state_dict = ckpt.get("model_state_dict", ckpt)

    # 2) grab your model’s dict
    model_dict = model.state_dict()

    # 3) filter out any keys that (a) aren’t in your model, or (b) don’t match in shape
    matched, skipped_ckpt, skipped_model = {}, [], []
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            matched[k] = v
        else:
            # keep track for reporting
            if k not in model_dict:
                skipped_ckpt.append(k)
            else:
                skipped_model.append(k)

    # 4) update your model’s dict, then load it
    model_dict.update(matched)
    model.load_state_dict(model_dict)

    # 5) report
    print(f"✅ Loaded {len(matched)}/{len(model_dict)} params from checkpoint")
    if skipped_ckpt:
        print(f"  • {len(skipped_ckpt)} keys in checkpoint did not match any parameter in the model:")
        for k in skipped_ckpt: print(f"      – {k}")
    if skipped_model:
        print(f"  • {len(skipped_model)} keys in checkpoint had wrong shape and were skipped:")
        for k in skipped_model: print(f"      – {k}")

def move_batch(b, device):
    out = {}
    for k, v in b.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out