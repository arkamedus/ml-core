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

# Update matplotlib configuration globally
plt.rcParams.update({
    # ─── resolution & size ───────────────────────────────────────────
    'figure.dpi':        300,               # default on-screen resolution
    'savefig.dpi':       300,               # default file‐save resolution
    'figure.figsize':    (8, 6),            # default width, height in inches
    'image.interpolation': 'nearest',

    # ─── fonts & text ───────────────────────────────────────────────
    'font.family':       ['monospace'],
    'font.monospace':    ['JetBrains Mono','Source Code Pro','DejaVu Sans Mono','Courier New'],
    'font.size':         11,                # base font size in points
    'axes.titlesize':    14,
    'axes.titleweight':  'bold',
    'axes.labelsize':    12,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,

    # ─── line & marker styles ───────────────────────────────────────
    'lines.linewidth':   1.5,
    'lines.markersize':  6,
    'lines.markeredgewidth': 0.5,

    # ─── axes & grid ────────────────────────────────────────────────
    'axes.prop_cycle':   plt.cycler(color=[
                            '#1f77b4','#ff7f0e','#2ca02c','#d62728',
                            '#9467bd','#8c564b','#e377c2','#7f7f7f',
                            '#bcbd22','#17becf'
                        ]),
    'axes.grid':         False,
    'grid.color':        '#cccccc',
    'grid.linestyle':    '--',
    'grid.linewidth':    0.5,
    'grid.alpha':        0.7,

    # ─── legend ─────────────────────────────────────────────────────
    'legend.fontsize':   10,
    'legend.frameon':    False,
    'legend.loc':        'best',

    # ─── savefig specifics ──────────────────────────────────────────
    'savefig.format':    'png',
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.1,
})


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
def get_device(device: DeviceLike = None, mps_fallback:bool = False) -> torch.device:
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
    if dev.type == "mps" and mps_fallback:
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




# chat formatting + tokenizer helpers

import re

def to_chat_format(text: str):
    pattern = r"<\|([A-Z]+)\|>"
    splits = re.split(pattern, text)

    messages = []
    i = 1
    while i < len(splits):
        role = splits[i].lower()
        content = splits[i + 1]

        # remove structural tokens
        content = content.replace("<|SEP|>", "").replace("<|EOS|>", "").strip()

        if content:
            messages.append({
                "role": role,
                "content": content
            })

        i += 2

    return messages

def from_chat_format(messages):
    parts = []
    i = 0
    n = len(messages)

    while i < n:
        m = messages[i]
        role = m["role"].lower()
        content = m["content"]

        if role == "user":
            parts.append(f"<|USER|>{content}")

            if i + 1 < n and messages[i + 1]["role"].lower() == "assistant":
                parts.append("<|SEP|>")
                parts.append(f"<|ASSISTANT|>{messages[i + 1]['content']}")
                parts.append("<|EOS|>")
                i += 2
            else:
                parts.append("<|SEP|>")
                i += 1

        elif role == "assistant":
            parts.append(f"<|ASSISTANT|>{content}")
            parts.append("<|EOS|>")
            i += 1

        elif role == "think":
            parts.append(f"<|THINK|>{content}")
            i += 1

        else:
            parts.append(f"<|{role.upper()}|>{content}")
            i += 1

    return "".join(parts)


def to_text_sample(x):
    """
    Supports:
      - raw string
      - list[{"role","content"}]
      - dict with text
      - dict with messages
    """
    if isinstance(x, str):
        return x

    if isinstance(x, list):
        return from_chat_format(x)

    if isinstance(x, dict):
        if "messages" in x and isinstance(x["messages"], list):
            return from_chat_format(x["messages"])
        if "conversation" in x and isinstance(x["conversation"], list):
            return from_chat_format(x["conversation"])
        if "conversations" in x and isinstance(x["conversations"], list):
            return from_chat_format(x["conversations"])
        if "text" in x:
            return str(x["text"])
        if "content" in x:
            return str(x["content"])

    return str(x)

def plot_token_embeddings_3d_clustered_with_centroid_tokens(
        model,
        vocab,
        num_tokens: int = 9700,
        n_clusters: int = 64,
        expand_factor: float = 1.5,
        device: str = 'cpu',
        pca_svd_solver: str = "full",
        pca_iterated_power: int = 7,
        pca_whiten: bool = False,
        use_float64: bool = True,  # will be ignored on MPS
        kmeans_n_init: int|str = 20,
        kmeans_max_iter: int = 3000,
        kmeans_random_state: int = 0
):
    import time
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import numpy as np
    import torch
    import plotly.graph_objects as go

    t0 = time.time()
    model.eval()

    with torch.no_grad():
        W_t = model.head.weight[:num_tokens].to(device)

        # If running on MPS, disable float64
        if device == "mps":
            W_t = W_t.float()
        elif use_float64:
            W_t = W_t.double()

        W = W_t.detach().cpu().numpy()  # (n, D)

    # PCA
    if pca_svd_solver == "randomized":
        pca = PCA(
            n_components=3,
            svd_solver="randomized",
            iterated_power=pca_iterated_power,
            whiten=pca_whiten,
            random_state=kmeans_random_state
        )
    else:
        # "full" or "arpack" don’t accept iterated_power
        pca = PCA(
            n_components=3,
            svd_solver=pca_svd_solver,
            whiten=pca_whiten,
            random_state=kmeans_random_state
        )

    coords = pca.fit_transform(W)

    # KMeans clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=kmeans_random_state,
        max_iter=kmeans_max_iter,
        n_init=kmeans_n_init
    ).fit(coords)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # expand spacing
    coords_expanded = centers[labels] + (coords - centers[labels]) * expand_factor

    # find centroid exemplars
    centroid_indices = {}
    for cid in range(n_clusters):
        members = np.where(labels == cid)[0]
        if members.size == 0:
            continue
        pts = coords[members]
        dists = np.linalg.norm(pts - centers[cid], axis=1)
        exemplar_idx = members[np.argmin(dists)]
        centroid_indices[cid] = exemplar_idx
    centroid_tokens = {cid: vocab.detokenize([centroid_indices[cid]]) for cid in centroid_indices}

    tokens = [vocab.detokenize([i]) for i in range(coords.shape[0])]
    cluster_ids = labels.astype(str)
    centroid_labels = [centroid_tokens.get(lab, "<none>") for lab in labels]

    SYMBOLS = ['circle', 'cross', 'diamond', 'square', 'x']
    marker_symbols = [SYMBOLS[c % len(SYMBOLS)] for c in labels]

    fig = go.Figure(data=go.Scatter3d(
        x=coords_expanded[:, 0],
        y=coords_expanded[:, 1],
        z=coords_expanded[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=labels,
            colorscale='Portland',
            symbol=marker_symbols,
            opacity=0.8,
            showscale=False
        ),
        hovertext=tokens,
        customdata=np.stack([cluster_ids, centroid_labels], axis=1),
        hovertemplate=(
            "Token: %{hovertext}<br>"
            #"Cluster: %{customdata[0]}<br>"
            "Centroid token: %{customdata[1]}<extra></extra>"
        )
    ))
    fig.update_layout(
        title=f"3D PCA of first {num_tokens} tokens — {n_clusters} clusters (×{expand_factor} spacing)",
        width=900, height=650,
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='white'
        )
    )
    print(f"PCA+KMeans total time: {time.time() - t0:.2f}s")
    fig.show()


import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_hidden_heatmap(x, title="Hidden states / embeddings heatmap", max_rows=None, max_cols=None):
    """
    Plot a stacked matplotlib heatmap for embeddings/hidden states.

    Accepts:
      [hidden]
      [seq, hidden]
      [batch, seq, hidden]
      [layers, seq, hidden]
      [layers, batch, seq, hidden]

    For 3D tensors like [N, 1, D], it squeezes the middle dim and plots [N, D].
    For 3D tensors like [B, T, D], it stacks B*T rows.
    """

    if isinstance(x, torch.Tensor):
        arr = x.detach().float().cpu().numpy()
    else:
        arr = np.asarray(x, dtype=np.float32)

    original_shape = arr.shape

    # Make it 2D: rows x hidden_dim
    if arr.ndim == 1:
        arr = arr[None, :]

    elif arr.ndim == 2:
        pass

    elif arr.ndim == 3:
        # Common case from your example: [N, 1, D] -> [N, D]
        if arr.shape[1] == 1:
            arr = arr[:, 0, :]
        else:
            # [B, T, D] or [layers, T, D] -> [B*T, D]
            arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])

    elif arr.ndim == 4:
        # [L, B, T, D] -> [L*B*T, D]
        arr = arr.reshape(arr.shape[0] * arr.shape[1] * arr.shape[2], arr.shape[3])

    else:
        raise ValueError(f"Unsupported tensor shape: {original_shape}")

    if max_rows is not None:
        arr = arr[:max_rows, :]

    if max_cols is not None:
        arr = arr[:, :max_cols]

    plt.figure(figsize=(14, max(3, min(12, arr.shape[0] * 0.35))))
    im = plt.imshow(arr, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)

    plt.colorbar(im, label="activation")
    plt.title(f"{title}\noriginal shape={original_shape}, plotted shape={arr.shape}")
    plt.xlabel("hidden dimension")
    plt.ylabel("stacked row")
    plt.tight_layout()
    plt.show()