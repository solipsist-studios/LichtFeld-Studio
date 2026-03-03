# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""OMG4 4D Gaussian Splat model loader plugin for LichtFeld Studio.

This plugin loads OMG4 model files and converts them to SplatData4D objects
that can be rendered by the 4D Gaussian Splatting pipeline.

Supported formats
-----------------
- ``.pth``  PyTorch checkpoint files containing 4D Gaussian parameters
- ``.xz``   LZMA-compressed OMG4 format (requires OMG4 reference decoder)

Usage
-----
The plugin is called by the C++ I/O layer when it cannot natively decode
a `.pth` or `.xz` file, and it provides numpy arrays back to the C++ side
to construct a `SplatData4D` object.

References
----------
- OMG4 paper: https://arxiv.org/html/2510.03857v1
- Reference implementation: https://github.com/MinShirley/OMG4
"""

from __future__ import annotations

import logging
import os
import struct
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_pth_tensors(path: Path) -> dict[str, np.ndarray]:
    """Load a PyTorch .pth checkpoint using PyTorch and return tensors as numpy arrays.

    The checkpoint is expected to contain the standard 3D Gaussian Splat keys
    plus the 4D extension keys: ``t``, ``scaling_t``, ``rotation_r``.

    Returns a dict mapping key → numpy float32 array.
    """
    try:
        import torch  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "PyTorch is required to load .pth OMG4 checkpoints. "
            "Install with: pip install torch"
        ) from e

    LOG.info("Loading .pth checkpoint with PyTorch: %s", path)

    # Load checkpoint (may contain nested dicts under 'gaussian_model' etc.)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

    # Flatten: some checkpoints wrap parameters under a state_dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    # Strip common key prefixes (e.g., "gaussian_model._xyz" → "_xyz")
    def _flatten(d: dict) -> dict:
        result = {}
        for k, v in d.items():
            # Strip leading underscore and common prefixes
            clean_k = k
            for prefix in ("gaussian_model.", "_gaussian_model.", "model."):
                if clean_k.startswith(prefix):
                    clean_k = clean_k[len(prefix):]
                    break
            # Remove leading underscore
            if clean_k.startswith("_"):
                clean_k = clean_k[1:]
            if isinstance(v, torch.Tensor):
                result[clean_k] = v.float().numpy()
            elif isinstance(v, np.ndarray):
                result[clean_k] = v.astype(np.float32)
        return result

    tensors = _flatten(ckpt)
    LOG.debug("Checkpoint keys after flatten: %s", list(tensors.keys()))
    return tensors


def _load_xz_model(path: Path) -> dict[str, np.ndarray]:
    """Load an OMG4 LZMA-compressed (.xz) model.

    Requires the OMG4 reference decoder (``OMG4_FTGS`` or ``scene`` module).
    Falls back to an error if the decoder is not available.
    """
    try:
        import lzma
        import pickle
    except ImportError as e:
        raise ImportError("LZMA/pickle support not available") from e

    LOG.info("Decompressing OMG4 .xz model: %s", path)

    with lzma.open(str(path), "rb") as f:
        data = pickle.load(f)

    # data should be a dict with SVQ-encoded fields
    # For M1 we require the OMG4 reference GaussianModel.decode() to be available
    try:
        # Try to import OMG4 reference implementation
        from scene.gaussian_model import GaussianModel  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "OMG4 reference implementation not found. "
            "Clone https://github.com/MinShirley/OMG4 and add it to your PYTHONPATH "
            "to load .xz compressed OMG4 models."
        )

    # Use OMG4's decode() to reconstruct the Gaussian parameters
    model = GaussianModel(sh_degree=3)
    model.decode(data)

    tensors = {
        "xyz": model.get_xyz.detach().cpu().numpy().astype(np.float32),
        "features_dc": model.get_features[:, :1, :].detach().cpu().numpy().astype(np.float32),
        "features_rest": model.get_features[:, 1:, :].detach().cpu().numpy().astype(np.float32),
        "scaling": model.get_scaling.log().detach().cpu().numpy().astype(np.float32),  # back to log space
        "rotation": model.get_rotation.detach().cpu().numpy().astype(np.float32),
        "opacity": model.get_opacity.logit().detach().cpu().numpy().astype(np.float32),
        "t": model.get_t.detach().cpu().numpy().astype(np.float32),
        "scaling_t": model.get_scaling_t.log().detach().cpu().numpy().astype(np.float32),
        "rotation_r": model.get_rotation_r.detach().cpu().numpy().astype(np.float32),
    }
    return tensors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class OMG4LoadResult:
    """Container for loaded 4D Gaussian Splat model data."""

    def __init__(self, tensors: dict[str, np.ndarray], time_range: tuple[float, float]):
        self.tensors = tensors
        self.time_range = time_range

    @property
    def n_gaussians(self) -> int:
        return int(self.tensors.get("xyz", np.array([])).shape[0])

    @property
    def sh_degree(self) -> int:
        rest = self.tensors.get("features_rest", None)
        if rest is None or rest.size == 0:
            return 0
        # features_rest shape: [N, K, 3] where K = sh_coeffs for (degree)
        # sh_coeffs: 1->0, 4->1(3), 9->2(8), 16->3(15)
        k = rest.shape[1]
        if k == 0:
            return 0
        elif k <= 3:
            return 1
        elif k <= 8:
            return 2
        else:
            return 3

    def get_numpy(self, key: str) -> Optional[np.ndarray]:
        """Get a tensor by key, returning None if not present."""
        return self.tensors.get(key)

    def __repr__(self) -> str:
        return (
            f"OMG4LoadResult(n={self.n_gaussians}, "
            f"sh_degree={self.sh_degree}, "
            f"time_range={self.time_range})"
        )


def load_omg4_model(path: str | Path) -> OMG4LoadResult:
    """Load an OMG4 4D Gaussian Splat model from a .pth or .xz file.

    Parameters
    ----------
    path : str or Path
        Path to the model file (.pth PyTorch checkpoint or .xz compressed).

    Returns
    -------
    OMG4LoadResult
        Loaded model data with numpy arrays ready to pass to C++.

    Raises
    ------
    ValueError
        If the file format is not recognised or the required keys are missing.
    ImportError
        If PyTorch or OMG4 decoder is not available.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pth":
        tensors = _load_pth_tensors(path)
    elif ext == ".xz":
        tensors = _load_xz_model(path)
    else:
        raise ValueError(f"Unsupported OMG4 file extension: {ext}. Expected .pth or .xz")

    # Verify required 4D keys
    required_4d_keys = {"t", "scaling_t", "rotation_r"}
    present_keys = set(tensors.keys())
    missing = required_4d_keys - present_keys
    if missing:
        raise ValueError(
            f"Missing 4D keys in checkpoint: {missing}. "
            f"Available keys: {sorted(present_keys)}"
        )

    # Infer time range from t values
    t_arr = tensors["t"]
    if t_arr.ndim == 2:
        t_arr = t_arr[:, 0]
    t_min = float(t_arr.min()) if len(t_arr) > 0 else 0.0
    t_max = float(t_arr.max()) if len(t_arr) > 0 else 1.0
    # Add small margin
    margin = max((t_max - t_min) * 0.05, 0.1)
    time_range = (t_min - margin, t_max + margin)

    LOG.info(
        "Loaded OMG4 model: %d Gaussians, time=[%.3f, %.3f]",
        tensors["xyz"].shape[0] if "xyz" in tensors else 0,
        time_range[0],
        time_range[1],
    )

    return OMG4LoadResult(tensors=tensors, time_range=time_range)
