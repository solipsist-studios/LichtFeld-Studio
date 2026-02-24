# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Flipbook (per-time-step) training orchestration for 4D datasets.

Milestone 2 implementation: train a standard 3D Gaussian Splat model for each
discrete time step in a 4D SequenceDataset, producing a TimeSampledSplatModel
that can be played back via the Sequencer playhead.

Architecture overview
---------------------
TimeSampledSplatModel  – Pure-Python container: ordered list of (timestamp, model_path)
                         pairs.  Mirrors the C++ class of the same name; the Python side
                         stores only on-disk paths so that per-frame memory usage stays
                         bounded regardless of the number of frames.

FlipbookTrainer        – Drives the per-frame training loop.  For each time step selected
                         by the keyframe stride it:
                           1. Assembles a per-frame camera/image dataset from the
                              SequenceDataset time-slice.
                           2. Delegates to the application's existing training subsystem
                              (via callback) to run a standard 3D splat training pass.
                           3. Records the resulting model path in the TimeSampledSplatModel.
                           4. Optionally warm-starts the next frame from the previous one.

Playback               – The sequencer_section uses ``TimeSampledSplatModel.get_model_for_time``
                         to resolve the current Sequencer playhead to the nearest frame.
"""

from __future__ import annotations

import bisect
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TimeSampledSplatModel – Python-side representation
# ---------------------------------------------------------------------------


class TimeSampledSplatModel:
    """Ordered sequence of (timestamp, model_path) entries.

    This is the Python counterpart of the C++ ``lfs::training::TimeSampledSplatModel``.
    It stores lightweight references (on-disk paths) rather than in-memory GPU data
    so that all frames can be held simultaneously without exhausting VRAM.

    Entry ordering is enforced: timestamps must be monotonically increasing.
    """

    @dataclass
    class Entry:
        timestamp: float
        model_path: Path

    def __init__(self) -> None:
        self._entries: list[TimeSampledSplatModel.Entry] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_entry(self, timestamp: float, model_path: Path) -> None:
        """Append a time-step entry.

        :param timestamp: Playhead time in seconds.  Must be >= last timestamp.
        :param model_path: On-disk path for the trained model (PLY/SOG/SPZ).
        :raises ValueError: if *timestamp* is less than the previous entry's timestamp.
        """
        if self._entries and timestamp < self._entries[-1].timestamp:
            raise ValueError(
                f"TimeSampledSplatModel: timestamps must be monotonically increasing "
                f"(got {timestamp} after {self._entries[-1].timestamp})"
            )
        self._entries.append(TimeSampledSplatModel.Entry(timestamp, Path(model_path)))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __iter__(self) -> Iterator[Entry]:
        return iter(self._entries)

    @property
    def entries(self) -> list[Entry]:
        return list(self._entries)

    def get_timestamp(self, index: int) -> float:
        return self._entries[index].timestamp

    def get_model_index_for_time(self, time_seconds: float) -> int:
        """Return the index of the entry whose timestamp is nearest to *time_seconds*.

        Uses nearest-neighbour semantics identical to the C++ implementation.
        Returns 0 when the model is empty.
        """
        if not self._entries:
            return 0
        timestamps = [e.timestamp for e in self._entries]
        pos = bisect.bisect_left(timestamps, time_seconds)
        if pos == len(timestamps):
            return len(timestamps) - 1
        if pos == 0:
            return 0
        upper = pos
        lower = pos - 1
        d_lower = time_seconds - timestamps[lower]
        d_upper = timestamps[upper] - time_seconds
        return lower if d_lower <= d_upper else upper

    def get_model_for_time(self, time_seconds: float) -> Optional[Path]:
        """Return the on-disk model path nearest to *time_seconds*, or None."""
        if not self._entries:
            return None
        idx = self.get_model_index_for_time(time_seconds)
        return self._entries[idx].model_path

    def get_entry(self, index: int) -> Entry:
        return self._entries[index]

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_manifest(self) -> dict:
        """Serialise to a plain-dict manifest suitable for JSON export."""
        return {
            "version": 1,
            "entries": [
                {"timestamp": e.timestamp, "model_path": str(e.model_path)}
                for e in self._entries
            ],
        }

    @classmethod
    def from_manifest(cls, data: dict) -> "TimeSampledSplatModel":
        """Deserialise from a manifest dict (e.g. loaded from JSON)."""
        model = cls()
        for raw in data.get("entries", []):
            model.add_entry(float(raw["timestamp"]), Path(raw["model_path"]))
        return model

    def save_manifest(self, path: Path) -> None:
        """Write a JSON manifest to *path*."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_manifest(), f, indent=2)
        LOG.info("TimeSampledSplatModel manifest saved to %s", path)

    @classmethod
    def load_manifest(cls, path: Path) -> "TimeSampledSplatModel":
        """Load a JSON manifest from *path*."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_manifest(data)


# ---------------------------------------------------------------------------
# FlipbookParameters
# ---------------------------------------------------------------------------


@dataclass
class FlipbookParameters:
    """Runtime parameters controlling the Flipbook training loop.

    These mirror ``lfs::core::param::FlipbookParameters`` on the C++ side.
    """

    #: Train every K-th time step (1 = all, 2 = every other, …).
    keyframe_stride: int = 1
    #: Warm-start each frame from the previous frame's final model parameters.
    warm_start: bool = False
    #: Per-frame iteration count (0 = use the global OptimizationParameters value).
    iterations_per_frame: int = 0
    #: Write per-frame PLY files + a manifest JSON to the output directory.
    export_per_frame: bool = False


# ---------------------------------------------------------------------------
# TrainFrameCallback type alias (for documentation purposes)
# ---------------------------------------------------------------------------
# The FlipbookTrainer calls this for each selected time step.
# Signature: (time_idx, timestamp, image_paths, camera_data, warm_start_path) -> output_model_path
TrainFrameCallbackT = Callable[
    [int, float, Sequence[str], object, Optional[Path]], Path
]


# ---------------------------------------------------------------------------
# FlipbookTrainer
# ---------------------------------------------------------------------------


class FlipbookTrainer:
    """Orchestrates per-time-step training over a 4D SequenceDataset.

    The FlipbookTrainer is intentionally decoupled from the GPU-resident Trainer
    class: it drives the high-level loop, constructs per-frame datasets, and
    records results; the actual GPU training is delegated to *train_frame_fn*.

    This separation keeps the implementation consistent with the existing
    Trainer/strategy abstractions and avoids invasive refactors.

    Usage::

        params = FlipbookParameters(keyframe_stride=2, warm_start=True)
        trainer = FlipbookTrainer(sequence_dataset, params, output_dir)
        result = trainer.run(train_frame_fn=my_training_function)
        # result is a TimeSampledSplatModel
    """

    def __init__(
        self,
        sequence_dataset,  # lfs::training::SequenceDataset (Python binding or duck-type)
        params: FlipbookParameters,
        output_dir: Path,
        *,
        on_frame_start: Optional[Callable[[int, float], None]] = None,
        on_frame_done: Optional[Callable[[int, float, Path], None]] = None,
    ) -> None:
        """
        :param sequence_dataset: A SequenceDataset (or compatible duck-type) that
            provides ``num_timesteps()``, ``get_timestamp(t)``, and
            ``get_time_slice(t)`` (returns list of SequenceFrameEntry).
        :param params: Flipbook training parameters.
        :param output_dir: Directory where per-frame model files are written.
        :param on_frame_start: Optional callback invoked before training each frame;
            receives (time_idx, timestamp).
        :param on_frame_done: Optional callback invoked after each frame with
            (time_idx, timestamp, model_path).
        """
        self._dataset = sequence_dataset
        self._params = params
        self._output_dir = Path(output_dir)
        self._on_frame_start = on_frame_start
        self._on_frame_done = on_frame_done

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def selected_time_indices(self) -> list[int]:
        """Return the list of time indices that will be trained (respects stride)."""
        n = self._dataset.num_timesteps()
        stride = max(1, self._params.keyframe_stride)
        return list(range(0, n, stride))

    def run(self, train_frame_fn: TrainFrameCallbackT) -> TimeSampledSplatModel:
        """Execute the Flipbook training loop.

        For each selected time step this method:
          1. Calls *train_frame_fn* with the per-frame data.
          2. Records the resulting model path in the output ``TimeSampledSplatModel``.
          3. Optionally saves a manifest JSON alongside the per-frame models.

        :param train_frame_fn: Callable that trains one time step.  Signature::

            def train_frame_fn(
                time_idx: int,
                timestamp: float,
                image_paths: list[str],       # one per camera
                camera_data: list,             # SequenceFrameEntry list from the dataset
                warm_start_path: Path | None,  # previous frame's model, or None
            ) -> Path:                         # path to the trained model file

        :returns: Populated ``TimeSampledSplatModel``.
        """
        result = TimeSampledSplatModel()
        time_indices = self.selected_time_indices()

        if not time_indices:
            LOG.warning("FlipbookTrainer: no time steps selected (empty dataset?)")
            return result

        LOG.info(
            "FlipbookTrainer: starting – %d frames selected (stride=%d, warm_start=%s)",
            len(time_indices),
            self._params.keyframe_stride,
            self._params.warm_start,
        )

        self._output_dir.mkdir(parents=True, exist_ok=True)
        prev_model_path: Optional[Path] = None

        for step_num, time_idx in enumerate(time_indices):
            timestamp = self._dataset.get_timestamp(time_idx)
            frame_slice = self._dataset.get_time_slice(time_idx)
            image_paths = [str(entry.image_path) for entry in frame_slice]

            warm_start_path = prev_model_path if self._params.warm_start else None

            if self._on_frame_start is not None:
                self._on_frame_start(time_idx, timestamp)

            LOG.info(
                "FlipbookTrainer: frame %d/%d  t=%.3fs  t_idx=%d",
                step_num + 1,
                len(time_indices),
                timestamp,
                time_idx,
            )

            model_path = train_frame_fn(
                time_idx,
                timestamp,
                image_paths,
                frame_slice,
                warm_start_path,
            )
            model_path = Path(model_path)
            result.add_entry(timestamp, model_path)
            prev_model_path = model_path

            if self._on_frame_done is not None:
                self._on_frame_done(time_idx, timestamp, model_path)

        if self._params.export_per_frame:
            manifest_path = self._output_dir / "flipbook_manifest.json"
            result.save_manifest(manifest_path)

        LOG.info(
            "FlipbookTrainer: complete – %d frames trained, manifest at %s",
            len(result),
            self._output_dir / "flipbook_manifest.json" if self._params.export_per_frame else "(not exported)",
        )
        return result


# ---------------------------------------------------------------------------
# Playback helpers
# ---------------------------------------------------------------------------


def get_model_path_for_time(
    model: TimeSampledSplatModel, time_seconds: float
) -> Optional[Path]:
    """Return the on-disk model path nearest to *time_seconds*.

    Convenience wrapper around ``TimeSampledSplatModel.get_model_for_time``.
    Returns None when *model* is empty.
    """
    return model.get_model_for_time(time_seconds)
