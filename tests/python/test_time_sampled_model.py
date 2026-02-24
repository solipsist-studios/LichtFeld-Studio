# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the TimeSampledSplatModel Python container (Milestone 2).

These tests exercise the pure-Python ``TimeSampledSplatModel`` class defined in
``lfs_plugins.flipbook_trainer`` and do not require the C++ lichtfeld extension
to be built or available.
"""

import json
import sys
from pathlib import Path

import pytest

# Allow import directly from the source tree without the C++ build.
_SRC = Path(__file__).parent.parent.parent / "src" / "python"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lfs_plugins.flipbook_trainer import (  # noqa: E402
    FlipbookParameters,
    FlipbookTrainer,
    TimeSampledSplatModel,
    get_model_path_for_time,
)


# ---------------------------------------------------------------------------
# TimeSampledSplatModel – construction and basic accessors
# ---------------------------------------------------------------------------


class TestTimeSampledSplatModelConstruction:
    def test_empty_on_construction(self):
        m = TimeSampledSplatModel()
        assert len(m) == 0
        assert not m

    def test_add_single_entry(self):
        m = TimeSampledSplatModel()
        m.add_entry(0.0, Path("frame_0.ply"))
        assert len(m) == 1
        assert bool(m)

    def test_add_multiple_entries(self):
        m = TimeSampledSplatModel()
        for i in range(5):
            m.add_entry(float(i) * 0.5, Path(f"frame_{i}.ply"))
        assert len(m) == 5

    def test_entries_property(self):
        m = TimeSampledSplatModel()
        m.add_entry(0.0, Path("a.ply"))
        m.add_entry(1.0, Path("b.ply"))
        entries = m.entries
        assert len(entries) == 2
        assert entries[0].timestamp == pytest.approx(0.0)
        assert entries[1].timestamp == pytest.approx(1.0)

    def test_get_timestamp(self):
        m = TimeSampledSplatModel()
        m.add_entry(1.5, Path("x.ply"))
        assert m.get_timestamp(0) == pytest.approx(1.5)

    def test_entry_model_path_stored_as_path(self):
        m = TimeSampledSplatModel()
        m.add_entry(0.0, "/some/path/model.ply")
        assert isinstance(m.get_entry(0).model_path, Path)
        assert m.get_entry(0).model_path == Path("/some/path/model.ply")


# ---------------------------------------------------------------------------
# TimeSampledSplatModel – monotonicity enforcement
# ---------------------------------------------------------------------------


class TestTimeSampledSplatModelMonotonicity:
    def test_non_monotonic_raises(self):
        m = TimeSampledSplatModel()
        m.add_entry(1.0, Path("a.ply"))
        with pytest.raises(ValueError, match="monotonically increasing"):
            m.add_entry(0.5, Path("b.ply"))

    def test_equal_timestamp_allowed(self):
        """Equal timestamps are allowed (degenerate case, e.g. duplicate frame)."""
        m = TimeSampledSplatModel()
        m.add_entry(1.0, Path("a.ply"))
        m.add_entry(1.0, Path("b.ply"))  # should not raise
        assert len(m) == 2


# ---------------------------------------------------------------------------
# TimeSampledSplatModel – time selection (get_model_index_for_time)
# ---------------------------------------------------------------------------


class TestTimeSampledSplatModelTimeSelection:
    @pytest.fixture()
    def model_4(self):
        """A model with 4 entries at t=0.0, 0.5, 1.0, 1.5."""
        m = TimeSampledSplatModel()
        for i in range(4):
            m.add_entry(float(i) * 0.5, Path(f"frame_{i}.ply"))
        return m

    def test_empty_model_returns_zero(self):
        m = TimeSampledSplatModel()
        assert m.get_model_index_for_time(1.0) == 0

    def test_exact_match_first(self, model_4):
        assert model_4.get_model_index_for_time(0.0) == 0

    def test_exact_match_middle(self, model_4):
        assert model_4.get_model_index_for_time(0.5) == 1

    def test_exact_match_last(self, model_4):
        assert model_4.get_model_index_for_time(1.5) == 3

    def test_before_start_clamps_to_first(self, model_4):
        assert model_4.get_model_index_for_time(-999.0) == 0

    def test_beyond_end_clamps_to_last(self, model_4):
        assert model_4.get_model_index_for_time(999.0) == 3

    def test_nearest_rounds_down(self, model_4):
        # 0.24 is closer to 0.0 (d=0.24) than to 0.5 (d=0.26)
        assert model_4.get_model_index_for_time(0.24) == 0

    def test_nearest_rounds_up(self, model_4):
        # 0.26 is closer to 0.5 (d=0.24) than to 0.0 (d=0.26)
        assert model_4.get_model_index_for_time(0.26) == 1

    def test_midpoint_picks_lower(self, model_4):
        # Exactly at midpoint 0.25: d_lower == d_upper → pick lower (index 0)
        assert model_4.get_model_index_for_time(0.25) == 0

    def test_get_model_for_time_returns_path(self, model_4):
        p = model_4.get_model_for_time(0.5)
        assert p == Path("frame_1.ply")

    def test_get_model_for_time_none_on_empty(self):
        m = TimeSampledSplatModel()
        assert m.get_model_for_time(0.0) is None

    def test_get_entry_for_time(self, model_4):
        entry = model_4.get_entry(model_4.get_model_index_for_time(1.0))
        assert entry.timestamp == pytest.approx(1.0)
        assert entry.model_path == Path("frame_2.ply")


# ---------------------------------------------------------------------------
# TimeSampledSplatModel – serialisation (manifest)
# ---------------------------------------------------------------------------


class TestTimeSampledSplatModelSerialisation:
    def test_to_manifest_structure(self):
        m = TimeSampledSplatModel()
        m.add_entry(0.0, Path("a.ply"))
        m.add_entry(1.0, Path("b.ply"))
        manifest = m.to_manifest()
        assert manifest["version"] == 1
        assert len(manifest["entries"]) == 2
        assert manifest["entries"][0]["timestamp"] == pytest.approx(0.0)
        assert manifest["entries"][0]["model_path"] == "a.ply"

    def test_round_trip_from_manifest(self):
        m = TimeSampledSplatModel()
        m.add_entry(0.0, Path("x.ply"))
        m.add_entry(2.5, Path("y.ply"))

        restored = TimeSampledSplatModel.from_manifest(m.to_manifest())
        assert len(restored) == 2
        assert restored.get_timestamp(0) == pytest.approx(0.0)
        assert restored.get_timestamp(1) == pytest.approx(2.5)
        assert restored.get_entry(1).model_path == Path("y.ply")

    def test_save_and_load_manifest(self, tmp_path):
        m = TimeSampledSplatModel()
        m.add_entry(0.0, Path("frame_0.ply"))
        m.add_entry(0.5, Path("frame_1.ply"))

        manifest_path = tmp_path / "flipbook_manifest.json"
        m.save_manifest(manifest_path)
        assert manifest_path.exists()

        loaded = TimeSampledSplatModel.load_manifest(manifest_path)
        assert len(loaded) == 2
        assert loaded.get_timestamp(0) == pytest.approx(0.0)
        assert loaded.get_timestamp(1) == pytest.approx(0.5)

    def test_save_creates_parent_dirs(self, tmp_path):
        m = TimeSampledSplatModel()
        m.add_entry(0.0, Path("f.ply"))
        nested = tmp_path / "a" / "b" / "manifest.json"
        m.save_manifest(nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# FlipbookParameters defaults
# ---------------------------------------------------------------------------


class TestFlipbookParametersDefaults:
    def test_default_stride(self):
        p = FlipbookParameters()
        assert p.keyframe_stride == 1

    def test_default_warm_start(self):
        p = FlipbookParameters()
        assert p.warm_start is False

    def test_default_iterations_per_frame(self):
        p = FlipbookParameters()
        assert p.iterations_per_frame == 0

    def test_default_export(self):
        p = FlipbookParameters()
        assert p.export_per_frame is False


# ---------------------------------------------------------------------------
# FlipbookTrainer – time-step selection
# ---------------------------------------------------------------------------


class _MockSequenceDataset:
    """Minimal duck-typed SequenceDataset for testing."""

    class _Entry:
        def __init__(self, image_path):
            self.image_path = image_path

    def __init__(self, num_cams, num_times):
        self._timestamps = [float(t) * 0.5 for t in range(num_times)]
        self._num_cams = num_cams

    def num_timesteps(self):
        return len(self._timestamps)

    def get_timestamp(self, t):
        return self._timestamps[t]

    def get_time_slice(self, t):
        return [
            self._Entry(f"img_t{t}_c{c}.jpg")
            for c in range(self._num_cams)
        ]


class TestFlipbookTrainerTimeIndices:
    def test_all_frames_stride_1(self):
        ds = _MockSequenceDataset(2, 5)
        trainer = FlipbookTrainer(ds, FlipbookParameters(keyframe_stride=1), Path("/tmp"))
        assert trainer.selected_time_indices() == [0, 1, 2, 3, 4]

    def test_stride_2(self):
        ds = _MockSequenceDataset(2, 6)
        trainer = FlipbookTrainer(ds, FlipbookParameters(keyframe_stride=2), Path("/tmp"))
        assert trainer.selected_time_indices() == [0, 2, 4]

    def test_stride_larger_than_dataset(self):
        ds = _MockSequenceDataset(2, 3)
        trainer = FlipbookTrainer(ds, FlipbookParameters(keyframe_stride=10), Path("/tmp"))
        assert trainer.selected_time_indices() == [0]

    def test_empty_dataset(self):
        ds = _MockSequenceDataset(2, 0)
        trainer = FlipbookTrainer(ds, FlipbookParameters(), Path("/tmp"))
        assert trainer.selected_time_indices() == []


# ---------------------------------------------------------------------------
# FlipbookTrainer – run() integration stub
# ---------------------------------------------------------------------------


class TestFlipbookTrainerRun:
    def _make_stub_train_fn(self, output_dir: Path):
        """Return a stub train_frame_fn that writes empty files."""
        calls = []

        def train_frame_fn(time_idx, timestamp, image_paths, frame_slice, warm_start_path):
            calls.append({
                "time_idx": time_idx,
                "timestamp": timestamp,
                "image_paths": image_paths,
                "warm_start_path": warm_start_path,
            })
            out = output_dir / f"frame_{time_idx:04d}.ply"
            out.touch()
            return out

        return train_frame_fn, calls

    def test_run_produces_correct_number_of_entries(self, tmp_path):
        ds = _MockSequenceDataset(3, 4)
        params = FlipbookParameters(keyframe_stride=1)
        trainer = FlipbookTrainer(ds, params, tmp_path)
        fn, calls = self._make_stub_train_fn(tmp_path)

        result = trainer.run(fn)

        assert len(result) == 4
        assert len(calls) == 4

    def test_run_with_stride(self, tmp_path):
        ds = _MockSequenceDataset(2, 6)
        params = FlipbookParameters(keyframe_stride=2)
        trainer = FlipbookTrainer(ds, params, tmp_path)
        fn, calls = self._make_stub_train_fn(tmp_path)

        result = trainer.run(fn)

        assert len(result) == 3
        assert [e.timestamp for e in result] == pytest.approx([0.0, 1.0, 2.0])

    def test_run_timestamps_are_monotonic(self, tmp_path):
        ds = _MockSequenceDataset(2, 5)
        params = FlipbookParameters()
        trainer = FlipbookTrainer(ds, params, tmp_path)
        fn, _ = self._make_stub_train_fn(tmp_path)

        result = trainer.run(fn)

        timestamps = [e.timestamp for e in result]
        assert timestamps == sorted(timestamps)

    def test_warm_start_passes_prev_path(self, tmp_path):
        ds = _MockSequenceDataset(1, 3)
        params = FlipbookParameters(warm_start=True)
        trainer = FlipbookTrainer(ds, params, tmp_path)
        fn, calls = self._make_stub_train_fn(tmp_path)

        trainer.run(fn)

        # First frame: no warm-start
        assert calls[0]["warm_start_path"] is None
        # Second frame: warm-start from first frame's output
        assert calls[1]["warm_start_path"] == tmp_path / "frame_0000.ply"
        # Third frame: warm-start from second frame's output
        assert calls[2]["warm_start_path"] == tmp_path / "frame_0001.ply"

    def test_no_warm_start_passes_none(self, tmp_path):
        ds = _MockSequenceDataset(1, 3)
        params = FlipbookParameters(warm_start=False)
        trainer = FlipbookTrainer(ds, params, tmp_path)
        fn, calls = self._make_stub_train_fn(tmp_path)

        trainer.run(fn)

        for call in calls:
            assert call["warm_start_path"] is None

    def test_callbacks_invoked(self, tmp_path):
        ds = _MockSequenceDataset(2, 2)
        params = FlipbookParameters()
        start_events = []
        done_events = []
        trainer = FlipbookTrainer(
            ds, params, tmp_path,
            on_frame_start=lambda ti, ts: start_events.append((ti, ts)),
            on_frame_done=lambda ti, ts, p: done_events.append((ti, ts, p)),
        )
        fn, _ = self._make_stub_train_fn(tmp_path)
        trainer.run(fn)

        assert len(start_events) == 2
        assert len(done_events) == 2

    def test_export_writes_manifest(self, tmp_path):
        ds = _MockSequenceDataset(1, 2)
        params = FlipbookParameters(export_per_frame=True)
        trainer = FlipbookTrainer(ds, params, tmp_path)
        fn, _ = self._make_stub_train_fn(tmp_path)

        trainer.run(fn)

        manifest_path = tmp_path / "flipbook_manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert len(data["entries"]) == 2

    def test_empty_dataset_returns_empty_model(self, tmp_path):
        ds = _MockSequenceDataset(2, 0)
        params = FlipbookParameters()
        trainer = FlipbookTrainer(ds, params, tmp_path)

        result = trainer.run(lambda *a, **kw: tmp_path / "x.ply")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# get_model_path_for_time helper
# ---------------------------------------------------------------------------


class TestGetModelPathForTime:
    def test_returns_nearest_path(self):
        m = TimeSampledSplatModel()
        m.add_entry(0.0, Path("a.ply"))
        m.add_entry(1.0, Path("b.ply"))
        assert get_model_path_for_time(m, 0.4) == Path("a.ply")
        assert get_model_path_for_time(m, 0.6) == Path("b.ply")

    def test_returns_none_for_empty(self):
        assert get_model_path_for_time(TimeSampledSplatModel(), 0.0) is None
