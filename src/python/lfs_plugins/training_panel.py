# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Training Panel - RmlUI with native data binding."""

import os
import time

import lichtfeld as lf

from .flipbook_trainer import FlipbookParameters, TimeSampledSplatModel
from .types import Panel, RmlPanel
from .ui.state import AppState


def tr(key):
    result = lf.ui.tr(key)
    return result if result else key


class IterationRateTracker:
    WINDOW_SECONDS = 5.0

    def __init__(self):
        self.samples = []

    def add_sample(self, iteration):
        now = time.monotonic()
        self.samples.append((iteration, now))
        self.samples = [(i, t) for i, t in self.samples if now - t <= self.WINDOW_SECONDS]

    def get_rate(self):
        if len(self.samples) < 2:
            return 0.0
        oldest = self.samples[0]
        newest = self.samples[-1]
        iter_diff = newest[0] - oldest[0]
        time_diff = newest[1] - oldest[1]
        return iter_diff / time_diff if time_diff > 0 else 0.0

    def clear(self):
        self.samples = []


_rate_tracker = IterationRateTracker()

LOCALE_KEYS = {
    "hdr_basic_params": "training.section.basic_params",
    "hdr_advanced_params": "training.section.advanced_params",
    "hdr_dataset": "training.section.dataset",
    "hdr_optimization": "training.section.optimization",
    "hdr_bilateral": "training.section.bilateral_grid",
    "hdr_losses": "training.section.losses",
    "hdr_init": "training.section.initialization",
    "hdr_adc": "training_panel.pruning_growing",
    "hdr_sparsity": "training_panel.sparsity",
    "hdr_save_steps": "training_panel.save_steps",
    "strategy": "training_params.strategy",
    "iterations": "training_params.iterations",
    "max_cap": "training_params.max_gaussians",
    "sh_degree": "training_params.sh_degree",
    "tile_mode": "training_params.tile_mode",
    "steps_scaler": "training_params.steps_scaler",
    "bilateral_grid": "training_params.bilateral_grid",
    "mask_mode": "training_params.mask_mode",
    "invert_masks": "training_params.invert_masks",
    "use_alpha_as_mask": "training_params.use_alpha_as_mask",
    "sparsity": "training_params.sparsity",
    "gut": "training_params.gut",
    "undistort": "training_params.undistort",
    "mip_filter": "training_params.mip_filter",
    "ppisp": "training_params.ppisp",
    "ppisp_controller": "training_params.ppisp_controller",
    "ppisp_auto": "common.auto",
    "ppisp_activation_step": "training_params.ppisp_activation_step",
    "ppisp_controller_lr": "training_params.ppisp_controller_lr",
    "ppisp_freeze_gaussians": "training_params.ppisp_freeze_gaussians",
    "bg_mode": "training_params.bg_mode",
    "bg_color": "training_params.bg_color",
    "bg_image": "training_params.bg_image",
    "dataset_path": "training.dataset.path",
    "dataset_images": "training.dataset.images",
    "resize_factor": "training.dataset.resize_factor",
    "max_width": "training.dataset.max_width",
    "cpu_cache": "training.dataset.cpu_cache",
    "fs_cache": "training.dataset.fs_cache",
    "dataset_output": "training.dataset.output",
    "no_dataset": "training_panel.no_dataset_loaded",
    "opt_strategy": "training_params.strategy",
    "lr_header": "training.opt.learning_rates",
    "means_lr": "training.opt.lr.position",
    "shs_lr": "training.opt.lr.sh_coeff",
    "opacity_lr": "training.opt.lr.opacity",
    "scaling_lr": "training.opt.lr.scaling",
    "rotation_lr": "training.opt.lr.rotation",
    "refinement_header": "training.section.refinement",
    "refine_every": "training.refinement.refine_every",
    "start_refine": "training.refinement.start_refine",
    "stop_refine": "training.refinement.stop_refine",
    "grad_threshold": "training.refinement.gradient_thr",
    "reset_every": "training.refinement.reset_every",
    "sh_degree_interval": "training.refinement.sh_upgrade_every",
    "bilateral_grid_x": "training.bilateral.grid_x",
    "bilateral_grid_y": "training.bilateral.grid_y",
    "bilateral_grid_w": "training.bilateral.grid_w",
    "bilateral_grid_lr": "training.bilateral.learning_rate",
    "lambda_dssim": "training.losses.lambda_dssim",
    "opacity_reg": "training.losses.opacity_reg",
    "scale_reg": "training.losses.scale_reg",
    "tv_loss_weight": "training.losses.tv_loss_weight",
    "init_opacity": "training.init.init_opacity",
    "init_scaling": "training.init.init_scaling",
    "random_init": "training.init.random_init",
    "init_num_pts": "training.init.num_points",
    "init_extent": "training.init.extent",
    "min_opacity": "training.thresholds.min_opacity",
    "prune_opacity": "training.thresholds.prune_opacity",
    "grow_scale3d": "training.thresholds.grow_scale_3d",
    "grow_scale2d": "training.thresholds.grow_scale_2d",
    "prune_scale3d": "training.thresholds.prune_scale_3d",
    "prune_scale2d": "training.thresholds.prune_scale_2d",
    "pause_refine_after_reset": "training.thresholds.pause_after_reset",
    "revised_opacity": "training.thresholds.revised_opacity",
    "sparsify_steps": "training_params.sparsify_steps",
    "init_rho": "training_params.init_rho",
    "prune_ratio": "training_params.prune_ratio",
    "no_trainer": "training_panel.no_trainer_loaded",
    "no_params": "training_panel.parameters_unavailable",
    "no_save_steps": "training_panel.no_save_steps",
    "save_checkpoint": "training_panel.save_checkpoint",
    "checkpoint_saved": "training_panel.checkpoint_saved",
    "add": "common.add",
    "remove": "common.remove",
    "bg_browse": "training_params.bg_image_browse",
    "bg_clear": "training_params.bg_image_clear",
}

class _FlipbookState:
    """Persistent UI state for the Flipbook training section."""

    params = FlipbookParameters()
    is_running = False
    result: TimeSampledSplatModel | None = None


_flipbook_state = _FlipbookState()


PARAM_BOOL_PROPS = [
    "use_bilateral_grid", "invert_masks", "use_alpha_as_mask",
    "enable_sparsity", "gut", "undistort", "mip_filter",
    "ppisp", "ppisp_use_controller", "ppisp_freeze_gaussians",
    "random", "revised_opacity",
]

DATASET_BOOL_PROPS = ["use_cpu_cache", "use_fs_cache"]

# (prop, type, format, min, max)
NUM_PROP_DEFS = [
    # (name, dtype, format, min, max, step)
    ("iterations", int, "%d", 1, None, 100),
    ("max_cap", int, "%d", 1, None, 100000),
    ("steps_scaler", float, "%.2f", 0.01, None, 0.1),
    ("means_lr", float, "%.6f", 0, None, 0.00001),
    ("shs_lr", float, "%.4f", 0, None, 0.001),
    ("opacity_lr", float, "%.4f", 0, None, 0.001),
    ("scaling_lr", float, "%.4f", 0, None, 0.001),
    ("rotation_lr", float, "%.4f", 0, None, 0.001),
    ("refine_every", int, "%d", 1, None, 10),
    ("start_refine", int, "%d", 0, None, 100),
    ("stop_refine", int, "%d", 0, None, 100),
    ("grad_threshold", float, "%.6f", 0, None, 0.00001),
    ("reset_every", int, "%d", 1, None, 100),
    ("sh_degree_interval", int, "%d", 1, None, 100),
    ("bilateral_grid_x", int, "%d", 1, None, 1),
    ("bilateral_grid_y", int, "%d", 1, None, 1),
    ("bilateral_grid_w", int, "%d", 1, None, 1),
    ("bilateral_grid_lr", float, "%.6f", 0, None, 0.00001),
    ("opacity_reg", float, "%.4f", 0, None, 0.001),
    ("scale_reg", float, "%.4f", 0, None, 0.001),
    ("tv_loss_weight", float, "%.1f", 0, None, 0.5),
    ("init_scaling", float, "%.3f", 0.001, None, 0.01),
    ("init_num_pts", int, "%d", 1, None, 1000),
    ("init_extent", float, "%.1f", 0.1, None, 0.5),
    ("min_opacity", float, "%.4f", 0, None, 0.001),
    ("prune_opacity", float, "%.4f", 0, None, 0.001),
    ("grow_scale3d", float, "%.4f", 0, None, 0.001),
    ("grow_scale2d", float, "%.3f", 0, None, 0.01),
    ("prune_scale3d", float, "%.3f", 0, None, 0.01),
    ("prune_scale2d", float, "%.3f", 0, None, 0.01),
    ("pause_refine_after_reset", int, "%d", 0, None, 10),
    ("sparsify_steps", int, "%d", 1, None, 100),
    ("init_rho", float, "%.4f", 0, None, 0.001),
    ("ppisp_controller_lr", float, "%.5f", 0, None, 0.0001),
]

_NUM_PROP_LOOKUP = {name: (dtype, fmt, min_v, max_v, step)
                    for name, dtype, fmt, min_v, max_v, step in NUM_PROP_DEFS}

SLIDER_PROPS = ["lambda_dssim", "init_opacity", "prune_ratio"]

DIRECT_SET_PROPS = {
    "iterations", "max_cap", "means_lr", "shs_lr",
    "opacity_lr", "scaling_lr", "rotation_lr", "ppisp_controller_lr",
}

RENDER_SYNC = {
    "gut": "gut",
    "mip_filter": "mip_filter",
    "ppisp": "apply_appearance_correction",
}

SECTIONS = [
    "basic_params", "advanced_params", "dataset", "optimization",
    "bilateral", "losses", "init", "adc", "sparsity", "save_steps",
]

INITIALLY_COLLAPSED = {
    "advanced_params", "dataset", "optimization", "bilateral",
    "losses", "init", "adc", "sparsity", "save_steps",
}




def _color_to_hex(c):
    return f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"


def _hex_to_color(h):
    h = h.lstrip("#")
    if len(h) != 6:
        return None
    try:
        return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)
    except ValueError:
        return None


class TrainingPanel(RmlPanel):
    idname = "lfs.training"
    label = "Training"
    space = "MAIN_PANEL_TAB"
    order = 20
    rml_template = "rmlui/training.rml"
    rml_height_mode = "content"

    def __init__(self):
        self._handle = None
        self._checkpoint_saved_time = 0.0
        self._new_save_step = 7000
        self._auto_scaled_for_cameras = 0
        self._last_state = ""
        self._last_save_steps = []
        self._color_edit_prop = None
        self._picker_click_handled = False
        self._collapsed = set(INITIALLY_COLLAPSED)
        self._last_iteration = -1
        self._last_num_gaussians = -1
        self._last_progress_frac = -1.0
        self._last_bg_color = None
        self._doc = None
        self._popup_el = None
        self._loss_graph_el = None
        self._loss_label_el = None
        self._tick_els = []
        self._step_repeat_prop = None
        self._step_repeat_dir = 0
        self._step_repeat_start = 0.0
        self._step_repeat_last = 0.0

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("training")
        if model is None:
            return

        p = lf.optimization_params
        d = lf.dataset_params

        self._bind_labels(model)
        self._bind_sections(model)
        self._bind_visibility(model, p, d)
        self._bind_disabled(model, p)
        self._bind_bool_props(model, p)
        self._bind_dataset_bools(model, d)
        self._bind_select_props(model, p, d)
        self._bind_num_props(model, p, d)
        self._bind_slider_props(model, p)
        self._bind_color(model, p)
        self._bind_status(model, p)
        self._bind_display(model, p, d)
        self._bind_events(model)
        self._handle = model.get_handle()

    def _bind_labels(self, model):
        for label_id, key in LOCALE_KEYS.items():
            model.bind_func(f"label_{label_id}", lambda k=key: tr(k))

        model.bind_func("label_reset", lambda: tr("training_panel.reset"))
        model.bind_func("label_clear", lambda: tr("training_panel.clear"))
        model.bind_func("label_pause", lambda: tr("training_panel.pause"))
        model.bind_func("label_resume", lambda: tr("training_panel.resume"))
        model.bind_func("label_stop", lambda: tr("training_panel.stop"))
        model.bind_func("label_switch_edit", lambda: tr("training_panel.switch_edit_mode"))
        model.bind_func("label_status_completed", lambda: tr("status.complete"))
        model.bind_func("label_status_stopped", lambda: tr("status.stopped"))
        model.bind_func("label_status_error", lambda: tr("status.error"))
        model.bind_func("label_status_stopping", lambda: tr("status.stopping"))

        def _btn_start():
            it = AppState.iteration.value
            return tr("training_panel.resume_training") if it > 0 else tr("training_panel.start_training")
        model.bind_func("btn_start", _btn_start)

    def _bind_sections(self, model):
        for sec in SECTIONS:
            model.bind(f"sec_{sec}_collapsed",
                       lambda n=sec: n in self._collapsed)
            model.bind_func(f"sec_{sec}_arrow",
                            lambda n=sec: "\u25B6" if n in self._collapsed else "\u25BC")

    def _bind_visibility(self, model, p, d):
        def _state():
            return AppState.trainer_state.value
        def _iteration():
            return AppState.iteration.value

        model.bind_func("show_no_trainer",
                         lambda: not AppState.has_trainer.value)
        model.bind_func("show_no_params",
                         lambda: AppState.has_trainer.value and not (p() and p().has_params()))
        model.bind_func("show_main",
                         lambda: AppState.has_trainer.value and p() is not None and p().has_params())

        for state_name in ["ready", "running", "paused", "completed", "stopped", "error", "stopping"]:
            model.bind_func(f"show_ctrl_{state_name}",
                            lambda s=state_name: _state() == s)

        model.bind_func("show_reset_ready",
                         lambda: _state() == "ready" and _iteration() > 0)
        model.bind_func("show_checkpoint",
                         lambda: _state() in ("running", "paused"))
        model.bind_func("show_checkpoint_saved",
                         lambda: _state() in ("running", "paused") and
                                 time.time() - self._checkpoint_saved_time < 2.0)

        model.bind_func("dep_mask_mode",
                         lambda: p() is not None and p().has_params() and p().mask_mode.value != 0)
        model.bind_func("dep_ppisp",
                         lambda: p() is not None and p().has_params() and p().ppisp)
        model.bind_func("dep_ppisp_controller",
                         lambda: p() is not None and p().has_params() and p().ppisp_use_controller)
        model.bind_func("dep_ppisp_manual_step",
                         lambda: p() is not None and p().has_params() and
                                 p().ppisp_controller_activation_step >= 0)
        model.bind_func("dep_bg_color",
                         lambda: p() is not None and p().has_params() and p().bg_mode.value in (0, 1))
        model.bind_func("dep_bg_image",
                         lambda: p() is not None and p().has_params() and p().bg_mode.value == 2)
        model.bind_func("has_bg_clear",
                         lambda: p() is not None and p().has_params() and bool(p().bg_image_path))
        model.bind_func("dep_bilateral",
                         lambda: p() is not None and p().has_params() and p().use_bilateral_grid)
        model.bind_func("dep_adc",
                         lambda: p() is not None and p().has_params() and p().strategy == "adc")
        model.bind_func("dep_sparsity",
                         lambda: p() is not None and p().has_params() and p().enable_sparsity)
        model.bind_func("dep_random",
                         lambda: p() is not None and p().has_params() and p().random)
        model.bind_func("show_progress",
                         lambda: AppState.max_iterations.value > 0 and _iteration() > 0)
        model.bind_func("has_dataset",
                         lambda: d() is not None and d().has_params())
        model.bind_func("show_dataset_no_data",
                         lambda: d() is None or not d().has_params())

        model.bind_func("save_edit_mode",
                         lambda: _state() == "ready" and _iteration() == 0)
        model.bind_func("save_readonly_mode",
                         lambda: _state() != "ready" or _iteration() != 0)
        model.bind_func("no_save_steps",
                         lambda: _state() == "ready" and _iteration() == 0 and
                                 p() is not None and p().has_params() and not list(p().save_steps))
        model.bind_func("no_save_steps_ro",
                         lambda: (_state() != "ready" or _iteration() != 0) and
                                 p() is not None and p().has_params() and not list(p().save_steps))
        model.bind_func("has_save_steps",
                         lambda: p() is not None and p().has_params() and bool(list(p().save_steps)))
        model.bind_string_list("save_steps_list")

    def _bind_disabled(self, model, p):
        model.bind_func("struct_disabled",
                         lambda: not (AppState.trainer_state.value == "ready" and
                                      AppState.iteration.value == 0))
        model.bind_func("live_disabled",
                         lambda: AppState.trainer_state.value not in ("ready", "running", "paused"))
        model.bind_func("adv_disabled",
                         lambda: not (AppState.trainer_state.value == "ready" and
                                      AppState.iteration.value == 0))
        model.bind_func("gut_disabled",
                         lambda: p() is not None and p().has_params() and p().strategy == "adc")
        model.bind_func("dataset_disabled",
                         lambda: not (lf.dataset_params() is not None and
                                      lf.dataset_params().has_params() and
                                      lf.dataset_params().can_edit()))

    def _bind_bool_props(self, model, p):
        for prop in PARAM_BOOL_PROPS:
            model.bind(prop,
                       lambda pr=prop: getattr(p(), pr, False) if p() and p().has_params() else False,
                       lambda v, pr=prop: self._set_bool_prop(pr, v))

        model.bind("ppisp_auto_step",
                    lambda: p() is not None and p().has_params() and
                            p().ppisp_controller_activation_step < 0,
                    lambda v: self._set_ppisp_auto_step(v))

    def _bind_dataset_bools(self, model, d):
        def _set_dataset_bool(v, pr):
            dp = d()
            if dp and dp.has_params():
                try:
                    setattr(dp, pr, v)
                except RuntimeError:
                    pass

        for prop in DATASET_BOOL_PROPS:
            model.bind(prop,
                       lambda pr=prop: getattr(d(), pr, False) if d() and d().has_params() else False,
                       lambda v, pr=prop: _set_dataset_bool(v, pr))

    def _bind_select_props(self, model, p, d):
        model.bind("strategy",
                    lambda: p().strategy if p() and p().has_params() else "mcmc",
                    lambda v: self._set_strategy(v))
        model.bind("sh_degree_str",
                    lambda: str(p().sh_degree) if p() and p().has_params() else "0",
                    lambda v: self._set_int_param("sh_degree", v))
        model.bind("tile_mode_str",
                    lambda: str(p().tile_mode) if p() and p().has_params() else "1",
                    lambda v: self._set_int_param("tile_mode", v))
        model.bind("mask_mode_str",
                    lambda: str(p().mask_mode.value) if p() and p().has_params() else "0",
                    lambda v: self._set_mask_mode(v))
        model.bind("bg_mode_str",
                    lambda: str(p().bg_mode.value) if p() and p().has_params() else "0",
                    lambda v: self._set_bg_mode(v))
        model.bind("resize_factor_str",
                    lambda: str(d().resize_factor) if d() and d().has_params() else "-1",
                    lambda v: self._set_resize_factor(v))

    def _bind_num_props(self, model, p, d):
        for prop, dtype, fmt, min_v, max_v, _step in NUM_PROP_DEFS:
            model.bind(
                f"{prop}_str",
                lambda pr=prop, f=fmt: f % getattr(p(), pr, 0) if p() and p().has_params() else "",
                lambda v, pr=prop, dt=dtype, mn=min_v, mx=max_v: self._set_num_prop(pr, v, dt, mn, mx))

        model.bind("ppisp_activation_step_str",
                    lambda: str(p().ppisp_controller_activation_step)
                            if p() and p().has_params() and p().ppisp_controller_activation_step >= 0
                            else "",
                    lambda v: self._set_ppisp_activation_step(v))

        model.bind("max_width_str",
                    lambda: "%d" % d().max_width if d() and d().has_params() else "",
                    lambda v: self._set_max_width(v))

        model.bind("new_step_str",
                    lambda: str(self._new_save_step),
                    lambda v: self._set_new_step_val(v))

    def _bind_slider_props(self, model, p):
        for prop in SLIDER_PROPS:
            model.bind(
                prop,
                lambda pr=prop: float(getattr(p(), pr, 0.0))
                                if p() and p().has_params() else 0.0,
                lambda v, pr=prop: self._set_slider_prop(pr, v))

    def _bind_color(self, model, p):
        def _bg():
            return getattr(p(), "bg_color", (0, 0, 0)) if p() and p().has_params() else (0, 0, 0)

        model.bind_func("bg_color_r", lambda: f"R:{int(_bg()[0]*255):>3d}")
        model.bind_func("bg_color_g", lambda: f"G:{int(_bg()[1]*255):>3d}")
        model.bind_func("bg_color_b", lambda: f"B:{int(_bg()[2]*255):>3d}")
        model.bind("bg_color_hex",
                    lambda: _color_to_hex(_bg()),
                    lambda v: self._set_bg_color_hex(v))

        model.bind_func("picker_r",
                         lambda: float(_bg()[0]) if self._color_edit_prop else 0.0)
        model.bind_func("picker_g",
                         lambda: float(_bg()[1]) if self._color_edit_prop else 0.0)
        model.bind_func("picker_b",
                         lambda: float(_bg()[2]) if self._color_edit_prop else 0.0)

    def _bind_status(self, model, p):
        def _status_mode():
            state = AppState.trainer_state.value
            it = AppState.iteration.value
            labels = {
                "idle": tr("training_panel.idle"),
                "ready": tr("status.ready") if it == 0 else tr("training_panel.resume"),
                "running": tr("training_panel.running"),
                "paused": tr("status.paused"),
                "stopping": tr("status.stopping"),
                "completed": tr("status.complete"),
                "stopped": tr("status.stopped"),
                "error": tr("status.error"),
            }
            return f"{tr('status.mode')}: {labels.get(state, tr('status.unknown'))}"

        def _status_iteration():
            it = AppState.iteration.value
            _rate_tracker.add_sample(it)
            rate = _rate_tracker.get_rate()
            return f"{tr('status.iteration')} {it:,} ({rate:.1f} {tr('training_panel.iters_per_sec')})"

        def _status_gaussians():
            return tr("progress.num_splats") % f"{AppState.num_gaussians.value:,}"

        def _progress_text():
            it = AppState.iteration.value
            mx = AppState.max_iterations.value
            return f"{it:,}/{mx:,}" if mx > 0 else ""

        def _error_message():
            return lf.trainer_error() or ""

        model.bind_func("status_mode", _status_mode)
        model.bind_func("status_iteration", _status_iteration)
        model.bind_func("status_gaussians", _status_gaussians)
        model.bind_func("progress_text", _progress_text)
        model.bind_func("error_message", _error_message)

        model.bind_func("save_steps_display",
                         lambda: ", ".join(str(s) for s in p().save_steps)
                                 if p() and p().has_params() else "")

    def _bind_display(self, model, p, d):
        model.bind_func("opt_strategy_display",
                         lambda: p().strategy.upper() if p() and p().has_params() else "")

        model.bind_func("dataset_path_display",
                         lambda: os.path.basename(d().data_path) if d() and d().has_params() and d().data_path
                                 else tr("training.value.none"))
        model.bind_func("dataset_images_display",
                         lambda: d().images if d() and d().has_params() and d().images
                                 else tr("training.value.default"))
        model.bind_func("dataset_output_display",
                         lambda: os.path.basename(d().output_path) if d() and d().has_params() and d().output_path
                                 else tr("training.value.not_set"))
        model.bind_func("bg_image_path_display",
                         lambda: os.path.basename(p().bg_image_path) if p() and p().has_params() and p().bg_image_path
                                 else tr("training.value.none"))

    def _bind_events(self, model):
        model.bind_event("toggle_section", self._on_toggle_section)
        model.bind_event("color_click", self._on_color_click)
        model.bind_event("picker_change", self._on_picker_change)
        model.bind_event("action", self._on_action)
        model.bind_event("remove_step", self._on_remove_step_event)
        model.bind_event("num_step", self._on_num_step)

    def on_load(self, doc):
        self._doc = doc
        self._popup_el = doc.get_element_by_id("color-picker-popup")
        if self._popup_el:
            self._popup_el.add_event_listener("click", self._on_popup_click)
        body = doc.get_element_by_id("body")
        if body:
            body.add_event_listener("click", self._on_body_click)
            body.add_event_listener("mouseup", self._on_step_mouseup)
        self._loss_graph_el = doc.get_element_by_id("loss-graph-el")
        self._loss_label_el = doc.get_element_by_id("loss-label")
        self._tick_els = [
            doc.get_element_by_id("loss-tick-max"),
            doc.get_element_by_id("loss-tick-mid"),
            doc.get_element_by_id("loss-tick-min"),
        ]

    def on_update(self, doc):
        if not self._handle:
            return

        state = AppState.trainer_state.value
        if state != self._last_state:
            self._last_state = state
            self._handle.dirty_all()
        else:
            it = AppState.iteration.value
            if it != self._last_iteration:
                self._last_iteration = it
                self._handle.dirty("status_iteration")
                self._handle.dirty("progress_text")
                self._handle.dirty("show_progress")

            ng = AppState.num_gaussians.value
            if ng != self._last_num_gaussians:
                self._last_num_gaussians = ng
                self._handle.dirty("status_gaussians")

            self._handle.dirty("show_checkpoint_saved")

        if state == "ready" and AppState.iteration.value == 0:
            params = lf.optimization_params()
            if params and params.has_params():
                self._try_auto_scale_steps(params)

        self._update_step_repeat()
        self._update_progress(doc)
        self._update_save_steps(doc)
        self._update_color_swatch(doc)
        self._update_loss_graph()

        if layout.collapsing_header("4D Flipbook##py", default_open=False):
            self._draw_flipbook_section(layout, state)

        self._draw_status(layout, state, iteration)
        
    def _update_progress(self, doc):
        it = AppState.iteration.value
        mx = AppState.max_iterations.value
        if mx > 0 and it > 0:
            frac = it / mx
            if frac != self._last_progress_frac:
                self._last_progress_frac = frac
                prog = doc.get_element_by_id("training-progress")
                if prog:
                    prog.set_attribute("value", str(frac))

    def _update_save_steps(self, doc):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return

        state = AppState.trainer_state.value
        can_edit = state == "ready" and AppState.iteration.value == 0
        if not can_edit:
            return

        steps = list(params.save_steps)
        if steps != self._last_save_steps:
            self._last_save_steps = steps[:]
            self._handle.update_string_list("save_steps_list", [str(s) for s in steps])
            self._handle.dirty("no_save_steps")
            self._handle.dirty("has_save_steps")
            self._handle.dirty("save_steps_display")

    def _update_color_swatch(self, doc):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        c = tuple(params.bg_color)
        if c == self._last_bg_color:
            return
        self._last_bg_color = c
        swatch = doc.get_element_by_id("swatch-bg_color")
        if swatch:
            r, g, b = int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)
            swatch.set_property("background-color", f"rgb({r},{g},{b})")

    def on_scene_changed(self, doc):
        if self._handle:
            self._handle.dirty_all()

    def on_unload(self, doc):
        doc.remove_data_model("training")
        self._handle = None

    def _update_loss_graph(self):
        if not self._loss_graph_el:
            return
        loss_data = lf.loss_buffer()
        if not loss_data:
            return
        data_min, data_max = lf.push_loss_to_element(self._loss_graph_el, loss_data)
        if self._loss_label_el:
            self._loss_label_el.set_inner_rml(f"{tr('status.loss')}: {loss_data[-1]:.4f}")
        fmt = "%.4f" if data_max < 0.1 else ("%.3f" if data_max < 1.0 else "%.2f")
        mid = data_min + (data_max - data_min) * 0.5
        tick_values = [data_max, mid, data_min]
        for el, val in zip(self._tick_els, tick_values):
            if el:
                el.set_inner_rml(fmt % val)

    def _on_picker_change(self, handle, event, args):
        params = lf.optimization_params()
        if not params or not params.has_params() or not event or not self._color_edit_prop:
            return
        r = float(event.get_parameter("red", "0"))
        g = float(event.get_parameter("green", "0"))
        b = float(event.get_parameter("blue", "0"))
        setattr(params, self._color_edit_prop, (r, g, b))
        rs = lf.get_render_settings()
        if rs and self._color_edit_prop == "bg_color":
            rs.set("background_color", (r, g, b))
        if self._handle:
            self._handle.dirty_all()

    def _on_popup_click(self, event):
        self._picker_click_handled = True

    def _on_body_click(self, event):
        if hasattr(self, '_picker_click_handled') and self._picker_click_handled:
            self._picker_click_handled = False
            return
        if hasattr(self, '_popup_el') and self._popup_el:
            self._popup_el.set_class("visible", False)
            self._color_edit_prop = None

    # ── Setters ────────────────────────────────────────────

    def _set_bool_prop(self, prop, val):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        setattr(params, prop, val)
        rs = lf.get_render_settings()
        if rs and prop in RENDER_SYNC:
            rs.set(RENDER_SYNC[prop], val)
        if self._handle:
            self._handle.dirty_all()

    def _set_ppisp_auto_step(self, val):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        if val:
            params.ppisp_controller_activation_step = -1
        else:
            params.ppisp_controller_activation_step = max(1, int(params.iterations) - 5000)
        if self._handle:
            self._handle.dirty_all()

    def _set_strategy(self, val):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        if val == "adc" and params.gut:
            btn_gut = tr("training.conflict.btn_disable_gut")
            btn_cancel = tr("training.conflict.btn_cancel")

            def _on_conflict(button, _gut=btn_gut):
                p = lf.optimization_params()
                if button == _gut:
                    p.gut = False
                    p.set_strategy("adc")
                    if self._handle:
                        self._handle.dirty_all()

            lf.ui.confirm_dialog(
                tr("training.error.adc_gut_title"),
                tr("training.conflict.adc_gut_strategy_message"),
                [btn_gut, btn_cancel],
                _on_conflict)
        else:
            params.set_strategy(val)
            if self._handle:
                self._handle.dirty_all()

    def _set_int_param(self, prop, val_str):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        try:
            setattr(params, prop, int(val_str))
        except (ValueError, TypeError):
            pass

    def _set_mask_mode(self, val_str):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        try:
            params.mask_mode = lf.MaskMode(int(val_str))
        except (ValueError, TypeError):
            pass
        if self._handle:
            self._handle.dirty_all()

    def _set_bg_mode(self, val_str):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        try:
            params.bg_mode = lf.BackgroundMode(int(val_str))
        except (ValueError, TypeError):
            pass
        if self._handle:
            self._handle.dirty_all()

    def _set_resize_factor(self, val_str):
        d = lf.dataset_params()
        if not d or not d.has_params():
            return
        try:
            d.resize_factor = int(val_str)
        except (ValueError, TypeError, RuntimeError):
            pass

    def _set_num_prop(self, prop, val_str, dtype, min_v, max_v):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        try:
            val = dtype(val_str)
        except (ValueError, TypeError):
            return
        if min_v is not None:
            val = max(val, dtype(min_v))
        if max_v is not None:
            val = min(val, dtype(max_v))

        if prop == "steps_scaler":
            params.apply_step_scaling(val)
        elif prop in DIRECT_SET_PROPS:
            setattr(params, prop, val)
        else:
            params.set(prop, val)

    def _set_ppisp_activation_step(self, val_str):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        try:
            params.ppisp_controller_activation_step = max(1, int(val_str))
        except (ValueError, TypeError):
            pass

    def _set_max_width(self, val_str):
        d = lf.dataset_params()
        if not d or not d.has_params():
            return
        try:
            val = int(val_str)
            if 0 < val <= 4096:
                d.max_width = val
        except (ValueError, TypeError, RuntimeError):
            pass

    def _set_new_step_val(self, val_str):
        try:
            self._new_save_step = max(1, int(val_str))
        except (ValueError, TypeError):
            pass

    def _set_slider_prop(self, prop, val):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        try:
            params.set(prop, float(val))
        except (ValueError, TypeError):
            pass

    def _set_bg_color_hex(self, hex_val):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        color = _hex_to_color(hex_val)
        if color:
            params.bg_color = color
            rs = lf.get_render_settings()
            if rs:
                rs.set("background_color", color)
            if self._handle:
                self._handle.dirty_all()

    # ── Event handlers ─────────────────────────────────────

    def _on_num_step(self, handle, event, args):
        if len(args) < 2:
            return
        prop = str(args[0])
        direction = int(args[1])
        self._apply_num_step(prop, direction)
        now = time.monotonic()
        self._step_repeat_prop = prop
        self._step_repeat_dir = direction
        self._step_repeat_start = now
        self._step_repeat_last = now

    def _apply_num_step(self, prop, direction):
        entry = _NUM_PROP_LOOKUP.get(prop)
        if entry:
            params = lf.optimization_params()
            if not params or not params.has_params():
                return
            dtype, fmt, min_v, max_v, step = entry
            current = getattr(params, prop, 0)
            new_val = dtype(current + step * direction)
            if min_v is not None:
                new_val = max(new_val, dtype(min_v))
            if max_v is not None:
                new_val = min(new_val, dtype(max_v))
            self._set_num_prop(prop, str(new_val), dtype, min_v, max_v)
            if self._handle:
                self._handle.dirty(f"{prop}_str")
            return

        if prop == "ppisp_activation_step":
            params = lf.optimization_params()
            if not params or not params.has_params():
                return
            current = params.ppisp_controller_activation_step
            if current < 0:
                return
            params.ppisp_controller_activation_step = max(1, current + 100 * direction)
            if self._handle:
                self._handle.dirty("ppisp_activation_step_str")
        elif prop == "max_width":
            d = lf.dataset_params()
            if not d or not d.has_params():
                return
            d.max_width = max(1, min(4096, d.max_width + 16 * direction))
            if self._handle:
                self._handle.dirty("max_width_str")
        elif prop == "new_step":
            self._new_save_step = max(1, self._new_save_step + 100 * direction)
            if self._handle:
                self._handle.dirty("new_step_str")

    def _on_step_mouseup(self, event):
        self._step_repeat_prop = None

    def _update_step_repeat(self):
        if not self._step_repeat_prop:
            return
        now = time.monotonic()
        if now - self._step_repeat_start < 0.15:
            return
        if now - self._step_repeat_last < 0.01:
            return
        self._step_repeat_last = now
        self._apply_num_step(self._step_repeat_prop, self._step_repeat_dir)

    def _on_toggle_section(self, handle, event, args):
        if not args:
            return
        name = str(args[0])
        if name in self._collapsed:
            self._collapsed.discard(name)
        else:
            self._collapsed.add(name)
        handle.dirty(f"sec_{name}_collapsed")
        handle.dirty(f"sec_{name}_arrow")

    def _on_color_click(self, handle, event, args):
        if not args:
            return
        prop_id = str(args[0])
        if self._color_edit_prop == prop_id:
            self._color_edit_prop = None
            if hasattr(self, '_popup_el') and self._popup_el:
                self._popup_el.set_class("visible", False)
        else:
            self._color_edit_prop = prop_id
            if hasattr(self, '_popup_el') and self._popup_el and event:
                mx = event.get_parameter("mouse_x", "0")
                my = event.get_parameter("mouse_y", "0")
                self._popup_el.set_property("left", f"{mx}px")
                self._popup_el.set_property("top", f"{int(float(my)) + 2}px")
                self._popup_el.set_class("visible", True)
                handle.dirty("picker_r")
                handle.dirty("picker_g")
                handle.dirty("picker_b")
            self._picker_click_handled = True

    def _on_action(self, handle, event, args):
        if not args:
            return
        action = str(args[0])

        if action == "start":
            self._action_start()
        elif action == "pause":
            lf.pause_training()
        elif action == "resume":
            lf.resume_training()
        elif action == "stop":
            lf.stop_training()
        elif action == "reset":
            lf.reset_training()
        elif action == "clear":
            lf.clear_scene()
        elif action == "switch_edit":
            lf.switch_to_edit_mode()
        elif action == "save_checkpoint":
            lf.save_checkpoint()
            self._checkpoint_saved_time = time.time()
        elif action == "browse_bg":
            selected = lf.ui.open_image_file_dialog("")
            if selected:
                params = lf.optimization_params()
                if params and params.has_params():
                    params.bg_image_path = selected
                    if self._handle:
                        self._handle.dirty_all()
        elif action == "clear_bg":
            params = lf.optimization_params()
            if params and params.has_params():
                params.bg_image_path = ""
                if self._handle:
                    self._handle.dirty_all()
        elif action == "add_step":
            params = lf.optimization_params()
            if params and params.has_params() and self._new_save_step > 0:
                params.add_save_step(self._new_save_step)
                self._last_save_steps = []

    def _action_start(self):
        params = lf.optimization_params()
        error = params.validate() if params and params.has_params() else ""
        if error:
            btn_mcmc = tr("training.conflict.btn_use_mcmc")
            btn_gut = tr("training.conflict.btn_disable_gut")
            btn_cancel = tr("training.conflict.btn_cancel")

            def _on_conflict(button, _mcmc=btn_mcmc, _gut=btn_gut):
                p = lf.optimization_params()
                if button == _mcmc:
                    p.set_strategy("mcmc")
                    lf.start_training()
                elif button == _gut:
                    p.gut = False
                    lf.start_training()

            lf.ui.confirm_dialog(
                tr("training.error.adc_gut_title"),
                tr("training.conflict.adc_gut_start_message"),
                [btn_mcmc, btn_gut, btn_cancel],
                _on_conflict)
        else:
            lf.start_training()

    def _on_remove_step_event(self, handle, event, args):
        if not args:
            return
        try:
            idx = int(args[0])
        except (ValueError, TypeError):
            return
        self._on_step_remove(idx)

    def _on_step_remove(self, idx):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        steps = list(params.save_steps)
        if 0 <= idx < len(steps):
            params.remove_save_step(steps[idx])
            self._last_save_steps = []

    def _try_auto_scale_steps(self, params):
        scene = lf.get_scene()
        if scene is None:
            return
        camera_count = scene.active_camera_count
        if camera_count == 0 or camera_count == self._auto_scaled_for_cameras:
            return
        self._auto_scaled_for_cameras = camera_count
        params.auto_scale_steps(camera_count)

    def _draw_controls(self, layout, state, iteration):
        if state == "ready":
            label = tr("training_panel.resume_training") if iteration > 0 else tr("training_panel.start_training")
            if layout.button_styled(label, "success", FULL_WIDTH):
                params = lf.optimization_params()
                error = params.validate() if params.has_params() else ""
                if error:
                    btn_mcmc = tr("training.conflict.btn_use_mcmc")
                    btn_gut = tr("training.conflict.btn_disable_gut")
                    btn_cancel = tr("training.conflict.btn_cancel")
                    def _on_start_conflict(button, _mcmc=btn_mcmc, _gut=btn_gut):
                        p = lf.optimization_params()
                        if button == _mcmc:
                            p.set_strategy("mcmc")
                            lf.start_training()
                        elif button == _gut:
                            p.gut = False
                            lf.start_training()
                    lf.ui.confirm_dialog(
                        tr("training.error.adc_gut_title"),
                        tr("training.conflict.adc_gut_start_message"),
                        [btn_mcmc, btn_gut, btn_cancel],
                        _on_start_conflict)
                else:
                    lf.start_training()
            if iteration > 0:
                if layout.button_styled(tr("training_panel.reset"), "secondary", FULL_WIDTH):
                    lf.reset_training()
            if layout.button_styled(tr("training_panel.clear"), "error", FULL_WIDTH):
                lf.clear_scene()

        elif state == "running":
            if layout.button_styled(tr("training_panel.pause"), "warning", FULL_WIDTH):
                lf.pause_training()

        elif state == "paused":
            if layout.button_styled(tr("training_panel.resume"), "success", FULL_WIDTH):
                lf.resume_training()
            if layout.button_styled(tr("training_panel.reset"), "secondary", FULL_WIDTH):
                lf.reset_training()
            if layout.button_styled(tr("training_panel.stop"), "error", FULL_WIDTH):
                lf.stop_training()

        elif state in ("completed", "stopped"):
            if state == "completed":
                layout.text_colored(tr("status.complete"), COLOR_SUCCESS)
            else:
                layout.text_colored(tr("status.stopped"), COLOR_MUTED)
            if layout.button_styled(tr("training_panel.switch_edit_mode"), "success", FULL_WIDTH):
                lf.switch_to_edit_mode()
            if layout.button_styled(tr("training_panel.reset"), "secondary", FULL_WIDTH):
                lf.reset_training()
            if layout.button_styled(tr("training_panel.clear"), "error", FULL_WIDTH):
                lf.clear_scene()

        elif state == "error":
            layout.text_colored(tr("status.error"), COLOR_ERROR)
            if error_msg := lf.trainer_error():
                layout.text_wrapped(error_msg)
            if layout.button_styled(tr("training_panel.reset"), "secondary", FULL_WIDTH):
                lf.reset_training()
            if layout.button_styled(tr("training_panel.clear"), "error", FULL_WIDTH):
                lf.clear_scene()

        elif state == "stopping":
            layout.text_colored(tr("status.stopping"), COLOR_MUTED)

        if state in ("running", "paused"):
            if layout.button_styled(tr("training_panel.save_checkpoint"), "primary", FULL_WIDTH):
                lf.save_checkpoint()
                self._checkpoint_saved_time = time.time()

            if time.time() - self._checkpoint_saved_time < 2.0:
                theme = lf.ui.theme()
                layout.text_colored(tr("training_panel.checkpoint_saved"), theme.palette.success)

    def _draw_basic_params(self, layout, state, iteration, params):
        can_edit = (state == "ready") and (iteration == 0)
        can_edit_live = state in ("ready", "running", "paused")

        if layout.begin_table("PyBasicParamsTable", 2):
            layout.table_setup_column(tr("common.column_label"), 120.0)
            layout.table_setup_column(tr("common.column_control"), 0.0)

            # -- Structural params (only before training starts) --
            layout.begin_disabled(not can_edit)

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.strategy"))
            layout.table_next_column()
            layout.push_item_width(-1)
            strategy_items = [
                tr("training.options.strategy.mcmc"),
                tr("training.options.strategy.adc"),
            ]
            strategy_idx = 0 if params.strategy == "mcmc" else 1
            changed, new_idx = layout.combo("##py_strategy", strategy_idx, strategy_items)
            if changed:
                if new_idx == 1 and params.gut:
                    btn_gut = tr("training.conflict.btn_disable_gut")
                    btn_cancel = tr("training.conflict.btn_cancel")
                    def _on_strategy_conflict(button, _gut=btn_gut):
                        p = lf.optimization_params()
                        if button == _gut:
                            p.gut = False
                            p.set_strategy("adc")
                    lf.ui.confirm_dialog(
                        tr("training.error.adc_gut_title"),
                        tr("training.conflict.adc_gut_strategy_message"),
                        [btn_gut, btn_cancel],
                        _on_strategy_conflict)
                else:
                    params.set_strategy("mcmc" if new_idx == 0 else "adc")
            layout.pop_item_width()
            if layout.is_item_hovered():
                tooltip = tr("training.tooltip.strategy_gut_conflict") if params.gut else tr("training.tooltip.strategy")
                layout.set_tooltip(tooltip)

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.iterations"))
            layout.table_next_column()
            layout.push_item_width(-1)
            changed, new_val = layout.input_int_formatted("##py_iterations", int(params.iterations), 1000, 5000)
            if changed and new_val > 0:
                params.iterations = new_val
            layout.pop_item_width()
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.iterations"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.max_gaussians"))
            layout.table_next_column()
            layout.push_item_width(-1)
            changed, new_val = layout.input_int_formatted("##py_max_cap", params.max_cap, 10000, 100000)
            if changed and new_val > 0:
                params.max_cap = new_val
            layout.pop_item_width()
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.max_gaussians"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.sh_degree"))
            layout.table_next_column()
            layout.push_item_width(-1)
            changed, new_idx = layout.combo("##py_sh_degree", params.sh_degree, SH_DEGREE_ITEMS)
            if changed:
                params.sh_degree = new_idx
            layout.pop_item_width()
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.sh_degree"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.tile_mode"))
            layout.table_next_column()
            layout.push_item_width(-1)
            tile_idx = {1: 0, 2: 1, 4: 2}.get(params.tile_mode, 0)
            tile_mode_items = [
                tr("training.options.tile.full"),
                tr("training.options.tile.half"),
                tr("training.options.tile.quarter"),
            ]
            changed, new_idx = layout.combo("##py_tile_mode", tile_idx, tile_mode_items)
            if changed:
                params.tile_mode = [1, 2, 4][new_idx]
            layout.pop_item_width()
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.tile_mode"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.steps_scaler"))
            layout.table_next_column()
            layout.push_item_width(-1)
            changed, new_val = layout.input_float("##py_steps_scaler", params.steps_scaler, 0.1, 0.5, "%.2f")
            if changed:
                params.apply_step_scaling(new_val)
            layout.pop_item_width()
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.steps_scaler"))

            layout.end_disabled()

            # -- Live-editable params (available during training) --
            layout.begin_disabled(not can_edit_live)

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.bilateral_grid"))
            layout.table_next_column()
            changed, new_val = layout.checkbox("##py_bilateral_grid", params.use_bilateral_grid)
            if changed:
                params.use_bilateral_grid = new_val
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.bilateral_grid"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.mask_mode"))
            layout.table_next_column()
            layout.push_item_width(-1)
            mask_idx = params.mask_mode.value
            mask_mode_items = [
                tr("training.options.mask.none"),
                tr("training.options.mask.segment"),
                tr("training.options.mask.ignore"),
                tr("training.options.mask.alpha_consistent"),
            ]
            changed, new_idx = layout.combo("##py_mask_mode", mask_idx, mask_mode_items)
            if changed:
                params.mask_mode = lf.MaskMode(new_idx)
            layout.pop_item_width()
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.mask_mode"))

            if params.mask_mode.value != 0:
                layout.table_next_row()
                layout.table_next_column()
                layout.label(tr("training_params.invert_masks"))
                layout.table_next_column()
                changed, new_val = layout.checkbox("##py_invert_masks", params.invert_masks)
                if changed:
                    params.invert_masks = new_val
                if layout.is_item_hovered():
                    layout.set_tooltip(tr("training.tooltip.invert_masks"))

                layout.table_next_row()
                layout.table_next_column()
                layout.label(tr("training_params.use_alpha_as_mask"))
                layout.table_next_column()
                changed, new_val = layout.checkbox("##py_use_alpha_as_mask", params.use_alpha_as_mask)
                if changed:
                    params.use_alpha_as_mask = new_val
                if layout.is_item_hovered():
                    layout.set_tooltip(tr("training.tooltip.use_alpha_as_mask"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.sparsity"))
            layout.table_next_column()
            changed, new_val = layout.checkbox("##py_sparsity", params.enable_sparsity)
            if changed:
                params.enable_sparsity = new_val
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.sparsity"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.gut"))
            layout.table_next_column()
            gut_disabled = (params.strategy == "adc")
            if gut_disabled:
                layout.begin_disabled(True)
            changed, new_val = layout.checkbox("##py_gut", params.gut)
            if changed:
                params.gut = new_val
                self._sync_render_setting("gut", new_val)
            if gut_disabled:
                layout.end_disabled()
            if layout.is_item_hovered():
                tooltip = tr("training.tooltip.gut_adc_conflict") if gut_disabled else tr("training.tooltip.gut")
                layout.set_tooltip(tooltip)

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.undistort"))
            layout.table_next_column()
            changed, new_val = layout.checkbox("##py_undistort", params.undistort)
            if changed:
                params.undistort = new_val
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.undistort"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.mip_filter"))
            layout.table_next_column()
            changed, new_val = layout.checkbox("##py_mip_filter", params.mip_filter)
            if changed:
                params.mip_filter = new_val
                self._sync_render_setting("mip_filter", new_val)
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.mip_filter"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.ppisp"))
            layout.table_next_column()
            changed, new_val = layout.checkbox("##py_ppisp", params.ppisp)
            if changed:
                params.ppisp = new_val
                self._sync_render_setting("apply_appearance_correction", new_val)
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.ppisp"))

            if params.ppisp:
                layout.table_next_row()
                layout.table_next_column()
                layout.label(tr("training_params.ppisp_controller"))
                layout.table_next_column()
                changed, new_val = layout.checkbox("##py_ppisp_controller", params.ppisp_use_controller)
                if changed:
                    params.ppisp_use_controller = new_val
                if layout.is_item_hovered():
                    layout.set_tooltip(tr("training.tooltip.ppisp_controller"))

                if params.ppisp_use_controller:
                    layout.table_next_row()
                    layout.table_next_column()
                    layout.label(tr("training_params.ppisp_activation_step"))
                    layout.table_next_column()
                    is_auto = params.ppisp_controller_activation_step < 0
                    changed, new_auto = layout.checkbox(f"{tr('common.auto')}##py_ppisp_auto_step", is_auto)
                    if changed:
                        params.ppisp_controller_activation_step = -1 if new_auto else max(1, int(params.iterations) - 5000)
                    if not is_auto:
                        layout.same_line()
                        layout.push_item_width(-1)
                        changed, new_val = layout.input_int_formatted("##py_ppisp_ctrl_step", params.ppisp_controller_activation_step, 1000, 5000)
                        if changed:
                            params.ppisp_controller_activation_step = max(1, new_val)
                        layout.pop_item_width()
                    if layout.is_item_hovered():
                        layout.set_tooltip(tr("training.tooltip.ppisp_activation_step"))

                    layout.table_next_row()
                    layout.table_next_column()
                    layout.label(tr("training_params.ppisp_controller_lr"))
                    layout.table_next_column()
                    layout.push_item_width(-1)
                    changed, new_val = layout.input_float("##py_ppisp_ctrl_lr", params.ppisp_controller_lr, 0.0001, 0.001, "%.5f")
                    if changed:
                        params.ppisp_controller_lr = new_val
                    layout.pop_item_width()
                    if layout.is_item_hovered():
                        layout.set_tooltip(tr("training.tooltip.ppisp_controller_lr"))

                    layout.table_next_row()
                    layout.table_next_column()
                    layout.label(tr("training_params.ppisp_freeze_gaussians"))
                    layout.table_next_column()
                    changed, new_val = layout.checkbox("##py_ppisp_freeze", params.ppisp_freeze_gaussians)
                    if changed:
                        params.ppisp_freeze_gaussians = new_val
                    if layout.is_item_hovered():
                        layout.set_tooltip(tr("training.tooltip.ppisp_freeze_gaussians"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.bg_mode"))
            layout.table_next_column()
            layout.push_item_width(-1)
            bg_idx = params.bg_mode.value
            bg_mode_items = [
                tr("training.options.bg.color"),
                tr("training.options.bg.modulation"),
                tr("training.options.bg.image"),
                tr("training.options.bg.random"),
            ]
            changed, new_idx = layout.combo("##py_bg_mode", bg_idx, bg_mode_items)
            if changed:
                params.bg_mode = lf.BackgroundMode(new_idx)
            layout.pop_item_width()

            bg_mode_val = params.bg_mode.value
            if bg_mode_val in (0, 1):
                layout.table_next_row()
                layout.table_next_column()
                layout.label(tr("training_params.bg_color"))
                layout.table_next_column()
                layout.push_item_width(-1)
                changed, new_color = layout.color_edit3("##py_bg_color", params.bg_color)
                if changed:
                    params.bg_color = new_color
                    self._sync_render_setting("background_color", new_color)
                layout.pop_item_width()

            if bg_mode_val == 2:
                layout.table_next_row()
                layout.table_next_column()
                layout.label(tr("training_params.bg_image"))
                layout.table_next_column()
                layout.push_item_width(-1)
                img_path = params.bg_image_path
                display = os.path.basename(img_path) if img_path else tr("training.value.none")
                layout.label(display)
                layout.pop_item_width()

                layout.table_next_row()
                layout.table_next_column()
                layout.table_next_column()
                if layout.button(tr("training_params.bg_image_browse") + "##py_bg_browse"):
                    selected = lf.ui.open_image_file_dialog("")
                    if selected:
                        params.bg_image_path = selected
                layout.same_line()
                if img_path and layout.button(tr("training_params.bg_image_clear") + "##py_bg_clear"):
                    params.bg_image_path = ""

            layout.end_disabled()
            layout.end_table()

    def _draw_advanced_params(self, layout, state, iteration, params):
        can_edit = (state == "ready") and (iteration == 0)
        dataset = lf.dataset_params()
        dataset_can_edit = dataset.can_edit() if dataset.has_params() else False

        if layout.tree_node(tr("training.section.dataset") + "##py"):
            table_open = False
            try:
                if dataset.has_params():
                    table_open = layout.begin_table("PyDatasetTable", 2)
                    if table_open:
                        layout.table_setup_column(tr("common.column_label"), 120.0)
                        layout.table_setup_column(tr("common.column_control"), 0.0)

                        data_path = dataset.data_path
                        self._table_text(layout, tr("training.dataset.path"), os.path.basename(data_path) if data_path else tr("training.value.none"))

                        images = dataset.images
                        self._table_text(layout, tr("training.dataset.images"), images if images else tr("training.value.default"))

                        layout.table_next_row()
                        layout.table_next_column()
                        layout.label(tr("training.dataset.resize_factor"))
                        layout.table_next_column()
                        if dataset_can_edit:
                            layout.push_item_width(-1)
                            resize_options = [-1, 1, 2, 4, 8]
                            resize_labels = [tr("common.auto"), "1", "2", "4", "8"]
                            current_idx = resize_options.index(dataset.resize_factor) if dataset.resize_factor in resize_options else 0
                            changed, new_idx = layout.combo("##py_resize_factor", current_idx, resize_labels)
                            if changed:
                                dataset.resize_factor = resize_options[new_idx]
                            layout.pop_item_width()
                        else:
                            layout.label(tr("common.auto") if dataset.resize_factor < 0 else str(dataset.resize_factor))

                        layout.table_next_row()
                        layout.table_next_column()
                        layout.label(tr("training.dataset.max_width"))
                        layout.table_next_column()
                        if dataset_can_edit:
                            layout.push_item_width(-1)
                            changed, new_val = layout.input_int("##py_max_width", dataset.max_width, 80, 400)
                            if changed and 0 < new_val <= 4096:
                                dataset.max_width = new_val
                            layout.pop_item_width()
                        else:
                            layout.label(str(dataset.max_width))

                        layout.table_next_row()
                        layout.table_next_column()
                        layout.label(tr("training.dataset.cpu_cache"))
                        layout.table_next_column()
                        if dataset_can_edit:
                            changed, new_val = layout.checkbox("##py_cpu_cache", dataset.use_cpu_cache)
                            if changed:
                                dataset.use_cpu_cache = new_val
                        else:
                            layout.label(tr("training.status.enabled") if dataset.use_cpu_cache else tr("training.status.disabled"))

                        layout.table_next_row()
                        layout.table_next_column()
                        layout.label(tr("training.dataset.fs_cache"))
                        layout.table_next_column()
                        if dataset_can_edit:
                            changed, new_val = layout.checkbox("##py_fs_cache", dataset.use_fs_cache)
                            if changed:
                                dataset.use_fs_cache = new_val
                        else:
                            layout.label(tr("training.status.enabled") if dataset.use_fs_cache else tr("training.status.disabled"))

                        out_path = dataset.output_path
                        self._table_text(layout, tr("training.dataset.output"),
                                       os.path.basename(out_path) if out_path else tr("training.value.not_set"))
                else:
                    layout.label(tr("training_panel.no_dataset_loaded"))
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if layout.tree_node(tr("training.section.optimization") + "##py"):
            table_open = False
            try:
                table_open = layout.begin_table("PyOptTable", 2)
                if table_open:
                    layout.table_setup_column(tr("common.column_label"), 120.0)
                    layout.table_setup_column(tr("common.column_control"), 0.0)

                    layout.begin_disabled(not can_edit)
                    self._table_text(layout, tr("training_params.strategy"), params.strategy.upper())

                    layout.table_next_row()
                    layout.table_next_column()
                    layout.text_colored(tr("training.opt.learning_rates"), (0.6, 0.6, 0.6, 1.0))
                    layout.table_next_column()

                    self._input_float_row(layout, tr("training.opt.lr.position"), "means_lr", params, params.means_lr, 0.000001, 0.00001, "%.6f")
                    self._input_float_row(layout, tr("training.opt.lr.sh_coeff"), "shs_lr", params, params.shs_lr, 0.0001, 0.001, "%.4f")
                    self._input_float_row(layout, tr("training.opt.lr.opacity"), "opacity_lr", params, params.opacity_lr, 0.001, 0.01, "%.4f")
                    self._input_float_row(layout, tr("training.opt.lr.scaling"), "scaling_lr", params, params.scaling_lr, 0.0001, 0.001, "%.4f")
                    self._input_float_row(layout, tr("training.opt.lr.rotation"), "rotation_lr", params, params.rotation_lr, 0.0001, 0.001, "%.4f")

                    layout.table_next_row()
                    layout.table_next_column()
                    layout.text_colored(tr("training.section.refinement"), (0.6, 0.6, 0.6, 1.0))
                    layout.table_next_column()

                    self._input_int_row(layout, tr("training.refinement.refine_every"), "refine_every", params, 10, 100)
                    self._input_int_row(layout, tr("training.refinement.start_refine"), "start_refine", params, 100, 500)
                    self._input_int_row(layout, tr("training.refinement.stop_refine"), "stop_refine", params, 1000, 5000)
                    self._input_float_prop_row(layout, tr("training.refinement.gradient_thr"), "grad_threshold", params, 0.00001, 0.0001, "%.6f")
                    self._input_int_row(layout, tr("training.refinement.reset_every"), "reset_every", params, 100, 1000)
                    self._input_int_row(layout, tr("training.refinement.sh_upgrade_every"), "sh_degree_interval", params, 100, 500)
                    layout.end_disabled()
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if params.use_bilateral_grid and layout.tree_node(tr("training.section.bilateral_grid") + "##py"):
            table_open = False
            try:
                table_open = layout.begin_table("PyBilateralTable", 2)
                if table_open:
                    layout.table_setup_column(tr("common.column_label"), 140.0)
                    layout.table_setup_column(tr("common.column_control"), 0.0)
                    layout.begin_disabled(not can_edit)
                    self._table_prop(layout, params, "bilateral_grid_x", tr("training.bilateral.grid_x"))
                    self._table_prop(layout, params, "bilateral_grid_y", tr("training.bilateral.grid_y"))
                    self._table_prop(layout, params, "bilateral_grid_w", tr("training.bilateral.grid_w"))
                    self._table_prop(layout, params, "bilateral_grid_lr", tr("training.bilateral.learning_rate"))
                    layout.end_disabled()
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if layout.tree_node(tr("training.section.losses") + "##py"):
            table_open = False
            try:
                table_open = layout.begin_table("PyLossTable", 2)
                if table_open:
                    layout.table_setup_column(tr("common.column_label"), 140.0)
                    layout.table_setup_column(tr("common.column_control"), 0.0)
                    layout.begin_disabled(not can_edit)
                    self._slider_float_row(layout, tr("training.losses.lambda_dssim"), "lambda_dssim", params, 0.0, 1.0)
                    self._input_float_prop_row(layout, tr("training.losses.opacity_reg"), "opacity_reg", params, 0.001, 0.01, "%.4f")
                    self._input_float_prop_row(layout, tr("training.losses.scale_reg"), "scale_reg", params, 0.001, 0.01, "%.4f")
                    self._input_float_prop_row(layout, tr("training.losses.tv_loss_weight"), "tv_loss_weight", params, 1.0, 5.0, "%.1f")
                    layout.end_disabled()
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if layout.tree_node(tr("training.section.initialization") + "##py"):
            table_open = False
            try:
                table_open = layout.begin_table("PyInitTable", 2)
                if table_open:
                    layout.table_setup_column(tr("common.column_label"), 140.0)
                    layout.table_setup_column(tr("common.column_control"), 0.0)
                    layout.begin_disabled(not can_edit)
                    self._slider_float_row(layout, tr("training.init.init_opacity"), "init_opacity", params, 0.01, 1.0)
                    self._input_float_prop_row(layout, tr("training.init.init_scaling"), "init_scaling", params, 0.01, 0.1, "%.3f")

                    layout.table_next_row()
                    layout.table_next_column()
                    layout.label(tr("training.init.random_init"))
                    layout.table_next_column()
                    changed, new_val = layout.checkbox("##py_random", params.random)
                    if changed:
                        params.random = new_val

                    if params.random:
                        self._input_int_row(layout, tr("training.init.num_points"), "init_num_pts", params, 10000, 50000)
                        self._input_float_prop_row(layout, tr("training.init.extent"), "init_extent", params, 0.5, 1.0, "%.1f")
                    layout.end_disabled()
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if params.strategy == "adc" and layout.tree_node(tr("training_panel.pruning_growing") + "##py"):
            table_open = False
            try:
                table_open = layout.begin_table("PyADCTable", 2)
                if table_open:
                    layout.table_setup_column(tr("common.column_label"), 140.0)
                    layout.table_setup_column(tr("common.column_control"), 0.0)
                    layout.begin_disabled(not can_edit)
                    self._input_float_prop_row(layout, tr("training.thresholds.min_opacity"), "min_opacity", params, 0.001, 0.01, "%.4f", min_val=0.0)
                    self._input_float_prop_row(layout, tr("training.thresholds.prune_opacity"), "prune_opacity", params, 0.001, 0.01, "%.4f", min_val=0.0)
                    self._input_float_prop_row(layout, tr("training.thresholds.grow_scale_3d"), "grow_scale3d", params, 0.001, 0.01, "%.4f", min_val=0.0)
                    self._input_float_prop_row(layout, tr("training.thresholds.grow_scale_2d"), "grow_scale2d", params, 0.01, 0.05, "%.3f", min_val=0.0)
                    self._input_float_prop_row(layout, tr("training.thresholds.prune_scale_3d"), "prune_scale3d", params, 0.01, 0.1, "%.3f", min_val=0.0)
                    self._input_float_prop_row(layout, tr("training.thresholds.prune_scale_2d"), "prune_scale2d", params, 0.01, 0.1, "%.3f", min_val=0.0)
                    self._input_int_row(layout, tr("training.thresholds.pause_after_reset"), "pause_refine_after_reset", params, 100, 500)
                    self._table_prop(layout, params, "revised_opacity", tr("training.thresholds.revised_opacity"))
                    layout.end_disabled()
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if params.enable_sparsity and layout.tree_node(tr("training_panel.sparsity") + "##py"):
            table_open = False
            try:
                table_open = layout.begin_table("PySparsityTable", 2)
                if table_open:
                    layout.table_setup_column(tr("common.column_label"), 140.0)
                    layout.table_setup_column(tr("common.column_control"), 0.0)
                    layout.begin_disabled(not can_edit)
                    self._input_int_row(layout, tr("training_params.sparsify_steps"), "sparsify_steps", params, 1000, 5000)
                    self._input_float_prop_row(layout, tr("training_params.init_rho"), "init_rho", params, 0.001, 0.01, "%.4f")
                    self._slider_float_row(layout, tr("training_params.prune_ratio"), "prune_ratio", params, 0.0, 1.0)
                    layout.end_disabled()
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if layout.tree_node(tr("training_panel.save_steps") + "##py"):
            try:
                self._draw_save_steps(layout, params, can_edit)
            finally:
                layout.tree_pop()

    def _draw_save_steps(self, layout, params, can_edit):
        theme = lf.ui.theme()
        steps = list(params.save_steps)

        if can_edit:
            _, self._new_save_step = layout.input_int_formatted("##py_new_step", self._new_save_step, 100, 1000)
            layout.same_line()
            if layout.button(tr("common.add") + "##py_add"):
                if self._new_save_step > 0:
                    params.add_save_step(self._new_save_step)

            layout.separator()

            for i, step in enumerate(steps):
                layout.push_id(f"py_step_{i}")
                layout.set_next_item_width(100)
                changed, new_val = layout.input_int_formatted("##step", step, 0, 0)
                if changed and new_val > 0 and new_val != step:
                    params.remove_save_step(step)
                    params.add_save_step(new_val)
                layout.same_line()
                if layout.button(tr("common.remove") + "##rm"):
                    params.remove_save_step(step)
                layout.pop_id()

            if not steps:
                layout.text_colored(tr("training_panel.no_save_steps"), theme.palette.text_dim)
        else:
            if steps:
                layout.label(", ".join(str(s) for s in steps))
            else:
                layout.text_colored(tr("training_panel.no_save_steps"), theme.palette.text_dim)

    def _input_int_row(self, layout, label, prop_id, params, step, step_fast):
        layout.table_next_row()
        layout.table_next_column()
        layout.label(label)
        layout.table_next_column()
        layout.push_item_width(-1)
        current_val = params.get(prop_id)
        if current_val is None:
            current_val = 0
        changed, new_val = layout.input_int_formatted(f"##py_{prop_id}", int(current_val), step, step_fast)
        if changed and new_val >= 0:
            params.set(prop_id, new_val)
        layout.pop_item_width()

    def _input_float_prop_row(self, layout, label, prop_id, params, step, step_fast, fmt, min_val=None):
        layout.table_next_row()
        layout.table_next_column()
        layout.label(label)
        layout.table_next_column()
        layout.push_item_width(-1)
        current_val = params.get(prop_id)
        if current_val is None:
            current_val = 0.0
        changed, new_val = layout.input_float(f"##py_{prop_id}", float(current_val), step, step_fast, fmt)
        if changed:
            if min_val is not None:
                new_val = max(min_val, new_val)
            params.set(prop_id, new_val)
        layout.pop_item_width()

    def _slider_float_row(self, layout, label, prop_id, params, min_val, max_val):
        layout.table_next_row()
        layout.table_next_column()
        layout.label(label)
        layout.table_next_column()
        layout.push_item_width(-1)
        current_val = params.get(prop_id)
        if current_val is None:
            current_val = 0.0
        changed, new_val = layout.slider_float(f"##py_{prop_id}", float(current_val), min_val, max_val)
        if changed:
            params.set(prop_id, new_val)
        layout.pop_item_width()

    def _input_float_row(self, layout, label, prop_id, params, value, step, step_fast, fmt):
        layout.table_next_row()
        layout.table_next_column()
        layout.label(label)
        layout.table_next_column()
        layout.push_item_width(-1)
        changed, new_val = layout.input_float(f"##py_{prop_id}", value, step, step_fast, fmt)
        if changed:
            setattr(params, prop_id, new_val)
        layout.pop_item_width()

    def _table_prop(self, layout, params, prop_id, label):
        layout.table_next_row()
        layout.table_next_column()
        layout.label(label)
        layout.table_next_column()
        layout.push_item_width(-1)
        layout.push_id(f"py_{prop_id}")
        layout.prop(params, prop_id)
        layout.pop_id()
        layout.pop_item_width()

    def _table_text(self, layout, label, value):
        layout.table_next_row()
        layout.table_next_column()
        layout.label(label)
        layout.table_next_column()
        layout.label(value)

    def _draw_flipbook_section(self, layout, state):
        """Draw the 4D Flipbook training controls."""
        fb = _flipbook_state
        can_start = (state in ("ready", "completed", "stopped")) and not fb.is_running

        layout.spacing()
        layout.text_disabled("Train one model per time step (4D dataset required).")
        layout.spacing()

        # Keyframe stride
        layout.push_item_width(-1)
        stride_labels = ["1 (all frames)", "2", "4", "8"]
        stride_values = [1, 2, 4, 8]
        cur_stride_idx = stride_values.index(fb.params.keyframe_stride) if fb.params.keyframe_stride in stride_values else 0
        changed, new_idx = layout.combo("Keyframe Stride##fb", cur_stride_idx, stride_labels)
        if changed:
            fb.params.keyframe_stride = stride_values[new_idx]
        layout.pop_item_width()
        if layout.is_item_hovered():
            layout.set_tooltip("Train every N-th time step (1 = all frames)")

        # Warm start
        _, fb.params.warm_start = layout.checkbox("Warm-Start from Prev Frame##fb", fb.params.warm_start)
        if layout.is_item_hovered():
            layout.set_tooltip("Initialise each frame's model from the previous frame's trained parameters")

        # Per-frame iterations
        _, fb.params.iterations_per_frame = layout.input_int(
            "Iters/Frame##fb", fb.params.iterations_per_frame, 1000, 5000
        )
        fb.params.iterations_per_frame = max(0, fb.params.iterations_per_frame)
        if layout.is_item_hovered():
            layout.set_tooltip("Per-frame iteration budget (0 = use global Iterations setting)")

        # Export per-frame models
        _, fb.params.export_per_frame = layout.checkbox("Export Per-Frame Models##fb", fb.params.export_per_frame)
        if layout.is_item_hovered():
            layout.set_tooltip("Write PLY files + manifest JSON for each trained frame")

        layout.spacing()

        # Status / result summary
        if fb.result is not None:
            layout.text_colored(
                f"Result: {len(fb.result)} frame(s) available for playback.",
                COLOR_SUCCESS,
            )
        elif fb.is_running:
            layout.text_colored("Flipbook training in progress…", COLOR_MUTED)

        layout.spacing()

        avail_w, _ = layout.get_content_region_avail()
        if not can_start:
            layout.begin_disabled()
        if layout.button_styled("Start Flipbook Training##fb", "primary", (avail_w, 0)):
            lf.log_info("Flipbook training requested via UI (stride=%d, warm_start=%s)",
                        fb.params.keyframe_stride, fb.params.warm_start)
            # Actual orchestration is delegated to the application layer;
            # the UI only sets intent and relies on the app to call FlipbookTrainer.
            # Here we emit a named event that the application can subscribe to.
            try:
                lf.ui.emit_event("flipbook_train_requested", {
                    "keyframe_stride": fb.params.keyframe_stride,
                    "warm_start": fb.params.warm_start,
                    "iterations_per_frame": fb.params.iterations_per_frame,
                    "export_per_frame": fb.params.export_per_frame,
                })
            except AttributeError:
                # lf.ui.emit_event may not exist in all versions; log and continue.
                lf.log_info("Flipbook: emit_event not available; start training manually.")
        if not can_start:
            layout.end_disabled()

    def _draw_status(self, layout, state, iteration):
        layout.separator()

        state_labels = {
            "idle": tr("training_panel.idle"),
            "ready": tr("status.ready") if iteration == 0 else tr("training_panel.resume"),
            "running": tr("training_panel.running"),
            "paused": tr("status.paused"),
            "stopping": tr("status.stopping"),
            "completed": tr("status.complete"),
            "stopped": tr("status.stopped"),
            "error": tr("status.error"),
        }
        unknown_state = tr("status.unknown")
        layout.label(f"{tr('status.mode')}: {state_labels.get(state, unknown_state)}")

        _rate_tracker.add_sample(iteration)
        rate = _rate_tracker.get_rate()
        layout.label(f"{tr('status.iteration')} {iteration:,} ({rate:.1f} {tr('training_panel.iters_per_sec')})")
        layout.label(tr("progress.num_splats") % f"{AppState.num_gaussians.value:,}")

        max_iter = AppState.max_iterations.value
        if max_iter > 0 and iteration > 0:
            layout.progress_bar(iteration / max_iter, f"{iteration:,}/{max_iter:,}")

        loss_data = lf.loss_buffer()
        if loss_data:
            min_val = min(loss_data)
            max_val = max(loss_data)
            if min_val == max_val:
                min_val -= 1.0
                max_val += 1.0
            else:
                margin = (max_val - min_val) * 0.05
                min_val -= margin
                max_val += margin
            loss_label = f"{tr('status.loss')}: {loss_data[-1]:.4f}"
            layout.plot_lines(loss_label, loss_data, min_val, max_val, (-1, 60))
