/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "sequencer/sequencer_controller.hpp"

namespace lfs::vis::gui {
    class PanelLayoutManager;
} // namespace lfs::vis::gui

namespace lfs::vis {

    // Global, Sequencer-driven time context.
    //
    // Provides a central query API for the current playhead time (sourced from
    // SequencerController) and a toggle to show/hide the Sequencer panel.
    // Subsystems that need to be time-aware (rendering, dataset preview, future
    // 4D systems) can query this context without depending on the GUI layer
    // directly.
    //
    // Lifecycle: bind() must be called after the GuiManager is initialised.
    class GlobalTimeContext {
    public:
        // Bind to the sequencer controller and panel layout manager that back
        // this context.  Safe to call more than once (rebinds in place).
        void bind(const SequencerController* controller,
                  gui::PanelLayoutManager* layout) {
            controller_ = controller;
            layout_ = layout;
        }

        // Returns the current playhead position in seconds.
        // Returns 0.0f when no controller is bound or the timeline is empty.
        [[nodiscard]] float currentTime() const {
            return controller_ ? controller_->playhead() : 0.0f;
        }

        // Returns true while the sequencer is actively playing.
        [[nodiscard]] bool isPlaying() const {
            return controller_ && controller_->isPlaying();
        }

        // Returns true when at least one keyframe has been added to the
        // timeline (i.e. time-aware operations have meaningful data).
        [[nodiscard]] bool hasTimeline() const {
            return controller_ && !controller_->timeline().empty();
        }

        // Returns true when the Sequencer panel is currently shown.
        // Implemented in global_time_context.cpp to avoid pulling in imgui.h.
        [[nodiscard]] bool isSequencerVisible() const;

        // Show or hide the Sequencer panel globally.
        // Implemented in global_time_context.cpp to avoid pulling in imgui.h.
        void setSequencerVisible(bool visible);

    private:
        const SequencerController* controller_ = nullptr;
        gui::PanelLayoutManager* layout_ = nullptr;
    };

} // namespace lfs::vis
