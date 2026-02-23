/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/global_time_context.hpp"
#include "gui/panel_layout.hpp"

namespace lfs::vis {

    bool GlobalTimeContext::isSequencerVisible() const {
        return layout_ && layout_->isShowSequencer();
    }

    void GlobalTimeContext::setSequencerVisible(const bool visible) {
        if (layout_)
            layout_->setShowSequencer(visible);
    }

} // namespace lfs::vis
