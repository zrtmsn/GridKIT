# ─────────────────────────────────────────────────────────────
# map_ui/map_widget_sections.py
#
# Compatibility wrapper for map_widget.py.
# The actual section implementations are split into smaller files.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from map_ui.map_widget_households import show_household_configuration
from map_ui.map_widget_training import show_training_section


__all__ = [
    "show_household_configuration",
    "show_training_section",
]