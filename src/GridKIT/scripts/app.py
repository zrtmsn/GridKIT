# scripts/app.py
# ─────────────────────────────────────────────────────────────
# GridKIT — unified app: Karte (build a network, configure households, save
# & train) and Dashboard (browse saved runs, watch training progress, view
# results) as two pages of ONE Streamlit app/process, so building a network
# and then checking on its training doesn't need two separate `streamlit
# run` processes / browser tabs.
#
# Each page's actual content lives in its own module and stays independently
# runnable (map_ui/map_widget.py, dashboard/app.py) — this file only composes
# them via st.navigation. It is allowed to import across map_ui/dashboard
# (unlike those modules themselves, which only import from core) because it
# is an orchestrator app, not a shared module.
#
# Run:  streamlit run src/GridKIT/scripts/app.py
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import sys
from pathlib import Path

# ── path bootstrap (so `streamlit run` finds the packages) ──
_SCRIPT_DIR = Path(__file__).resolve().parent.parent   # src/GridKIT/
_SRC_DIR = _SCRIPT_DIR.parent                            # src/
for _p in (str(_SCRIPT_DIR), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from map_ui.map_widget import render_map_ui
from dashboard.app import render_dashboard


def main() -> None:
    st.set_page_config(page_title="GridKIT", layout="wide")
    nav = st.navigation({
        "GridKIT": [
            st.Page(render_map_ui, title="Karte", icon="🗺️"),
            st.Page(render_dashboard, title="Dashboard", icon="📊"),
        ],
    })
    nav.run()


if __name__ == "__main__":
    main()
