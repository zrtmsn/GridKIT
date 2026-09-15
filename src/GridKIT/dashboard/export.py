# dashboard/export.py
# ─────────────────────────────────────────────────────────────
# Download-Schaltflächen für die Tabellen hinter den Diagrammen.
#
# CSV here means CSV as German Excel expects it: semicolon separator, decimal
# comma, and a UTF-8 BOM. Without the BOM Excel guesses the codepage and turns
# "Wärmepumpe" into "WÃ¤rmepumpe"; with a plain comma separator it drops every
# row into a single column. Both make the export useless to exactly the people
# who would open it.
#
# JSON is offered alongside for anything that will be read by a program rather
# than a spreadsheet, and keeps plain dots and unescaped umlauts.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import pandas as pd

CSV_SEPARATOR = ";"
CSV_DECIMAL = ","
#: utf-8-sig writes the byte-order mark Excel needs to detect UTF-8.
CSV_ENCODING = "utf-8-sig"


# ══════════════════════════════════════════════════════════════
# Reine Helfer (kein Streamlit — unit-testbar)
# ══════════════════════════════════════════════════════════════
def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """CSV a German Excel opens correctly, as bytes."""
    return frame.to_csv(
        index=False, sep=CSV_SEPARATOR, decimal=CSV_DECIMAL,
    ).encode(CSV_ENCODING)


def to_json_bytes(frame: pd.DataFrame) -> bytes:
    """Records-oriented JSON, umlauts intact."""
    return frame.to_json(
        orient="records", indent=2, force_ascii=False,
    ).encode("utf-8")


_UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def filename(basename: str, suffix: str) -> str:
    """Download filename from a German label ('Überlast-Matrix' → 'ueberlast-matrix.csv').

    Umlauts are transliterated rather than stripped, so the name still reads as
    the thing it came from once it is sitting in a downloads folder.
    """
    name = basename.strip().lower().translate(_UMLAUTS)
    name = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)
    while "__" in name:
        name = name.replace("__", "_")
    name = name.strip("_")
    return f"{name or 'export'}.{suffix}"


# ══════════════════════════════════════════════════════════════
# Streamlit-Ansicht
# ══════════════════════════════════════════════════════════════
def download_pair(frame: pd.DataFrame, basename: str, key: str,
                  label: str = "Daten zu diesem Diagramm") -> None:  # pragma: no cover (UI)
    """A CSV/JSON download pair for the data behind a chart.

    Rendered inside an expander so it never competes with the chart itself —
    the numbers are there when someone wants them, invisible when they don't.
    """
    import streamlit as st

    if frame is None or frame.empty:
        return
    with st.expander(f"⤓ {label}"):
        col_csv, col_json = st.columns(2)
        col_csv.download_button(
            "CSV (Excel)", data=to_csv_bytes(frame),
            file_name=filename(basename, "csv"), mime="text/csv",
            key=f"{key}_csv", width="stretch",
        )
        col_json.download_button(
            "JSON", data=to_json_bytes(frame),
            file_name=filename(basename, "json"), mime="application/json",
            key=f"{key}_json", width="stretch",
        )
        st.caption(
            f"CSV mit Semikolon und Dezimalkomma, so öffnet Excel die Datei "
            f"direkt richtig. {len(frame)} Zeilen."
        )
