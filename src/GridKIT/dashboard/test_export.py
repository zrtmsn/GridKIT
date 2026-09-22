# dashboard/test_export.py
import io
import json

import pandas as pd

from dashboard.export import (
    CSV_DECIMAL,
    CSV_SEPARATOR,
    filename,
    to_csv_bytes,
    to_json_bytes,
)


def _frame():
    return pd.DataFrame({
        "Gerät": ["Wärmepumpe", "Batterie"],
        "Leistung": [1.5, -2.25],
    })


# ── CSV as German Excel expects it ───────────────────────────
def test_csv_uses_semicolon_and_decimal_comma():
    text = to_csv_bytes(_frame()).decode("utf-8-sig")
    assert f"Gerät{CSV_SEPARATOR}Leistung" in text
    assert f"1{CSV_DECIMAL}5" in text


def test_csv_starts_with_a_bom_so_excel_detects_utf8():
    # without it Excel guesses the codepage and mangles every umlaut
    assert to_csv_bytes(_frame()).startswith(b"\xef\xbb\xbf")


def test_csv_keeps_umlauts_readable_after_the_bom():
    text = to_csv_bytes(_frame()).decode("utf-8-sig")
    assert "Wärmepumpe" in text


def test_csv_has_no_index_column():
    first = to_csv_bytes(_frame()).decode("utf-8-sig").splitlines()[0]
    assert first.startswith("Gerät")


def test_csv_round_trips_with_matching_settings():
    raw = to_csv_bytes(_frame()).decode("utf-8-sig")
    back = pd.read_csv(io.StringIO(raw), sep=CSV_SEPARATOR, decimal=CSV_DECIMAL)
    assert back["Leistung"].tolist() == [1.5, -2.25]


# ── JSON ─────────────────────────────────────────────────────
def test_json_is_records_with_real_umlauts():
    data = json.loads(to_json_bytes(_frame()).decode("utf-8"))
    assert data[0]["Gerät"] == "Wärmepumpe"


def test_json_keeps_dots_as_decimal_separator():
    data = json.loads(to_json_bytes(_frame()).decode("utf-8"))
    assert data[1]["Leistung"] == -2.25


# ── Filenames ────────────────────────────────────────────────
def test_filename_transliterates_umlauts_instead_of_dropping_them():
    assert filename("Überlast-Matrix", "csv") == "ueberlast-matrix.csv"


def test_filename_replaces_spaces_and_collapses_runs():
    assert filename("Geräte  &  Haushalte", "json") == "geraete_haushalte.json"


def test_filename_strips_path_characters():
    assert "/" not in filename("a/b:c", "csv")
    assert filename("a/b:c", "csv") == "a_b_c.csv"


def test_filename_falls_back_when_the_label_is_empty():
    assert filename("   ", "csv") == "export.csv"


def test_filename_keeps_the_suffix():
    assert filename("Tagesverlauf", "json").endswith(".json")
