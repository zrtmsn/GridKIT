# dashboard/app.py
# ─────────────────────────────────────────────────────────────
# Streamlit dashboard for the GridKIT scenario × penetration experiment.
# Reads outputs/summary.json + outputs/timelines.json (produced by
# scripts/run_experiment.py) and renders the §14a curtailment comparison.
#
# Run:  streamlit run src/GridKIT/dashboard/app.py
#
# UI text is German; SCENARIO_ORDER/SCENARIO_COLORS keys stay in English
# on purpose — they match the raw "scenario" strings written by the
# backend's summary.json/timelines.json. Translating those keys would
# silently break every filter/lookup against the data. SCENARIO_LABELS_DE
# is the only place scenario names are translated, and it's used purely
# for display (chart legends, selectbox labels via format_func).
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ── path bootstrap (so `streamlit run` finds the packages) ──
# Needed because this file is run directly as a script, not imported as part of
# the package: without it `from dashboard...` below fails unless PYTHONPATH
# happens to be set outside. Same bootstrap as scripts/grid_designer.py.
_PKG_DIR = Path(__file__).resolve().parent.parent   # src/GridKIT/
_SRC_DIR = _PKG_DIR.parent                          # src/
for _p in (str(_PKG_DIR), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.auslastung import render_auslastung
from dashboard.controls import scenario_penetration_picker
from dashboard.geraete import render_geraete
from dashboard.training import render_training
from dashboard.ueberblick import render_ueberblick

OUTPUT_DIR = Path(os.environ.get("GRIDKIT_OUTPUT_DIR", "outputs"))
EPISODE_START_HOUR = 12
TIMESTEP_HOURS = 0.25

SCENARIO_ORDER = [
    "1: flat / immediate",
    "2: price-follow (manual)",
    "2: price-follow (automated)",
    "3: selfish RL",
]
SCENARIO_COLORS = {
    "1: flat / immediate": "#6c757d",
    "2: price-follow (manual)": "#f4a261",
    "2: price-follow (automated)": "#e76f51",
    "3: selfish RL": "#2a9d8f",
}
SCENARIO_LABELS_DE = {
    "1: flat / immediate": "1: konstant / sofort",
    "2: price-follow (manual)": "2: preisorientiert (manuell)",
    "2: price-follow (automated)": "2: preisorientiert (automatisiert)",
    "3: selfish RL": "3: eigennütziges RL",
}


def _de(scenario: str) -> str:
    """German display label for a raw scenario key; falls back to the key itself."""
    return SCENARIO_LABELS_DE.get(scenario, scenario)


def _load(name: str):
    """Read one result file; None when it is missing, unreadable or malformed.

    A run killed mid-write leaves truncated JSON behind, and that used to take
    the whole dashboard down with a raw parser traceback. Returning None lets
    each view say what is missing instead.
    """
    path = OUTPUT_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        st.error(
            f"`{name}` konnte nicht gelesen werden ({type(exc).__name__}). "
            "Die Datei ist vermutlich unvollständig — das passiert, wenn ein Lauf "
            "abgebrochen wurde. Experiment erneut ausführen."
        )
        return None


def _hours_axis(n: int) -> np.ndarray:
    return (EPISODE_START_HOUR + np.arange(n) * TIMESTEP_HOURS)


# ══════════════════════════════════════════════════════════════
# Überlastungskarte — WO das Netz litt, nicht nur wie oft
# ══════════════════════════════════════════════════════════════
#: Auslastung (p.u.) → Farbe. Ein reales NS-Netz ist kabellimitiert, lange bevor
#: der Transformator etwas merkt — die Karte muss ein 0,9-p.u.-Kabel sichtbar machen.
_LOAD_COLORS = ((1.0, "#d7191c", 5.0), (0.9, "#fdae61", 4.0), (0.7, "#ffd54f", 3.0), (0.0, "#7cb342", 2.5))


def _load_style(pu: float, tripped: bool = False) -> tuple[str, float]:
    """(Farbe, Linienstärke) für eine Auslastung in p.u.

    `tripped` erzwingt die Überlastungsfarbe. Der gespeicherte Spitzenwert ist ein
    MITTELWERT über die Seeds — ein Kabel, das in zwei von sechs Läufen seine Grenze
    verletzt hat, liegt im Mittel unter 1.0. Es muss trotzdem als Verletzung erkennbar
    sein, sonst würde die Karte der danebenstehenden Überlastungszahl widersprechen.
    """
    if tripped:
        return _LOAD_COLORS[0][1], _LOAD_COLORS[0][2]
    for floor, color, weight in _LOAD_COLORS:
        if pu > floor:
            return color, weight
    return "#7cb342", 2.5


def render_overload_map(network, summary, key: str = "overload_map") -> None:
    """Zeichnet das Netz und hebt die Kabel hervor, die heiß liefen oder auslösten.

    `network` ist das gespeicherte GridNetwork des Laufs; `summary` liefert die
    pro Leitung gespeicherte Spitzenauslastung je Szenario. Nicht aufgeführte
    Leitungen blieben unter der Warnschwelle und werden blass im Hintergrund
    gezeichnet, damit der Blick direkt zum kritischen Pfad geht.
    """
    import folium
    from streamlit_folium import st_folium

    if network is None or not getattr(network, "buses", None):
        st.info("Kein gespeichertes Netz für diesen Lauf — die Überlastungskarte benötigt `network.json`.")
        return

    rows = [r for r in summary if r.get("line_peak_loading_pu")]
    if not rows:
        st.info("Dieser Lauf stammt aus der Zeit vor der Leitungs-Aufzeichnung, daher gibt es "
                "keine Überlastungskarte dafür. Neue Läufe speichern sie automatisch.")
        return

    labels = [r["scenario"] for r in rows]
    chosen = st.selectbox("Szenario", labels, index=len(labels) - 1, key=f"{key}_scenario", format_func=_de)
    row = next(r for r in rows if r["scenario"] == chosen)
    peaks: dict = row.get("line_peak_loading_pu") or {}
    steps: dict = row.get("line_overload_steps") or {}

    coords = {b.bus_id: (b.y_coord, b.x_coord)
              for b in network.buses if b.x_coord is not None and b.y_coord is not None}
    if not coords:
        st.info("Das gespeicherte Netz hat keine Koordinaten und kann daher nicht auf einer Karte dargestellt werden.")
        return

    centre = (float(np.mean([c[0] for c in coords.values()])),
              float(np.mean([c[1] for c in coords.values()])))
    m = folium.Map(location=centre, zoom_start=15, tiles="cartodbpositron")

    n_over = 0
    for ln in network.lines:
        a, b = coords.get(ln.from_bus), coords.get(ln.to_bus)
        if not a or not b:
            continue
        pu = peaks.get(ln.line_id)
        if pu is None:
            folium.PolyLine([a, b], color="#c3c9d1", weight=1.2, opacity=0.5).add_to(m)
            continue
        trips = steps.get(ln.line_id, 0)
        color, weight = _load_style(pu, tripped=trips > 0)
        n_over += trips > 0
        folium.PolyLine(
            [a, b], color=color, weight=weight, opacity=0.95,
            tooltip=f"{ln.line_id} — Spitzenwert {pu:.2f} p.u." + (f", {trips:.1f} Schritte überlastet" if trips else ""),
        ).add_to(m)

    for t in network.transformers:
        tb = coords.get(t.lv_bus)
        if tb:
            folium.Marker(tb, icon=folium.Icon(color="black", icon="bolt", prefix="fa"),
                          tooltip=f"{t.trafo_id} — {t.s_nom_mva*1000:.0f} kVA").add_to(m)

    worst = max(peaks.values()) if peaks else 0.0
    st.caption(
        f"**{_de(chosen)}** — schlechtestes Kabel {worst:.2f} p.u. · {n_over} Kabel über der Nennlast. "
        "🟥 >1,0 überlastet · 🟧 >0,9 · 🟨 >0,7 · 🟩 belastet, aber unauffällig · grau = unter Warnschwelle. "
        "⚡ = Transformator. Zum Anzeigen des Spitzenwerts über ein Segment fahren."
    )
    st_folium(m, height=520, width=None, returned_objects=[], key=key)


def render_results(summary, timelines, network=None) -> None:
    """Rendert das Szenario-Vergleichs-Dashboard aus Daten im Arbeitsspeicher.

    Wird sowohl von der einheitlichen Web-App (Ergebnisse je Lauf) als auch vom
    eigenständigen Dashboard `main()` genutzt, das die Ausgabe des Batch-Experiments
    lädt. `network` ist optional: wenn übergeben, wird zusätzlich die Überlastungskarte
    gezeichnet.
    """
    if not summary:
        st.info("Für diese Auswahl liegen noch keine Ergebnisse vor.")
        return

    _render_glossary()

    tab_ueberblick, tab_last, tab_geraete, tab_vergleich, tab_training = st.tabs(
        ["Überblick", "Netzauslastung", "Geräte & Haushalte", "Szenarienvergleich", "Training"]
    )
    with tab_ueberblick:
        render_ueberblick(summary, timelines)
    with tab_last:
        render_auslastung(timelines or [])
    with tab_geraete:
        render_geraete(summary, timelines or [])
    with tab_vergleich:
        _render_comparison(summary, timelines, network)
    with tab_training:
        render_training(OUTPUT_DIR / "checkpoints")


def _render_glossary() -> None:
    """Die Begriffe, ohne die keine Zahl auf dieser Seite lesbar ist."""
    with st.expander("Wie lese ich das? — Begriffe in einem Satz"):
        left, right = st.columns(2)
        left.markdown(
            "**Auslastung (%)** — Belastung im Verhältnis zur Nennleistung. "
            "100 % heißt genau ausgelastet, darüber ist Überlast.\n\n"
            "**Transformator vs. Leitung** — beide können überlasten. Im "
            "Niederspannungsnetz erreicht meist das **Kabel** zuerst seine "
            "Grenze, während der Transformator noch entspannt aussieht.\n\n"
            "**EV-Anteil** — wie viele Haushalte ein Elektroauto haben. "
            "Der Härtegrad des Tests."
        )
        right.markdown(
            "**§14a EnWG** — erlaubt dem Netzbetreiber, steuerbare Geräte "
            "gedrosselt zu betreiben, wenn das Netz sonst überlastet. Ein "
            "„Eingriff“ ist eine Viertelstunde, in der das passiert.\n\n"
            "**Ladestand (SoC)** — Füllstand von Autobatterie oder Speicher, "
            "0 bis 1.\n\n"
            "**Szenarien** — die Regelstrategie: *konstant/sofort* lädt ohne "
            "Rücksicht, *preisorientiert* wartet auf günstigen Strom, "
            "*eigennütziges RL* ist die gelernte Strategie."
        )


def _render_comparison(summary, timelines, network=None) -> None:
    """Der bisherige Szenarienvergleich — unverändert, nur in einen Reiter verschoben."""
    df = pd.DataFrame(summary)
    penetrations = sorted(df["penetration"].unique())

    if network is not None:
        st.header("Wo das Netz überlastet war")
        render_overload_map(network, summary)

    # ── 1. Abregelungsvergleich über den EV-Anteil ──────────
    st.header("§14a-Eingriffe nach Szenario und EV-Anteil")
    st.write(
        "Zeitschritte pro 24-h-Episode, in denen die §14a-Abregelung ausgelöst wurde "
        "(Mittelwert ± Std. über die Seeds). Die Geschichte dahinter: naives "
        "**automatisiertes** Preisfolgeverhalten synchronisiert sich in das günstige "
        "Nachtfenster und verursacht die meisten Eingriffe; der eigennützige "
        "**RL**-Agent, der lokale Spannung und vergangene Abregelungen wahrnimmt, "
        "lernt, sich zu entzerren."
    )
    fig, ax = plt.subplots(figsize=(9, 4.2))
    scenarios = [s for s in SCENARIO_ORDER if s in df["scenario"].unique()]
    x = np.arange(len(penetrations))
    width = 0.8 / max(1, len(scenarios))
    for i, scen in enumerate(scenarios):
        sub = df[df["scenario"] == scen].set_index("penetration").reindex(penetrations)
        ax.bar(x + i * width, sub["curtailment_mean"], width,
               yerr=sub["curtailment_std"], capsize=3,
               label=_de(scen), color=SCENARIO_COLORS.get(scen, None))
    ax.set_xticks(x + width * (len(scenarios) - 1) / 2)
    ax.set_xticklabels([f"{p:.0%}" for p in penetrations])
    ax.set_xlabel("EV-Anteil")
    ax.set_ylabel("§14a-Eingriffe / Episode")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)

    # ── 2. SoC-Zielerreichung (die Kosten aus Kundensicht) ──────────
    st.header("SoC-Zielerreichung — wurden Kund:innen rechtzeitig geladen?")
    pivot = df.pivot_table(index="penetration", columns="scenario", values="soc_mean")
    pivot = pivot.reindex(columns=[s for s in SCENARIO_ORDER if s in pivot.columns])
    pivot.columns = [_de(c) for c in pivot.columns]
    pivot.index = [f"{p:.0%}" for p in pivot.index]
    st.dataframe(pivot.style.format("{:.2f}").background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
                 width="stretch")

    # ── 3. Repräsentative 24-h-Episode ───────────────────────
    if timelines:
        st.header("Eine repräsentative 24-h-Episode")
        tdf = pd.DataFrame(timelines)
        sel_scen, sel_pen = scenario_penetration_picker(timelines, "vergleich")

        row = tdf[(tdf["penetration"] == sel_pen) & (tdf["scenario"] == sel_scen)]
        if row.empty:
            st.info("Für diese Kombination liegt keine Episode vor.")
        else:
            rec = row.iloc[0]
            hours = _hours_axis(len(rec["transformer_loading"]))

            has_devices = "ev_power" in rec and rec["ev_power"] is not None
            n_panels = 4 if has_devices else 3
            fig2, axes = plt.subplots(n_panels, 1, figsize=(10, 2.6 * n_panels), sharex=True)
            axa, axb, axd = axes[0], axes[1], axes[2]
            axa.plot(hours, rec["transformer_loading"], label="Trafo-Auslastung", color="#264653")
            axa.plot(hours, rec["max_line_loading"], label="max. Leitungsauslastung", color="#e76f51")
            axa.axhline(1.0, ls="--", color="red", lw=1, label="Überlastungsschwelle")
            curt = np.array(rec["curtailment"], dtype=bool)
            axa.fill_between(hours, 0, 1.4, where=curt, color="red", alpha=0.12, label="§14a-Abregelung")
            axa.set_ylabel("Auslastung (p.u.)")
            axa.set_ylim(0, 1.4)
            axa.legend(fontsize=8, ncol=2)
            axa.grid(alpha=0.3)
            axa.set_title(f"{_de(sel_scen)} @ {sel_pen:.0%} EV-Anteil")

            # Preis + PV-Erzeugung (warum Eigenverbrauch vs. Einspeisung eine Rolle spielt)
            axb.plot(hours, rec["price"], label="Preis (€/kWh)", color="#2a9d8f")
            axb.set_ylabel("Preis (€/kWh)", color="#2a9d8f")
            axb.tick_params(axis="y", labelcolor="#2a9d8f")
            if "pv_generation" in rec and rec["pv_generation"] is not None:
                axpv = axb.twinx()
                axpv.fill_between(hours, 0, rec["pv_generation"], color="#e9c46a", alpha=0.4, label="PV (kW)")
                axpv.set_ylabel("PV-Erzeugung (kW)", color="#b8860b")
                axpv.tick_params(axis="y", labelcolor="#b8860b")
            axb.grid(alpha=0.3)

            # Grundlast + Außentemperatur (Treiber für die Wärmepumpe)
            axd.plot(hours, rec["base_load"], label="Grundlast (kW)", color="#6c757d", ls=":")
            axd.set_ylabel("Grundlast (kW)", color="#6c757d")
            if "temperature" in rec and rec["temperature"] is not None:
                axt = axd.twinx()
                axt.plot(hours, rec["temperature"], label="Temperatur (°C)", color="#e76f51")
                axt.set_ylabel("Temperatur (°C)", color="#e76f51")
                axt.tick_params(axis="y", labelcolor="#e76f51")
            axd.grid(alpha=0.3)

            # Entscheidungen je Gerätetyp (Summe über den ganzen Feeder)
            if has_devices:
                axe = axes[3]
                axe.plot(hours, rec["ev_power"], label="EV-Ladung", color="#457b9d")
                axe.plot(hours, rec["hp_power"], label="Wärmepumpe", color="#e63946")
                axe.plot(hours, rec["battery_power"], label="Batterie (+laden / −entladen)", color="#2a9d8f")
                if "pv_generation" in rec and rec["pv_generation"] is not None:
                    axe.plot(hours, [-p for p in rec["pv_generation"]], label="PV (−Erzeugung)",
                             color="#e9c46a", ls="--", alpha=0.7)
                axe.axhline(0, color="k", lw=0.6)
                axe.set_ylabel("Geräteleistung (kW)")
                axe.legend(fontsize=8, ncol=2)
                axe.grid(alpha=0.3)
                axe.set_title("Entscheidungen je Gerätetyp (Feeder gesamt)", fontsize=9)
            # Episode läuft von Mittag bis Mittag, die Rohachse ist also 12..36.
            # Umbeschriften auf Uhrzeit (mod 24) und Mitternacht markieren, damit
            # es sich intuitiv liest.
            import matplotlib.ticker as mticker
            for ax in axes:
                ax.axvline(24, color="k", ls=":", lw=0.9, alpha=0.5)
            axes[-1].xaxis.set_major_locator(mticker.MultipleLocator(6))
            axes[-1].xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda h, _: f"{int(round(h)) % 24:02d}:00"))
            axes[-1].set_xlabel("Tageszeit  (Episode läuft von Mittag → Mitternacht ┊ → Mittag am Folgetag)")
            st.pyplot(fig2)

            # ── ein repräsentativer Haushalt: exakte Geräteleistung + EV-Verfügbarkeit + SoC ──
            if "house_ev_power" in rec and rec["house_ev_power"] is not None:
                st.subheader("Ein einzelner repräsentativer Haushalt — exakte Geräteleistung")
                st.caption(
                    "Die Geräteentscheidungen eines einzelnen Haushalts (nicht die Feeder-Summe "
                    "oben). Das schattierte Band markiert, wann das EV zu Hause / angesteckt ist; "
                    "das untere Feld verfolgt den Zustand jedes Geräts."
                )
                figh, (hp1, hp2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                hp1.plot(hours, rec["house_ev_power"], label="EV-Ladung", color="#457b9d")
                hp1.plot(hours, rec["house_hp_power"], label="Wärmepumpe", color="#e63946")
                hp1.plot(hours, rec["house_battery_power"], label="Batterie (+laden / −entladen)", color="#2a9d8f")
                hp1.plot(hours, [-p for p in rec["house_pv"]], label="PV (−Erzeugung)", color="#e9c46a", ls="--", alpha=0.7)
                hp1.axhline(0, color="k", lw=0.6)
                ymin, ymax = hp1.get_ylim()
                avail = np.array(rec["house_ev_available"], dtype=bool)
                hp1.fill_between(hours, ymin, ymax, where=avail, color="#457b9d", alpha=0.08,
                                 label="EV zu Hause / angesteckt")
                hp1.set_ylim(ymin, ymax)
                hp1.set_ylabel("Geräteleistung (kW)")
                hp1.legend(fontsize=8, ncol=2)
                hp1.grid(alpha=0.3)

                hp2.plot(hours, rec["house_ev_soc"], label="EV-Ladezustand", color="#457b9d")
                hp2.axhline(0.80, ls=":", color="#457b9d", lw=1, alpha=0.7)   # EV-Ziel
                hp2.plot(hours, rec["house_battery_soc"], label="Batterie-Ladezustand", color="#2a9d8f")
                hp2.plot(hours, rec["house_hp_soc"], label="WP-Wärmespeicher", color="#e63946")
                hp2.axhline(0.30, ls=":", color="#e63946", lw=1, alpha=0.7)   # Komfort-Untergrenze WP
                hp2.set_ylabel("Ladezustand")
                hp2.set_ylim(0, 1.05)
                hp2.legend(fontsize=8, ncol=3)
                hp2.grid(alpha=0.3)

                for ax in (hp1, hp2):
                    ax.axvline(24, color="k", ls=":", lw=0.9, alpha=0.5)
                hp2.xaxis.set_major_locator(mticker.MultipleLocator(6))
                hp2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda h, _: f"{int(round(h)) % 24:02d}:00"))
                hp2.set_xlabel("Tageszeit  (Mittag → Mittag am Folgetag)")
                st.pyplot(figh)
            st.metric("SoC-Zielerreichung (diese Episode)", f"{rec['soc_satisfaction_rate']:.0%}")

    with st.expander("Rohdaten-Übersichtstabelle"):
        st.dataframe(df, width="stretch")


def main() -> None:
    st.set_page_config(page_title="GridKIT — §14a bei Multi-Geräte-Flexibilität", layout="wide")
    st.title("GridKIT — hält die §14a-Abregelung mit EV + Batterie + Wärmepumpe + PV stand?")
    st.caption(
        "Jeder Haushalt betreibt drei steuerbare Geräte-Agenten (EV, Batterie, Wärmepumpe) "
        "sowie eine exogene Dach-PV-Anlage, auf Basis realer, wetterabhängiger Profile "
        "(GridCreator/pyCity). Der Mechanismus wird unter vereinfachten Annahmen gezeigt "
        "(Ersatz-Lastfluss) — ein relativer Vergleich, keine Prognose."
    )
    summary = _load("summary.json")
    timelines = _load("timelines.json")
    if not summary:
        st.warning(f"Keine Ergebnisse in `{OUTPUT_DIR}/`. Zuerst `python -m GridKIT.scripts.run_experiment` ausführen.")
        return
    render_results(summary, timelines)


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    main()
elif _running_under_streamlit():
    # `streamlit run` importiert das Modul (name != __main__); nur dann automatisch
    # ausführen, NICHT bei einem einfachen Import (z. B. wenn die Web-App render_results importiert).
    main()
