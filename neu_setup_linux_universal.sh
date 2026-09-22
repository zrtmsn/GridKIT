#!/usr/bin/env bash
# setup_linux_universal.sh — Minimal-Skript v2.0 (universell für Linux)
# ─────────────────────────────────────────────────────────────
# Gleicher Vertrag wie setup_linux_minimal.sh (set -e, keine
# Fehlerbehandlung — die zuletzt gedruckte "Schritt N/7"-Zeile
# zeigt an, wo der Fehler auftrat), aber UNIVERSELL:
#
#   - Downloads über curl ODER wget — eines von beiden ist auf
#     allen gängigen Distributionen vorinstalliert (Fedora/RHEL:
#     curl, Debian/Ubuntu: wget, openSUSE: beide; Ausnahme Arch
#     Linux (die Distribution, nicht ARM!): dort kann beides
#     fehlen → vorher per Pacman nachinstallieren)
#   - conda über den OFFIZIELLEN Miniconda-Installer (Batch-Modus
#     -b: keine interaktiven Fragen) nach ~/miniconda3 — KEIN
#     sudo, KEIN Paketmanager, identisch auf jeder Distribution
#   - conda-PATH dauerhaft in der Shell-rc hinterlegt (~/.bashrc
#     bzw. ~/.zshrc), damit GridKIT in neuen Terminals
#     'conda run -n GridCreator' ausführen kann
#
# Voraussetzungen: Linux x86_64 (kein macOS, kein ARM/Raspberry
# Pi), git (zum Klonen des Repos ohnehin nötig), curl oder wget,
# ~40 GB freier Speicher, Internet.
#
# Aufruf (vom Repo-Root):
#   ./setup_linux_universal.sh                      # alles
#   ./setup_linux_universal.sh --skip-input-data    # Schritt 7 überspringen
# ─────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")"    # Repo-Root (Skript liegt im Wurzelverzeichnis)

# Mindestens eines der beiden Download-Tools muss vorhanden sein:
command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 \
    || { echo "FEHLER: Weder curl noch wget gefunden."; exit 1; }

# Download-Helfer: curl wenn vorhanden, sonst wget.
# Aufruf: download <url> <zieldatei>
download() {
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --retry-delay 5 -o "$2" "$1"
    else
        wget --tries=3 -O "$2" "$1"
    fi
}

echo "=== Schritt 1/7: Submodule initialisieren (vendor/GridCreator) ==="
git submodule update --init --recursive

echo "=== Schritt 2/7: uv installieren (falls noch nicht vorhanden) ==="
if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/uv" ]]; then
    download https://astral.sh/uv/install.sh /tmp/uv_install.sh
    sh /tmp/uv_install.sh
    rm -f /tmp/uv_install.sh
fi
export PATH="$HOME/.local/bin:$PATH"    # (frisch installiertes) uv bekannt machen

echo "=== Schritt 3/7: GridKIT-venv erstellen (uv sync) und aktivieren ==="
uv sync
source .venv/bin/activate

echo "=== Schritt 4/7: Verifikation: Test-Suite (pytest) ==="
pytest

echo "=== Schritt 5/7: conda (offizieller Miniconda-Installer — kein sudo) ==="
if command -v conda >/dev/null 2>&1; then
    echo "conda bereits vorhanden: $(command -v conda)"
else
    # Nur Linux x86_64 — kein macOS, kein ARM (Raspberry Pi & Co.
    # sind für GridKIT ohnehin zu schwach). Der uname -s-Check
    # fängt auch Intel-Macs ab (dort wäre uname -m ebenfalls x86_64):
    [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] \
        || { echo "FEHLER: Nur Linux x86_64 unterstützt (gefunden: $(uname -s) $(uname -m))."; exit 1; }
    download https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"    # -b = Batch: keine interaktiven Fragen
    rm -f /tmp/miniconda.sh
    export PATH="$HOME/miniconda3/bin:$PATH"           # conda in DIESER Skript-Shell
    # Und dauerhaft für zukünftige Shells (idempotent — kein Doppel-Eintrag
    # bei erneutem Lauf). Ohne diesen Eintrag fänden neue Terminals — und
    # damit GridKITs 'conda run -n GridCreator' — kein conda:
    RC_FILE="$HOME/.bashrc"
    if [[ "$SHELL" == */zsh ]]; then RC_FILE="$HOME/.zshrc"; fi
    grep -q 'miniconda3/bin' "$RC_FILE" 2>/dev/null \
        || echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> "$RC_FILE"
    echo "PATH-Eintrag in $RC_FILE hinterlegt (aktiv in neuen Terminals)."
fi
# conda in DIESE Skript-Shell laden — ersetzt den Shell-Neustart:
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== Schritt 6/7: GridCreator-conda-Env anlegen (python 3.12.11) ==="
# conda >= 25 fragt sonst interaktiv nach den Nutzungsbedingungen
# und würde das Skript anhalten:
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=true
# Achtung (verifiziert an conda 25.11): -y überschreibt ein bereits
# existierendes Env 'GridCreator' stillschweigend (neu bauen + frisches
# pip install). Auf einer frischen Maschine gewollt.
conda create -y -n GridCreator python=3.12.11
conda activate GridCreator
pip install -r vendor/GridCreator/requirements.txt
conda deactivate

echo "=== Schritt 7/7: GridCreator-Input-Daten (2.9 GB Download, 18 GB entpackt) ==="
# '--skip-input-data' als Argument überspringt nur diesen Schritt
# (die 18 GB später nachholen mit ./download_gridcreator_input.sh):
# 'bash …' statt './…': robust gegen ein fehlendes Exec-Bit (git speichert
# Exec-Bits nur, wenn die Datei als ausführbar committet wurde — sonst
# scheitert './…' mit "Permission denied" und set -e bricht hier ab).
[[ "$*" == *--skip-input-data* ]] || bash download_gridcreator_input.sh

echo ""
echo "Installation abgeschlossen."
