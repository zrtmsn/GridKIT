#!/usr/bin/env bash
# download_gridcreator_input.sh
# ─────────────────────────────────────────────────────────────
# Lädt die GridCreator-Eingabedaten (2.9 GB) automatisch von
# Zenodo herunter, prüft die Checksumme und entpackt sie genau
# dorthin, wo GridCreator sie erwartet: vendor/GridCreator/input/
#
# Warum das funktioniert (verifiziert 2026-09):
#   - Zenodo erlaubt direkte File-Downloads ohne Login
#     (HTTP 200, application/octet-stream)
#   - Range-Requests werden unterstützt → abgebrochene Downloads
#     werden mit `curl -C -` nahtlos fortgesetzt
#   - Das Zip enthält den Ordner "input/" an der Wurzel → es muss
#     NACH vendor/GridCreator/ entpackt werden (nicht hinein nach
#     input/!), sonst entsteht input/input/...
#
# Lizenzhinweis: Die Daten (Zenodo-Record 17884917) stehen unter
# GPLv3. Dieses Skript lädt sie nur herunter (Nutzung ist frei);
# bereitgestellt von Grimm/Behr/Hofmann, Uni Freiburg (NFDI4Energy).
#
# Aufruf (vom Repo-Root):
#   ./download_gridcreator_input.sh
#   ./download_gridcreator_input.sh --target vendor/GridCreator
#   ./download_gridcreator_input.sh --force     # existierendes input/ überschreiben
# ─────────────────────────────────────────────────────────────
set -euo pipefail

ZENODO_URL="https://zenodo.org/records/17884917/files/input.zip?download=1"
EXPECT_MD5="af37b52a652a712b9f3df067b2ec769a"
TARGET_DIR="vendor/GridCreator"
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET_DIR="$2"; shift 2 ;;
        --force)  FORCE=1; shift ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "Unbekannte Option: $1 (siehe --help)"; exit 1 ;;
    esac
done

INPUT_DIR="${TARGET_DIR}/input"

# ── 1. Vorab-Checks ──────────────────────────────────────────
if [[ ! -d "$TARGET_DIR" ]]; then
    echo "FEHLER: Zielverzeichnis $TARGET_DIR existiert nicht."
    echo "       Erst das Repo klonen und 'git submodule update --init --recursive' ausführen."
    exit 1
fi

if [[ -d "$INPUT_DIR" ]] && [[ -n "$(ls -A "$INPUT_DIR" 2>/dev/null)" ]] && [[ $FORCE -eq 0 ]]; then
    echo "Hinweis: $INPUT_DIR ist bereits gefüllt — übersprungen."
    echo "        (Erneut herunterladen mit: $0 --force)"
    exit 0
fi

command -v curl >/dev/null || { echo "FEHLER: curl wird benötigt."; exit 1; }

# ── 1b. Platz-Check: Zip (2.9 GB) + entpackte Daten (~18 GB) brauchen ~25 GB ──
# (Ohne diesen Check stirbt unzip mitten im Entpacken mit "disk full?" —
#  real gemessen 2026-09: entpackt 18 GB, Zip + Daten zusammen 21 GB.)
AVAILABLE_KB=$(df -Pk "$TARGET_DIR" | awk 'NR==2 {print $4}')
REQUIRED_KB=$((25 * 1024 * 1024))   # 25 GB mit Sicherheitsreserve
if [[ "$AVAILABLE_KB" -lt "$REQUIRED_KB" ]]; then
    echo "FEHLER: Zu wenig freier Speicher auf $(df -P "$TARGET_DIR" | awk 'NR==2 {print $6}')."
    echo "   benötigt: ~25 GB (2.9 GB Zip + ~18 GB entpackt + Reserve)"
    echo "   frei    : $((AVAILABLE_KB / 1024 / 1024)) GB"
    exit 1
fi

# ── 2. Download (mit Resume-Fähigkeit) ───────────────────────
ZIP_PATH="${TARGET_DIR}/input.zip"
echo ">> Lade input.zip (2.9 GB) von Zenodo herunter …"
echo "   (Bei Abbruch einfach erneut ausführen — der Download wird fortgesetzt.)"
if ! curl -fL -C - --retry 5 --retry-delay 5 \
        --speed-time 30 --speed-limit 10240 \
        --progress-bar -o "$ZIP_PATH" "$ZENODO_URL"; then
    echo "FEHLER: Download abgebrochen. Erneut ausführen, um fortzusetzen: $0"
    exit 1
fi

# ── 3. Checksumme prüfen (Zenodo-veröffentlichte MD5) ────────
echo ">> Prüfe MD5 …"
if command -v md5sum >/dev/null 2>&1; then
    ACTUAL_MD5="$(md5sum "$ZIP_PATH" | cut -d' ' -f1)"
elif command -v md5 >/dev/null 2>&1; then
    ACTUAL_MD5="$(md5 -q "$ZIP_PATH")"           # macOS
else
    ACTUAL_MD5="$(python3 -c "import hashlib,sys; print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$ZIP_PATH")"
fi

if [[ "$ACTUAL_MD5" != "$EXPECT_MD5" ]]; then
    echo "FEHLER: MD5 stimmt nicht überein!"
    echo "   erwartet: $EXPECT_MD5"
    echo "   erhalten: $ACTUAL_MD5"
    echo "   Die Datei ist unvollständig/beschädigt und wird gelöscht — bitte erneut ausführen."
    rm -f "$ZIP_PATH"
    exit 1
fi
echo "   MD5 ok ($ACTUAL_MD5)"

# ── 4. Entpacken nach TARGET_DIR (Zip enthält input/ an der Wurzel) ──
# stdin </dev/null: sonst bleibt unzip bei Fehlern (z.B. Disk full)
# interaktiv an einer Continue?-Abfrage hängen — in einem Skript tödlich.
echo ">> Entpacke nach $TARGET_DIR …"
if command -v unzip >/dev/null 2>&1; then
    unzip -oq "$ZIP_PATH" -d "$TARGET_DIR" < /dev/null
else
    # Fallback ohne unzip: Python kann Zips entpacken
    python3 -m zipfile -e "$ZIP_PATH" "$TARGET_DIR/"
fi

rm -f "$ZIP_PATH"

# ── 5. Verifikation ──────────────────────────────────────────
if [[ -d "$INPUT_DIR/grids" && -d "$INPUT_DIR/weather_2013" && -d "$INPUT_DIR/zensus_daten" ]]; then
    echo ">> Fertig. Erwartete Struktur liegt in $INPUT_DIR:"
    echo "   $(ls "$INPUT_DIR" | head -8 | tr '\n' ' ') …"
else
    echo "WARNUNG: Entpackt, aber die erwarteten Unterordner"
    echo "   (grids/, weather_2013/, zensus_daten/) fehlen in $INPUT_DIR."
    exit 1
fi
