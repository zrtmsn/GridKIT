# setup_windows_GER.ps1 – Setup mit deutschem Console-Output / Dokumentation
#
# NOTE: For an English setup script
# with English documentation see setup_windows_EN.ps1
#
# Ziele der deutschen Dokumentation:
# - User beim Troubleshooting helfen;
# - Konsistenz mit deutschem Console-Output, der wiederum
#   konsistent mit GridKIT’s deutscher UI ist.
# ----------------------------------------------------------------------
#
# Voraussetzungen: 
# - Windows 10 (ab 1803) / 11 x64
# - Git fuer Windows (zum Klonen des Repos ohnehin noetig)
# - ~40 GB freier Speicher,
# - Internetverbindung
# - VC++ 2015-2022 x64 (siehe Vorab-Check)
#
#
# ----------------------------------------------------------------------
param([switch]$SkipInputData)
$ErrorActionPreference = 'Stop'          # PowerShell-Analogon zu 'set -e'
Set-Location -LiteralPath $PSScriptRoot  # Repo-Root (Skript liegt im Wurzelverzeichnis)

# Externe Programme (git, uv, conda, curl.exe, tar.exe) loesen bei
# Fehlern KEINE PowerShell-Exception aus - deshalb explizite Pruefung
# des Exit-Codes nach jedem nativen Befehl:
function Assert-Exit ([string]$Schritt) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FEHLER in $Schritt (Exit-Code $LASTEXITCODE) - Skript bricht ab."
        exit $LASTEXITCODE
    }
}

# ---- Vorab-Check: VC++ Redistributable 2015-2022 (x64) --------------
# Ohne diese Laufzeitbibliotheken schlagen Importe von torch & Co. fehl

Write-Host "=== Vorab-Check: VC++ Redistributable 2015-2022 (x64) ==="
$vcFehlt = @('msvcp140.dll','vcruntime140.dll','vcruntime140_1.dll') |
    Where-Object { -not (Test-Path (Join-Path "$env:SystemRoot\System32" $_)) }
if ($vcFehlt) {
    Write-Host "FEHLER: VC++-Laufzeitbibliotheken fehlen: $($vcFehlt -join ', ')"
    Write-Host "  Einmalig VOR dem naechsten Skript-Lauf selbst installieren (dafuer Admin-Rechte noetig):"
    Write-Host "    1. Download: https://aka.ms/vs/17/release/vc_redist.x64.exe"
    Write-Host "    2. Datei ausfuehren, UAC bestaetigen - fertig."
    Write-Host "    3. Dieses Skript erneut starten."
    exit 1
}
Write-Host "OK: VC++-Laufzeitbibliotheken vorhanden."

Write-Host "=== Schritt 1/7: Submodule initialisieren (vendor/GridCreator) ==="
git submodule update --init --recursive
Assert-Exit "Schritt 1/7 (git submodule)"

Write-Host "=== Schritt 2/7: uv installieren (falls noch nicht vorhanden) ==="
if (-not (Get-Command uv -ErrorAction SilentlyContinue) -and -not (Test-Path "$env:USERPROFILE\.local\bin\uv.exe")) {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"  # (frisch installiertes) uv bekannt machen

Write-Host "=== Schritt 3/7: GridKIT-venv erstellen (uv sync) und aktivieren ==="
uv sync
Assert-Exit "Schritt 3/7 (uv sync)"

$env:PATH = "$PSScriptRoot\.venv\Scripts;$env:PATH"

Write-Host "=== Schritt 4/7: Verifikation: Test-Suite (pytest) ==="
pytest
Assert-Exit "Schritt 4/7 (pytest)"

Write-Host "=== Schritt 5/7: conda (offizieller Miniconda-Installer - keine Admin-Rechte) ==="
if (Get-Command conda -ErrorAction SilentlyContinue) {
    Write-Host "conda bereits vorhanden."
    $condaBase = ((& conda info --base) -join '').Trim()
} else {
    if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
        Write-Host "FEHLER: Nur Windows x64 unterstuetzt (gefunden: $env:PROCESSOR_ARCHITECTURE)."
        exit 1
    }
    curl.exe -fL --retry 3 --retry-delay 5 -o "$env:TEMP\miniconda.exe" https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
    Assert-Exit "Schritt 5/7 (Miniconda-Download)"
    # Silent-Installation pro Benutzer (offizielle Anaconda-Doku-Route).
    # /D= muss LETZTES Argument sein und darf NICHT in Anfuehrungszeichen
    # stehen (NSIS-Regel)
    $p = Start-Process -Wait -PassThru -FilePath "$env:TEMP\miniconda.exe" -ArgumentList '/InstallationType=JustMe','/RegisterPython=0','/S',"/D=$env:USERPROFILE\miniconda3"
    if ($p.ExitCode -ne 0) {
        Write-Host "FEHLER: Miniconda-Installer (Exit-Code $($p.ExitCode)) - Skript bricht ab."
        exit $p.ExitCode
    }
    Remove-Item "$env:TEMP\miniconda.exe"
    $condaBase = "$env:USERPROFILE\miniconda3"
    $env:PATH = "$condaBase\condabin;$condaBase\Scripts;$env:PATH"  # conda in DIESER Skript-Shell
    # Und dauerhaft fuer zukuenftige Terminals - Windows-Analogon zur
    # ~/.bashrc-Zeile des Linux-Skripts (idempotent). Bewusst NICHT
    $userPath = [Environment]::GetEnvironmentVariable('Path','User')
    if ("$userPath" -notlike '*miniconda3*') {
        # condabin UND Scripts eintragen: condabin enthaelt nur conda.bat,
        # aber subprocess.run() (CreateProcess, ohne Shell) loest 'conda'
        # nur zur conda.exe in Scripts auf - ohne Scripts-Eintrag schlaegt
        # GridKITs subprocess-Aufruf mit WinError 2 fehl.
        [Environment]::SetEnvironmentVariable('Path', "$userPath;$condaBase\condabin;$condaBase\Scripts", 'User')
        Write-Host "PATH-Eintraege dauerhaft hinterlegt (aktiv in neuen Terminals)."
    }
}
# conda in DIESE Shell laden - Windows-Analogon zu 'source .../conda.sh':
(& "$condaBase\Scripts\conda.exe" 'shell.powershell' 'hook') | Out-String | Invoke-Expression

Write-Host "=== Schritt 6/7: GridCreator-conda-Env anlegen (python 3.12.11) ==="
# conda >= 25 fragt sonst interaktiv nach den Nutzungsbedingungen
# und wuerde das Skript anhalten:
$env:CONDA_PLUGINS_AUTO_ACCEPT_TOS = 'true'
# Achtung: -y ueberschreibt ein bereits existierendes Env 'GridCreator'
# Wenn nicht mehr gewünscht, entfernen
conda create -y -n GridCreator python=3.12.11
Assert-Exit "Schritt 6/7 (conda create)"
conda activate GridCreator
pip install -r vendor\GridCreator\requirements.txt
Assert-Exit "Schritt 6/7 (pip install)"
conda deactivate

Write-Host "=== Schritt 7/7: GridCreator-Input-Daten (2.9 GB Download, 18 GB entpackt) ==="
# Umsetzung des GridCreator-Input-Downloads
if ($SkipInputData) {
    Write-Host "Uebersprungen (-SkipInputData). Spaeter nachholen: Skript ohne Switch erneut ausfuehren."
} else {
    $targetDir = 'vendor\GridCreator'
    $inputDir  = "$targetDir\input"
    $zipPath   = "$targetDir\input.zip"
    $zenodoUrl = 'https://zenodo.org/records/17884917/files/input.zip?download=1'
    $expectMd5 = 'af37b52a652a712b9f3df067b2ec769a'
    if ((Test-Path $inputDir) -and (@(Get-ChildItem $inputDir -ErrorAction SilentlyContinue).Count -gt 0)) {
        Write-Host "Hinweis: $inputDir ist bereits gefuellt - uebersprungen."
    } else {
        # Platz-Check: Zip (2.9 GB) + entpackt (~18 GB) brauchen ~25 GB:
        $laufwerk = (Get-Location).Path.Substring(0,1)
        $freiGB   = [math]::Round((Get-PSDrive $laufwerk).Free / 1GB)
        if ($freiGB -lt 25) {
            Write-Host "FEHLER: Zu wenig freier Speicher auf Laufwerk $laufwerk (noetig: ~25 GB, frei: $freiGB GB)."
            exit 1
        }
        Write-Host ">> Lade input.zip (2.9 GB) von Zenodo herunter ..."
        Write-Host "   (Bei Abbruch erneut ausfuehren - curl.exe -C - setzt den Download fort.)"
        curl.exe -fL -C - --retry 5 --retry-delay 5 --speed-time 30 --speed-limit 10240 -o $zipPath $zenodoUrl
        Assert-Exit "Schritt 7/7 (Zenodo-Download)"
        Write-Host ">> Pruefe MD5 ..."
        $md5 = (Get-FileHash -Algorithm MD5 $zipPath).Hash.ToLower()
        if ($md5 -ne $expectMd5) {
            Write-Host "FEHLER: MD5 stimmt nicht ueberein (erwartet: $expectMd5, erhalten: $md5)."
            Write-Host "   Die Datei ist unvollstaendig/beschaedigt und wird geloescht - bitte erneut ausfuehren."
            Remove-Item $zipPath
            exit 1
        }
        Write-Host "   MD5 ok ($md5)"
        Write-Host ">> Entpacke nach $targetDir ... (das Zip enthaelt input/ an der Wurzel)"
        tar.exe -xf $zipPath -C $targetDir
        Assert-Exit "Schritt 7/7 (Entpacken)"
        Remove-Item $zipPath
        if ((Test-Path "$inputDir\grids") -and (Test-Path "$inputDir\weather_2013") -and (Test-Path "$inputDir\zensus_daten")) {
            Write-Host ">> Fertig. Erwartete Struktur liegt in $inputDir."
        } else {
            Write-Host "FEHLER: Entpackt, aber die erwarteten Unterordner (grids/, weather_2013/, zensus_daten/) fehlen in $inputDir."
            exit 1
        }
    }
}

Write-Host ""
Write-Host "Installation abgeschlossen. Bitte eine neue Shell oeffnen und zur GridKIT-Root navigieren."
Write-Host "Anschliessend folgende Befehle ausfuehren:"
Write-Host ".venv\Scripts\activate"
Write-Host "streamlit run src/GridKIT/scripts/app.py"