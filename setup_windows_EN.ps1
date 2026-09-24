# setup_windows_EN.ps1 – Setup with English console output / documentation
# Goals of the English documentation:
# - Consistency with English console output -> setup_windows_GER.ps1 is
# the exception, but serves a specific purpose:
# a language-consistent user experience both in setup and web app
# ----------------------------------------------------------------------
#
# Requirements:
# - Windows 10 (1803 or later) / 11 x64
# - Git for Windows (needed anyway to clone the repo)
# - ~40 GB of free storage,
# - Internet connection
# - VC++ 2015-2022 x64 (see pre-check)
#
#
# ----------------------------------------------------------------------
param([switch]$SkipInputData)
$ErrorActionPreference = 'Stop'          # PowerShell equivalent of 'set -e'
Set-Location -LiteralPath $PSScriptRoot  # Repo root (script is located in the root directory)

# External programs (git, uv, conda, curl.exe, tar.exe) do NOT trigger
# a PowerShell exception on errors - therefore explicit check of the
# exit code after every native command:
function Assert-Exit ([string]$Schritt) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR in $Schritt (exit code $LASTEXITCODE) - script aborts."
        exit $LASTEXITCODE
    }
}

# ---- Pre-check: VC++ Redistributable 2015-2022 (x64) ----------------
# Without these runtime libraries, imports of torch & co. fail

Write-Host "=== Pre-check: VC++ Redistributable 2015-2022 (x64) ==="
$vcFehlt = @('msvcp140.dll','vcruntime140.dll','vcruntime140_1.dll') |
    Where-Object { -not (Test-Path (Join-Path "$env:SystemRoot\System32" $_)) }
if ($vcFehlt) {
    Write-Host "ERROR: VC++ runtime libraries missing: $($vcFehlt -join ', ')"
    Write-Host "  Install them yourself once BEFORE the next run of this script (admin rights required):"
    Write-Host "    1. Download: https://aka.ms/vs/17/release/vc_redist.x64.exe"
    Write-Host "    2. Run the file, confirm the UAC prompt - done."
    Write-Host "    3. Start this script again."
    exit 1
}
Write-Host "OK: VC++ runtime libraries present."

Write-Host "=== Step 1/7: Initialize submodules (vendor/GridCreator) ==="
git submodule update --init --recursive
Assert-Exit "Step 1/7 (git submodule)"

Write-Host "=== Step 2/7: Install uv (if not yet present) ==="
if (-not (Get-Command uv -ErrorAction SilentlyContinue) -and -not (Test-Path "$env:USERPROFILE\.local\bin\uv.exe")) {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"  # make (freshly installed) uv known

Write-Host "=== Step 3/7: Create GridKIT venv (uv sync) and activate it ==="
uv sync
Assert-Exit "Step 3/7 (uv sync)"

$env:PATH = "$PSScriptRoot\.venv\Scripts;$env:PATH"

Write-Host "=== Step 4/7: Verification: test suite (pytest) ==="
pytest
Assert-Exit "Step 4/7 (pytest)"

Write-Host "=== Step 5/7: conda (official Miniconda installer - no admin rights) ==="
if (Get-Command conda -ErrorAction SilentlyContinue) {
    Write-Host "conda already present."
    $condaBase = ((& conda info --base) -join '').Trim()
} else {
    if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
        Write-Host "ERROR: Only Windows x64 is supported (found: $env:PROCESSOR_ARCHITECTURE)."
        exit 1
    }
    curl.exe -fL --retry 3 --retry-delay 5 -o "$env:TEMP\miniconda.exe" https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
    Assert-Exit "Step 5/7 (Miniconda download)"
    # Per-user silent installation (official Anaconda documentation route).
    # /D= must be the LAST argument and must NOT be wrapped in quotes
    # (NSIS rule)
    $p = Start-Process -Wait -PassThru -FilePath "$env:TEMP\miniconda.exe" -ArgumentList '/InstallationType=JustMe','/RegisterPython=0','/S',"/D=$env:USERPROFILE\miniconda3"
    if ($p.ExitCode -ne 0) {
        Write-Host "ERROR: Miniconda installer (exit code $($p.ExitCode)) - script aborts."
        exit $p.ExitCode
    }
    Remove-Item "$env:TEMP\miniconda.exe"
    $condaBase = "$env:USERPROFILE\miniconda3"
    $env:PATH = "$condaBase\condabin;$condaBase\Scripts;$env:PATH"  # conda in THIS script shell
    # And permanently for future terminals - Windows equivalent of the
    # ~/.bashrc line in the Linux script (idempotent). Deliberately NOT
    $userPath = [Environment]::GetEnvironmentVariable('Path','User')
    if ("$userPath" -notlike '*miniconda3*') {
        # Add BOTH condabin AND Scripts to the PATH: condabin only contains conda.bat,
        # but subprocess.run() (CreateProcess, without a shell) resolves 'conda'
        # only to the conda.exe in Scripts - without a Scripts entry,
        # GridKIT's subprocess call fails with WinError 2.
        [Environment]::SetEnvironmentVariable('Path', "$userPath;$condaBase\condabin;$condaBase\Scripts", 'User')
        Write-Host "PATH entries stored permanently (active in new terminals)."
    }
}
# Load conda into THIS shell - Windows equivalent of 'source .../conda.sh':
(& "$condaBase\Scripts\conda.exe" 'shell.powershell' 'hook') | Out-String | Invoke-Expression

Write-Host "=== Step 6/7: Create GridCreator conda env (python 3.12.11) ==="
# Otherwise conda >= 25 asks interactively for the terms of service
# and would halt the script:
$env:CONDA_PLUGINS_AUTO_ACCEPT_TOS = 'true'
# Note: -y overwrites an already existing env 'GridCreator'
# Remove this if no longer desired
conda create -y -n GridCreator python=3.12.11
Assert-Exit "Step 6/7 (conda create)"
conda activate GridCreator
pip install -r vendor\GridCreator\requirements.txt
Assert-Exit "Step 6/7 (pip install)"
conda deactivate

Write-Host "=== Step 7/7: GridCreator input data (2.9 GB download, 18 GB extracted) ==="
# Implementation of the GridCreator input download
if ($SkipInputData) {
    Write-Host "Skipped (-SkipInputData). Catch up later: run the script again without the switch."
} else {
    $targetDir = 'vendor\GridCreator'
    $inputDir  = "$targetDir\input"
    $zipPath   = "$targetDir\input.zip"
    $zenodoUrl = 'https://zenodo.org/records/17884917/files/input.zip?download=1'
    $expectMd5 = 'af37b52a652a712b9f3df067b2ec769a'
    if ((Test-Path $inputDir) -and (@(Get-ChildItem $inputDir -ErrorAction SilentlyContinue).Count -gt 0)) {
        Write-Host "Note: $inputDir is already populated - skipped."
    } else {
        # Space check: zip (2.9 GB) + extracted (~18 GB) need ~25 GB:
        $laufwerk = (Get-Location).Path.Substring(0,1)
        $freiGB   = [math]::Round((Get-PSDrive $laufwerk).Free / 1GB)
        if ($freiGB -lt 25) {
            Write-Host "ERROR: Not enough free space on drive $laufwerk (required: ~25 GB, free: $freiGB GB)."
            exit 1
        }
        Write-Host ">> Downloading input.zip (2.9 GB) from Zenodo ..."
        Write-Host "   (If interrupted, run again - curl.exe -C - resumes the download.)"
        curl.exe -fL -C - --retry 5 --retry-delay 5 --speed-time 30 --speed-limit 10240 -o $zipPath $zenodoUrl
        Assert-Exit "Step 7/7 (Zenodo download)"
        Write-Host ">> Checking MD5 ..."
        $md5 = (Get-FileHash -Algorithm MD5 $zipPath).Hash.ToLower()
        if ($md5 -ne $expectMd5) {
            Write-Host "ERROR: MD5 mismatch (expected: $expectMd5, got: $md5)."
            Write-Host "   The file is incomplete/corrupted and will be deleted - please run again."
            Remove-Item $zipPath
            exit 1
        }
        Write-Host "   MD5 ok ($md5)"
        Write-Host ">> Extracting to $targetDir ... (the zip contains input/ at its root)"
        tar.exe -xf $zipPath -C $targetDir
        Assert-Exit "Step 7/7 (extraction)"
        Remove-Item $zipPath
        if ((Test-Path "$inputDir\grids") -and (Test-Path "$inputDir\weather_2013") -and (Test-Path "$inputDir\zensus_daten")) {
            Write-Host ">> Done. Expected structure is in place in $inputDir."
        } else {
            Write-Host "ERROR: Extracted, but the expected subfolders (grids/, weather_2013/, zensus_daten/) are missing in $inputDir."
            exit 1
        }
    }
}

Write-Host ""
Write-Host "Installation complete. Please open a new shell and navigate to the GridKIT root folder."
Write-Host "Then execute the following commands:"
Write-Host ".venv\Scripts\activate"
Write-Host "streamlit run src/GridKIT/scripts/app.py"