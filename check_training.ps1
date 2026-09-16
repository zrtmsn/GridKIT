# check_training.ps1 - wie weit ist der Trainingslauf?
#
#   powershell -File check_training.ps1
#
# Zeigt, welche EV-Anteile fertig sind, wie lange sie gebraucht haben und ob
# der Lauf noch arbeitet. Der Lauf schreibt summary.json/timelines.json erst,
# wenn ALLE drei Stufen durch sind - vorher gibt es nur die Trainingsmetriken
# je Stufe.
#
# ASCII only: PowerShell 5.1 liest .ps1 ohne BOM als ANSI, Umlaute wuerden
# dann als Parserfehler ankommen.
$out = Join-Path $PSScriptRoot "outputs_full"
$levels = @("pen_20", "pen_40", "pen_60")

Write-Host ""
Write-Host "GridKIT - Trainingslauf" -ForegroundColor Cyan
Write-Host ("-" * 46)

if (-not (Test-Path $out)) {
    Write-Host "  Noch nichts geschrieben (${out} fehlt)."
    return
}

foreach ($lvl in $levels) {
    $dir = Join-Path $out "checkpoints\$lvl"
    $metrics = Join-Path $dir "iteration_metrics.json"
    if (Test-Path $metrics) {
        $iters = (Get-Content $metrics -Raw | ConvertFrom-Json).Count
        $took = [math]::Round(((Get-Item $metrics).LastWriteTime - (Get-Item $dir).CreationTime).TotalMinutes)
        Write-Host ("  {0,-8} fertig   {1} Iterationen, {2} min" -f $lvl, $iters, $took) -ForegroundColor Green
    }
    elseif (Test-Path $dir) {
        $running = [math]::Round(((Get-Date) - (Get-Item $dir).CreationTime).TotalMinutes)
        Write-Host ("  {0,-8} laeuft   seit {1} min" -f $lvl, $running) -ForegroundColor Yellow
    }
    else {
        Write-Host ("  {0,-8} wartet" -f $lvl) -ForegroundColor DarkGray
    }
}

Write-Host ("-" * 46)
if (Test-Path (Join-Path $out "summary.json")) {
    Write-Host "  FERTIG - Ergebnisse liegen vor." -ForegroundColor Green
    Write-Host ""
    Write-Host "  Dashboard darauf ansehen:"
    Write-Host '    $env:GRIDKIT_OUTPUT_DIR="outputs_full"; uv run streamlit run src/GridKIT/dashboard/app.py'
}
else {
    $procs = @(Get-Process python -ErrorAction SilentlyContinue)
    if ($procs.Count -gt 0) {
        $cpu = [math]::Round((($procs | Measure-Object CPU -Sum).Sum) / 60)
        Write-Host ("  laeuft noch ({0} Prozesse, {1} min CPU-Zeit)" -f $procs.Count, $cpu)
    }
    else {
        Write-Host "  Kein Python-Prozess und keine summary.json - Lauf wurde abgebrochen." -ForegroundColor Red
    }
}
Write-Host ""
