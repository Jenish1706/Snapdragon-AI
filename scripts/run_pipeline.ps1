# End-to-end empirical benchmark and evaluation pipeline.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_pipeline.ps1 [-From <step>]
param([string]$From = "")
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

# Locate Python environment
$py = "python"
if (Test-Path (Join-Path $root ".venv\Scripts\python.exe")) {
    $py = Join-Path $root ".venv\Scripts\python.exe"
} elseif (Test-Path (Join-Path $root "cache\venv-win\Scripts\python.exe")) {
    $py = Join-Path $root "cache\venv-win\Scripts\python.exe"
}

New-Item -ItemType Directory -Force (Join-Path $root "cache\logs") | Out-Null
$env:PYTHONUNBUFFERED = "1"

$steps = @(
    @("singles", "scripts/torch_ppl.py singles"),
    @("kl-proxy", "scripts/torch_ppl.py kl"),
    @("allayer-svd", "scripts/torch_ppl.py allayer --keep 0.9"),
    @("pairs", "scripts/torch_ppl.py pairs"),
    @("matched-bytes", "scripts/matched_bytes.py"),
    @("additivity", "scripts/s3_additivity.py"),
    @("allayer-mpo", "scripts/torch_ppl.py allayer --method mpo --keep 0.9"),
    @("closed-loop", "scripts/loop_plan.py --real")
)

$go = $From -eq ""
foreach ($s in $steps) {
    if (-not $go -and $s[0] -eq $From) { $go = $true }
    if (-not $go) { continue }
    "[$(Get-Date -Format HH:mm:ss)] start $($s[0])"
    $p = Start-Process -FilePath $py -ArgumentList $s[1] -WorkingDirectory $root -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $root "cache\logs\$($s[0]).log") -RedirectStandardError (Join-Path $root "cache\logs\$($s[0]).err")
    if ($p.ExitCode -ne 0) { "[$(Get-Date -Format HH:mm:ss)] FAILED $($s[0]) (exit $($p.ExitCode))"; exit $p.ExitCode }
    "[$(Get-Date -Format HH:mm:ss)] done $($s[0])"
}
"[$(Get-Date -Format HH:mm:ss)] pipeline finished successfully"
