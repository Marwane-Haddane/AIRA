$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $projectRoot '.venv'
$pythonPath = Join-Path $venvPath 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    py -3 -m venv $venvPath
}

& $pythonPath -m pip install --disable-pip-version-check -r (Join-Path $projectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install the local app dependencies.' }

Set-Location $projectRoot
$port = if ($env:ROBOGENIUS_PORT) { $env:ROBOGENIUS_PORT } else { '8767' }
Write-Host "RoboGenius opens at http://127.0.0.1:$port/"
& $pythonPath (Join-Path $projectRoot 'dobot_service.py')
