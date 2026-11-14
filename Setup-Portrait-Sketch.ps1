$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$modelPath = Join-Path $projectRoot 'models\U2Net_Portrait_Sketch.tflite'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    py -3 -m venv (Join-Path $projectRoot '.venv')
}

& $pythonPath -m pip install --disable-pip-version-check --no-cache-dir -r (Join-Path $projectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install the local portrait sketch runtime.' }

if (-not (Test-Path -LiteralPath $modelPath)) {
    $modelDir = Split-Path -Parent $modelPath
    New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
    Write-Host 'Downloading the 176 MB U²-Net portrait sketch model…'
    & curl.exe -L --fail --retry 3 -o $modelPath 'https://huggingface.co/litert-community/U2Net-Portrait-Sketch-LiteRT/resolve/main/portrait.tflite?download=true'
    if ($LASTEXITCODE -ne 0 -or (Get-Item -LiteralPath $modelPath).Length -lt 170000000) {
        Remove-Item -LiteralPath $modelPath -Force -ErrorAction SilentlyContinue
        throw 'The portrait sketch model download did not complete. Re-run this setup script to try again.'
    }
}

Write-Host 'Portrait Sketch is installed. Restart Run-RoboGenius.ps1 to use it.'
