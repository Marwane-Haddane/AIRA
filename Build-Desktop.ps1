# Builds the AIRA desktop app (one folder, AIRA.exe) with PyInstaller.
# Prerequisites: .\Setup-Clean-Line.ps1 has been run (creates .venv and downloads the models).
# Output: dist\AIRA\AIRA.exe  (zip the whole dist\AIRA folder to share it)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run .\Setup-Clean-Line.ps1 first (it creates .venv).' }

& $python -m pip install --disable-pip-version-check 'pywebview>=5,<7' 'pyinstaller>=6.10,<7'
if ($LASTEXITCODE -ne 0) { throw 'Could not install pywebview / PyInstaller.' }

$icon = Join-Path $PSScriptRoot 'docs\aira.ico'
$iconArgs = @()
if (Test-Path -LiteralPath $icon) { $iconArgs = @('--icon', $icon, '--add-data', "$icon;docs") }

& $python -m PyInstaller --noconfirm --clean --windowed --name AIRA `
    --contents-directory . `
    --add-data 'robogenius_dobot_magician_3d_digital_twin_studio.html;.' `
    --add-data '.env.example;.' `
    --collect-data HersheyFonts `
    --collect-data faster_whisper `
    --collect-submodules skimage `
    --hidden-import portrait_lineart.service `
    --hidden-import portrait_lineart.create `
    --hidden-import portrait_lineart.archive `
    --hidden-import portrait_lineart.cloud_ai `
    --hidden-import portrait_lineart.pipeline `
    --hidden-import portrait_lineart.vectorize `
    @iconArgs `
    aira_desktop.py
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }

# PyInstaller may bundle an older MSVC runtime than ctranslate2 (Whisper) needs; ship the system's
# current one (the Visual C++ runtime is redistributable)
foreach ($dll in 'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll') {
    $sys = Join-Path $env:WINDIR "System32\$dll"
    $out = Join-Path $PSScriptRoot "dist\AIRA\$dll"
    if ((Test-Path -LiteralPath $sys) -and (Test-Path -LiteralPath $out) -and
        ([version](Get-Item $sys).VersionInfo.ProductVersion -gt [version](Get-Item $out).VersionInfo.ProductVersion)) {
        Copy-Item -LiteralPath $sys -Destination $out -Force
    }
}

# Models the desktop engines use (ArtLine / U2-Net Portrait Sketch need their own Python and stay out)
$dest = Join-Path $PSScriptRoot 'dist\AIRA\models'
New-Item -ItemType Directory -Force -Path $dest | Out-Null
foreach ($name in 'face_detection_yunet_2023mar.onnx', 'face_parsing_resnet18.onnx', 'informative-drawings.onnx', 'u2net_human_seg.onnx') {
    $src = Join-Path $PSScriptRoot "models\$name"
    if (Test-Path -LiteralPath $src) { Copy-Item -LiteralPath $src -Destination $dest -Force }
}
$whisper = Join-Path $PSScriptRoot 'models\whisper-base'
if (Test-Path -LiteralPath $whisper) { Copy-Item -LiteralPath $whisper -Destination $dest -Recurse -Force }

Write-Host ''
Write-Host 'Built: dist\AIRA\AIRA.exe'
Write-Host 'Open it, then add your Cloudflare key in Settings (gear icon) to enable AI Coloring.'
