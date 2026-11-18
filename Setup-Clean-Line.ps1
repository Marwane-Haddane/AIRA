$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$modelDir = Join-Path $projectRoot 'models'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    py -3 -m venv (Join-Path $projectRoot '.venv')
}

& $pythonPath -m pip install --disable-pip-version-check --no-cache-dir -r (Join-Path $projectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install the Clean Line runtime.' }

New-Item -ItemType Directory -Path $modelDir -Force | Out-Null

# name, url, minimum size in bytes, description
$models = @(
    @('face_parsing_resnet18.onnx', 'https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx', 50000000, '53 MB BiSeNet face parsing'),
    @('u2net_human_seg.onnx', 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx', 170000000, '176 MB person matting'),
    @('informative-drawings.onnx', 'https://huggingface.co/rocca/informative-drawings-line-art-onnx/resolve/main/informative-drawings.onnx', 16000000, '17 MB line-art network'),
    @('face_detection_yunet_2023mar.onnx', 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx', 200000, '0.2 MB YuNet face detector')
)

foreach ($m in $models) {
    $target = Join-Path $modelDir $m[0]
    if (Test-Path -LiteralPath $target) { continue }
    Write-Host "Downloading the $($m[3]) model…"
    $part = "$target.part"
    & curl.exe -L --fail --retry 3 -o $part $m[1]
    if ($LASTEXITCODE -ne 0 -or (Get-Item -LiteralPath $part).Length -lt $m[2]) {
        Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
        throw "The $($m[0]) download did not complete. Re-run this setup script to try again."
    }
    Move-Item -LiteralPath $part -Destination $target -Force
}

Write-Host 'Clean Line is installed. Restart Run-RoboGenius.ps1 to use it.'
