$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonInstall = Get-ItemProperty 'Registry::HKEY_CURRENT_USER\Software\Python\PythonCore\3.8\InstallPath' -ErrorAction SilentlyContinue
$runtimeDir = if ($pythonInstall) { $pythonInstall.'(default)' } else { '' }
$venvDir = Join-Path $projectRoot '.artline-venv'
$modelDir = Join-Path $projectRoot 'models'
$pythonExe = if ($runtimeDir) { Join-Path $runtimeDir 'python.exe' } else { '' }
$venvPython = Join-Path $venvDir 'Scripts\python.exe'
$installer = Join-Path $env:TEMP 'robogenius-python-3.8.10-amd64.exe'
$checkpoint = Join-Path $modelDir 'ArtLine_650.pkl'

if (-not (Test-Path $pythonExe)) {
  Invoke-WebRequest 'https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe' -OutFile $installer
  $installArgs = '/quiet InstallAllUsers=0 Include_launcher=0 Include_test=0 Include_doc=0 Include_tcltk=0'
  $install = Start-Process -FilePath $installer -ArgumentList $installArgs -Wait -PassThru -WindowStyle Hidden
  if ($install.ExitCode -ne 0) { throw "Python 3.8 setup failed with exit code $($install.ExitCode)." }
  $pythonInstall = Get-ItemProperty 'Registry::HKEY_CURRENT_USER\Software\Python\PythonCore\3.8\InstallPath' -ErrorAction SilentlyContinue
  $runtimeDir = if ($pythonInstall) { $pythonInstall.'(default)' } else { '' }
  $pythonExe = if ($runtimeDir) { Join-Path $runtimeDir 'python.exe' } else { '' }
  if (-not (Test-Path $pythonExe)) { throw 'Could not locate the Python 3.8 installation.' }
}

if (-not (Test-Path $venvPython)) { & $pythonExe -m venv $venvDir }
& $venvPython -m pip install --upgrade 'pip<25'
if ($LASTEXITCODE -ne 0) { throw 'Could not prepare pip for ArtLine.' }
& $venvPython -m pip install 'torch==1.6.0+cpu' 'torchvision==0.7.0+cpu' -f 'https://download.pytorch.org/whl/torch_stable.html'
if ($LASTEXITCODE -ne 0) { throw 'Could not install the ArtLine PyTorch runtime.' }
& $venvPython -m pip install 'fastai==1.0.61' 'numpy==1.19.5' 'pandas==1.1.2' 'Pillow<10'
if ($LASTEXITCODE -ne 0) { throw 'Could not install FastAI for ArtLine.' }

New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
if (-not (Test-Path $checkpoint)) {
  $checkpointPart = "$checkpoint.part"
  Write-Host 'Downloading ArtLine portrait weights (about 356 MB)…'
  & curl.exe -L --fail --retry 5 -o $checkpointPart 'https://www.dropbox.com/s/starqc9qd2e1lg1/ArtLine_650.pkl?dl=1'
  if ($LASTEXITCODE -ne 0 -or (Get-Item $checkpointPart -ErrorAction SilentlyContinue).Length -lt 300000000) {
    throw 'ArtLine checkpoint download did not complete. Re-run this setup script to download it again.'
  }
  Move-Item -LiteralPath $checkpointPart -Destination $checkpoint -Force
}

Write-Host 'ArtLine is installed for RoboGenius. Restart Run-RoboGenius.ps1 to use it.'
