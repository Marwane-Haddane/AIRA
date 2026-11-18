# Builds the Windows installer: installer\Output\AIRA-Setup-<version>.exe
# 1. builds dist\AIRA with Build-Desktop.ps1 (skip with -SkipAppBuild)
# 2. packs it with Inno Setup 6 (winget install JRSoftware.InnoSetup)
param(
    [string]$Version = '1.0.0',
    [switch]$SkipAppBuild
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if (-not $SkipAppBuild) {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'Build-Desktop.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Desktop build failed.' }
}
if (-not (Test-Path -LiteralPath 'dist\AIRA\AIRA.exe')) { throw 'dist\AIRA\AIRA.exe is missing: run Build-Desktop.ps1 first.' }

$iscc = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $iscc) { throw 'Inno Setup 6 not found. Install it with: winget install JRSoftware.InnoSetup' }

& $iscc "/DAppVersion=$Version" (Join-Path $PSScriptRoot 'installer\AIRA.iss')
if ($LASTEXITCODE -ne 0) { throw 'Inno Setup failed.' }
Write-Host "Built: installer\Output\AIRA-Setup-$Version.exe"
