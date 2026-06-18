#Requires -Version 5.1
$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "==> Creating virtual environment"
python -m venv .venv

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"

Write-Host "==> Installing project dependencies"
& $Python -m pip install --upgrade pip
& $Pip install -e ".[dev]"

Write-Host "==> Building executable with PyInstaller"
$PyInstallerArgs = @("quicktag.spec", "--distpath", "dist/win", "--noconfirm")
if ($env:CI -ne "true") {
    $PyInstallerArgs += "--clean"
}
& $Python -m PyInstaller @PyInstallerArgs

$DistDir = Join-Path $ProjectRoot "dist\win\quicktag"
New-Item -ItemType Directory -Force -Path (Join-Path $DistDir "input") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $DistDir "output") | Out-Null
Copy-Item -Path (Join-Path $ProjectRoot "config.example.yaml") -Destination (Join-Path $DistDir "config.yaml") -Force
Copy-Item -Path (Join-Path $ProjectRoot "tags.example.yaml") -Destination (Join-Path $DistDir "tags.yaml") -Force
Copy-Item -Path (Join-Path $ProjectRoot "docs\DIST_README_WIN.md") -Destination (Join-Path $DistDir "README.md") -Force

Write-Host "==> Build complete: $DistDir"
