[CmdletBinding()]
param(
    [string]$RepositoryUrl = "https://github.com/Saroswat/nasa-cmapss-predictive-maintenance.git",
    [string]$InstallDirectory = (Join-Path (Get-Location) "nasa-cmapss-predictive-maintenance")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required. Install Git for Windows from https://git-scm.com/download/win"
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "Node.js 22.13 or newer is required for the web dashboard. Install it from https://nodejs.org"
}

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    Write-Host "Installing uv..."
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $uvCandidates = @(
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe")
    )
    $uvPath = $uvCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $uvPath) {
        throw "uv was installed but could not be found. Open a new PowerShell window and rerun this script."
    }
} else {
    $uvPath = $uvCommand.Source
}

$gitDirectory = Join-Path $InstallDirectory ".git"
if (Test-Path $gitDirectory) {
    git -C $InstallDirectory pull --ff-only
} elseif (Test-Path $InstallDirectory) {
    throw "Install path exists but is not a Git repository: $InstallDirectory"
} else {
    git clone $RepositoryUrl $InstallDirectory
}

Push-Location $InstallDirectory
try {
    & $uvPath sync --extra notebook --extra dev
    & $uvPath run cmapss-maintenance download
    npm --prefix web ci
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Setup complete: $InstallDirectory" -ForegroundColor Green
Write-Host "Run the project: cd `"$InstallDirectory`"; uv run cmapss-maintenance run"
Write-Host "Open the notebook: uv run jupyter lab notebooks/01_modern_predictive_maintenance.ipynb"
Write-Host "Open the dashboard: npm --prefix web run dev"
