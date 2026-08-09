[CmdletBinding()]
param(
    [string]$RepositoryUrl = "https://github.com/Saroswat/nasa-cmapss-predictive-maintenance.git",
    [string]$InstallDirectory = (Join-Path (Get-Location) "nasa-cmapss-predictive-maintenance"),
    [switch]$RunExperiment,
    [switch]$StartDashboard
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
    }
}

$gitCommand = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCommand) {
    throw "Git is required. Install Git for Windows from https://git-scm.com/download/win"
}

$nodeCommand = Get-Command node -ErrorAction SilentlyContinue
$npmCommand = Get-Command npm -ErrorAction SilentlyContinue
if (-not $nodeCommand -or -not $npmCommand) {
    throw "Node.js 22.13 or newer is required. Install it from https://nodejs.org"
}

$nodeVersionText = (& $nodeCommand.Source --version).TrimStart("v")
$nodeVersion = [Version]$nodeVersionText
if ($nodeVersion -lt [Version]"22.13.0") {
    throw "Node.js 22.13 or newer is required. Found v$nodeVersionText."
}

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    Write-Host "Installing uv..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
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
    Write-Host "Updating existing repository..."
    Invoke-NativeCommand $gitCommand.Source @("-C", $InstallDirectory, "pull", "--ff-only")
} elseif (Test-Path $InstallDirectory) {
    throw "Install path exists but is not a Git repository: $InstallDirectory"
} else {
    Write-Host "Cloning repository..."
    Invoke-NativeCommand $gitCommand.Source @("clone", $RepositoryUrl, $InstallDirectory)
}

Push-Location $InstallDirectory
try {
    Write-Host "Installing Python dependencies..."
    Invoke-NativeCommand $uvPath @("sync", "--extra", "notebook", "--extra", "dev")

    Write-Host "Downloading and verifying NASA C-MAPSS FD001..."
    Invoke-NativeCommand $uvPath @("run", "cmapss-maintenance", "download")

    Write-Host "Installing dashboard dependencies..."
    Invoke-NativeCommand $npmCommand.Source @("--prefix", "web", "ci")

    if ($RunExperiment) {
        Write-Host "Training models and refreshing dashboard data..."
        Invoke-NativeCommand $uvPath @("run", "cmapss-maintenance", "run")
        Invoke-NativeCommand $uvPath @("run", "python", "scripts/export_dashboard_data.py")
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Setup complete: $InstallDirectory" -ForegroundColor Green
Write-Host "Run experiment:  cd `"$InstallDirectory`"; uv run cmapss-maintenance run"
Write-Host "Open notebook:   cd `"$InstallDirectory`"; uv run jupyter lab notebooks/01_modern_predictive_maintenance.ipynb"
Write-Host "Open dashboard:  cd `"$InstallDirectory`"; npm --prefix web run dev"
Write-Host "Dashboard URL:   http://localhost:3000"

if ($StartDashboard) {
    Write-Host ""
    Write-Host "Starting dashboard..."
    Push-Location $InstallDirectory
    try {
        Invoke-NativeCommand $npmCommand.Source @("--prefix", "web", "run", "dev")
    } finally {
        Pop-Location
    }
}
