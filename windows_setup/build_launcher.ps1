# Build the Isaac Sim Bridge Launcher and output BridgeLauncher.exe to BridgeLauncher/bin/publish/.
# Requires .NET 8 SDK: https://dotnet.microsoft.com/download
#
# Usage: Right-click -> "Run with PowerShell"  (no Admin needed)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projFile  = Join-Path $scriptDir "BridgeLauncher\BridgeLauncher.csproj"
$outDir    = Join-Path $scriptDir "BridgeLauncher\bin\publish"

Write-Host ""
Write-Host "=== Building Isaac Sim Bridge Launcher ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $projFile)) {
    Write-Error "Project not found: $projFile"
    exit 1
}

dotnet publish $projFile -c Release -r win-x64 --self-contained false -o $outDir --nologo

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Error "Build failed."
    exit 1
}

Write-Host ""
Write-Host "Done. Run: $outDir\BridgeLauncher.exe" -ForegroundColor Green
Write-Host ""
