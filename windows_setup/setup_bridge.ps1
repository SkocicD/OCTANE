# Isaac Sim ROS2 Bridge - Windows Setup
#
# What this does:
#   1. Verifies ROS2 Humble is installed on Windows
#   2. Writes Isaac Sim's FastDDS profile (UDPv4 only, no shared memory)
#   3. Sets user-level environment variables Isaac Sim will read on next launch
#   4. Adds a Windows Firewall inbound rule for DDS UDP traffic
#
# Both Isaac Sim and ROS2 run natively on Windows — no WSL2 or Docker needed
# for simulation. DDS discovery works over localhost between them directly.
#
# Run with: Right-click -> "Run with PowerShell" (as Administrator for firewall step)
# Re-run if you reset environment variables or reinstall Isaac Sim.

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== Isaac Sim Bridge Setup ===" -ForegroundColor Cyan
Write-Host ""

# 1. Check ROS2 installation
$ros2Candidates = @(
    "C:\opt\ros\humble\x64\setup.bat",
    "C:\opt\ros2\humble\setup.bat",
    "C:\dev\ros2_humble\setup.bat",
    "E:\ros2_humble\setup.bat"
)

$ros2SetupBat = $ros2Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($ros2SetupBat) {
    Write-Host "ROS2 Humble found: $ros2SetupBat" -ForegroundColor Green
} else {
    Write-Warning "ROS2 Humble not found at any known path."
    Write-Warning "Install from: https://docs.ros.org/en/humble/Installation/Windows-Install-Binary.html"
    Write-Warning "Setup will continue to configure Isaac Sim environment variables."
}

# 2. Write FastDDS profile
# Uses NVIDIA's recommended profile: UDPv4 only, shared memory disabled.
# Isaac Sim and Windows ROS2 discover each other over localhost automatically.
$profilePath  = Join-Path $PSScriptRoot "fastdds_isaac_sim.xml"
$templatePath = Join-Path $PSScriptRoot "fastdds_isaac_sim.xml.template"

if (-not (Test-Path $templatePath)) {
    Write-Error "Template not found: $templatePath"
    exit 1
}

$xmlContent = [System.IO.File]::ReadAllText($templatePath)

# Write UTF-8 without BOM, LF line endings (Isaac Sim XML parser requires this)
$encoding = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($profilePath, $xmlContent.Replace("`r`n", "`n"), $encoding)

Write-Host "FastDDS profile written: $profilePath" -ForegroundColor Green

# 3. Set environment variables
[System.Environment]::SetEnvironmentVariable("ROS_DOMAIN_ID",                  "0",                "User")
[System.Environment]::SetEnvironmentVariable("RMW_IMPLEMENTATION",             "rmw_fastrtps_cpp", "User")
[System.Environment]::SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", $profilePath,       "User")

Write-Host "Environment variables set (user-level):" -ForegroundColor Green
Write-Host "  ROS_DOMAIN_ID                  = 0"
Write-Host "  RMW_IMPLEMENTATION             = rmw_fastrtps_cpp"
Write-Host "  FASTRTPS_DEFAULT_PROFILES_FILE = $profilePath"

# 4. Firewall rule
Write-Host ""
$ruleName = "ROS2 DDS UDP (Isaac Sim Bridge)"

try {
    $existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "Firewall rule already exists: '$ruleName'" -ForegroundColor Yellow
    } else {
        New-NetFirewallRule `
            -DisplayName $ruleName `
            -Direction   Inbound `
            -Protocol    UDP `
            -LocalPort   7400-7500 `
            -Action      Allow | Out-Null
        Write-Host "Firewall rule added: '$ruleName' (UDP 7400-7500 inbound)" -ForegroundColor Green
    }
} catch {
    Write-Warning "Could not add firewall rule - re-run as Administrator to apply it."
    Write-Warning "($_)"
}

# Done
Write-Host ""
Write-Host "Done." -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Restart Isaac Sim to pick up the new environment variables."
Write-Host "  2. In a terminal with ROS2 sourced, run:"
Write-Host "       ros2 topic list"
Write-Host "  3. Hit Play in Isaac Sim -- topics should appear."
Write-Host ""
