# Isaac Sim ROS 2 Bridge - Windows Setup
#
# What this does:
#   1. Writes Isaac Sim's FastDDS profile (UDPv4 only, no shared memory)
#   2. Sets user-level environment variables Isaac Sim will read on next launch
#   3. Adds a Windows Firewall inbound rule for DDS UDP traffic
#
# Why no container IP detection?
#   Docker Desktop puts containers on a separate virtual network (192.168.65.x)
#   that Isaac Sim cannot reach for DDS discovery. The solution is to run ROS2
#   directly in WSL2 -- DDS multicast works between Windows and WSL2 over the
#   Hyper-V virtual switch. No hardcoded peer IPs needed.
#
# Run with: Right-click -> "Run with PowerShell" (as Administrator for firewall step)
# Re-run if you reset environment variables or reinstall Isaac Sim.

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== Isaac Sim Bridge Setup ===" -ForegroundColor Cyan
Write-Host ""

# 1. Write FastDDS profile
# Uses NVIDIA's recommended profile: UDPv4 only, shared memory disabled.
# No initialPeersList -- DDS multicast over the WSL2 Hyper-V virtual switch
# handles discovery automatically between Isaac Sim and WSL2 ROS2.
$profilePath = Join-Path $PSScriptRoot "fastdds_isaac_sim.xml"
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

# 2. Set environment variables
[System.Environment]::SetEnvironmentVariable("ROS_DOMAIN_ID",                  "0",                "User")
[System.Environment]::SetEnvironmentVariable("RMW_IMPLEMENTATION",             "rmw_fastrtps_cpp", "User")
[System.Environment]::SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", $profilePath,       "User")

Write-Host "Environment variables set (user-level):" -ForegroundColor Green
Write-Host "  ROS_DOMAIN_ID                  = 0"
Write-Host "  RMW_IMPLEMENTATION             = rmw_fastrtps_cpp"
Write-Host "  FASTRTPS_DEFAULT_PROFILES_FILE = $profilePath"

# 3. Firewall rule
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
Write-Host "  2. In WSL2 (not inside the Docker container), run:"
Write-Host "       source /opt/ros/humble/setup.bash"
Write-Host "       ros2 topic list"
Write-Host "  3. Hit Play in Isaac Sim -- topics should appear."
Write-Host ""
