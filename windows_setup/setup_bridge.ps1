# Isaac Sim ROS 2 Bridge - Windows Setup
#
# What this does:
#   1. Detects the current WSL2 IP (changes on every WSL2 restart)
#   2. Sets user-level environment variables Isaac Sim will read on next launch
#   3. Adds a Windows Firewall inbound rule for DDS UDP traffic
#
# Run with: Right-click -> "Run with PowerShell" (as Administrator for firewall step)
# Re-run each time WSL2 restarts, then relaunch Isaac Sim.

$ErrorActionPreference = "Stop"

# 1. Detect WSL2 IP
Write-Host ""
Write-Host "=== Isaac Sim Bridge Setup ===" -ForegroundColor Cyan
Write-Host ""

$wsl2IP = (wsl -- hostname -I 2>$null).Trim().Split()[0]

if (-not $wsl2IP) {
    Write-Error "Could not detect WSL2 IP. Make sure WSL2 is running (open a WSL terminal first)."
    exit 1
}

Write-Host "WSL2 IP detected: $wsl2IP" -ForegroundColor Green

# 2. Generate FastDDS XML profile with the real WSL2 IP substituted in.
# $ENV{VAR} expansion is not supported in FastDDS address fields (same issue as the
# container side) so we write the IP directly into the XML here using PowerShell.
$profilePath = Join-Path $PSScriptRoot "fastdds_isaac_sim.xml"

$xmlContent = @"
<?xml version="1.0" encoding="UTF-8" ?>
<dds>
    <profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
        <participant profile_name="isaac_sim_bridge_windows" is_default_profile="true">
            <rtps>
                <builtin>
                    <initialPeersList>
                        <locator>
                            <udpv4>
                                <address>$wsl2IP</address>
                                <port>7412</port>
                            </udpv4>
                        </locator>
                        <locator>
                            <udpv4>
                                <address>$wsl2IP</address>
                                <port>7410</port>
                            </udpv4>
                        </locator>
                    </initialPeersList>
                </builtin>
            </rtps>
        </participant>
    </profiles>
</dds>
"@

$xmlContent | Set-Content -Path $profilePath -Encoding UTF8
Write-Host "FastDDS profile written: $profilePath" -ForegroundColor Green

# 3. Set environment variables
[System.Environment]::SetEnvironmentVariable("WSL2_HOST",                      $wsl2IP,            "User")
[System.Environment]::SetEnvironmentVariable("ROS_DOMAIN_ID",                  "0",                "User")
[System.Environment]::SetEnvironmentVariable("RMW_IMPLEMENTATION",             "rmw_fastrtps_cpp", "User")
[System.Environment]::SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", $profilePath,       "User")

Write-Host "Environment variables set (user-level):" -ForegroundColor Green
Write-Host "  WSL2_HOST                      = $wsl2IP"
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
Write-Host "Restart Isaac Sim to pick up the new environment variables."
Write-Host "Re-run this script each time WSL2 restarts (the IP changes)."
Write-Host ""
