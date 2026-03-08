# Isaac Sim ROS 2 Bridge - Windows Setup
#
# What this does:
#   1. Gets the ros2_bridge container's actual IP (Docker Desktop assigns its own subnet)
#   2. Writes Isaac Sim's FastDDS profile with that real container IP
#   3. Sets user-level environment variables Isaac Sim will read on next launch
#   4. Adds a Windows Firewall inbound rule for DDS UDP traffic
#
# IMPORTANT: Start the container first (bash run_bridge.sh in WSL2), then run this script.
# Run with: Right-click -> "Run with PowerShell" (as Administrator for firewall step)
# Re-run each time WSL2 restarts or the container is recreated.

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== Isaac Sim Bridge Setup ===" -ForegroundColor Cyan
Write-Host ""

# 1. Get the container's actual IP via docker inspect.
# On Docker Desktop for Windows, network_mode:host does not share the Windows/WSL2 network
# interface -- the container gets its own IP in the Docker Desktop VM subnet (192.168.65.x).
# Isaac Sim must send DDS packets to this IP, not the WSL2 IP.
Write-Host "Detecting container IP..." -ForegroundColor White

$containerIP = $null
try {
    $containerIP = (docker inspect ros2_bridge --format "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" 2>$null).Trim()
} catch {}

# With network_mode:host the above may return empty -- fall back to hostname -I inside container
if (-not $containerIP) {
    try {
        $containerIP = (docker exec ros2_bridge hostname -I 2>$null).Trim().Split()[0]
    } catch {}
}

if (-not $containerIP) {
    Write-Error "Could not detect container IP. Make sure the ros2_bridge container is running first (bash run_bridge.sh in WSL2)."
    exit 1
}

Write-Host "Container IP detected: $containerIP" -ForegroundColor Green

# 2. Generate FastDDS XML profile pointing Isaac Sim at the real container IP
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
                                <address>$containerIP</address>
                                <port>7412</port>
                            </udpv4>
                        </locator>
                        <locator>
                            <udpv4>
                                <address>$containerIP</address>
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
Write-Host "FastDDS profile written with container IP ($containerIP): $profilePath" -ForegroundColor Green

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
Write-Host "Restart Isaac Sim to pick up the new environment variables."
Write-Host "Re-run this script each time the container is restarted (IP may change)."
Write-Host ""
