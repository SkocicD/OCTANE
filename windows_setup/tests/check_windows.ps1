# Isaac Sim Bridge - Windows Pre-flight Checks
#
# Run this on Windows PowerShell before launching Isaac Sim.
# Verifies environment variables, the FastDDS XML profile, and the firewall rule.
#
# Usage: Right-click -> "Run with PowerShell"  (no Admin needed - read-only checks)

$pass = 0
$fail = 0

function Check {
    param($Label, [bool]$Condition, $Hint = "")
    if ($Condition) {
        Write-Host "  [PASS] $Label" -ForegroundColor Green
        $script:pass++
    } else {
        Write-Host "  [FAIL] $Label" -ForegroundColor Red
        if ($Hint) { Write-Host "         -> $Hint" -ForegroundColor Yellow }
        $script:fail++
    }
}

Write-Host ""
Write-Host "=== Windows Pre-flight Checks ===" -ForegroundColor Cyan
Write-Host ""

# Environment variables
Write-Host "Environment variables:" -ForegroundColor White

$domainId    = [System.Environment]::GetEnvironmentVariable("ROS_DOMAIN_ID",                  "User")
$rmw         = [System.Environment]::GetEnvironmentVariable("RMW_IMPLEMENTATION",             "User")
$profileFile = [System.Environment]::GetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", "User")

Check "ROS_DOMAIN_ID = 0           ($domainId)"   ($domainId -eq "0")                          "Run windows_setup/setup_bridge.ps1 first"
Check "RMW_IMPLEMENTATION set      ($rmw)"        ($rmw -eq "rmw_fastrtps_cpp")                "Run windows_setup/setup_bridge.ps1 first"
Check "FASTRTPS_DEFAULT_PROFILES_FILE is set"     (-not [string]::IsNullOrEmpty($profileFile)) "Run windows_setup/setup_bridge.ps1 first"

# FastDDS XML profile
Write-Host ""
Write-Host "FastDDS profile:" -ForegroundColor White

if (-not [string]::IsNullOrEmpty($profileFile)) {
    $xmlExists = Test-Path $profileFile
    Check "XML file exists at path" $xmlExists "Expected: $profileFile"

    if ($xmlExists) {
        # Verify it uses UDPv4 transport and does NOT contain initialPeersList
        $xmlText = [System.IO.File]::ReadAllText($profileFile)
        Check "Profile uses UDPv4 transport"  ($xmlText -match "UDPv4")             "Re-run setup_bridge.ps1"
        Check "No initialPeersList (crash risk)" ($xmlText -notmatch "initialPeersList") "Re-run setup_bridge.ps1 to regenerate profile"
    }
} else {
    Check "XML file exists at path" $false "FASTRTPS_DEFAULT_PROFILES_FILE is not set"
}

# Firewall
Write-Host ""
Write-Host "Firewall:" -ForegroundColor White

$ruleName = "ROS2 DDS UDP (Isaac Sim Bridge)"
$rule = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
Check "Inbound UDP 7400-7500 rule exists" ($null -ne $rule) "Re-run setup_bridge.ps1 as Administrator"

# Summary
Write-Host ""
Write-Host "-------------------------------------" -ForegroundColor DarkGray
if ($fail -eq 0) {
    Write-Host "All $pass checks passed. Safe to launch Isaac Sim." -ForegroundColor Green
    Write-Host ""
    Write-Host "Reminder: open a WSL2 terminal (not inside the Docker container) and run:"
    Write-Host "  source /opt/ros/humble/setup.bash"
    Write-Host "  ros2 topic list"
} else {
    Write-Host "$fail of $($pass + $fail) check(s) failed - fix issues above before launching Isaac Sim." -ForegroundColor Red
}
Write-Host ""
