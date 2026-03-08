# Isaac Sim Bridge — Windows Pre-flight Checks
#
# Run this on Windows PowerShell before launching Isaac Sim.
# Verifies environment variables, the FastDDS XML profile, WSL2 reachability,
# and the firewall rule are all in order.
#
# Usage: Right-click → "Run with PowerShell"  (no Admin needed — read-only checks)

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

# ── Environment variables ─────────────────────────────────────────────────────
Write-Host "Environment variables:" -ForegroundColor White

$wsl2Host   = [System.Environment]::GetEnvironmentVariable("WSL2_HOST",                      "User")
$domainId   = [System.Environment]::GetEnvironmentVariable("ROS_DOMAIN_ID",                  "User")
$rmw        = [System.Environment]::GetEnvironmentVariable("RMW_IMPLEMENTATION",             "User")
$profileFile= [System.Environment]::GetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", "User")

Check "WSL2_HOST is set            ($wsl2Host)"    (-not [string]::IsNullOrEmpty($wsl2Host))    "Run windows_setup/setup_bridge.ps1 first"
Check "ROS_DOMAIN_ID = 0           ($domainId)"    ($domainId -eq "0")                          "Run windows_setup/setup_bridge.ps1 first"
Check "RMW_IMPLEMENTATION set      ($rmw)"         ($rmw -eq "rmw_fastrtps_cpp")                "Run windows_setup/setup_bridge.ps1 first"
Check "FASTRTPS_DEFAULT_PROFILES_FILE is set"      (-not [string]::IsNullOrEmpty($profileFile)) "Run windows_setup/setup_bridge.ps1 first"

# ── FastDDS XML profile ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "FastDDS profile:" -ForegroundColor White

if (-not [string]::IsNullOrEmpty($profileFile)) {
    Check "XML file exists at path" (Test-Path $profileFile) "Expected: $profileFile"
} else {
    Check "XML file exists at path" $false "FASTRTPS_DEFAULT_PROFILES_FILE is not set"
}

# ── Network ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Network:" -ForegroundColor White

if (-not [string]::IsNullOrEmpty($wsl2Host)) {
    $reachable = Test-Connection -ComputerName $wsl2Host -Count 1 -Quiet -ErrorAction SilentlyContinue
    Check "WSL2 host ($wsl2Host) is pingable" $reachable "Make sure WSL2 is running. Re-run setup_bridge.ps1 if IP changed."
}

# ── Firewall ──────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Firewall:" -ForegroundColor White

$ruleName = "ROS2 DDS UDP (Isaac Sim Bridge)"
$rule = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
Check "Inbound UDP 7400-7500 rule exists" ($null -ne $rule) "Re-run setup_bridge.ps1 as Administrator"

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "─────────────────────────────────────" -ForegroundColor DarkGray
if ($fail -eq 0) {
    Write-Host "All $pass checks passed. Safe to launch Isaac Sim." -ForegroundColor Green
} else {
    Write-Host "$fail of $($pass + $fail) check(s) failed — fix issues above before launching Isaac Sim." -ForegroundColor Red
}
Write-Host ""
