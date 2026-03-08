using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace BridgeLauncher;

public enum BridgeStatus { Unknown, Running, Ready, Error }

/// <summary>
/// Handles all process execution and state management for the bridge.
/// All events are raised on background threads — callers must dispatch to UI thread.
/// </summary>
public class BridgeManager : IDisposable
{
    private readonly string _repoRoot;
    private readonly string _setupScriptPath;
    private CancellationTokenSource? _monitorCts;

    public event Action<string>?         LogMessage;
    public event Action<BridgeStatus>?   SetupStatusChanged;
    public event Action<BridgeStatus>?   RosStatusChanged;
    public event Action<BridgeStatus>?   IsaacStatusChanged;
    public event Action<List<string>>?   TopicsUpdated;

    public BridgeManager()
    {
        _repoRoot        = FindRepoRoot();
        _setupScriptPath = Path.Combine(_repoRoot, "windows_setup", "setup_bridge.ps1");
        Log($"Repo root: {_repoRoot}");
    }

    // ── Setup ────────────────────────────────────────────────────────────────

    public async Task RunSetupAsync()
    {
        SetupStatusChanged?.Invoke(BridgeStatus.Running);
        Log("Running windows_setup/setup_bridge.ps1 ...");

        try
        {
            bool ok = await RunPowerShellFileAsync(_setupScriptPath);
            SetupStatusChanged?.Invoke(ok ? BridgeStatus.Ready : BridgeStatus.Error);
            Log(ok ? "Setup complete." : "Setup finished with errors — check log above.");
        }
        catch (Exception ex)
        {
            SetupStatusChanged?.Invoke(BridgeStatus.Error);
            Log($"Setup exception: {ex.Message}");
        }
    }

    // ── ROS2 Monitoring ──────────────────────────────────────────────────────

    public async Task StartRosMonitoringAsync()
    {
        RosStatusChanged?.Invoke(BridgeStatus.Running);
        Log("Checking ROS2 in WSL2 ...");

        // Verify ros2 is available
        var (checkOut, _) = await RunWslAsync(
            "source /opt/ros/humble/setup.bash 2>/dev/null && ros2 --help > /dev/null 2>&1 && echo OK");

        if (!checkOut.Contains("OK"))
        {
            RosStatusChanged?.Invoke(BridgeStatus.Error);
            Log("ROS2 not found in WSL2. Run: sudo apt install ros-humble-ros-base");
            return;
        }

        RosStatusChanged?.Invoke(BridgeStatus.Ready);
        Log("ROS2 ready. Waiting for Isaac Sim topics...");

        _monitorCts = new CancellationTokenSource();
        _ = PollTopicsAsync(_monitorCts.Token);
    }

    private async Task PollTopicsAsync(CancellationToken ct)
    {
        // Build workspace source path in WSL2 format
        var wslInstall = ToWslPath(Path.Combine(_repoRoot, "workspace", "install", "setup.bash"));

        while (!ct.IsCancellationRequested)
        {
            try
            {
                var cmd =
                    $"source /opt/ros/humble/setup.bash && " +
                    $"source {wslInstall} 2>/dev/null; " +
                    $"RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=0 " +
                    $"FASTRTPS_DEFAULT_PROFILES_FILE='' " +          // don't inherit Windows profile
                    $"ros2 topic list 2>/dev/null";

                var (output, _) = await RunWslAsync(cmd);

                var topics = new List<string>();
                foreach (var raw in output.Split('\n', StringSplitOptions.RemoveEmptyEntries))
                {
                    var t = raw.Trim();
                    if (t.StartsWith('/')) topics.Add(t);
                }

                TopicsUpdated?.Invoke(topics);

                bool isaacConnected = topics.Exists(
                    t => t != "/parameter_events" && t != "/rosout");
                IsaacStatusChanged?.Invoke(
                    isaacConnected ? BridgeStatus.Ready : BridgeStatus.Unknown);
            }
            catch { /* swallow transient errors */ }

            try { await Task.Delay(2000, ct); }
            catch (OperationCanceledException) { break; }
        }
    }

    // ── Terminal ─────────────────────────────────────────────────────────────

    public void OpenTerminal()
    {
        var wslWorkspace = ToWslPath(Path.Combine(_repoRoot, "workspace"));
        var wslInstall   = ToWslPath(Path.Combine(_repoRoot, "workspace", "install", "setup.bash"));

        // Write init to a temp script — avoids wt.exe treating semicolons as its own command separators
        var tempScript    = Path.Combine(Path.GetTempPath(), "bridge_init.sh");
        var wslTempScript = ToWslPath(tempScript);
        File.WriteAllText(tempScript, string.Join("\n",
            "#!/bin/bash",
            "source /opt/ros/humble/setup.bash",
            $"source {wslInstall} 2>/dev/null",
            $"cd {wslWorkspace}",
            "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp",
            "export ROS_DOMAIN_ID=0",
            "echo ''",
            "echo '  ROS2 sourced | workspace ready | domain 0'",
            "echo ''",
            "exec bash"
        ));

        // Try Windows Terminal first, fall back to bare wsl.exe window
        try
        {
            var psi = new ProcessStartInfo("wt.exe") { UseShellExecute = true };
            psi.ArgumentList.Add("wsl.exe");
            psi.ArgumentList.Add("bash");
            psi.ArgumentList.Add(wslTempScript);
            Process.Start(psi);
        }
        catch
        {
            var psi = new ProcessStartInfo("wsl.exe") { UseShellExecute = true };
            psi.ArgumentList.Add("bash");
            psi.ArgumentList.Add(wslTempScript);
            Process.Start(psi);
        }
    }

    // ── Process Helpers ──────────────────────────────────────────────────────

    private async Task<bool> RunPowerShellFileAsync(string scriptPath)
    {
        var psi = new ProcessStartInfo("powershell.exe")
        {
            RedirectStandardOutput = true,
            RedirectStandardError  = true,
            UseShellExecute        = false,
            CreateNoWindow         = true
        };
        psi.ArgumentList.Add("-NonInteractive");
        psi.ArgumentList.Add("-ExecutionPolicy");
        psi.ArgumentList.Add("Bypass");
        psi.ArgumentList.Add("-File");
        psi.ArgumentList.Add(scriptPath);

        using var process = Process.Start(psi)!;

        // Stream stdout line by line so the log fills in real time
        var readTask = Task.Run(async () =>
        {
            string? line;
            while ((line = await process.StandardOutput.ReadLineAsync()) != null)
            {
                if (!string.IsNullOrWhiteSpace(line)) Log(line);
            }
        });

        var errTask = Task.Run(async () =>
        {
            var err = await process.StandardError.ReadToEndAsync();
            if (!string.IsNullOrWhiteSpace(err)) Log($"[stderr] {err.Trim()}");
        });

        await Task.WhenAll(readTask, errTask);
        await process.WaitForExitAsync();
        return process.ExitCode == 0;
    }

    private async Task<(string output, string error)> RunWslAsync(string bashCommand)
    {
        var psi = new ProcessStartInfo("wsl.exe")
        {
            RedirectStandardOutput = true,
            RedirectStandardError  = true,
            UseShellExecute        = false,
            CreateNoWindow         = true
        };
        psi.ArgumentList.Add("bash");
        psi.ArgumentList.Add("-c");
        psi.ArgumentList.Add(bashCommand);

        using var process = Process.Start(psi)!;
        var output = await process.StandardOutput.ReadToEndAsync();
        var error  = await process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();
        return (output, error);
    }

    // ── Utilities ────────────────────────────────────────────────────────────

    /// <summary>Walks up from the app directory to find the .git root.</summary>
    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, ".git")))
                return dir.FullName;
            dir = dir.Parent;
        }
        // Fallback: assume app is somewhere inside the repo
        return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));
    }

    /// <summary>Converts a Windows path to its /mnt/... WSL2 equivalent.</summary>
    private static string ToWslPath(string windowsPath)
    {
        if (windowsPath.Length >= 2 && windowsPath[1] == ':')
        {
            var drive = char.ToLower(windowsPath[0]);
            var rest  = windowsPath[2..].Replace('\\', '/');
            return $"/mnt/{drive}{rest}";
        }
        return windowsPath.Replace('\\', '/');
    }

    private void Log(string message) =>
        LogMessage?.Invoke($"[{DateTime.Now:HH:mm:ss}] {message}");

    public void Dispose() => _monitorCts?.Cancel();
}
