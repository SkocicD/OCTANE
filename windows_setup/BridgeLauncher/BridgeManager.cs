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
    private readonly string  _repoRoot;
    private readonly string  _setupScriptPath;
    private readonly string? _ros2SetupBat;
    private CancellationTokenSource? _monitorCts;

    public event Action<string>?        LogMessage;
    public event Action<BridgeStatus>?  SetupStatusChanged;
    public event Action<BridgeStatus>?  RosStatusChanged;
    public event Action<BridgeStatus>?  IsaacStatusChanged;
    public event Action<List<string>>?  TopicsUpdated;

    public BridgeManager()
    {
        _repoRoot        = FindRepoRoot();
        _setupScriptPath = Path.Combine(_repoRoot, "windows_setup", "setup_bridge.ps1");
        _ros2SetupBat    = FindRos2SetupBat();
        Log($"Repo root: {_repoRoot}");
        Log(_ros2SetupBat != null
            ? $"ROS2 found: {_ros2SetupBat}"
            : "ROS2 not found — install ROS2 Humble for Windows, then re-run setup.");
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
        Log("Checking ROS2 ...");

        if (_ros2SetupBat == null)
        {
            RosStatusChanged?.Invoke(BridgeStatus.Error);
            Log("ROS2 not found. Install ROS2 Humble for Windows:");
            Log("  https://docs.ros.org/en/humble/Installation/Windows-Install-Binary.html");
            return;
        }

        var (checkOut, _) = await RunRos2Async("ros2 --version");
        if (string.IsNullOrWhiteSpace(checkOut))
        {
            RosStatusChanged?.Invoke(BridgeStatus.Error);
            Log("ROS2 not responding. Check your installation.");
            return;
        }

        RosStatusChanged?.Invoke(BridgeStatus.Ready);
        Log($"ROS2 ready ({checkOut.Trim()}). Waiting for Isaac Sim topics...");

        _monitorCts = new CancellationTokenSource();
        _ = PollTopicsAsync(_monitorCts.Token);
    }

    private async Task PollTopicsAsync(CancellationToken ct)
    {
        var workspaceInstall = Path.Combine(_repoRoot, "workspace", "install", "setup.bat");

        while (!ct.IsCancellationRequested)
        {
            try
            {
                var sourceWorkspace = File.Exists(workspaceInstall)
                    ? $"call \"{workspaceInstall}\" && "
                    : "";

                var (output, _) = await RunRos2Async($"{sourceWorkspace}ros2 topic list");

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
        if (_ros2SetupBat == null) { Log("ROS2 not found."); return; }

        var workspace        = Path.Combine(_repoRoot, "workspace");
        var workspaceInstall = Path.Combine(workspace, "install", "setup.bat");
        var sourceWorkspace  = File.Exists(workspaceInstall)
            ? $"call \"{workspaceInstall}\" && "
            : "";

        var init =
            $"call \"{_ros2SetupBat}\" && " +
            $"{sourceWorkspace}" +
            $"cd /d \"{workspace}\" && " +
            $"set RMW_IMPLEMENTATION=rmw_fastrtps_cpp && " +
            $"set ROS_DOMAIN_ID=0 && " +
            $"echo. && echo   ROS2 sourced ^| workspace ready ^| domain 0 && echo.";

        try
        {
            var psi = new ProcessStartInfo("wt.exe") { UseShellExecute = true };
            psi.ArgumentList.Add("cmd.exe");
            psi.ArgumentList.Add("/k");
            psi.ArgumentList.Add(init);
            Process.Start(psi);
        }
        catch
        {
            var psi = new ProcessStartInfo("cmd.exe") { UseShellExecute = true };
            psi.ArgumentList.Add("/k");
            psi.ArgumentList.Add(init);
            Process.Start(psi);
        }
    }

    // ── RViz2 ────────────────────────────────────────────────────────────────

    public void OpenRviz2()
    {
        if (_ros2SetupBat == null) { Log("ROS2 not found."); return; }

        Log("Launching rviz2...");
        var cmd =
            $"call \"{_ros2SetupBat}\" && " +
            $"set RMW_IMPLEMENTATION=rmw_fastrtps_cpp && " +
            $"set ROS_DOMAIN_ID=0 && " +
            $"start \"\" ros2 run rviz2 rviz2";

        var psi = new ProcessStartInfo("cmd.exe")
        {
            UseShellExecute = false,
            CreateNoWindow  = true
        };
        psi.ArgumentList.Add("/c");
        psi.ArgumentList.Add(cmd);
        Process.Start(psi);
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

        var readTask = Task.Run(async () =>
        {
            string? line;
            while ((line = await process.StandardOutput.ReadLineAsync()) != null)
                if (!string.IsNullOrWhiteSpace(line)) Log(line);
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

    private Task<(string output, string error)> RunRos2Async(string ros2Command) =>
        RunCmdAsync(
            $"call \"{_ros2SetupBat}\" && " +
            $"set RMW_IMPLEMENTATION=rmw_fastrtps_cpp && " +
            $"set ROS_DOMAIN_ID=0 && " +
            ros2Command);

    private async Task<(string output, string error)> RunCmdAsync(string command)
    {
        var psi = new ProcessStartInfo("cmd.exe")
        {
            RedirectStandardOutput = true,
            RedirectStandardError  = true,
            UseShellExecute        = false,
            CreateNoWindow         = true
        };
        psi.ArgumentList.Add("/c");
        psi.ArgumentList.Add(command);

        using var process = Process.Start(psi)!;
        var output = await process.StandardOutput.ReadToEndAsync();
        var error  = await process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();
        return (output, error);
    }

    // ── Utilities ────────────────────────────────────────────────────────────

    /// <summary>Finds the ROS2 Humble setup.bat in common install locations.</summary>
    private static string? FindRos2SetupBat()
    {
        string[] candidates =
        [
            @"C:\opt\ros\humble\x64\setup.bat",
            @"C:\opt\ros2\humble\setup.bat",
            @"C:\dev\ros2_humble\setup.bat",
            @"E:\ros2_humble\setup.bat",
        ];
        return Array.Find(candidates, File.Exists);
    }

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
        return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));
    }

    private void Log(string message) =>
        LogMessage?.Invoke($"[{DateTime.Now:HH:mm:ss}] {message}");

    public void Dispose() => _monitorCts?.Cancel();
}
