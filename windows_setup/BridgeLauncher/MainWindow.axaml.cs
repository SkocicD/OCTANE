using Avalonia.Controls;
using Avalonia.Controls.Shapes;
using Avalonia.Interactivity;
using Avalonia.Media;
using Avalonia.Threading;
using System.Collections.Generic;
using System.Text;

namespace BridgeLauncher;

public partial class MainWindow : Window
{
    private readonly BridgeManager _bridge;
    private readonly StringBuilder _logBuffer = new();

    // ── Chip colour definitions ──────────────────────────────────
    private record ChipTheme(string Bg, string Border, string Dot, string Label);

    private static readonly ChipTheme ThemeUnknown = new("#161616", "#222222", "#3a3a3a", "#555555");
    private static readonly ChipTheme ThemeRunning = new("#161100", "#2a2200", "#f5a623", "#f5a623");
    private static readonly ChipTheme ThemeReady   = new("#0d1300", "#1e2e00", "#76b900", "#76b900");
    private static readonly ChipTheme ThemeError   = new("#160a0a", "#2e1212", "#e74c3c", "#e74c3c");

    public MainWindow()
    {
        InitializeComponent();

        _bridge = new BridgeManager();

        _bridge.LogMessage         += msg    => Dispatcher.UIThread.Post(() => AppendLog(msg));
        _bridge.SetupStatusChanged += status => Dispatcher.UIThread.Post(() =>
        {
            SetChip(ChipSetup, DotSetup, LabelSetup, status);
            if (status == BridgeStatus.Ready)
            {
                Badge02.Background  = SolidBrush("#76b900");
                ((TextBlock)Badge02.Child!).Foreground = SolidBrush("#ffffff");
                BtnStartRos.IsEnabled = true;
            }
        });
        _bridge.RosStatusChanged += status => Dispatcher.UIThread.Post(() =>
        {
            SetChip(ChipRos, DotRos, LabelRos, status);
            if (status == BridgeStatus.Ready)
            {
                MonitorSubtitle.Text     = "Polling every 2 s...";
                MonitorSubtitle.Foreground = SolidBrush("#3a5a00");
                LogStatus.Text = "Monitoring";
            }
        });
        _bridge.IsaacStatusChanged += status => Dispatcher.UIThread.Post(() =>
            SetChip(ChipIsaac, DotIsaac, LabelIsaac, status));
        _bridge.TopicsUpdated += topics => Dispatcher.UIThread.Post(() =>
            UpdateTopics(topics));

        AppendLog("Ready. Click 01 Windows Setup, then 02 Start Monitoring.");
        AppendLog("After that, launch Isaac Sim and hit Play — topics will appear.");
    }

    // ── Handlers ────────────────────────────────────────────────

    private void OnSetupClick(object? sender, RoutedEventArgs e)
    {
        BtnSetup.IsEnabled = false;
        LogStatus.Text = "Running setup...";
        _ = _bridge.RunSetupAsync().ContinueWith(_ =>
            Dispatcher.UIThread.Post(() =>
            {
                BtnSetup.IsEnabled = true;
                LogStatus.Text = "Setup complete";
            }));
    }

    private void OnStartRosClick(object? sender, RoutedEventArgs e)
    {
        BtnStartRos.IsEnabled = false;
        BtnStartRos.Content   = BuildMonitoringContent();
        _ = _bridge.StartRosMonitoringAsync();
    }

    private void OnOpenTerminalClick(object? sender, RoutedEventArgs e) =>
        _bridge.OpenTerminal();

    private void OnOpenRviz2Click(object? sender, RoutedEventArgs e) =>
        _bridge.OpenRviz2();

    // ── UI helpers ───────────────────────────────────────────────

    private void AppendLog(string message)
    {
        if (_logBuffer.Length > 0) _logBuffer.AppendLine();
        _logBuffer.Append(message);
        LogBox.Text = _logBuffer.ToString();
        LogBox.CaretIndex = LogBox.Text.Length;
    }

    private void UpdateTopics(List<string> topics)
    {
        bool hasTopics = topics.Count > 0;
        TopicsPlaceholder.IsVisible = !hasTopics;
        TopicsScroller.IsVisible    = hasTopics;
        TopicCountLabel.Text = hasTopics
            ? $"{topics.Count} topic{(topics.Count == 1 ? "" : "s")}"
            : "";
        TopicsList.ItemsSource = topics;
    }

    private static void SetChip(Border chip, Ellipse dot, TextBlock label, BridgeStatus status)
    {
        var t = status switch
        {
            BridgeStatus.Running => ThemeRunning,
            BridgeStatus.Ready   => ThemeReady,
            BridgeStatus.Error   => ThemeError,
            _                    => ThemeUnknown
        };
        chip.Background  = SolidBrush(t.Bg);
        chip.BorderBrush = SolidBrush(t.Border);
        dot.Fill         = SolidBrush(t.Dot);
        label.Foreground = SolidBrush(t.Label);
    }

    private static ISolidColorBrush SolidBrush(string hex) =>
        new SolidColorBrush(Color.Parse(hex));

    // Rebuild the "02 Start Monitoring" button content after it's clicked
    private static object BuildMonitoringContent()
    {
        var grid = new Grid();
        grid.ColumnDefinitions.Add(new ColumnDefinition(38, GridUnitType.Pixel));
        grid.ColumnDefinitions.Add(new ColumnDefinition(1,  GridUnitType.Star));

        var badge = new Border
        {
            Width = 28, Height = 28,
            CornerRadius = new Avalonia.CornerRadius(14),
            Background = SolidBrush("#76b900"),
            VerticalAlignment = Avalonia.Layout.VerticalAlignment.Center,
            Child = new TextBlock
            {
                Text = "02", FontSize = 10,
                FontWeight = FontWeight.Bold,
                Foreground = SolidBrush("#ffffff"),
                HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center,
                VerticalAlignment   = Avalonia.Layout.VerticalAlignment.Center
            }
        };
        Grid.SetColumn(badge, 0);

        var stack = new StackPanel { Spacing = 2, VerticalAlignment = Avalonia.Layout.VerticalAlignment.Center };
        stack.Children.Add(new TextBlock
        {
            Text = "Monitoring", FontSize = 12,
            FontWeight = FontWeight.SemiBold,
            Foreground = SolidBrush("#d8d8d8")
        });
        stack.Children.Add(new TextBlock
        {
            Text = "Polling every 2 s...", FontSize = 10,
            Foreground = SolidBrush("#3a5a00")
        });
        Grid.SetColumn(stack, 1);

        grid.Children.Add(badge);
        grid.Children.Add(stack);
        return grid;
    }
}
