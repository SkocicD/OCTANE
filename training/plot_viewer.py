"""Live loss plot viewer. Launched automatically by train.py --view.
Reads training/loss_log.csv and updates the plot every 2 seconds.
Can also be run standalone: python training/plot_viewer.py"""
import sys
import os
import csv
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def _read_status(status_path: str) -> str:
    try:
        with open(status_path) as f:
            s = json.load(f)
        if s.get('status') == 'done':
            return f"Training complete — {s['total']} epochs"
        return f"Epoch {s['epoch']} / {s['total']} in progress..."
    except Exception:
        return "Waiting for training to start..."


def _read_log(log_path: str):
    episodes, train_losses, val_episodes, val_losses = [], [], [], []
    try:
        with open(log_path, newline='') as f:
            for row in csv.DictReader(f):
                ep = int(row['episodes'])
                train_val = row['train']
                val_val   = row['val']
                if train_val:
                    episodes.append(ep)
                    train_losses.append(float(train_val))
                if val_val:
                    val_episodes.append(ep)
                    val_losses.append(float(val_val))
    except Exception:
        pass
    return episodes, train_losses, val_episodes, val_losses


def main(log_path: str):
    status_path = os.path.join(os.path.dirname(os.path.abspath(log_path)), 'train_status.json')

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlabel('Episodes seen', fontsize=12)
    ax.set_ylabel('Loss (mean)', fontsize=12)
    ax.set_title('Terrain Model — Training Progress', fontsize=14)
    ax.grid(True, alpha=0.3)

    train_line, = ax.plot([], [], color='steelblue',  linewidth=1.5, label='train (running mean)', alpha=0.85)
    val_line,   = ax.plot([], [], color='darkorange', linewidth=2,   label='val (per epoch)', marker='o', markersize=6)
    best_marker, = ax.plot([], [], '*', color='green', markersize=14, label='best val', zorder=5)
    ax.legend(fontsize=11)

    status_text = ax.text(0.01, 0.97, _read_status(status_path),
                          transform=ax.transAxes, fontsize=10,
                          verticalalignment='top', color='dimgray')
    fig.tight_layout()

    def update(_frame):
        status_text.set_text(_read_status(status_path))

        if not os.path.exists(log_path):
            return train_line, val_line, best_marker, status_text

        episodes, train_losses, val_episodes, val_losses = _read_log(log_path)

        if episodes:
            train_line.set_data(episodes, train_losses)

        if val_losses:
            val_line.set_data(val_episodes, val_losses)
            best_idx = val_losses.index(min(val_losses))
            best_marker.set_data([val_episodes[best_idx]], [val_losses[best_idx]])
            status_text.set_text(
                f"{_read_status(status_path)}  |  "
                f"best val {min(val_losses):.4f} @ {val_episodes[best_idx]:,} episodes"
            )

        if episodes or val_losses:
            ax.relim()
            ax.autoscale_view()

        return train_line, val_line, best_marker, status_text

    ani = animation.FuncAnimation(fig, update, interval=2000, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == '__main__':
    log_path = sys.argv[1] if len(sys.argv) > 1 else 'training/loss_log.csv'
    main(os.path.abspath(log_path))
