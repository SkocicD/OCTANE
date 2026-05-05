"""Live loss plot viewer. Launched automatically by train.py --view.
Reads training/loss_log.csv and updates the plot every 2 seconds.
Can also be run standalone: python training/plot_viewer.py"""
import sys
import os
import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main(log_path: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Terrain Model — Training Progress', fontsize=14)
    ax.grid(True, alpha=0.3)
    train_line, = ax.plot([], [], label='train', color='steelblue',  linewidth=2)
    val_line,   = ax.plot([], [], label='val',   color='darkorange', linewidth=2)
    best_marker, = ax.plot([], [], 'o', color='green', markersize=8,
                           label='best val', zorder=5)
    ax.legend(fontsize=11)
    status_text = ax.text(0.02, 0.97, 'Waiting for training to start...',
                          transform=ax.transAxes, fontsize=10,
                          verticalalignment='top', color='gray')
    fig.tight_layout()

    def update(_frame):
        if not os.path.exists(log_path):
            return train_line, val_line, best_marker, status_text

        try:
            epochs, train_losses, val_losses = [], [], []
            with open(log_path, newline='') as f:
                for row in csv.DictReader(f):
                    epochs.append(int(row['epoch']))
                    train_losses.append(float(row['train']))
                    val_losses.append(float(row['val']))

            if not epochs:
                return train_line, val_line, best_marker, status_text

            train_line.set_data(epochs, train_losses)
            val_line.set_data(epochs, val_losses)

            # Mark the best val loss epoch
            best_idx = val_losses.index(min(val_losses))
            best_marker.set_data([epochs[best_idx]], [val_losses[best_idx]])

            ax.relim()
            ax.autoscale_view()

            status_text.set_text(
                f"Epoch {epochs[-1]}  |  "
                f"train {train_losses[-1]:.4f}  |  "
                f"val {val_losses[-1]:.4f}  |  "
                f"best val {min(val_losses):.4f} @ epoch {epochs[best_idx]}"
            )
            status_text.set_color('black')

        except Exception:
            pass

        return train_line, val_line, best_marker, status_text

    ani = animation.FuncAnimation(fig, update, interval=2000, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == '__main__':
    log_path = sys.argv[1] if len(sys.argv) > 1 else 'training/loss_log.csv'
    main(log_path)
