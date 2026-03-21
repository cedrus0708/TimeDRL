from rich.table import Table
from rich.console import Console
import matplotlib.pyplot as plt


STYLE_COLOR = {"train": "blue", "valid": "green", "test": "red"}


def show_table(history):
    # Extract metrics and modes
    metrics = history["test"].keys()
    modes = history.keys()

    # Show table
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Epoch")
    for mode in modes:
        for metric in metrics:
            table.add_column(
                f"{mode.capitalize()} {metric.upper()}", style=STYLE_COLOR[mode]
            )
    for epoch in range(len(history["test"]["loss"])):
        row = [str(epoch + 1)]
        for mode in modes:
            for metric in metrics:
                value = history[mode][metric][epoch]
                row.append(f"{value:.3f}")
        table.add_row(*row)
    console.print(table)


def show_plot(history, save_path="./img/training_history"):
    metrics = list(history["test"].keys())
    modes = list(history.keys())

    print("metrics:", metrics)
    print("modes:", modes)

    epochs = list(range(1, len(history["test"]["loss"]) + 1))

    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5))

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for mode in modes:
            ax.plot(
                epochs,
                history[mode][metric],
                color=STYLE_COLOR[mode],
                label=f"{mode.capitalize()} {metric.upper()}",
            )

        ax.set_title(f"{metric.upper()} vs Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.upper())
        ax.set_xticks(epochs)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to: {save_path}+{epochs}.png")