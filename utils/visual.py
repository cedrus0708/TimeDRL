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
    

def show_plot(history):
    metrics = list(history["test"].keys())
    modes = list(history.keys())

    print(metrics, modes)

    #plt.clf()
    

    for i, metric in enumerate(metrics):
        epochs = [epoch + 1 for epoch in range(len(history["test"]["loss"]))]
        plt.subplot(1, len(metrics), i + 1)

        plt.title(f"{metric.upper()} vs Epoch")
        plt.xticks(epochs)

        for mode in modes:
            plt.plot(
                epochs,
                history[mode][metric],
                color=STYLE_COLOR[mode],
                label=f"{mode.capitalize()} {metric.upper()}",
            )

        plt.xlabel("Epoch")
        plt.ylabel(metric.upper())
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.show()