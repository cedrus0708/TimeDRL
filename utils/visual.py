from rich.table import Table
from rich.console import Console
import matplotlib.pyplot as plt
import numpy as np

from utils.saver import Saver


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



def show_pretrain_plot(pretrain_losses, saver: Saver, pretrain_epoch):
    # x tengely: 1, 2, 3, ..., N
    epochs = list(range(1, len(pretrain_losses) + 1))

    # ábra
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, pretrain_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Pretrain loss")
    plt.title("Pretraining loss curve")
    plt.grid(True)

    # mentési útvonal
    save_path = saver.get_path("learning_curves", f"pretrain_loss_curve_{pretrain_epoch}.png")

    # mentés
    plt.savefig(save_path, bbox_inches="tight")
    

    

def show_plot(history, saver: Saver, pretrain_epoch):
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

    # mentési útvonal
    save_path = saver.get_path("learning_curves", f"linear_eval_curve_{pretrain_epoch}.png")

    # mentés
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def show_final_linear_eval_plot(linear_eval_history, saver: Saver):
    best_test_mse = list(linear_eval_history["best_test_mse"])
    best_test_mae = list(linear_eval_history["best_test_mae"])
    epochs = list(range(1, len(best_test_mse) + 1))

    plt.figure(figsize=(8, 5))

    if not np.all(np.isnan(best_test_mse)):
        plt.plot(epochs, best_test_mse, marker="o", label="Best test MSE")

    if not np.all(np.isnan(best_test_mae)):
        plt.plot(epochs, best_test_mae, marker="o", label="Best test MAE")

    plt.xlabel("Pretrain epoch")
    plt.ylabel("Metric value")
    plt.title("Final linear evaluation history")
    plt.grid(True)
    plt.legend()

    save_path = saver.get_path("learning_curves", "final_linear_eval_history.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()