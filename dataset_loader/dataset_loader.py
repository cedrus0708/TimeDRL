from pathlib import Path
import os
import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import numpy as np
from rich.table import Table
from rich.console import Console
import warnings
import numpy as np
import plotly.graph_objects as go

from dataset_loader.forecasting_loader import data_provider, arg_setup_forecasting
from dataset_loader.classification_loader import (
    load_all_datasets,
    arg_setup_classification,
)


def show_dataset_stats(train_dataset, valid_dataset, test_dataset, show_K=True):
    # * Combine all datasets
    all_dataset = ConcatDataset([train_dataset, valid_dataset, test_dataset])

    # * N
    print(
        f"N: {len(all_dataset)} (train: {len(train_dataset)}, "
        f"val: {len(valid_dataset)}, test: {len(test_dataset)})"
    )

    # * C
    print(f"C: {all_dataset[0][0].shape[1]}")

    # * K
    if show_K:
        unique_labels = set()
        for item in all_dataset:
            unique_labels.add(item[1].item())
        print(f"K: {len(unique_labels)}")

    # * T
    print(f"T: {all_dataset[0][0].shape[0]}")

    # * Show the mean and std of all the samples in the training set
    x_list = []  # [(1, T, C), ...]
    for idx in range(len(train_dataset)):
        x = train_dataset[idx][0]
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        x_list.append(x.unsqueeze(0))
    all_x = torch.cat(x_list, dim=0)  # (N, T, C)
    means = all_x.mean(dim=[0, 1])  # (C,)
    stds = all_x.std(dim=[0, 1])  # (C,)
    # print("Mean:", means.numpy())
    # print("Std:", stds.numpy())
    # assert torch.allclose(
    #     means, torch.zeros_like(means), atol=1
    # ), f"Mean is not 0 but {means}"
    # assert torch.allclose(
    #     stds, torch.ones_like(stds), atol=1
    # ), f"Std is not 1 but {stds}"
    if not torch.allclose(means, torch.zeros_like(means), atol=1):
        warnings.warn(f"Mean is not 0 but {means}")
    if not torch.allclose(stds, torch.ones_like(stds), atol=1):
        warnings.warn(f"Std is not 1 but {stds}")

    # Class Distribution Analysis for each dataset
    if show_K:
        console = Console()
        datasets = {
            "Train": train_dataset,
            "Validation": valid_dataset,
            "Test": test_dataset,
        }
        unique_labels = set()
        for dataset in datasets.values():
            for _, label in dataset:
                label = label.item() if isinstance(label, torch.Tensor) else label
                unique_labels.add(label)

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Class", justify="right")
        for name in datasets:
            table.add_column(name, justify="right")

        class_distributions = {
            label: {name: 0 for name in datasets} for label in unique_labels
        }
        for name, dataset in datasets.items():
            for _, label in dataset:
                label = label.item() if isinstance(label, torch.Tensor) else label
                class_distributions[label][name] += 1

        for label in sorted(unique_labels):
            row = [str(label)]
            for name in datasets:
                total_samples = len(datasets[name])
                count = class_distributions[label][name]
                percentage = count / total_samples * 100
                row.append(f"{count} ({percentage:.2f}%)")
            table.add_row(*row)

        console.print("\nClass Distribution Across Datasets:")
        console.print(table)

        return class_distributions
    else:
        return None


def load_forecasting_dataloader(args, mode="pretrain", visualize=False):
    assert args.task_name == "forecasting"
    train_dataset, train_loader = data_provider(args, mode, flag="train")
    valid_dataset, valid_loader = data_provider(args, mode, flag="val")
    test_dataset, test_loader = data_provider(args, mode, flag="test")

    if visualize:
        visualize_validation_dataset(valid_dataset, args)

    # Show dataset statistics (N, C, K, T, mean, std)
    print("----------------------------------------")
    print(f"### {args.data_name} ###")
    show_dataset_stats(train_dataset, valid_dataset, test_dataset, show_K=False)

    return train_loader, valid_loader, test_loader


def visualize_validation_dataset(valid_dataset, args):
    """
    Show an interactive plot of the whole validation dataset.
    Assumes the dataset stores the raw split in a common attribute like data_x.
    """

    # 1) Try common attribute names
    if hasattr(valid_dataset, "data_x"):
        data = valid_dataset.data_x
    elif hasattr(valid_dataset, "data_y"):
        data = valid_dataset.data_y
    elif hasattr(valid_dataset, "x"):
        data = valid_dataset.x
    else:
        raise AttributeError(
            "Couldn't find raw validation data in the dataset. "
            "Expected one of: data_x, data_y, x"
        )

    # Convert to numpy
    if hasattr(data, "values"):  # pandas DataFrame
        data = data.values
    data = np.asarray(data)

    # Ensure shape [T, C]
    if data.ndim == 1:
        data = data[:, None]

    T, C = data.shape

    fig = go.Figure()

    for c in range(C):
        fig.add_trace(
            go.Scatter(
                x=np.arange(T),
                y=data[:, c],
                mode="lines",
                name=f"channel_{c}",
                visible=True if c == 0 else "legendonly",
            )
        )

    fig.update_layout(
        title=f"Validation dataset - {args.data_name}",
        xaxis_title="Time index",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    fig.update_xaxes(rangeslider_visible=True)
    fig.show()

    print(f"Validation data shape: T={T}, C={C}")


def load_classification_dataloader(args, mode="pretrain"):
    assert args.task_name == "classification"
    dataset_folder = args.root_folder / "dataset" / "classification" / args.data_name
    (
        train_dataset,
        train_loader,
        valid_dataset,
        valid_loader,
        test_dataset,
        test_loader,
    ) = load_all_datasets(args, mode, dataset_folder)

    # Show dataset statistics (N, C, K, T, mean, std)
    print("----------------------------------------")
    print(f"### {args.data_name} ###")
    class_distributions = show_dataset_stats(
        train_dataset, valid_dataset, test_dataset, show_K=True
    )

    return train_loader, valid_loader, test_loader, class_distributions


def update_args_from_dataset(args):
    if args.task_name == "forecasting":
        # Set data_path, data, C
        args = arg_setup_forecasting(args)
    elif args.task_name == "classification":
        # Set C, K, seq_len
        args = arg_setup_classification(args)
    else:
        raise NotImplementedError

    return args
