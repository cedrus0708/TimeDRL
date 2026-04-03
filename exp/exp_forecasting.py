import math

import torch
import torch.nn as nn
from torch import optim
import os
from pathlib import Path
import time
import warnings
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from dataset_loader.dataset_loader import load_forecasting_dataloader
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.tools import print_params, print_formatted_dict
from utils.visual import show_final_linear_eval_plot, show_pretrain_plot, show_table, show_plot
from models import linear_forecaster as linear_eval
from layers.Embed import Patching

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")


class Exp_Forecasting(Exp_Basic):
    def __init__(self, args, saver):
        super().__init__(args, saver)
        # 1. set args, model_dict, device into self
        # 2. build model

    def _build_model(self):
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # Feed `args` as `config`
        print_params(model)
        return model

    def _build_linear_eval(self):
        D, C, T_out = self.args.d_model, self.args.C, self.args.pred_len
        self.linear_eval = (
            linear_eval.Model(
                D,
                C,
                self.args.T_p,
                T_out,
                self.args.dropout,
                self.args.enable_channel_independence,
            )
            .float()
            .to(self.device)
        )

    def train_together(self, use_tqdm=False):
        #! The purpose of this function is to find a good encoder for the downstream task
        ### Training ###
        print(
            f">>>>> start training together (forecasting: {self.args.pred_len}) : {self.args.setting}>>>>>"
        )
        """
        For each epoch
        1. Train the encoder along with the linear_eval model
        """

        # Get data
        train_loader, valid_loader, test_loader = load_forecasting_dataloader(self.args)

        # Define the linear_eval model (we've already defined the encoder in the previous step)
        self._build_linear_eval()

        checkpoint_path = Path(self.args.checkpoints, self.args.setting)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        # * Define optimizers and early stopping (for both models)
        shared_optim = getattr(optim, self.args.pretrain_optim)(
            list(self.model.parameters()) + list(self.linear_eval.parameters()),
            lr=self.args.pretrain_learning_rate,
            weight_decay=self.args.pretrain_weight_decay,
        )
        shared_early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # Automatic Mixed Precision (some op. are fp32, some are fp16)
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)  # type: ignore

        # * The whole training process (pretrain + linear_eval)
        shared_history = {
            "train": {"loss": [], "mae": []},
            "valid": {"loss": [], "mae": []},
            "test": {"loss": [], "mae": []},
        }
        shared_epochs = self.args.pretrain_epochs
        for shared_epoch in range(shared_epochs):
            self.model.train()
            self.linear_eval.train()

            iter_data = (
                tqdm(
                    train_loader,
                    desc=f"Epoch {shared_epoch + 1}/{shared_epochs}, Training Loss: {0}",
                )
                if use_tqdm
                else train_loader
            )
            train_losses = []
            for i, (batch_x, batch_y, _, _) in enumerate(iter_data):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # ? 1. Zero grad
                shared_optim.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                    # ? 2. Call the model
                    # Encoder (including instance normalization)
                    t_1, t_2, _, _, _, _, _, _ = self.model(batch_x)

                    # Linear eval
                    y_pred_1 = self.linear_eval(t_1)
                    y_pred_2 = self.linear_eval(t_2)

                    # Reverse Instance Normalization
                    y_pred_1 = self.model.revin(y_pred_1, "denorm")
                    y_pred_2 = self.model.revin(y_pred_2, "denorm")

                    # ? 3. Calculate loss
                    loss = (
                        nn.MSELoss()(y_pred_1, batch_y)
                        + nn.MSELoss()(y_pred_2, batch_y)
                    ) / 2
                    train_losses.append(loss.item())

                # ? 4. Backward
                scaler.scale(loss).backward()  # type: ignore
                scaler.step(shared_optim)
                scaler.update()

                if use_tqdm:
                    iter_data.set_description(  # type: ignore
                        f"Epoch {shared_epoch + 1}/{shared_epochs}, "
                        f"Training Loss: {np.mean(train_losses):.3f}"
                    )

                # if i == 10:
                #     break

            # * At the end of each epoch, we get all the metrics
            print(">>>>> Calculate training metrics >>>>>")
            train_loss, train_mae = self.get_metrics(train_loader, use_tqdm)
            shared_history["train"]["loss"].append(train_loss)
            shared_history["train"]["mae"].append(train_mae)
            print(">>>>> Calculate validation metrics >>>>>")
            valid_loss, valid_mae = self.get_metrics(valid_loader, use_tqdm)
            shared_history["valid"]["loss"].append(valid_loss)
            shared_history["valid"]["mae"].append(valid_mae)
            print(">>>>> Calculate testing metrics >>>>>")
            test_loss, test_mae = self.get_metrics(test_loader, use_tqdm)
            shared_history["test"]["loss"].append(test_loss)
            shared_history["test"]["mae"].append(test_mae)

            # * Show metrics for all the previous epochs
            show_table(shared_history)
            show_plot(shared_history)

            # * Early stopping
            shared_early_stopping(valid_loss)
            if shared_early_stopping.early_stop:
                print("Early stopping")
                break

            # * Adjust learning rate
            adjust_learning_rate(
                shared_optim,
                shared_epoch + 1,
                self.args.pretrain_lradj,
                self.args.pretrain_learning_rate,
            )
            print("------------------------------------------------------------------")

        best_mse_epoch = np.nanargmin(shared_history["test"]["loss"])
        metrics = {
            "best_test_mse": shared_history["test"]["loss"][best_mse_epoch],
            "best_test_mae": shared_history["test"]["mae"][best_mse_epoch],
            "best_shared_epoch": best_mse_epoch + 1,
        }
        print("===============================")
        print_formatted_dict(metrics)
        print("===============================")

        self.spent_time = time.time() - start_time

        return metrics

    def train(self, use_tqdm=False):
        #! The purpose of this function is to train the given good encoder
        ### Training ###
        print(
            f">>>>> start training (forecasting: {self.args.pred_len}) : {self.args.setting}>>>>>"
        )
        """
        For each epoch
        1. (Unfreeze the encoder first) Train the encoder with the pretext tasks
        2. Freeze the encoder
        3. Get the linear_eval model and train it along with the encoder for the downstream task
        """

        # Get data
        train_loader, valid_loader, test_loader = load_forecasting_dataloader(
            self.args, mode="pretrain"
        )
        linear_eval_train_loader, _, _ = load_forecasting_dataloader(
            self.args, mode="linear_eval"
        )

        # Define the linear_eval model (we've already defined the encoder in the previous step)
        self._build_linear_eval()

        # Define the patching layer (for the predictive task)
        patching = Patching(
            self.args.patch_len, self.args.stride, self.args.enable_channel_independence
        )

        checkpoint_path = Path(self.args.checkpoints, self.args.setting)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        # * Define optimizers and early stopping (for both models)
        model_optim = getattr(optim, self.args.pretrain_optim)(
            self.model.parameters(),
            lr=self.args.pretrain_learning_rate,
            weight_decay=self.args.pretrain_weight_decay,
        )
        linear_eval_optim = getattr(optim, self.args.linear_eval_optim)(
            self.linear_eval.parameters(),
            lr=self.args.linear_eval_learning_rate,
            weight_decay=self.args.linear_eval_weight_decay,
        )
        model_early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        linear_eval_early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True
        )

        # Automatic Mixed Precision (some op. are fp32, some are fp16)
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)  # type: ignore

        # * The whole training process (pretrain + linear_eval)
        pretrain_history = {
            "predictive_loss": [],
            "contrastive_loss": [],
            "pretrain_loss": [],
        }
        linear_eval_history = {"best_test_mse": [], "best_test_mae": []}

        for pretrain_epoch in range(self.args.pretrain_epochs):
            ###! 1. Pretrain ###
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True  # Unfreeze the encoder
            trainable_ratio = print_params(self.model, show_num_params=False)
            assert (
                trainable_ratio == 1.0
            ), f"The encoder is not fully trainable (trainable_ratio: {trainable_ratio})"
            assert (
                self.args.disable_predictive_loss == False
                or self.args.disable_contrastive_loss == False
            ), "Both predictive and contrastive losses are disabled"

            iter_data = (
                tqdm(
                    train_loader,
                    desc=f"[Pretrain] Epoch {pretrain_epoch + 1}/{self.args.pretrain_epochs}, "
                    f"Predictive Loss: {0}, Contrastive Loss: {0}, Pretrain Loss: {0}",
                )
                if use_tqdm
                else train_loader
            )
            predictive_losses, contrastive_losses, pretrain_losses = [], [], []
            for i, (batch_x, _, _, _) in enumerate(iter_data):
                batch_x = batch_x.float().to(self.device)

                # ? 1. Zero grad
                model_optim.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                    # ? 2. Call the model
                    # Encoder (including instance normalization)
                    (
                        _,
                        _,
                        x_pred_1,
                        x_pred_2,
                        i_1,
                        i_2,
                        i_1_pred,
                        i_2_pred,
                    ) = self.model(batch_x)

                    # ? 3. Calculate loss
                    # Predictive task
                    if self.args.disable_predictive_loss:
                        predictive_loss = 0
                    else:
                        predictive_loss = (
                            nn.MSELoss()(x_pred_1, patching(batch_x))
                            + nn.MSELoss()(x_pred_2, patching(batch_x))
                        ) / 2

                    # Contrastive task
                    if self.args.disable_contrastive_loss:
                        contrastive_loss = 0
                    else:
                        if not self.args.disable_stop_gradient:
                            i_1 = i_1.detach()
                            i_2 = i_2.detach()
                        cos_sim = nn.CosineSimilarity(dim=1)
                        contrastive_loss = (
                            -(
                                cos_sim(i_1, i_2_pred).mean()
                                + cos_sim(i_2, i_1_pred).mean()
                            )
                            * 0.5
                        )

                    pretrain_loss = (
                        predictive_loss
                        + self.args.contrastive_weight * contrastive_loss
                    )
                    if self.args.disable_predictive_loss:
                        pretrain_losses.append(0)
                    else:
                        predictive_losses.append(predictive_loss.item())  # type: ignore
                    if self.args.disable_contrastive_loss:
                        contrastive_losses.append(0)
                    else:
                        contrastive_losses.append(contrastive_loss.item())  # type: ignore
                    pretrain_losses.append(pretrain_loss.item())

                # ? 4. Backward
                scaler.scale(pretrain_loss).backward()  # type: ignore
                scaler.step(model_optim)
                scaler.update()

                if use_tqdm:
                    iter_data.set_description(  # type: ignore
                        f"[Pretrain] Epoch {pretrain_epoch + 1}/{self.args.pretrain_epochs}, "
                        f"Predictive Loss: {np.mean(predictive_losses):.3f}, "
                        f"Contrastive Loss: {np.mean(contrastive_losses):.3f}, "
                        f"Pretrain Loss: {np.mean(pretrain_losses):.3f}"
                    )


                # if i == 10:
                #     break

            ###! 2. Linear Eval ###
            local_linear_eval_history = {
                # "train": {"loss": [], "mae": []},
                "valid": {"loss": [], "mae": []},
                "test": {"loss": [], "mae": []},
            }

            show_pretrain_plot(pretrain_losses, self.saver, pretrain_epoch)

            self.linear_eval.train()
            if not self.args.disable_freeze_encoder:
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False  # Freeze the encoder
                trainable_ratio = print_params(self.model, show_num_params=False)
                assert (
                    trainable_ratio == 0.0
                ), f"The encoder is not fully frozen (trainable_ratio: {trainable_ratio})"

            for linear_eval_epoch in range(self.args.linear_eval_epochs):
                iter_data = (
                    tqdm(
                        linear_eval_train_loader,
                        desc=f"({pretrain_epoch + 1}/{self.args.pretrain_epochs}) "
                        f"[Linear Eval] Epoch {linear_eval_epoch + 1}/{self.args.linear_eval_epochs}, Training Loss: {0}",
                    )
                    if use_tqdm
                    else linear_eval_train_loader
                )
                train_losses = []
                for i, (batch_x, batch_y, _, _) in enumerate(iter_data):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    # ? 1. Zero grad
                    linear_eval_optim.zero_grad()

                    with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                        # ? 2. Call the model
                        # Encoder (including instance normalization)
                        t_1, t_2, _, _, _, _, _, _ = self.model(batch_x)

                        # Linear eval
                        y_pred_1 = self.linear_eval(t_1)
                        y_pred_2 = self.linear_eval(t_2)

                        # Reverse Instance Normalization
                        y_pred_1 = self.model.revin(y_pred_1, "denorm")
                        y_pred_2 = self.model.revin(y_pred_2, "denorm")

                        # ? 3. Calculate loss
                        loss = (
                            nn.MSELoss()(y_pred_1, batch_y)
                            + nn.MSELoss()(y_pred_2, batch_y)
                        ) / 2
                        train_losses.append(loss.item())

                    # ? 4. Backward
                    scaler.scale(loss).backward()  # type: ignore
                    scaler.step(linear_eval_optim)
                    scaler.update()

                    if use_tqdm:
                        iter_data.set_description(  # type: ignore
                            f"({pretrain_epoch + 1}/{self.args.pretrain_epochs}) "
                            f"[Linear Eval] Epoch {linear_eval_epoch + 1}/{self.args.linear_eval_epochs}, "
                            f"Training Loss: {np.mean(train_losses):.3f}"
                        )

                    # if i == 10:
                    #     break

                # * At the end of each epoch, we get all the metrics
                # print(">>>>> Calculate training metrics >>>>>")
                # train_loss, train_mae = self.get_metrics(train_loader, use_tqdm)
                # local_linear_eval_history["train"]["loss"].append(train_loss)
                # local_linear_eval_history["train"]["mae"].append(train_mae)
                print(">>>>> Calculate validation metrics >>>>>")
                valid_loss, valid_mae = self.get_metrics(
                    valid_loader, use_tqdm
                )  # We need this to do early stopping
                local_linear_eval_history["valid"]["loss"].append(valid_loss)
                local_linear_eval_history["valid"]["mae"].append(valid_mae)
                print(">>>>> Calculate testing metrics >>>>>")
                test_loss, test_mae = self.get_metrics(test_loader, use_tqdm)
                local_linear_eval_history["test"]["loss"].append(test_loss)
                local_linear_eval_history["test"]["mae"].append(test_mae)

                print("Forecast example plotting on test data")
                x_t, y_true_t, y_predself_t = self.collect_predictions(test_loader, max_batches=2)

                print("x_t:", x_t.shape)
                print("y_true_t:", y_true_t.shape)
                print("xy_predself_t_t:", y_predself_t.shape)

                #print()
                #print("x_t:", x_t)
                #print("y_true_t:", y_true_t)
                #print("xy_predself_t_t:", y_predself_t)

                save_name=f"forecast_example_pretrain{pretrain_epoch+1}_lineval{linear_eval_epoch+1}.png"

                self.plot_forecast_example(
                    x_t, 
                    y_true_t, 
                    y_predself_t, 
                    sample_idx=0,
                    title=f"forecast_example_pretrain{pretrain_epoch+1}_lineval{linear_eval_epoch+1}",
                    save_name=save_name,
                    
                ) # x, y_true, y_pred, sample_idx=0, title=None, save_name="forecast_example.png", normalize_per_channel=True, offset_scale=3.0,
                


                # * Show metrics for all the previous epochs
                show_table(local_linear_eval_history)
                #show_plot(local_linear_eval_history)


                # * Early stopping
                linear_eval_early_stopping(valid_loss)
                if linear_eval_early_stopping.early_stop:
                    print("Early stopping")
                    break

                # * Adjust learning rate
                adjust_learning_rate(
                    linear_eval_optim,
                    linear_eval_epoch + 1,
                    self.args.linear_eval_lradj,
                    self.args.linear_eval_learning_rate,
                )


            show_plot(local_linear_eval_history, self.saver, pretrain_epoch)

            # * At the end of each epoch, we get all the metrics (for both pretrain and linear_eval)
            # ? Pretrain
            predictive_loss = np.mean(predictive_losses)
            contrastive_loss = np.mean(contrastive_losses)
            pretrain_loss = np.mean(pretrain_losses)
            pretrain_history["predictive_loss"].append(predictive_loss)
            pretrain_history["contrastive_loss"].append(contrastive_loss)
            pretrain_history["pretrain_loss"].append(pretrain_loss)
            # ? Linear Eval
            if np.all(np.isnan(local_linear_eval_history["test"]["loss"])):
                # All the linear results for this epoch are NaN
                linear_eval_history["best_test_mse"].append(np.nan)
                linear_eval_history["best_test_mae"].append(np.nan)
            else:
                best_mse_epoch = np.nanargmin(local_linear_eval_history["test"]["loss"])
                best_test_mse = local_linear_eval_history["test"]["loss"][
                    best_mse_epoch
                ]
                best_test_mae = local_linear_eval_history["test"]["mae"][best_mse_epoch]
                linear_eval_history["best_test_mse"].append(best_test_mse)
                linear_eval_history["best_test_mae"].append(best_test_mae)

            # * Early stopping
            model_early_stopping(pretrain_loss)
            if model_early_stopping.early_stop:
                print("Early stopping")
                break

            # * Adjust learning rate
            adjust_learning_rate(
                model_optim,
                pretrain_epoch + 1,
                self.args.pretrain_lradj,
                self.args.pretrain_learning_rate,
            )
            print("------------------------------------------------------------------")

            _, vis_valid_loader, vis_test_loader = load_forecasting_dataloader(
            self.args, mode="pretrain"
            )
            self.visualize_validation(vis_valid_loader, pretrain_epoch)

        show_pretrain_plot(pretrain_history["pretrain_loss"], self.saver, "last")
        #show_plot(linear_eval_history["best_test_mae"], self.saver, pretrain_epoch)
        show_final_linear_eval_plot(linear_eval_history, self.saver)


        _, vis_valid_loader, vis_test_loader = load_forecasting_dataloader(
            self.args, mode="pretrain"
        )
        self.visualize_validation(vis_valid_loader, "final")


        best_pretrain_epoch = np.nanargmin(pretrain_history["pretrain_loss"])
        best_best_test_mse_epoch = np.nanargmin(linear_eval_history["best_test_mse"])
        metrics = {
            # ? Pretrain (train)
            "best_pretrain_loss": pretrain_history["pretrain_loss"][
                best_pretrain_epoch
            ],
            "predictive_loss": pretrain_history["predictive_loss"][best_pretrain_epoch],
            "contrastive_loss": pretrain_history["contrastive_loss"][
                best_pretrain_epoch
            ],
            # ? Linear Eval (test)
            "best_test_mse": linear_eval_history["best_test_mse"][
                best_best_test_mse_epoch
            ],
            "best_test_mae": linear_eval_history["best_test_mae"][
                best_best_test_mse_epoch
            ],
            "best_pretrain_epoch": best_pretrain_epoch + 1,
            "best_best_test_mse_epoch": best_best_test_mse_epoch + 1,
        }
        print("===============================")
        print(f"### pred_len: {self.args.pred_len} ###")
        print_formatted_dict(metrics)
        print("===============================")

        #print("plot history")
        #self.plot_linear_eval_history(linear_eval_history)

        self.spent_time = time.time() - start_time

        return metrics

    # this function will make a short video showing the model sliding through a validational batch after training.
    # it will whow the predictions as the window slides forward and the residual-based anomaly-score.
    #def visualize_validation(self,):







    def visualize_validation(self, val_loader: DataLoader, pretrain_epoch, num_windows=48, feature_idx="all"):
        """
        Show a short animation of the model sliding through validation windows.

        Parameters
        ----------
        num_windows : int
            How many validation windows to animate.
        feature_idx : "all" | int | list[int]
            - "all": show all channels
            - int: show only one channel
            - list[int]: show selected channels
        """

        # --------------------------------------------------
        # 1) Get validation loader
        # --------------------------------------------------
        batch_size = val_loader.batch_size if val_loader.batch_size is not None else 1
        max_batches = math.ceil(num_windows / batch_size)

        # --------------------------------------------------
        # 2) Collect predictions
        # --------------------------------------------------
        xs, ys, preds = self.collect_predictions(val_loader, max_batches=max_batches)

        if xs.numel() == 0:
            raise RuntimeError("No validation predictions were collected.")

        # Ensure shape: [N, T, C]
        if xs.ndim == 2:
            xs = xs.unsqueeze(-1)
            ys = ys.unsqueeze(-1)
            preds = preds.unsqueeze(-1)

        xs = xs[:num_windows].cpu()
        ys = ys[:num_windows].cpu()
        preds = preds[:num_windows].cpu()

        n_windows, seq_len, n_channels = xs.shape
        pred_len = ys.shape[1]

        # --------------------------------------------------
        # 3) Decide which channels to show
        # --------------------------------------------------
        if feature_idx == "all":
            plot_channels = list(range(n_channels))
        elif isinstance(feature_idx, int):
            plot_channels = [feature_idx]
        elif isinstance(feature_idx, (list, tuple)):
            plot_channels = list(feature_idx)
        else:
            raise ValueError("feature_idx must be 'all', int, or list/tuple of ints.")

        for ch in plot_channels:
            if ch < 0 or ch >= n_channels:
                raise ValueError(f"Invalid channel index: {ch}. n_channels={n_channels}")

        # --------------------------------------------------
        # 4) Residual-based anomaly score
        # --------------------------------------------------
        # global score: mean residual over all pred steps and all channels
        residuals = torch.abs(preds - ys)                         # [N, pred_len, C]
        window_scores = residuals.mean(dim=(1, 2)).numpy()       # [N]

        # optional: per-channel score
        channel_scores = residuals.mean(dim=1).numpy()           # [N, C]

        # --------------------------------------------------
        # 5) Prepare arrays
        # --------------------------------------------------
        xs_np = xs.numpy()
        ys_np = ys.numpy()
        preds_np = preds.numpy()

        x_input = np.arange(seq_len)
        x_future = np.arange(seq_len, seq_len + pred_len)

        # --------------------------------------------------
        # 6) Create figure
        # --------------------------------------------------
        n_plot_axes = len(plot_channels)
        fig_height = 2.5 * n_plot_axes + 2.5

        fig, axes = plt.subplots(
            n_plot_axes + 1,
            1,
            figsize=(11, fig_height),
            gridspec_kw={"height_ratios": [3] * n_plot_axes + [1.5]},
        )

        if n_plot_axes + 1 == 2:
            axes = np.array(axes)

        series_axes = axes[:-1]
        score_ax = axes[-1]

        line_handles = []

        for ax, ch in zip(series_axes, plot_channels):
            channel_x = xs_np[:, :, ch]
            channel_y = ys_np[:, :, ch]
            channel_pred = preds_np[:, :, ch]

            y_min = min(channel_x.min(), channel_y.min(), channel_pred.min())
            y_max = max(channel_x.max(), channel_y.max(), channel_pred.max())
            pad = 0.05 * max(y_max - y_min, 1e-8)

            line_input, = ax.plot([], [], label="Input window")
            line_true, = ax.plot([], [], label="True future")
            line_pred, = ax.plot([], [], "--", label="Predicted future")

            ax.axvline(seq_len - 0.5, linestyle=":", alpha=0.7)
            ax.set_xlim(0, seq_len + pred_len - 1)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_ylabel(f"Channel {ch}")
            ax.grid(True)

            line_handles.append((line_input, line_true, line_pred, ch))

        series_axes[0].set_title("Validation sliding forecast")
        series_axes[-1].set_xlabel("Relative time")
        series_axes[0].legend(loc="upper left")

        info_text = series_axes[0].text(
            0.02,
            0.98,
            "",
            transform=series_axes[0].transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        # --------------------------------------------------
        # 7) Score subplot
        # --------------------------------------------------
        global_score_line, = score_ax.plot([], [], marker="o", label="Global anomaly score")
        current_point, = score_ax.plot([], [], marker="o", linestyle="None", markersize=8)

        score_ax.set_xlim(0, n_windows - 1)
        score_ax.set_ylim(0, max(float(np.nanmax(window_scores)) * 1.1, 1e-8))
        score_ax.set_xlabel("Validation window index")
        score_ax.set_ylabel("Mean |residual|")
        score_ax.set_title("Residual-based anomaly score (all channels)")
        score_ax.grid(True)
        score_ax.legend(loc="upper left")

        # --------------------------------------------------
        # 8) Animation callbacks
        # --------------------------------------------------
        def init():
            artists = []

            for line_input, line_true, line_pred, _ in line_handles:
                line_input.set_data([], [])
                line_true.set_data([], [])
                line_pred.set_data([], [])
                artists.extend([line_input, line_true, line_pred])

            global_score_line.set_data([], [])
            current_point.set_data([], [])
            info_text.set_text("")
            artists.extend([global_score_line, current_point, info_text])

            return artists

        def update(frame_idx):
            artists = []

            for line_input, line_true, line_pred, ch in line_handles:
                line_input.set_data(x_input, xs_np[frame_idx, :, ch])
                line_true.set_data(x_future, ys_np[frame_idx, :, ch])
                line_pred.set_data(x_future, preds_np[frame_idx, :, ch])
                artists.extend([line_input, line_true, line_pred])

            global_score_line.set_data(np.arange(frame_idx + 1), window_scores[: frame_idx + 1])
            current_point.set_data([frame_idx], [window_scores[frame_idx]])

            per_channel_text = ", ".join(
                [f"ch{ch}: {channel_scores[frame_idx, ch]:.4f}" for ch in plot_channels]
            )

            info_text.set_text(
                f"window: {frame_idx + 1}/{n_windows}\n"
                f"global score: {window_scores[frame_idx]:.6f}\n"
                f"{per_channel_text}"
            )

            artists.extend([global_score_line, current_point, info_text])
            return artists

        anim = FuncAnimation(
            fig,
            update,
            frames=n_windows,
            init_func=init,
            interval=400,
            blit=False,
        )

        # --------------------------------------------------
        # 9) Save
        # --------------------------------------------------
        if feature_idx == "all":
            file_name = f"validation_sliding_forecast_all_channels_predlen_{self.args.pred_len}_{pretrain_epoch}.gif"
        elif isinstance(feature_idx, int):
            file_name = f"validation_sliding_forecast_channel_{feature_idx}_predlen_{self.args.pred_len}_{pretrain_epoch}.gif"
        else:
            channel_str = "_".join(map(str, plot_channels))
            file_name = f"validation_sliding_forecast_channels_{channel_str}_predlen_{self.args.pred_len}_{pretrain_epoch}.gif"

        save_path = self.saver.get_path("forecast_examples", file_name)

        anim.save(save_path, writer=PillowWriter(fps=3), dpi=120)
        plt.close(fig)

        print(f"Validation visualization saved to: {save_path}")
        return save_path


    def get_metrics(self, data_loader, use_tqdm=False):
        total_mse = 0
        total_mae = 0
        total_samples = 0

        self.model.eval()
        self.linear_eval.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(
                tqdm(data_loader) if use_tqdm else data_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # ? 1. Zero grad
                pass

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                    # ? 2. Call the model
                    # Encoder (including instance normalization)
                    t_1, t_2, _, _, _, _, _, _ = self.model(batch_x)

                    # Linear eval
                    y_pred_1 = self.linear_eval(t_1)
                    y_pred_2 = self.linear_eval(t_2)

                    # Reverse Instance Normalization
                    y_pred_1 = self.model.revin(y_pred_1, "denorm")
                    y_pred_2 = self.model.revin(y_pred_2, "denorm")

                pred_1 = y_pred_1.detach()
                pred_2 = y_pred_2.detach()
                true = batch_y.detach()

                batch_mse = (
                    torch.mean((pred_1 - true) ** 2).item()
                    + torch.mean((pred_2 - true) ** 2).item()
                ) / 2
                batch_mae = (
                    torch.mean(torch.abs(pred_1 - true)).item()
                    + torch.mean(torch.abs(pred_2 - true)).item()
                ) / 2

                total_mse += batch_mse * len(batch_x)
                total_mae += batch_mae * len(batch_x)
                total_samples += len(batch_x)

                # if i == 10:
                #     break

        mse = total_mse / total_samples
        mae = total_mae / total_samples

        return mse, mae




    def collect_predictions(self, data_loader, max_batches=1):
        xs, ys, preds = [], [], []

        self.model.eval()
        self.linear_eval.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                t_1, t_2, _, _, _, _, _, _ = self.model(batch_x)

                y_pred_1 = self.linear_eval(t_1)
                y_pred_2 = self.linear_eval(t_2)

                y_pred_1 = self.model.revin(y_pred_1, "denorm")
                y_pred_2 = self.model.revin(y_pred_2, "denorm")

                y_pred = (y_pred_1 + y_pred_2) / 2.0

                xs.append(batch_x.detach().cpu())
                ys.append(batch_y.detach().cpu())
                preds.append(y_pred.detach().cpu())

                if i + 1 >= max_batches:
                    break

        return (
            torch.cat(xs, dim=0),
            torch.cat(ys, dim=0),
            torch.cat(preds, dim=0),
        )
    


    def plot_forecast_example_old(self, x, y_true, y_pred, sample_idx=0, channel_idx=0, title=None):
        hist = x[sample_idx, :, channel_idx].numpy()
        true = y_true[sample_idx, :, channel_idx].numpy()
        pred = y_pred[sample_idx, :, channel_idx].numpy()

        t_hist = np.arange(len(hist))
        t_fut = np.arange(len(hist), len(hist) + len(true))

        plt.figure(figsize=(12, 4))
        plt.plot(t_hist, hist, label="input window")
        plt.plot(t_fut, true, label="ground truth")
        plt.plot(t_fut, pred, label="forecast")
        plt.axvline(len(hist) - 1, linestyle="--")
        plt.legend()
        plt.title(title or f"sample={sample_idx}, channel={channel_idx}")
        plt.show()

    def plot_forecast_example_newer_old(
        self,
        x,
        y_true,
        y_pred,
        sample_idx=0,
        channel_idx=0,
        title=None,
        save_name="forecast_example.png",
    ):

        hist = x[sample_idx, :, channel_idx].detach().cpu().numpy()
        true = y_true[sample_idx, :, channel_idx].detach().cpu().numpy()
        pred = y_pred[sample_idx, :, channel_idx].detach().cpu().numpy()

        t_hist = np.arange(len(hist))
        t_fut = np.arange(len(hist), len(hist) + len(true))

        plt.figure(figsize=(12, 4))
        plt.plot(t_hist, hist, label="input window")
        plt.plot(t_fut, true, label="ground truth")
        plt.plot(t_fut, pred, label="forecast")
        plt.axvline(len(hist) - 1, linestyle="--")
        plt.legend()
        plt.title(title or f"sample={sample_idx}, channel={channel_idx}")
        plt.tight_layout()
        plt.savefig(f"./plots/{save_name}", dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Forecast plot saved to: ./plots/{save_name}")


    def plot_forecast_example(
        self,
        x,
        y_true,
        y_pred,
        sample_idx=0,
        title=None,
        save_name="forecast_example.png",
        normalize_per_channel=True,
        offset_scale=3.0,
    ):
        #os.makedirs("./plots", exist_ok=True)

        # x: [B, T_in, C]
        # y_true: [B, T_out, C]
        # y_pred: [B, T_out, C]
        x_np = x[sample_idx].detach().cpu().numpy()         # [T_in, C]
        y_true_np = y_true[sample_idx].detach().cpu().numpy()  # [T_out, C]
        y_pred_np = y_pred[sample_idx].detach().cpu().numpy()  # [T_out, C]

        t_in, n_channels = x_np.shape
        t_out = y_true_np.shape[0]

        t_hist = np.arange(t_in)
        t_fut = np.arange(t_in, t_in + t_out)

        plt.figure(figsize=(14, max(6, 1.2 * n_channels)))

        cmap = plt.get_cmap("tab10")

        for ch in range(n_channels):
            hist = x_np[:, ch].copy()
            true = y_true_np[:, ch].copy()
            pred = y_pred_np[:, ch].copy()

            if normalize_per_channel:
                full = np.concatenate([hist, true, pred])
                mean = full.mean()
                std = full.std()
                if std < 1e-8:
                    std = 1.0
                hist = (hist - mean) / std
                true = (true - mean) / std
                pred = (pred - mean) / std

            offset = ch * offset_scale
            color = cmap(ch % 10)

            plt.plot(t_hist, hist + offset, linewidth=1.2, color=color)
            plt.plot(t_fut, true + offset, linewidth=1.2, linestyle="-", color=color)
            plt.plot(t_fut, pred + offset, linewidth=1.2, linestyle="--", color=color)

        plt.axvline(t_in - 1, linestyle="--", alpha=0.7)

        # y-tengelyre csatornanevek / indexek
        yticks = [ch * offset_scale for ch in range(n_channels)]
        ylabels = [f"ch {ch}" for ch in range(n_channels)]
        plt.yticks(yticks, ylabels)

        plt.xlabel("time step")
        plt.ylabel("channels")
        plt.title(title or f"Forecast example, sample={sample_idx}")

        # Egyszerű, tiszta legenda
        legend_handles = [
            Line2D([0], [0], linestyle="-", label="input / history"),
            Line2D([0], [0], linestyle="-", label="ground truth"),
            Line2D([0], [0], linestyle="--", label="forecast"),
        ]
        plt.legend(handles=legend_handles, loc="best")

        plt.tight_layout()
        full_path = self.saver.get_path("forecast_examples", save_name)
        plt.savefig(full_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Forecast plot saved to: {full_path}")


    def plot_pretrain_history(self, pretrain_history):
        plt.figure(figsize=(10, 4))
        for k, v in pretrain_history.items():
            plt.plot(v, label=k)
        plt.xlabel("pretrain epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title("Pretraining curves")
        plt.show()

    def plot_linear_eval_history(self, hist):
        plt.figure(figsize=(10, 4))
        for split in hist:
            for metric in hist[split]:
                plt.plot(hist[split][metric], label=f"{split}_{metric}")
        plt.xlabel("linear-eval epoch")
        plt.legend()
        plt.title("Linear evaluation curves")
        plt.show()