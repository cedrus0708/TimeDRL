import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset_loader.dataset_loader import load_forecasting_dataloader

def anomaly_plot(exp, args):
    train_loader, valid_loader, test_loader = load_forecasting_dataloader(args, mode="pretrain")
    plot_validation_series_and_residual_score(exp, valid_loader)

def plot_validation_series_and_residual_score(
    exp,
    valid_loader,
    use_abs=True,
    aggregate_channels="mean",   # "mean", "max", "sum"
    series_mode="lines",         # "lines" vagy "heatmap"
    max_channels_in_lines=None,  # pl. 20, ha túl sok csatorna van
    smooth_score=None,           # pl. 11 mozgóátlaghoz
    figsize=(18, 8),
):
    exp.model.eval()
    exp.linear_eval.eval()

    dataset = valid_loader.dataset
    full_series = dataset.data_x  # [T_valid, C]
    T_valid, C = full_series.shape

    seq_len = exp.args.seq_len
    pred_len = exp.args.pred_len

    score_sum = np.zeros(T_valid, dtype=np.float64)
    score_cnt = np.zeros(T_valid, dtype=np.float64)

    window_start = 0

    with torch.no_grad():
        for batch_x, batch_y, _, _ in valid_loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)

            with torch.cuda.amp.autocast(enabled=getattr(exp.args, "use_amp", False)):
                t_1, t_2, _, _, _, _, _, _ = exp.model(batch_x)

                y_pred_1 = exp.linear_eval(t_1)
                y_pred_2 = exp.linear_eval(t_2)

                y_pred_1 = exp.model.revin(y_pred_1, "denorm")
                y_pred_2 = exp.model.revin(y_pred_2, "denorm")

                pred = 0.5 * (y_pred_1 + y_pred_2)   # [B, T_out, C] tipikusan
                true = batch_y

                # Ha batch_y hosszabb lenne (pl. label_len + pred_len), csak a forecast részt vegyük
                if true.shape[1] != pred.shape[1]:
                    true = true[:, -pred.shape[1]:, :]

                resid = pred - true
                if use_abs:
                    resid = resid.abs()
                else:
                    resid = resid ** 2

                # csatorna aggregálás -> [B, T_out]
                if aggregate_channels == "mean":
                    resid = resid.mean(dim=-1)
                elif aggregate_channels == "max":
                    resid = resid.max(dim=-1).values
                elif aggregate_channels == "sum":
                    resid = resid.sum(dim=-1)
                else:
                    raise ValueError("aggregate_channels must be one of: mean, max, sum")

                resid = resid.detach().cpu().numpy()

            B, T_out = resid.shape

            for b in range(B):
                start_idx = window_start + b

                # a forecastolt jövő a seq után kezdődik
                t_idx = np.arange(start_idx + seq_len, start_idx + seq_len + T_out)

                valid_mask = (t_idx >= 0) & (t_idx < T_valid)
                t_idx = t_idx[valid_mask]
                r = resid[b][valid_mask]

                score_sum[t_idx] += r
                score_cnt[t_idx] += 1

            window_start += B

    score = np.divide(
        score_sum,
        np.maximum(score_cnt, 1.0),
        out=np.zeros_like(score_sum),
        where=score_cnt > 0
    )

    if smooth_score is not None and smooth_score > 1:
        kernel = np.ones(smooth_score) / smooth_score
        score = np.convolve(score, kernel, mode="same")

    fig, axes = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    x = np.arange(T_valid)

    # felső panel: teljes validációs idősor
    if series_mode == "lines":
        channels_to_plot = C
        if max_channels_in_lines is not None:
            channels_to_plot = min(C, max_channels_in_lines)

        for c in range(channels_to_plot):
            axes[0].plot(x, full_series[:, c], linewidth=0.8, alpha=0.7)

        if max_channels_in_lines is not None and C > max_channels_in_lines:
            axes[0].set_title(
                f"Validation series ({max_channels_in_lines}/{C} channels shown)"
            )
        else:
            axes[0].set_title(f"Validation series ({C} channels)")
        axes[0].set_ylabel("value")

    elif series_mode == "heatmap":
        im = axes[0].imshow(
            full_series.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest"
        )
        axes[0].set_title(f"Validation series heatmap ({C} channels)")
        axes[0].set_ylabel("channel")
        fig.colorbar(im, ax=axes[0], fraction=0.02, pad=0.01)

    else:
        raise ValueError("series_mode must be 'lines' or 'heatmap'")

    # alsó panel: anomaly score
    axes[1].plot(x, score, linewidth=1.2)
    axes[1].set_title("Residual-based anomaly score")
    axes[1].set_ylabel("score")
    axes[1].set_xlabel("time index")

    plt.tight_layout()
    plt.show()

    return full_series, score