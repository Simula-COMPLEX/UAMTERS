#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/1/5 下午2:12
# @Author  : Chengjie
# @File    : correlation_test.py
# @Software: PyCharm
import pandas as pd
from scipy.stats import spearmanr

img_metrics = {
    "[img_level] ms_miss": '$MS_{img}^{miss}$',
    "[img_level] ms_ghost": '$MS_{img}^{ghost}$',
    "[img_level] ms_miss_ghost": '$MS_{img}^{mg}$',
}

obj_metrics = {
    "[obj_level] kill_rate_miss_mean": '$MS_{obj}^{miss}$',
    "[obj_level] kill_rate_ghost_mean": '$MS_{obj}^{ghost}$',
    "[obj_level] kill_rate_miss_ghost_mean": '$MS_{obj}^{mg}$',
    # "[match] iou_mean",  # optional
}

iou_metrics = {
    "[match] iou_mean": '$MS_{iou}$',  # optional
}

# Optional mapping for LaTeX-friendly metric names
uc_metrics = {
    "[match] vr_diff": "$MS_{vr}^{match}$",
    "[match] ie_diff": "$MS_{ie}^{match}$",
    "[match] mi_diff": "$MS_{mi}^{match}$",
    "[match] var_diff": "$MS_{va}^{match}$",
    "[match] ps_diff": "$MS_{ps}^{match}$",
    "[miss] vr_diff": "$MS_{vr}^{miss}$",
    "[miss] ie_diff": "$MS_{ie}^{miss}$",
    "[miss] mi_diff": "$MS_{mi}^{miss}$",
    "[miss] var_diff": "$MS_{va}^{miss}$",
    "[miss] ps_diff": "$MS_{ps}^{miss}$",
    "[ghost] vr_diff": "$MS_{vr}^{ghost}$",
    "[ghost] ie_diff": "$MS_{ie}^{ghost}$",
    "[ghost] mi_diff": "$MS_{mi}^{ghost}$",
    "[ghost] var_diff": "$MS_{va}^{ghost}$",
    "[ghost] ps_diff": "$MS_{ps}^{ghost}$",
}
uc_metrics_match = {
    "[match] vr_diff": "$MS_{vr}^{match}$",
    "[match] ie_diff": "$MS_{ie}^{match}$",
    "[match] mi_diff": "$MS_{mi}^{match}$",
    "[match] var_diff": "$MS_{va}^{match}$",
    "[match] ps_diff": "$MS_{ps}^{match}$",
}

uc_metrics_miss = {
    "[miss] vr_diff": "$MS_{vr}^{miss}$",
    "[miss] ie_diff": "$MS_{ie}^{miss}$",
    "[miss] mi_diff": "$MS_{mi}^{miss}$",
    "[miss] var_diff": "$MS_{va}^{miss}$",
    "[miss] ps_diff": "$MS_{ps}^{miss}$",
}

uc_metrics_ghost = {
    "[ghost] vr_diff": "$MS_{vr}^{ghost}$",
    "[ghost] ie_diff": "$MS_{ie}^{ghost}$",
    "[ghost] mi_diff": "$MS_{mi}^{ghost}$",
    "[ghost] var_diff": "$MS_{va}^{ghost}$",
    "[ghost] ps_diff": "$MS_{ps}^{ghost}$",
}

model_names = {'best_added_more_noscrews_at_diff_exposure': r'ScDM1',
               'fasterrcnn_resnet50_fpn': r'StDM1', 'fasterrcnn_resnet50_fpn_v2': r'StDM2',
               'retinanet_resnet50_fpn': r'StDM3', 'retinanet_resnet50_fpn_v2': r'StDM4',
               'ssd300_vgg16': r'StDM5',
               'yunet_s': r'HFDM1', 'yunet_n': r'HFDM2'
               }

metrics_all = {**obj_metrics, **iou_metrics, **uc_metrics}


def correlation_test_mcb(metrics):
    """
    Spearman correlation between dropout_rate and metrics.
    Aggregation:
        - mean over block_size
    Per-model correlation test.

    Returns:
        corr_df: DataFrame with columns
            [model, metric, spearman_rho, p_value, significance]
    """

    # Load data
    df = pd.read_csv("./experiment_data/ms_results_formal_experiments_1/mc_dropout_mutation_ratio.csv")

    models = df["model"].unique()

    results = []

    for model in models:
        df_m = df[df["model"] == model]

        # average over block_size (important!)
        df_avg = (
            df_m
            .groupby(["dropout_rate"], as_index=False)[metrics]
            .mean()
        )

        x = df_avg["dropout_rate"]

        for metric in metrics:
            y = df_avg[metric]

            # Spearman test
            rho, p = spearmanr(x, y)

            # significance level
            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            else:
                sig = "n.s."

            results.append({
                "model": model_names[model],
                "metric": metrics_all[metric],
                "spearman_rho": rho,
                "p_value": p,
                "significance": sig
            })

    corr_df = pd.DataFrame(results)

    return corr_df


def format_rho_with_color(rho, p):
    """
    Return LaTeX-formatted rho with color if significant (p < 0.01)
    """
    if pd.isna(rho):
        return ""

    # 非显著：不加颜色
    if p >= 0.01:
        return f"{rho:.3f}"

    abs_rho = abs(rho)

    if abs_rho < 0.3:
        return f"{rho:.3f}"  # negligible → no color
    elif abs_rho < 0.5:
        color = "corrLow"
    elif abs_rho < 0.7:
        color = "corrModerate"
    elif abs_rho < 0.9:
        color = "corrHigh"
    else:
        color = "corrVeryHigh"

    return rf"\cellcolor{{{color}}}\textbf{{{rho:.3f}}}"


def spearman_to_wide_table(corr_df, metric_order):
    """
    metric_order: list of metrics in the desired (original) order
    """

    # 🔒 lock metric order
    corr_df["metric"] = pd.Categorical(
        corr_df["metric"],
        categories=metric_order,
        ordered=True
    )

    # p-value display (DO NOT change numeric p_value)
    corr_df["p_display"] = corr_df["p_value"].apply(
        lambda p: r"\textbf{<0.01}" if p < 0.01 else f"{p:.3f}"
    )

    # pivot rho and p
    rho_df = corr_df.pivot(
        index="metric",
        columns="model",
        values="spearman_rho"
    ).reindex(metric_order)

    p_df = corr_df.pivot(
        index="metric",
        columns="model",
        values="p_value"
    ).reindex(metric_order)

    p_disp_df = corr_df.pivot(
        index="metric",
        columns="model",
        values="p_display"
    ).reindex(metric_order)

    # build table
    columns = []
    data = []

    for metric in metric_order:
        row = []
        for model in rho_df.columns:
            p_val = p_df.loc[metric, model]
            rho_val = rho_df.loc[metric, model]

            # p column (string)
            row.append(p_disp_df.loc[metric, model])

            # rho column (colored if significant)
            row.append(format_rho_with_color(rho_val, p_val))

        data.append(row)

    for model in rho_df.columns:
        columns.append((model, r"$p$"))
        columns.append((model, r"$\rho$"))

    wide_df = pd.DataFrame(
        data,
        index=metric_order,
        columns=pd.MultiIndex.from_tuples(columns)
    ).reset_index()

    wide_df = wide_df.rename(columns={"index": "Metric"})
    return wide_df


metrics = list(obj_metrics.keys()) + list(iou_metrics.keys()) + list(uc_metrics.keys())
metrics_order = list(obj_metrics.values()) + list(iou_metrics.values()) + list(uc_metrics.values())

corr_dframe = correlation_test_mcb(metrics)
# corr_dframe["model"] = corr_dframe["model"].map(model_names)

wide_dframe = spearman_to_wide_table(corr_dframe, metrics_order)

latex = wide_dframe.to_latex(
    index=False,
    escape=False,
    multicolumn=True,
    multicolumn_format="c",
    float_format="%.3f",
    caption=(
        "Spearman correlation between dropout rate and metrics. "
        r"$\rho$ denotes Spearman’s rank correlation coefficient."
    ),
    label="tab:spearman_dropout_models",
    column_format="l" + "rr" * (len(wide_dframe.columns) - 1)
)

print(latex)
