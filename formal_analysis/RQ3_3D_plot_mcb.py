#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/1/23 上午10:43
# @Author  : 
# @File    : RQ3_3D_plots.py
# @Software: PyCharm


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D

# Load data
df = pd.read_csv('./ms_results_formal_experiments_1/mc_dropblock_mutation_ratio.csv')


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
# Combine all columns if needed
# uc_metrics = match_cols + miss_cols + ghost_cols

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

model_names = {'best_added_more_noscrews_at_diff_exposure': r'SCDM1',
               'fasterrcnn_resnet50_fpn': r'STDM1', 'fasterrcnn_resnet50_fpn_v2': r'STDM2',
               'retinanet_resnet50_fpn': r'STDM3', 'retinanet_resnet50_fpn_v2': r'STDM4',
               'ssd300_vgg16': r'STDM5',
               'yunet_s': r'HFDM1', 'yunet_n': r'HFDM2'
               }

models = df['model'].unique()


class OneSigFigFormatter(ScalarFormatter):
    def _set_format(self):
        self.format = "%.1f"


def subplot_3d(metrics):
    num_rows = len(metrics)
    num_cols = len(models)

    # Define ticks
    dropout_ticks = [0.1, 0.2, 0.3, 0.4, 0.5]
    block_size_ticks = [1, 3, 5, 7, 9]

    # Create figure using plt.subplots with constrained_layout Note: sharex=True is not fully supported for 3D plots
    # in the same way as 2D (it doesn't automatically hide ticks correctly for 3D usually). We will manually handle
    # tick visibility, but use constrained_layout as requested.
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(2.5 * num_cols, 2 * num_rows),
        # sharex=True, # Removed to avoid potential 3D axis errors; implemented manually below
        # sharey=False,
        # constrained_layout=True,
        subplot_kw={'projection': '3d'}
    )

    # axes is a 2D array [row, col]
    for i, metric in enumerate(metrics):
        ci_col = metric + '_CI95'

        for j, model in enumerate(models):
            ax = axes[i, j]

            subset = df[df['model'] == model]

            if not subset.empty:
                X = subset['dropout_rate']
                Y = subset['block_size']
                Z = subset[metric]

                # Plot Mean
                ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

                # Plot CI
                if ci_col in df.columns:
                    Z_ci = subset[ci_col]
                    Z_upper = Z + Z_ci
                    Z_lower = Z - Z_ci
                    ax.plot_trisurf(X, Y, Z_upper, color='gray', edgecolor='none', alpha=0.3)
                    ax.plot_trisurf(X, Y, Z_lower, color='gray', edgecolor='none', alpha=0.3)

            # Axis configuration
            ax.set_xticks(dropout_ticks)
            ax.set_yticks(block_size_ticks)

            # Only show labels and tick labels for the last row
            if i == num_rows - 1:
                ax.set_xlabel('Dropout Rate', labelpad=10, fontsize=14)
                ax.set_ylabel('Block Size', labelpad=6, fontsize=14)
            else:
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            ax.set_xticklabels([str(x) for x in dropout_ticks], rotation=30, fontsize=14)
            ax.set_yticklabels([str(x) for x in block_size_ticks], fontsize=14)
            ax.tick_params(axis='z', labelsize=14, pad=10)
            # ax.tick_params(axis='x', pad=14)

            # formatter = ScalarFormatter(useMathText=True)
            # formatter.set_scientific(True)
            # formatter.set_powerlimits((-2, 2))  # Adjusts when 10^n kicks in
            # ax.zaxis.set_major_formatter(formatter)
            # ax.zaxis.get_major_formatter().set_useOffset(True)
            # ax.tick_params(axis='z', labelsize=12)
            # ax.zaxis.labelpad = 12

            # Column Headers (Model Names) - Top Row Only
            if i == 0:
                ax.set_title(model_names[model], fontsize=22, pad=10)

            # Row Headers (Metric Names) - Left Column Only
            if j == 0:
                # Adjust position for constrained layout
                # 3D axes text2D coordinates might need tuning
                ax.text2D(-0.1, 0.5, metrics[metric], transform=ax.transAxes,
                          rotation=90, va='center', ha='right', fontsize=22)

    plt.subplots_adjust(left=0, right=1, bottom=0.04, top=0.97, wspace=-0.2, hspace=0.25)
    # plt.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)

    # Save
    plt.savefig('mcb_3d_all.pdf')
    print("Figure saved.")


metrics_obj_iou_match = {**obj_metrics, **iou_metrics, **uc_metrics_match}
metrics_miss_ghost = {**uc_metrics_miss, **uc_metrics_ghost}

metrics_all = {**obj_metrics, **iou_metrics, **uc_metrics_match, **uc_metrics_miss, **uc_metrics_ghost}

subplot_3d(metrics_all)
