#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/30 下午2:40
# @Author  :
# @File    : box_plot.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter


class OneSigFigFormatter(ScalarFormatter):
    def _set_format(self):
        self.format = "%.1f"


case_study = {
    'screw_detection':
        {
            'test_suites': {
                'normal': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_screw/easy.pkl',
                'different_exposure': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_screw/hard.pkl'},
            'case_study_name': 'screw_detection',
            'models': {'best_added_more_noscrews_at_diff_exposure': r'\textit{ScDM1}'}
        },
    'sticker_detection':
        {
            'test_suites': {
                'orig': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_sticker/orig.pkl',
                'dalle': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_sticker/dalle.pkl',
                'sd': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_sticker/sd.pkl'
            },
            'case_study_name': 'sticker_detection',
            'models': {
                'fasterrcnn_resnet50_fpn': r'\textit{StDM1}', 'fasterrcnn_resnet50_fpn_v2': r'\textit{StDM2}',
                'retinanet_resnet50_fpn': r'\textit{StDM3}', 'retinanet_resnet50_fpn_v2': r'\textit{StDM4}',
                'ssd300_vgg16': r'\textit{StDM5}',
                # 'ssdlite320_mobilenet_v3_large'
            }
        },
    'social_perception':
        {
            'test_suites': {
                'low': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_face/low_count.pkl',
                'meduim': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_face/medium_count.pkl',
                'high': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_face/high_count.pkl'
            },
            'case_study_name': 'social_perception',
            'models': {'yunet_s': r'\textit{HFDM1}', 'yunet_n': r'\textit{HFDM2}'}
        }
}

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

match_cols = [
    "[match] vr_diff", "[match] ie_diff", "[match] mi_diff", "[match] var_diff", "[match] ps_diff"
]

miss_cols = [
    "[miss] vr_diff", "[miss] ie_diff", "[miss] mi_diff", "[miss] var_diff", "[miss] ps_diff"
]

ghost_cols = [
    "[ghost] vr_diff", "[ghost] ie_diff", "[ghost] mi_diff", "[ghost] var_diff", "[ghost] ps_diff"
]

# Combine all columns if needed
# uc_metrics = match_cols + miss_cols + ghost_cols

# Optional mapping for LaTeX-friendly metric names
# uc_metrics = {
#     "[match] vr_diff": "$MS_{vr}^{match}$",
#     "[match] ie_diff": "$MS_{ie}^{match}$",
#     "[match] mi_diff": "$MS_{mi}^{match}$",
#     "[match] var_diff": "$MS_{va}^{match}$",
#     "[match] ps_diff": "$MS_{ps}^{match}$",
#     "[miss] vr_diff": "$MS_{vr}^{miss}$",
#     "[miss] ie_diff": "$MS_{ie}^{miss}$",
#     "[miss] mi_diff": "$MS_{mi}^{miss}$",
#     "[miss] var_diff": "$MS_{va}^{miss}$",
#     "[miss] ps_diff": "$MS_{ps}^{miss}$",
#     "[ghost] vr_diff": "$MS_{vr}^{ghost}$",
#     "[ghost] ie_diff": "$MS_{ie}^{ghost}$",
#     "[ghost] mi_diff": "$MS_{mi}^{ghost}$",
#     "[ghost] var_diff": "$MS_{va}^{ghost}$",
#     "[ghost] ps_diff": "$MS_{ps}^{ghost}$",
# }
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


def draw_line_chart_with_ci(metrics):
    # Load CSV
    df = pd.read_csv("./experiment_data/ms_results_formal_experiments_1/mc_dropout_mutation_ratio.csv")

    models = df['model'].unique()

    num_rows = len(metrics)
    num_cols = len(models)

    sns.set(style="whitegrid")
    # x_vals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    x_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    # LINE_COLOR = "#1f77b4"  # classic matplotlib blue
    # LINE_COLOR = "#ff7f0e"  # matplotlib default orange
    LINE_COLOR = "#E65C00"  # matplotlib red-orange / brick red

    # \definecolor{corrLow}{RGB}{255,237,213}
    # \definecolor{corrModerate}{RGB}{255,204,153}
    # \definecolor{corrHigh}{RGB}{255,153,51}
    # \definecolor{corrVeryHigh}{RGB}{230,92,0}

    # Smaller figure, no shared y-axis
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(2.5 * num_cols, 2 * num_rows),
        sharex=True,
        sharey=False,  # do not share y-axis
        constrained_layout=True
    )

    # Ensure axes is 2D
    if num_rows == 1:
        axes = [axes]
    if num_cols == 1:
        axes = [[ax] for ax in axes]

    # Plot each metric for each model
    for i, metric in enumerate(metrics):
        ci_col = f"{metric}_CI95"
        for j, model in enumerate(models):
            ax = axes[i][j]
            sub_df = df[df['model'] == model]

            # Mean line
            sns.lineplot(
                x="dropout_rate",
                y=metric,
                data=sub_df,
                ax=ax,
                marker='o',
                color=LINE_COLOR,
                linewidth=2
            )

            # 95% CI band
            if ci_col in sub_df.columns:
                ax.fill_between(
                    sub_df["dropout_rate"],
                    sub_df[metric] - sub_df[ci_col],
                    sub_df[metric] + sub_df[ci_col],
                    color=LINE_COLOR,
                    alpha=0.2
                )

            # ----- Black frame (spines) -----
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(1.0)

            # # ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            formatter = OneSigFigFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))  # always scientific notation

            ax.yaxis.set_major_formatter(formatter)

            # Make the offset text (×10^k) bold & readable
            ax.yaxis.get_offset_text().set_fontsize(16)
            ax.yaxis.get_offset_text().set_fontweight("bold")

            # Axis formatting
            # ax.set_xlim(min(x_vals), max(x_vals))
            ax.set_xticks(x_vals)
            ax.set_xticklabels([str(x) for x in x_vals], rotation=45, fontsize=22)

            # Titles and labels
            if i == 0:
                ax.set_title(model_names[model], fontsize=22)
            if j == 0:
                ax.set_ylabel(metrics[metric], fontsize=22)
            else:
                ax.set_ylabel('')
            ax.set_xlabel("Dropout Rate", fontsize=22)
            ax.tick_params(axis='both', which='major', labelsize=22)

    # ---- control spacing here ----
    # plt.tight_layout()
    # plt.subplots_adjust(
    #     wspace=0.30,  # ⬅️ make plots in same row closer
    #     hspace=0.25
    # )

    # Save as PDF
    fig.savefig("./RQ3.1_figs/mcd_all.pdf")
    plt.close(fig)


# metrics_all = {**obj_metrics, **iou_metrics, **uc_metrics}
# print(metrics)

metrics_obj_iou_match = {**obj_metrics, **iou_metrics, **uc_metrics_match}
metrics_miss_ghost = {**uc_metrics_miss, **uc_metrics_ghost}

metrics_all = {**obj_metrics, **iou_metrics, **uc_metrics_match, **uc_metrics_miss, **uc_metrics_ghost}

draw_line_chart_with_ci(metrics_all)
