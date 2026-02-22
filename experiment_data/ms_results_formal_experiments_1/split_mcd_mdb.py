#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/1/2 上午11:11
# @Author  : 
# @File    : split_mcd_mdb.py
# @Software: PyCharm
import numpy as np
import pandas as pd

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

all_metrics = {**obj_metrics, **iou_metrics, **uc_metrics}

import pandas as pd
import numpy as np
from scipy.stats import t


def split_and_combine(case_name, metrics_dict):
    """
    Aggregate metrics across test suites while preserving rows,
    add metadata columns, and combine all models into one DataFrame.
    Adds 95% confidence interval columns for each metric.

    Returns:
        combined_df: pandas DataFrame
    """
    data = case_study[case_name]
    models = list(data["models"].keys())
    test_suites = list(data["test_suites"].keys())

    df_all_models = []

    for model in models:
        df_list = []
        for ts in test_suites:
            f_path = f"./{case_name}/{model}/{ts}/ua_ms.csv"
            df = pd.read_csv(f_path)

            # keep metrics + first 9 rows (dropout rates)
            df = df[list(metrics_dict.keys())].iloc[9:].copy()
            df_list.append(df)

        # stack: shape -> (num_ts, num_dropout, num_metrics)
        stacked = np.stack([df.values for df in df_list], axis=0)
        num_ts = stacked.shape[0]

        # mean over test suites (axis=0)
        mean_values = stacked.mean(axis=0)

        # compute 95% CI
        ci_values = []
        alpha = 0.05
        for i in range(stacked.shape[1]):  # for each dropout rate
            ci_row = []
            for j in range(stacked.shape[2]):  # for each metric
                sample = stacked[:, i, j]
                mean = sample.mean()
                std_err = sample.std(ddof=1) / np.sqrt(num_ts)
                t_score = t.ppf(1 - alpha / 2, df=num_ts - 1)
                margin = t_score * std_err
                ci_row.append(margin)
            ci_values.append(ci_row)

        ci_values = np.array(ci_values)  # shape -> (num_dropout, num_metrics)

        # back to DataFrame
        mean_df = pd.DataFrame(
            mean_values,
            columns=df_list[0].columns,
            index=df_list[0].index
        )
        ci_df = pd.DataFrame(
            ci_values,
            columns=[f"{m}_CI95" for m in df_list[0].columns],
            index=df_list[0].index
        )

        # combine mean + CI
        mean_df = pd.concat([mean_df, ci_df], axis=1)

        # add metadata columns
        mean_df["case_study"] = case_name
        mean_df["model"] = model
        mean_df["dropout_rate"] = sorted(
            [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] * 5
        )
        mean_df["block_size"] = [1, 3, 5, 7, 9] * 9

        # reorder columns: metadata first
        mean_df = mean_df[
            ["case_study", "model", "dropout_rate", "block_size"] + \
            list(metrics_dict.keys()) + \
            [f"{m}_CI95" for m in metrics_dict.keys()]
            ]

        df_all_models.append(mean_df)

    # combine all models into single DataFrame
    combined_df = pd.concat(df_all_models, ignore_index=True)
    return combined_df


# def split_and_combine(case_name, metrics_dict):
#     """
#     Aggregate metrics across test suites while preserving rows,
#     add metadata columns, and combine all models into one DataFrame.
#
#     Returns:
#         combined_df: pandas DataFrame
#     """
#
#     data = case_study[case_name]
#     models = list(data["models"].keys())
#     test_suites = list(data["test_suites"].keys())
#
#     df_all_models = []
#
#     for model in models:
#
#         df_list = []
#         for ts in test_suites:
#             f_path = f"./{case_name}/{model}/{ts}/ua_ms.csv"
#             df = pd.read_csv(f_path)
#
#             # keep metrics + first 10 rows (10 dropout rates)
#             df = df[list(metrics_dict.keys())].iloc[:9].copy()
#             print(model, ts)
#             print(df)
#             df_list.append(df)
#
#         # stack: shape -> (num_ts, 10, num_metrics)
#         stacked = np.stack([df.values for df in df_list], axis=0)
#
#         # mean over test suites (axis=0)
#         mean_values = stacked.mean(axis=0)
#
#         # back to DataFrame, rows preserved
#         mean_df = pd.DataFrame(
#             mean_values,
#             columns=df_list[0].columns,
#             index=df_list[0].index
#         )
#
#         # add metadata columns
#         mean_df["case_study"] = case_name
#         mean_df["model"] = model
#         mean_df["dropout_rate"] = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
#
#         # reorder columns: put metadata first
#         mean_df = mean_df[
#             ["case_study", "model", "dropout_rate"] + list(metrics_dict.keys())
#             ]
#
#         # append to list
#         df_all_models.append(mean_df)
#
#     # combine all models into a single DataFrame
#     combined_df = pd.concat(df_all_models, ignore_index=True)
#
#     return combined_df

sp_df = split_and_combine('social_perception', all_metrics)
scdm_df = split_and_combine('screw_detection', all_metrics)
stdm_df = split_and_combine('sticker_detection', all_metrics)

combined_all_df = pd.concat([sp_df, scdm_df, stdm_df], ignore_index=True)
combined_all_df.to_csv('./mc_dropblock_mutation_ratio.csv', header=True, index=False)
