#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/28 下午10:35
# @Author  : Chengjie
# @File    : make_latex_tables.py
# @Software: PyCharm
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


def get_latex_table(case_name, metric):
    data = case_study[case_name]
    models = data["models"]

    model_tables = []

    for i, model in enumerate(models.keys()):
        f_path = (
            f"./ms_results_formal_experiments_1/"
            f"{case_name}/{model}/ua_ms_model.csv"
        )

        df = pd.read_csv(f_path)
        df = df.loc[:, ["test_suite"] + list(metric.keys())]

        # test_suite → columns
        df = df.set_index("test_suite")

        # rows = metrics, columns = test suites
        df = df.T

        # Pretty metric names
        df.index = [metric[m] for m in df.index]

        # Add metric as a real column only for the first model
        if i == 0:
            df.insert(0, "Metric", df.index)
            cols = [("", "Metric")] + [(models[model], c) for c in df.columns[1:]]
        else:
            # No Metric column for subsequent models
            cols = [(models[model], c) for c in df.columns]

        # MultiIndex columns: (Model, TestSuite)
        df.columns = pd.MultiIndex.from_tuples(cols, names=["Model", "TestSuite"])

        model_tables.append(df)

    # Concatenate models horizontally
    final_table = pd.concat(model_tables, axis=1)

    return final_table


def get_latex_table_by_model(case_name, metric, caption_prefix=""):
    """
    Generates LaTeX tables, one per model, for the given case study.

    Returns:
        dict: {model_name: LaTeX string}
    """
    data = case_study[case_name]
    models = data["models"]

    latex_tables = {}

    for model in models.keys():
        f_path = f"./ms_results_formal_experiments_1/{case_name}/{model}/ua_ms_model.csv"
        df = pd.read_csv(f_path)

        # Keep test_suite and metrics
        df = df.loc[:, ["test_suite"] + list(metric.keys())]

        # Prepare MultiIndex for metrics
        metrics_only = df[list(metric.keys())]
        metrics_only.columns = pd.MultiIndex.from_product(
            [[models[model]], [metric[c] for c in list(metric.keys())]],
            names=["Model", "Metric"]
        )

        # Combine test_suite and metrics
        final_table = pd.concat([df[["test_suite"]], metrics_only], axis=1)
        final_table.columns = pd.MultiIndex.from_tuples(
            [("", "test_suite") if c == "test_suite" else c for c in final_table.columns],
            names=["Model", "Metric"]
        )

        # Convert to LaTeX
        # Convert to LaTeX with 3 decimal digits
        latex_str = final_table.to_latex(
            index=False,
            escape=False,           # keep \textit{} etc.
            multicolumn=True,
            multicolumn_format="c",
            bold_rows=False,
            float_format="%.3f",
            # caption=f"{caption_prefix} {models[model]}",
            # column_format="l" + "r" * len(metric)  # first column left, metrics right
        )

        latex_tables[model] = latex_str

    return latex_tables


img_table = get_latex_table('sticker_detection', iou_metrics)  # social_perception, sticker_detection screw_detection
latex = img_table.to_latex(
    index=False,
    float_format="%.3f",
    escape=False,
    multicolumn=True,
    multicolumn_format="c",
    # booktabs=True
)

print(latex)


# table = get_latex_table_by_model('social_perception', uc_metrics)
#
# models = case_study['social_perception']['models']
# print(models)
# for model in models.keys():
#     print(table[model])

# test_suite,[img_level] ms_miss,[img_level] ms_ghost,[img_level] ms_miss_ghost,
# [obj_level] kill_rate_miss_mean,[obj_level] kill_rate_ghost_mean,[obj_level] kill_rate_miss_ghost_mean,
# [match] iou_mean,
# [match] vr_diff,[match] ie_diff,[match] mi_diff,[match] var_diff,[match] ps_diff,
# [miss] vr_diff,[miss] ie_diff,[miss] mi_diff,[miss] var_diff,[miss] ps_diff,
# [ghost] vr_diff,[ghost] ie_diff,[ghost] mi_diff,[ghost] var_diff,[ghost] ps_diff
