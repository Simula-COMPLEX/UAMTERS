#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/1/23 下午2:10
# @Author  : Chengjie Lu
# @File    : RQ3_multiple_corr_test.py
# @Software: PyCharm


import pandas as pd
import statsmodels.api as sm
import numpy as np


metrics = [
    '[obj_level] kill_rate_miss_mean',
    '[obj_level] kill_rate_ghost_mean',
    '[obj_level] kill_rate_miss_ghost_mean',
    '[match] iou_mean',
    '[match] vr_diff',
    '[match] ie_diff',
    '[match] mi_diff',
    '[match] var_diff',
    '[match] ps_diff',
    '[miss] vr_diff',
    '[miss] ie_diff',
    '[miss] mi_diff',
    '[miss] var_diff',
    '[miss] ps_diff',
    '[ghost] vr_diff',
    '[ghost] ie_diff',
    '[ghost] mi_diff',
    '[ghost] var_diff',
    '[ghost] ps_diff'
]


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

models_all = {'best_added_more_noscrews_at_diff_exposure': r'ScDM1',
               'fasterrcnn_resnet50_fpn': r'StDM1', 'fasterrcnn_resnet50_fpn_v2': r'StDM2',
               'retinanet_resnet50_fpn': r'StDM3', 'retinanet_resnet50_fpn_v2': r'StDM4',
               'ssd300_vgg16': r'StDM5',
               'yunet_s': r'HFDM1', 'yunet_n': r'HFDM2'
               }

metrics_all = {**obj_metrics, **iou_metrics, **uc_metrics}

# Re-calculate to ensure we have the data
df = pd.read_csv('./ms_results_formal_experiments_1/mc_dropblock_mutation_ratio.csv')

results_list = []
models = df['model'].unique()

for model_name in models:
    df_model = df[df['model'] == model_name]
    # Check if there is enough variance
    if df_model['dropout_rate'].nunique() <= 1 or df_model['block_size'].nunique() <= 1:
        continue

    X = sm.add_constant(df_model[['dropout_rate', 'block_size']])

    for metric in metrics:
        if metric in df_model.columns:
            y = df_model[metric]
            mask = ~np.isnan(y) & ~np.isnan(X['dropout_rate']) & ~np.isnan(X['block_size'])

            if mask.sum() > 3:
                try:
                    ols_model = sm.OLS(y[mask], X[mask]).fit()
                    R = np.sqrt(ols_model.rsquared)
                    p_value = ols_model.f_pvalue
                    results_list.append({
                        'Model': model_name,
                        'Metric': metric,
                        'R': R,
                        'p': p_value
                    })
                except:
                    results_list.append({'Model': model_name, 'Metric': metric, 'R': np.nan, 'p': np.nan})
            else:
                results_list.append({'Model': model_name, 'Metric': metric, 'R': np.nan, 'p': np.nan})

results_df = pd.DataFrame(results_list)

if not results_df.empty:
    pivot_df = results_df.pivot(index='Metric', columns='Model', values=['R', 'p'])
    pivot_df.columns = pivot_df.columns.swaplevel(0, 1)

    # Reindex Rows (Metrics) to match input order
    available_metrics = [m for m in metrics if m in pivot_df.index]
    pivot_df = pivot_df.reindex(available_metrics)

    # Reindex Columns (Models) to match input order (df['model'].unique())
    # Note: pivot_df.columns is a MultiIndex (Model, Measure)
    # We want to sort the top level (Model) according to 'models' list
    # Let's filter 'models' to those present in pivot_df columns
    present_models = [m for m in models if m in pivot_df.columns.levels[0]]

    # We need to construct a new column index that respects this order
    # The second level is ['p', 'R'] (or ['R', 'p'] depending on previous pivot)
    # Actually, pivot created (measure, model). swaplevel -> (model, measure).
    # The measures are 'p' and 'R'.

    # Let's create the sorted columns list
    new_columns = []
    for model in present_models:
        new_columns.append((model, 'p'))
        new_columns.append((model, 'R'))

    pivot_df = pivot_df.reindex(columns=new_columns)

    latex_str = "\\begin{table}[ht]\n\\centering\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{l" + "cc" * len(
        present_models) + "}\n\\toprule\n"

    # Header Row 1: Model names
    latex_str += "Metric"
    for model in present_models:
        # escaped_model = model.replace('_', r'\_')
        latex_str += f" & \\multicolumn{{2}}{{c}}{{{models_all[model]}}}"
    latex_str += " \\\\\n"

    # Header Row 2: p and R repeated
    latex_str += " & " + " & ".join(["$p$", "$R$"] * len(present_models)) + " \\\\\n\\midrule\n"

    # Data Rows
    for index, row in pivot_df.iterrows():
        # escaped_index = index.replace('_', r'\_')
        latex_str += f"{metrics_all[index]}"
        for model in present_models:
            # We accessed (model, 'p') and (model, 'R')
            p_val = row[(model, 'p')]
            r_val = row[(model, 'R')]

            # Format p
            if pd.isna(p_val):
                p_str = "-"
            elif p_val < 0.01:
                p_str = "\\textbf{$<0.01$}"
            else:
                p_str = f"{p_val:.3f}"

            # Format R
            if pd.isna(r_val):
                r_str = "-"
            else:
                if p_val < 0.01:
                    # Determine color
                    cell_color = ""
                    if r_val >= 0.90:
                        cell_color = "\\cellcolor{corrVeryHigh}"
                    elif r_val >= 0.70:
                        cell_color = "\\cellcolor{corrHigh}"
                    elif r_val >= 0.50:
                        cell_color = "\\cellcolor{corrModerate}"
                    elif r_val >= 0.30:
                        cell_color = "\\cellcolor{corrLow}"

                    r_str = f"{cell_color}\\textbf{{{r_val:.3f}}}"
                else:
                    r_str = f"{r_val:.3f}"

            latex_str += f" & {p_str} & {r_str}"
        latex_str += " \\\\\n"

    latex_str += "\\bottomrule\n\\end{tabular}}\n\\caption{p-values and Multiple Correlation Coefficient (R) for each model and metric.}\n\\end{table}"

    print(latex_str)
else:
    print("No results to display.")




#
#     # Data Rows
#     for index, row in pivot_df.iterrows():
#         escaped_index = index.replace('_', r'\_')
#         latex_str += f"{escaped_index}"
#         for model in present_models:
#             p_val = row[(model, 'p')]
#             r_val = row[(model, 'R')]
#
#             # Check significance threshold
#             if pd.isna(p_val) or p_val >= 0.01:
#                 # If not significant or NaN, show dashes
#                 p_str = "-"
#                 r_str = "-"
#             else:
#                 # Format p (it is < 0.01)
#                 if p_val < 0.001:
#                     p_str = "$<0.001$"
#                 else:
#                     p_str = f"{p_val:.3f}"
#
#                 # Format R with Color
#                 if pd.isna(r_val):
#                     r_str = "-"
#                 else:
#                     # Determine color
#                     cell_color = ""
#                     if r_val >= 0.90:
#                         cell_color = "\\cellcolor{corrVeryHigh}"
#                     elif r_val >= 0.70:
#                         cell_color = "\\cellcolor{corrHigh}"
#                     elif r_val >= 0.50:
#                         cell_color = "\\cellcolor{corrModerate}"
#                     elif r_val >= 0.30:
#                         cell_color = "\\cellcolor{corrLow}"
#
#                     r_str = f"{cell_color}{r_val:.3f}"
#
#             latex_str += f" & {p_str} & {r_str}"
#         latex_str += " \\\\\n"
#
#     latex_str += "\\bottomrule\n\\end{tabular}}\n\\caption{Multiple Correlation Coefficient (R) and p-values (shown only for $p < 0.01$). R values colored by magnitude: \\colorbox{corrLow}{Low}, \\colorbox{corrModerate}{Moderate}, \\colorbox{corrHigh}{High}, \\colorbox{corrVeryHigh}{Very High}.}\n\\end{table}"
#
#     print(latex_str)
# else:
# print("No results.")
