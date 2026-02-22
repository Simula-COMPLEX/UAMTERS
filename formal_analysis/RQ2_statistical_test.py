#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/29 下午2:41
# @Author  : 
# @File    : statistical_test.py
# @Software: PyCharm
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal

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

metrics = [
    "[obj_level] kill_rate_miss_mean",
    "[obj_level] kill_rate_ghost_mean",
    "[obj_level] kill_rate_miss_ghost_mean",
    "[match] iou_mean",
    "[match] vr_diff",
    "[match] ie_diff",
    "[match] mi_diff",
    "[match] var_diff",
    "[match] ps_diff",
    "[miss] vr_diff",
    "[miss] ie_diff",
    "[miss] mi_diff",
    "[miss] var_diff",
    "[miss] ps_diff",
    "[ghost] vr_diff",
    "[ghost] ie_diff",
    "[ghost] mi_diff",
    "[ghost] var_diff",
    "[ghost] ps_diff"
]


# A12 effect size
def a12(lst1, lst2):
    n1 = len(lst1)
    n2 = len(lst2)
    more = sum(x > y for x in lst1 for y in lst2)
    equal = sum(x == y for x in lst1 for y in lst2)
    return (more + 0.5 * equal) / (n1 * n2)


def statistical_test(case_name):
    test_suites = list(case_study[case_name]['test_suites'].keys())
    models = list(case_study[case_name]['models'].keys())

    results_dict = {}

    for metric in metrics:
        results_dict[metric] = {}  # metric level
        for model in models:
            results_dict[metric][model] = {}  # model level

            # Load all test suite data for this model and metric
            test_suite_data = {}
            for test_suite in test_suites:
                f_path = (
                    f"./ms_results_formal_experiments_1/"
                    f"{case_name}/{model}/{test_suite}/ua_ms.csv"
                )
                data = pd.read_csv(f_path)
                test_suite_data[test_suite] = data[metric].values

            # Pairwise comparison of test suites
            for i in range(len(test_suites)):
                for j in range(i + 1, len(test_suites)):
                    ts1 = test_suites[i]
                    ts2 = test_suites[j]
                    u_stat, p_value = mannwhitneyu(
                        test_suite_data[ts1], test_suite_data[ts2], alternative='two-sided'
                    )
                    effect_size = a12(test_suite_data[ts1], test_suite_data[ts2])

                    # Round to 3 digits
                    p_value = round(p_value, 3)
                    effect_size = round(effect_size, 3)

                    # Store results in nested dict
                    if ts1 not in results_dict[metric][model]:
                        results_dict[metric][model][ts1] = {}
                    results_dict[metric][model][ts1][ts2] = {
                        'p_value': p_value,
                        'A12': effect_size
                    }

    return results_dict


def save_results_to_csv(results_dict, case_name, output_path):
    rows = []
    for metric, metric_data in results_dict.items():
        for model, model_data in metric_data.items():
            for ts1, ts1_data in model_data.items():
                for ts2, values in ts1_data.items():
                    rows.append({
                        'case': case_name,
                        'metric': metric,
                        'model': model,
                        'test_suite_1': ts1,
                        'test_suite_2': ts2,
                        'p_value': values['p_value'],
                        'A12': values['A12']
                    })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def kruskal_global_test(test_suite_data):
    """
    test_suite_data: dict[test_suite] -> array of metric values
    """
    samples = list(test_suite_data.values())

    H, p_value = kruskal(*samples)

    k = len(samples)  # number of test suites
    n = sum(len(s) for s in samples)  # total number of samples

    # Eta-squared effect size for Kruskal–Wallis
    eta_squared = (H - k + 1) / (n - k)
    eta_squared = max(0.0, eta_squared)  # avoid negative values

    return {
        'H': round(H, 3),
        'p_value': round(p_value, 3),
        'eta_squared': round(eta_squared, 3)
    }


def interpret_eta_squared(eta_squared):
    if eta_squared < 0.01:
        return "negligible"
    elif eta_squared < 0.06:
        return "small"
    elif eta_squared < 0.14:
        return "medium"
    else:
        return "large"


def statistical_test_global(case_name):
    test_suites = list(case_study[case_name]['test_suites'].keys())
    models = list(case_study[case_name]['models'].keys())

    results = {}

    for metric in metrics:
        results[metric] = {}
        for model in models:
            test_suite_data = {}

            # Load all samples for this metric
            for test_suite in test_suites:
                f_path = (
                    f"./ms_results_formal_experiments_1/"
                    f"{case_name}/{model}/{test_suite}/ua_ms.csv"
                )
                data = pd.read_csv(f_path)
                test_suite_data[test_suite] = data[metric].dropna().values

            # Global test across ALL test suites
            results[metric][model] = kruskal_global_test(test_suite_data)

    return results


def statistical_test_global_merge(case_name):
    test_suites = list(case_study[case_name]['test_suites'].keys())
    models = list(case_study[case_name]['models'].keys())

    results = {}

    for metric in metrics:
        results[metric] = {}
        for model in models:
            test_suite_data = {}

            # Load all samples for this metric
            for test_suite in test_suites:
                dfs = []  # list to store each CSV for this test suite
                for d_r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                    f_path = (
                        f"./ms_results_formal_experiments_1/"
                        f"{case_name}/{model}/{test_suite}/mc_dropout/mutant_{d_r}.csv"
                    )
                    data = pd.read_csv(f_path)
                    # data['d_r'] = d_r  # optionally keep track of which b_s this row came from
                    dfs.append(data)

                for d_r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                    for b_s in [1, 3, 5, 7, 9]:
                        f_path = (
                            f"./ms_results_formal_experiments_1/"
                            f"{case_name}/{model}/{test_suite}/mc_dropblock/mutant_{d_r}_{b_s}.csv"
                        )
                        data = pd.read_csv(f_path)
                        # data['d_r'] = d_r  # optionally keep track of which b_s this row came from
                        dfs.append(data)
                test_suite_data[test_suite] = pd.concat(dfs, ignore_index=True)[metric].dropna().values

            # Global test across ALL test suites
            results[metric][model] = kruskal_global_test(test_suite_data)

    return results


def save_global_results_to_csv(results, case_name, output_path):
    rows = []

    for metric, metric_data in results.items():
        for model, stats in metric_data.items():
            rows.append({
                'case': case_name,
                'model': model,
                'metric': metric,
                'H': stats['H'],
                'p_value': stats['p_value'],
                'eta_squared': stats['eta_squared'],
                'effect_size': interpret_eta_squared(stats['eta_squared'])
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Global test results saved to {output_path}")


# r = statistical_test_global('sticker_detection')
r = statistical_test_global_merge('social_perception')
save_global_results_to_csv(r, 'social_perception', './RQ2/social_perception_global_merge.csv')

print(r)
