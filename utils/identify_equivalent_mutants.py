#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/11 上午11:28
# @Author  : Chengjie Lu
# @File    : identify_equivalent_mutants.py
# @Software: PyCharm
from pathlib import Path

import pandas as pd

case_study = {
    'screw_detection':
        {
            'case_study_name': 'screw_detection',
            'models': ['best_added_more_noscrews_at_diff_exposure']
        },
    'sticker_detection':
        {
            'case_study_name': 'sticker_detection',
            'models': [
                'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2',
                'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2',
                'ssd300_vgg16',
                # 'ssdlite320_mobilenet_v3_large'
            ]
        },
    'social_perception':
        {
            'case_study_name': 'social_perception',
            'models': ['yunet_s', 'yunet_n']
        }
}

d_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
b_sizes = [1, 3, 5, 7, 9]


def identify_for_one_mutant(mutation_operator, model, case_study_name):
    top_path = Path('/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/ms_results')
    if mutation_operator == "mc_dropout":
        mutation_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    else:
        mutation_rates = [(d, b) for d in d_rates for b in b_sizes]

    for mutation_rate in mutation_rates:
        # calculate mutation scores for each mutant
        tag_str = f"{(mutation_rate[0])}_{mutation_rate[1]}" if isinstance(mutation_rate, tuple) else str(
            mutation_rate)

        csv_fn = top_path / f'{case_study_name}/{model}/{mutation_operator}/mutant_{tag_str}.csv'
        data = pd.read_csv(filepath_or_buffer=csv_fn, sep=',')
        # [img_level] iskill_miss_ghost_count
        col = '[img_level] iskill_miss_ghost_count'

        # True if the entire column is all zeros
        is_kill = (data[col] == 0).all()

        if is_kill:
            print(csv_fn)
            print(is_kill)
        # return is_kill


def identify_for_case_study(case_study_name):
    models = case_study[case_study_name]['models']

    for model in models:
        identify_for_one_mutant(mutation_operator='mc_dropout', model=model, case_study_name=case_study_name)
        identify_for_one_mutant(mutation_operator='mc_dropblock', model=model, case_study_name=case_study_name)


if __name__ == '__main__':
    identify_for_case_study(case_study_name='social_perception')
