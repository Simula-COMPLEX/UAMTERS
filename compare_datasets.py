#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/12 下午2:29
# @Author  : 
# @File    : compare_datasets.py
# @Software: PyCharm
import pickle
import random
from pathlib import Path

import pandas as pd

case_study = {
    'screw_detection':
        {
            'dataset': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets',
            'test_suites': {
                'normal': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_screw/easy.pkl',
                'different_exposure': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_screw/hard.pkl'},
            'case_study_name': 'screw_detection',
            'models': ['best_added_more_noscrews_at_diff_exposure']
        },
    'sticker_detection':
        {
            'dataset': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets',
            'test_suites': {
                # 'easy': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_face/easy.pkl',
                # 'meduim': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_face/medium.pkl',
                # 'hard': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_face/hard.pkl'
                'orig': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_sticker/orig.pkl',
                'dalle': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_sticker/dalle.pkl',
                'sd': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_sticker/sd.pkl'
            },
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
            'dataset': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets',
            'test_suites': {
                # 'easy': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_face/easy.pkl',
                # 'meduim': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_face/medium.pkl',
                # 'hard': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_face/hard.pkl'
                'low': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_face/low_count.pkl',
                'meduim': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_face/medium_count.pkl',
                'high': '/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/test_suites_face/high_count.pkl'
            },
            'case_study_name': 'social_perception',
            'models': ['yunet_s', 'yunet_n']
        }
}

d_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
b_sizes = [1, 3, 5, 7, 9]


def sample_for_one_model(mutation_operator, model, test_suite, case_study_name, ts, ua_ms_fn, save_folder, ms_results):
    original_path = Path(f'/home/complexse/workspace/Chengjie/UAMTERS/{ms_results}')
    new_path = Path(
        f'./{save_folder}/{case_study_name}/{model}/{ts}/{mutation_operator}')

    new_path.mkdir(parents=True, exist_ok=True)
    if mutation_operator == "mc_dropout":
        mutation_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    else:
        mutation_rates = [(d, b) for d in d_rates for b in b_sizes]

    mutants_info_dict = {}
    for mutation_rate in mutation_rates:
        # calculate mutation scores for each mutant
        tag_str = f"{(mutation_rate[0])}_{mutation_rate[1]}" if isinstance(mutation_rate, tuple) else str(
            mutation_rate)

        csv_fn = original_path / f'{case_study_name}/{model}/{mutation_operator}/mutant_{tag_str}.csv'
        data = pd.read_csv(filepath_or_buffer=csv_fn, sep=',')
        # [img_level] iskill_miss_ghost_count

        df_subset = data[data['test_suite'].isin(test_suite)]

        img_miss_col = '[img_level] iskill_miss_count'
        img_ghost_col = '[img_level] iskill_ghost_count'
        img_miss_ghost_col = '[img_level] iskill_miss_ghost_count'
        # True if the entire column is all zeros

        is_kill_miss = (df_subset[img_miss_col] != 0).any()
        is_kill_ghost = (df_subset[img_ghost_col] != 0).any()
        is_kill_miss_ghost = (df_subset[img_miss_ghost_col] != 0).any()

        # is_equivalent = (df_subset[img_miss_ghost_col] == 0).all()
        is_equivalent = False

        means_dict = df_subset.iloc[:, 13:].mean().to_dict()

        # print(means_dict)

        # print(df_subset)
        new_csv_fn = new_path / f'mutant_{tag_str}.csv'

        mutant_id = f'mutant_{mutation_operator}_{tag_str}'
        mutants_info_dict.update({
            'mutant_id': mutant_id,
            'equivalent_mutant': is_equivalent,
            'iskill_miss': is_kill_miss,
            'iskill_ghost': is_kill_ghost,
            'iskill_miss_ghost': is_kill_miss_ghost
        })

        mutants_info_dict.update(means_dict)

        df = pd.DataFrame([mutants_info_dict])

        df.to_csv(ua_ms_fn, index=False, header=True, mode='w') if mutant_id == 'mutant_mc_dropout_0.1' else \
            df.to_csv(ua_ms_fn, index=False, header=False, mode='a')

        # exit(0)
        df_subset.to_csv(new_csv_fn, index=False, header=True)


def calcu_mean_ms(df, ts):
    ms_miss = df['iskill_miss'].sum() / (
            len(df['mutant_id']) - df['equivalent_mutant'].sum())
    ms_ghost = df['iskill_ghost'].sum() / (
            len(df['mutant_id']) - df['equivalent_mutant'].sum())
    ms_miss_ghost = df['iskill_miss_ghost'].sum() / (
            len(df['mutant_id']) - df['equivalent_mutant'].sum())

    # Filter rows
    filtered_data = df[df['equivalent_mutant'] == False]

    # Compute column means from column index 5 onward
    means_dict = filtered_data.iloc[:, 5:].mean().to_dict()

    # print(means_dict.values())
    ms_dict = {
        'test_suite': ts,
        '[img_level] ms_miss': ms_miss,
        '[img_level] ms_ghost': ms_ghost,
        '[img_level] ms_miss_ghost': ms_miss_ghost}

    ms_dict.update(means_dict)
    return ms_dict


# test_suite,[img_level] ms_miss,[img_level] ms_ghost,[img_level] ms_miss_ghost,
# [obj_level] kill_rate_miss_mean,[obj_level] kill_rate_ghost_mean,[obj_level] kill_rate_miss_ghost_mean,
# [match] iou_mean,
# [match] vr_diff,[match] ie_diff,[match] mi_diff,[match] var_diff,[match] ps_diff,
# [miss] vr_diff,[miss] ie_diff,[miss] mi_diff,[miss] var_diff,[miss] ps_diff,
# [ghost] vr_diff,[ghost] ie_diff,[ghost] mi_diff,[ghost] var_diff,[ghost] ps_diff


def calcu_ms_model(case_study_name, save_folder):
    models = case_study[case_study_name]['models']

    for model in models:
        ms_model_all_fn = Path(
            f'./{save_folder}/{case_study_name}/{model}')

        for i in range(len(case_study[case_study_name]['test_suites'].keys())):
            ts = list(case_study[case_study_name]['test_suites'].keys())[i]
            ms_model_fn = Path(
                f'./{save_folder}/{case_study_name}/{model}/{ts}/ua_ms.csv')
            # print(ms_model_fn)

            ms_data = pd.read_csv(filepath_or_buffer=ms_model_fn, sep=',')

            ms_dict = calcu_mean_ms(ms_data, ts)

            ms_dict_dropout = calcu_mean_ms(ms_data.head(9), ts)
            ms_dict_dropblock = calcu_mean_ms(ms_data.tail(45), ts)

            df_dropout = pd.DataFrame([ms_dict_dropout])
            df_dropblock = pd.DataFrame([ms_dict_dropblock])
            df_all = pd.DataFrame([ms_dict])

            df_dropout.to_csv(ms_model_all_fn / 'ua_ms_model_dropout.csv', index=False, header=True, mode='w') if i == 0 \
                else \
                df_dropout.to_csv(ms_model_all_fn / 'ua_ms_model_dropout.csv', index=False, header=False, mode='a')

            df_dropblock.to_csv(ms_model_all_fn / 'ua_ms_model_dropblock.csv', index=False, header=True,
                                mode='w') if i == 0 \
                else \
                df_dropblock.to_csv(ms_model_all_fn / 'ua_ms_model_dropblock.csv', index=False, header=False, mode='a')

            df_all.to_csv(ms_model_all_fn / 'ua_ms_model.csv', index=False, header=True, mode='w') if i == 0 \
                else df_all.to_csv(ms_model_all_fn / 'ua_ms_model.csv', index=False, header=False, mode='a')


def compare_ts(case_study_name, save_folder, ms_results):
    models = case_study[case_study_name]['models']
    test_suites = case_study[case_study_name]['test_suites']

    for model in models:
        dataset_path = Path(case_study[case_study_name]['dataset']) / f'common_images_{model}_filtered.pkl'
        with open(dataset_path, 'rb') as f_n:
            loaded_imgs = pickle.load(f_n)

        for ts in test_suites.keys():
            with open(test_suites[ts], 'rb') as f_n:
                ts_imgs = pickle.load(f_n)

            print(ts_imgs[0])
            print(loaded_imgs[0])
            # print('---------')
            ms_model_fn = Path(
                f'./{save_folder}/{case_study_name}/{model}/{ts}/ua_ms.csv')
            ts_imgs_common = [load_img.stem for load_img in loaded_imgs if load_img.name in ts_imgs]

            # Calculate 10% of the list size
            sample_size = max(1, int(len(ts_imgs_common)))  # at least 1 element

            # Randomly sample 10%
            ts_imgs_common = random.sample(ts_imgs_common, sample_size)

            print(ts_imgs_common)
            sample_for_one_model(mutation_operator='mc_dropout', model=model, test_suite=ts_imgs_common,
                                 case_study_name=case_study_name, ts=ts, ua_ms_fn=ms_model_fn,
                                 save_folder=save_folder, ms_results=ms_results)

            sample_for_one_model(mutation_operator='mc_dropblock', model=model, test_suite=ts_imgs_common,
                                 case_study_name=case_study_name, ts=ts, ua_ms_fn=ms_model_fn,
                                 save_folder=save_folder, ms_results=ms_results)


compare_ts(case_study_name='screw_detection', save_folder='ms_results_formal_experiments_1', ms_results='ms_results_1')
calcu_ms_model(case_study_name='screw_detection', save_folder='ms_results_formal_experiments_1')

compare_ts(case_study_name='sticker_detection', save_folder='ms_results_formal_experiments_1',
           ms_results='ms_results_1')
calcu_ms_model(case_study_name='sticker_detection', save_folder='ms_results_formal_experiments_1')

compare_ts(case_study_name='social_perception', save_folder='ms_results_formal_experiments_1',
           ms_results='ms_results_1')
calcu_ms_model(case_study_name='social_perception', save_folder='ms_results_formal_experiments_1')
