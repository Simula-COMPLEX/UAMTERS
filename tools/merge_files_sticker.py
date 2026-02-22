#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/13 下午12:51
# @Author  : 
# @File    : merge_files_sticker.py
# @Software: PyCharm


import os
import pandas as pd


path = '/home/complexse/workspace/RoboSapiens/DTI-Laptop-refubishment/experiment_results'
model_list = [
    'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2',
    'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2',
    'ssd300_vgg16',
    # 'ssdlite320_mobilenet_v3_large'
]


def merge_files(model_n, p):
    base_dir = f"{p}/experiment_results_mc_dropout/{model_n}"

    orig_fn = f"{base_dir}/logs_0_origimg-org.csv"
    dalle_fn = f"{base_dir}/logs_0.0_dalleimg-org.csv"
    sd_fn = f"{base_dir}/logs_0.0_sdimg-org.csv"

    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    print(f"Processing directory: {base_dir}")

    dfs = []
    for fn in [orig_fn, dalle_fn, sd_fn]:
        if os.path.exists(fn):
            df = pd.read_csv(fn)
            df = df.iloc[:-1]  # ✅ remove last row
            dfs.append(df)
        else:
            print(f"File not found, skipping: {fn}")

    if not dfs:
        print("No CSV files to merge.")
        return

    merged_df = pd.concat(dfs, ignore_index=True)

    out_fn = orig_fn
    merged_df.to_csv(out_fn, index=False)

    print(f"Merged CSV saved to: {out_fn}")

    return merged_df


for model in model_list:
    merge_files(model, path)
