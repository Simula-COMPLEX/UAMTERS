#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/9 下午1:03
# @Author  : Chengjie Lu
# @File    : rename.py
# @Software: PyCharm

import os

dropout_list = ['0.0', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5']
block_size_list = ['1.0', '3.0', '5.0', '7.0', '9.0']

# adv_list = [
#     'orig', 'adv_run_0', 'adv_run_1', 'adv_run_2', 'adv_run_3', 'adv_run_4',
#     'adv_run_5', 'adv_run_6', 'adv_run_7', 'adv_run_8', 'adv_run_9'
# ]


adv_list = [
    'orig'
]


def rename_dropblock(model_n, p):
    # Ensure this path is correct relative to where you run the script
    # p = '/home/complexse/workspace/RoboSapiens/Social_Perception/libfacedetection.train/experiment_results'
    base_dir = f"{p}/experiment_results_mc_dropblock/{model_n}"

    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    print(f"Processing directory: {base_dir}")

    for d in dropout_list:
        for b in block_size_list:

            # Convert '1.0' (str) -> 1.0 (float) -> 1 (int)
            d_int = '0' if d == '0.0' else d
            b_int = int(float(b))

            # Construct the prefixes we are looking for and replacing
            # Old: mc_dropblock_0.5_9.0_...
            prefix_old = f"mc_dropblock_{d}_{b_int}_"
            # New: mc_dropblock_0.5_9_...
            prefix_new = f"mc_dropblock_{d_int}_{b_int}_"

            for a in adv_list:
                # print(f"Processing dataset: {d}, {b}, {a}")

                # --- Part 1: Rename the CSV files ---
                old_fname = f"logs_{d}_{b_int}_sdimg-org.csv"
                new_fname = f"logs_{d_int}_{b_int}_sdimg-{a}.csv"

                old_path = os.path.join(base_dir, old_fname)
                new_path = os.path.join(base_dir, new_fname)

                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                    print(f"Renamed File: {old_fname} -> {new_fname}")
                elif os.path.exists(new_path):
                    # print(f"Skipping File: {new_fname} already exists.")
                    pass
                else:
                    # print(f"File not found: {old_fname}")
                    pass

                # --- Part 2: Rename the Subfolders (Completed Logic) ---
                folder_n = os.path.join(base_dir, f'dataset/{a}')

                print(folder_n)
                if os.path.exists(folder_n):
                    # Iterate through all actual items in the specific 'dataset/a' folder
                    # We use os.listdir to scan the directory content
                    for item_name in os.listdir(folder_n):

                        print(item_name)
                        # Check if this specific folder matches the current loop's d and b
                        if item_name.startswith(prefix_old):

                            # Create the new name by swapping the prefix
                            # .replace(old, new, 1) ensures we only replace the first occurrence
                            new_item_name = item_name.replace(prefix_old, prefix_new, 1)

                            if new_item_name.endswith(".jpg"):
                                new_item_name = new_item_name[:-4]  # strip last 4 chars

                            # print(item_name, new_item_name)
                            old_item_path = os.path.join(folder_n, item_name)
                            new_item_path = os.path.join(folder_n, new_item_name)

                            # Rename the folder
                            # We check new_item_path to avoid overwriting or crashes if run twice
                            if not os.path.exists(new_item_path):
                                os.rename(old_item_path, new_item_path)
                                # print(f"Renamed Folder in {a}: \n   {item_name} \n-> {new_item_name}")
                            else:
                                # print(f"Skipping Folder: {new_item_name} already exists.")
                                pass


def rename_dropout(model_n, p):
    # p = '/home/complexse/workspace/RoboSapiens/Social_Perception//experiment_results'
    base_dir = f"{p}/experiment_results_mc_dropout/{model_n}"

    for d in dropout_list:
        d_int = '0' if d == '0.0' else d

        prefix_old = f"mc_dropout_{d}_"
        # New: mc_dropblock_0.5_9_...
        prefix_new = f"mc_dropout_{d_int}_"

        for a in adv_list:
            print(f"Processing dataset: {d}, {a}")

            # --- Part 1: Rename the CSV files ---
            old_fname = f"logs_{d}_dalleimg-{a}.csv"
            new_fname = f"logs_{d_int}_dalleimg-{a}.csv"

            old_path = os.path.join(base_dir, old_fname)
            new_path = os.path.join(base_dir, new_fname)

            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                print(f"Renamed File: {old_fname} -> {new_fname}")
            elif os.path.exists(new_path):
                # print(f"Skipping File: {new_fname} already exists.")
                pass
            else:
                # print(f"File not found: {old_fname}")
                pass

            folder_n = os.path.join(base_dir, f'dataset/{a}')

            # Iterate through all actual items in the specific 'dataset/a' folder
            # We use os.listdir to scan the directory content
            for item_name in os.listdir(folder_n):

                # print(item_name)
                # Check if this specific folder matches the current loop's d and b
                if item_name.startswith(prefix_old):

                    # Create the new name by swapping the prefix
                    # .replace(old, new, 1) ensures we only replace the first occurrence
                    new_item_name = item_name.replace(prefix_old, prefix_new, 1)

                    if new_item_name.endswith(".jpg"):
                        new_item_name = new_item_name[:-4]  # strip last 4 chars

                    # print(item_name, new_item_name)
                    old_item_path = os.path.join(folder_n, item_name)
                    new_item_path = os.path.join(folder_n, new_item_name)

                    # print(item_name, new_item_name)
                    # Rename the folder
                    # We check new_item_path to avoid overwriting or crashes if run twice
                    if not os.path.exists(new_item_path):
                        os.rename(old_item_path, new_item_path)
                        # print(f"Renamed Folder in {a}: \n   {item_name} \n-> {new_item_name}")
                    else:
                        # print(f"Skipping Folder: {new_item_name} already exists.")
                        pass


if __name__ == '__main__':
    path = '/home/complexse/workspace/RoboSapiens/DTI-Laptop-refubishment/experiment_results'
    model_list = [
        'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2',
        'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2',
        'ssd300_vgg16',
        # 'ssdlite320_mobilenet_v3_large'
    ]

    # rename_dropblock(model_n='yunet_s')
    for model in model_list:
        # rename_dropblock(model_n=model, p=path)
        rename_dropout(model_n=model, p=path)