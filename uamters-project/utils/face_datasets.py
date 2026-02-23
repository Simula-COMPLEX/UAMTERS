#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/11 上午11:58
# @Author  : 
# @File    : face_datasets.py
# @Software: PyCharm
import json
import os
import pickle
import statistics

import numpy as np
from scipy.io import loadmat


def get_gt_boxes(gt_dir):
    """gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat,
    wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    # print(hard_mat.keys())
    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    cleaned = [s[0][0].split('--', 1)[1] for s in event_list]
    print(cleaned)
    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    # print(hard_mat.keys())
    length = 0
    for l in easy_mat['gt_list']:
        # print(l[0].shape)
        length += l[0].shape[0]
    # print(length)

    # print(medium_mat['file_list'])
    length = 0
    for l in medium_mat['gt_list']:
        length += l[0].shape[0]
    # print(length)

    length = 0
    for l in hard_mat['gt_list']:
        length += l[0].shape[0]

    # print(length)
    # print(easy_mat['file_list'])
    # print(medium_mat['file_list'])
    #
    # print(hard_mat['file_list'])

    def process_list(gt_list):
        merged_list = []

        for arr in gt_list:
            # Flatten the nested arrays
            for subarr in arr.flatten():
                merged_list.append(subarr[0][0][0])  # subarr is an array of one string
        return merged_list

    easy_list = process_list(easy_mat['file_list'])
    medium_list = process_list(medium_mat['file_list'])
    hard_list = process_list(hard_mat['file_list'])

    return easy_list, medium_list, hard_list


def count_wider_objects(label_file):
    counts = {}  # {image_name: num_objects}
    current_image = None
    current_count = 0

    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()

            # Start of a new image block
            if line.startswith("#"):
                # Save previous block
                if current_image is not None:
                    counts[current_image] = current_count

                # Parse new image path
                # Format: "# image_path width height"
                parts = line[1:].strip().split()
                current_image = parts[0].split('/')[-1].split('.')[0]
                current_count = 0

            # Bounding-box line (starts with a number)
            elif line and line[0].isdigit():
                current_count += 1

        # Save final image block
        if current_image is not None:
            counts[current_image] = current_count

    return counts


def split_images_by_count(label_file):
    counts = count_wider_objects(label_file)  # {image_name: num_objects}

    low_count = []  # images with count <= 3
    high_count = []  # images with count > 3

    for img_name, num_objects in counts.items():
        if num_objects <= 12:
            low_count.append(img_name + '.jpg')
        else:
            high_count.append(img_name + '.jpg')

    with open(f'low_count.pkl', 'wb') as f:
        pickle.dump(low_count, f)

    with open(f'high_count.pkl', 'wb') as f:
        pickle.dump(high_count, f)

    return low_count, high_count


def split_images_by_count2(label_file):
    counts = count_wider_objects(label_file)  # {image_name: num_objects}

    count_1_3 = []  # images with 1-3 objects
    count_3_9 = []  # images with 4-9 objects
    count_9_plus = []  # images with 10 or more objects

    for img_name, num_objects in counts.items():
        img_file = img_name + '.jpg'
        if 1 <= num_objects <= 3:
            count_1_3.append(img_file)
        elif 4 <= num_objects <= 9:
            count_3_9.append(img_file)
        elif num_objects >= 10:
            count_9_plus.append(img_file)

    # Save lists as pickle files
    with open('low_count.pkl', 'wb') as f:
        pickle.dump(count_1_3, f)

    with open('medium_count.pkl', 'wb') as f:
        pickle.dump(count_3_9, f)

    with open('high_count.pkl', 'wb') as f:
        pickle.dump(count_9_plus, f)

    return count_1_3, count_3_9, count_9_plus


# Example usage:
# count_1_3, count_3_9, count_9_plus = split_images_by_count(label_file)


# Example usage:
# label_file = "/path/to/label_file.txt"
# low_count_imgs, high_count_imgs = split_images_by_count(label_file)
# print("Images with <=3 objects:", low_count_imgs)
# print("Images with >3 objects:", high_count_imgs)


labels = '/home/complexse/workspace/RoboSapiens/Social_Perception/libfacedetection.train/data/widerface/labelv2/val' \
         '/labelv2.txt'

# low, medium, high = split_images_by_count2(label_file=labels)
# print(low)
# print(len(low))
# print(high)
# print(len(high))
#
# aaa
count = count_wider_objects(label_file=labels)

# Using statistics module
mean_val = statistics.mean(list(count.values()))
median_val = statistics.median(list(count.values()))
mode_val = None
try:
    mode_val = statistics.mode(list(count.values()))
except statistics.StatisticsError:
    mode_val = "No unique mode"

# Using numpy for additional stats
max_val = np.max(list(count.values()))
min_val = np.min(list(count.values()))
std_val = np.std(list(count.values()))
variance_val = np.var(list(count.values()))
percentiles = {
    '25%': np.percentile(list(count.values()), 25),
    '50%': np.percentile(list(count.values()), 50),
    '75%': np.percentile(list(count.values()), 75)
}

# Combine all stats in a dictionary
stats = {
    'mean': mean_val,
    'median': median_val,
    'mode': mode_val,
    'max': max_val,
    'min': min_val,
    'std': std_val,
    'variance': variance_val,
    'percentiles': percentiles,
    'count': len(list(count.values())),
    'sum': sum(list(count.values()))
}

data_fixed = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
              for k, v in stats.items()}

print(json.dumps(data_fixed, indent=4))

easy_list, medium_list, hard_list = get_gt_boxes(
    gt_dir='/home/complexse/workspace/RoboSapiens/Social_Perception/libfacedetection.train/data/widerface'
           '/labelv2/val/gt')
