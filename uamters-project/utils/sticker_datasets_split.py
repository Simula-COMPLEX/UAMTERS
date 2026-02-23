#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/13 下午1:25
# @Author  : 
# @File    : sticker_datasets_split.py
# @Software: PyCharm
import pickle
from pathlib import Path

model_list = [
    'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2',
    'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2',
    'ssd300_vgg16',
    # 'ssdlite320_mobilenet_v3_large'
]

path = '/home/complexse/workspace/Chengjie/UMATERS/experiment_data/datasets'

for model in model_list:
    filter_datasets = Path(path) / f'common_images_{model}_filtered.pkl'

    with open(filter_datasets, 'rb') as f_n:
        loaded_imgs = pickle.load(f_n)

    dalle = [img.name for img in loaded_imgs if 'dall' in img.name]
    sd = [img.name for img in loaded_imgs if 'stable' in img.name]
    orig = [img.name for img in loaded_imgs if 'dall' not in img.name and 'stable' not in img.name]

    print(len(dalle), len(sd), len(orig))
    with open('dalle.pkl', 'wb') as f:
        pickle.dump(dalle, f)

    with open('sd.pkl', 'wb') as f:
        pickle.dump(sd, f)

    with open('orig.pkl', 'wb') as f:
        pickle.dump(orig, f)
    # print(dalle)
    # print(loaded_imgs)
