#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/10 下午7:59
# @Author  : Chengjie Lu
# @File    : filter.py
# @Software: PyCharm
import json

from tools.common_tc import *
from utils.ms import *

import torchmetrics


# from utils.helper import *


def get_pass_img(test_images, path, models, test_datasets, labels_path, case_study):
    # print(len(test_images))
    filter_dataset = set()
    for uq in uq_ms:
        print(f"Checking UQ method: {uq}...")

        # Define method-specific parameters
        if uq == "mc_dropout":
            mutation_rates = [0]
        else:
            mutation_rates = [(0, 1)]

        for model in models:
            for test_dataset in test_datasets:
                for mutation_rate in mutation_rates:

                    # Format the tag string
                    tag_str = f"{(mutation_rate[0])}_{mutation_rate[1]}" if isinstance(mutation_rate, tuple) else str(
                        mutation_rate)

                    # Optimization: Iterate over a copy (list) of the currently valid images
                    # This automatically skips images that already failed a previous check
                    # from torchmetrics.detection.mean_ap import MeanAveragePrecision
                    # import torch

                    # all_preds, all_gts = {}, {}

                    for test_i in test_images:
                        all_preds = []
                        all_gts = []
                        metric = torchmetrics.detection.MeanAveragePrecision(max_detection_thresholds=[1, 5, 100],
                                                                             iou_thresholds=[0.5])

                        # print(test_i)
                        p_folder = f'{path}/experiment_results_{uq}/{model}/dataset/{test_dataset}/' \
                                   f'{uq}_{tag_str}_{test_i.stem}/prediction_0.json'

                        # print(p_folder)
                        with open(p_folder, "r") as f:
                            preds = json.load(f)

                        # print(test_i)
                        if case_study == 'screw_detection':
                            label_f = labels_path / (Path(test_i).stem + '.txt')
                            yolo_labels = get_labels(label_f)
                            gt_objs = convert_detection_output(yolo_to_absolute(yolo_labels))
                        if case_study == 'sticker_detection':
                            gt_objs = get_coco_objects(image_name=Path(test_i).stem + '.jpg', coco_file=labels_path)

                        # print(preds)
                        # print(gt_objs)
                        pred_boxes = torch.tensor([v["box"] for v in preds.values()], dtype=torch.float32)
                        pred_scores = torch.tensor([v["score"] for v in preds.values()], dtype=torch.float32)
                        pred_labels = torch.tensor([v["label"] for v in preds.values()], dtype=torch.int64)

                        all_preds.append({
                            "boxes": pred_boxes,
                            "scores": pred_scores,
                            "labels": pred_labels
                        })

                        gt_boxes = torch.tensor([v["box"] for v in gt_objs.values()], dtype=torch.float32)
                        gt_labels = torch.tensor([v["label"] for v in gt_objs.values()], dtype=torch.int64)

                        all_gts.append({
                            "boxes": gt_boxes,
                            "labels": gt_labels
                        })

                        # print(all_gts)
                        # print(all_preds)

                        metric.update(all_preds, all_gts)

                        # match, miss, ghost = identify_matches_misses_ghosts(gt_objs, preds)
                        # print(len(match), len(miss), len(ghost))

                        # if len(match) == len(gt_objs):
                        #     filter_dataset.add(test_i)
                        #     print(len(match), len(miss), len(ghost))

                        results = metric.compute()

                        # print("mAP@[0.5:0.95]:", results["map"])
                        # print("mAP@0.50:", results["map_50"])
                        if results['map_50'] == 1:
                            filter_dataset.add(test_i)
                        # print("mAP@0.75:", results["map_75"])
                        # print("Per-class:", results["map_per_class"])
    print(filter_dataset)
    # print(len(filter_dataset))

    # with open(f'common_images_{case_study}_filtered.pkl', 'wb') as f:
    #     pickle.dump(list(filter_dataset), f)

    print(f"Successfully saved {len(filter_dataset)} images to f'common_images_{case_study}_filtered.pkl'")


from pycocotools.coco import COCO


def get_coco_objects(image_name, coco_file):
    """
    Returns a dict containing bounding boxes and labels for a given image
    in the COCO annotation file.
    """
    coco = COCO(coco_file)

    # -------- Find image ID manually (COCO has no file_name search) --------
    img_id = None
    for img in coco.dataset["images"]:
        if img["file_name"] == image_name:
            img_id = img["id"]
            break

    if img_id is None:
        raise ValueError(f"Image {image_name} not found in annotations")

    # -------- Load annotations for this image --------
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)

    results = {}

    for i, ann in enumerate(anns):
        x, y, w, h = ann["bbox"]
        box_xyxy = [x, y, x + w, y + h]  # convert COCO format → xyxy

        label_id = ann["category_id"]  # integer label

        # Build output entry
        results[f"label_{i}"] = {
            "box": box_xyxy,
            "label": label_id,
        }

    return results


def iou_xyxy(box1, box2):
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def get_labels(label_fn):
    yolo_labels_l = []

    with open(label_fn, "r") as f:
        for line in f:
            # Remove leading/trailing whitespace and split by spaces
            parts = line.strip().split()
            if parts:  # skip empty lines
                # Convert each part to float, except the first which is int (class id)
                label = (int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                yolo_labels_l.append(label)
    return yolo_labels_l


def get_success_from_csv(exper_result, model):
    common_imgs_fn = f'/home/complexse/workspace/Chengjie/UAMTERS/experiment_data/datasets/common_images_{model}.pkl'
    with open(common_imgs_fn, 'rb') as f_n:
        loaded_imgs = pickle.load(f_n)

    print(len(loaded_imgs))

    csv_fn = Path(exper_result) / f'experiment_results_mc_dropblock/{model}/logs_0_1_origimg-orig.csv'
    df = pd.read_csv(filepath_or_buffer=csv_fn, sep=',')

    # print(len(df))
    # Filter image_name where MAP == 1
    image_names = df.loc[df['MAP'] == 1, 'image_name'].tolist()
    image_names = [i_name.split('/')[-1] for i_name in image_names]

    filtered_images = []
    for l_image in loaded_imgs:
        if l_image.name in image_names:
            filtered_images.append(l_image)
    print(filtered_images)
    # print(len(image_names))
    #
    # print(len(filtered_images))

    with open(f'common_images_{model}_filtered.pkl', 'wb') as f:
        pickle.dump(list(filtered_images), f)

    print(f"Successfully saved {len(filtered_images)} images to f'common_images_{model}_filtered.pkl'")


#
# screw_orig_label = Path('/home/complexse/workspace/RoboSapiens/Screw_Detection/screw_detection/uq_evaluation/origimg2'
#                         '/orig/labels')
# screw_dataset_path = '/home/complexse/workspace/Chengjie/UAMTERS/tools' \
#                      '/common_images_best_added_more_noscrews_at_diff_exposure.pkl'
# screw_results = '/home/complexse/workspace/RoboSapiens/Screw_Detection/screw_detection/experiment_results'
#
# with open(screw_dataset_path, 'rb') as f_n:
#     loaded_imgs = pickle.load(f_n)
#
# get_pass_img(loaded_imgs, path=screw_results, models=['best_added_more_noscrews_at_diff_exposure'],
#              test_datasets=['orig'], labels_path=screw_orig_label, case_study='screw_detection')

#
sticker = '/home/complexse/workspace/RoboSapiens/DTI-Laptop-refubishment/experiment_results/'
for stdm in sticker_models:
    get_success_from_csv(exper_result=sticker, model=stdm)


# face = '/home/complexse/workspace/RoboSapiens/Social_Perception/experiment_results'
# for fdm in face_models:
#     get_success_from_csv(exper_result=face, model=fdm)
