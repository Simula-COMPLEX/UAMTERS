#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/2 下午3:14
# @Author  : Chengjie Lu
# @File    : mutation_score.py
# @Software: PyCharm
from pathlib import Path

import numpy as np
import json
from scipy.optimize import linear_sum_assignment


# ==========================================
# 1. Helper Functions (Math & Geometry)
# ==========================================

def calculate_uncertainty(logits):
    """
    Calculates normalized entropy (u).
    Returns value between 0.0 (Certain) and 1.0 (Max Uncertainty).
    """
    probs = np.array(logits)
    if probs.sum() > 0:
        probs = probs / probs.sum()  # Normalize
    else:
        return 1.0

        # Entropy = -sum(p * log(p))
    entropy = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))

    # Normalize by log(num_classes)
    if len(probs) > 1:
        max_entropy = np.log(len(probs))
        return entropy / max_entropy
    return 0.0


def compute_iou(boxA, boxB):
    """Standard IoU calculation."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


# ==========================================
# 2. The Matching Logic (Bipartite + Threshold)
# ==========================================

def identify_matches_misses_ghosts(orig_objs, mut_objs, iou_threshold=0.5):
    """
    Performs Hungarian Matching and filters by IoU threshold.
    """
    n_org = len(orig_objs)
    n_mut = len(mut_objs)

    matches = []
    misses = []
    ghosts = []

    # Edge cases
    if n_org == 0: return [], [], mut_objs
    if n_mut == 0: return [], orig_objs, []

    # Build Cost Matrix (Cost = 1 - IoU)
    cost_matrix = np.ones((n_org, n_mut))
    iou_matrix = np.zeros((n_org, n_mut))

    for i, o in enumerate(orig_objs):
        for j, m in enumerate(mut_objs):
            iou = compute_iou(o['box'], m['box'])
            iou_matrix[i, j] = iou
            cost_matrix[i, j] = 1.0 - iou

    # Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_org_indices = set()
    matched_mut_indices = set()

    # Filter by Threshold
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] > iou_threshold:
            matches.append((orig_objs[r], mut_objs[c]))
            matched_org_indices.add(r)
            matched_mut_indices.add(c)

    # Identify Leftovers
    for i in range(n_org):
        if i not in matched_org_indices: misses.append(orig_objs[i])

    for j in range(n_mut):
        if j not in matched_mut_indices: ghosts.append(mut_objs[j])

    return matches, misses, ghosts


# ==========================================
# 3. The U-ODDS Calculator (The Core Logic)
# ==========================================

def calculate_u_odds(orig_data, mut_data,
                     alpha=1.0,  # Weight: Class Label Change
                     beta=1.0,  # Weight: Localization Shift
                     lam=1.0,  # Weight: Uncertainty Divergence (Lambda)
                     gamma_miss=1.0,  # Weight: Missing Object
                     gamma_ghost=1.0  # Weight: Hallucinating Object
                     ):
    # 1. Parse and Inject Uncertainty
    orig_objs = list(orig_data.values())
    mut_objs = list(mut_data.values())

    for obj in orig_objs: obj['u_val'] = calculate_uncertainty(obj['logit'])
    for obj in mut_objs:  obj['u_val'] = calculate_uncertainty(obj['logit'])

    # 2. Get Sets: Matches, Misses, Ghosts
    matches, misses, ghosts = identify_matches_misses_ghosts(orig_objs, mut_objs, iou_threshold=0.5)

    # 3. Calculate Components
    sum_cost_matches = 0.0
    sum_cost_misses = 0.0
    sum_cost_ghosts = 0.0

    # --- A. Matched Pair Cost (Stability) ---
    # Cost = alpha * ClassDiff + beta * LocDiff + lambda * |u - u'|
    for (o, m) in matches:
        # Classification
        c_class = 1.0 if o['label'] != m['label'] else 0.0

        # Localization (1 - IoU)
        iou = compute_iou(o['box'], m['box'])
        c_loc = 1.0 - iou

        # Uncertainty Divergence (|u - u'|)
        c_unc = abs(o['u_val'] - m['u_val'])

        pair_cost = (alpha * c_class) + (beta * c_loc) + (lam * c_unc)
        sum_cost_matches += pair_cost

    # --- B. Miss Cost (Contextual Severity) ---
    # Cost = gamma * (1 - u_org)
    # Punish missing confident anchors heavily.
    for o in misses:
        c_miss = gamma_miss * (1.0 - o['u_val'])
        sum_cost_misses += c_miss

    # --- C. Ghost Cost (Hallucination Severity) ---
    # Cost = gamma * (1 - u_mut)
    # Punish confident hallucinations heavily.
    for m in ghosts:
        c_ghost = gamma_ghost * (1.0 - m['u_val'])
        sum_cost_ghosts += c_ghost

    # 4. Normalization
    # Sum of original confidences (1 - u_org)
    complexity_sum = sum([(1.0 - o['u_val']) for o in orig_objs])
    epsilon = 1e-6

    numerator = sum_cost_matches + sum_cost_misses + sum_cost_ghosts
    denominator = complexity_sum + epsilon

    u_odds_score = numerator / denominator

    # Return details for debugging
    return {
        "score": u_odds_score,
        "details": {
            "matches": matches,
            "misses": misses,
            "ghosts": ghosts,
            "cost_match_term": sum_cost_matches,
            "cost_miss_term": sum_cost_misses,
            "cost_ghost_term": sum_cost_ghosts,
            "denominator": denominator
        }
    }


def run_evaluation():
    test_path = Path('../origimg2/{}'.format('orig'))
    test_images_path = test_path / 'images'
    # test_labels_path = test_path / 'labels'
    test_images = list(test_images_path.glob('*.png')) + list(test_images_path.glob('*.jpg'))
    print(test_images)

    for test_i in test_images:
        file_path = test_i.stem
        orig_path = f"../../experiment_results/experiment_results_mc_dropout/best_initial/dataset/orig/mc_dropout_0_{file_path}/prediction_0.json"
        mut_path = f"../../experiment_results/experiment_results_mc_dropout/best_initial/dataset/orig/mc_dropout_0.5_{file_path}/prediction_0.json"

        with open(orig_path, 'r') as f:
            orig_json = json.load(f)
        with open(mut_path, 'r') as f:
            mut_json = json.load(f)

        result = calculate_u_odds(orig_json, mut_json, alpha=1.0, beta=1.0, lam=1.0)

        print("-" * 30)
        print(f"Final U-MS Score: {result['score']:.4f}")
        print("\n--- Breakdown ---")
        print(f"Matched Pairs: {result['details']['matches']} (Cost: {result['details']['cost_match_term']:.2f})")
        print(f"Missed Objects: {result['details']['misses']} (Cost: {result['details']['cost_miss_term']:.2f})")
        print(f"Ghosts Found:   {result['details']['ghosts']} (Cost: {result['details']['cost_ghost_term']:.2f})")
        # print(f"Denominator:    {result['details']['denominator']:.2f}")
        print("-" * 30)


# ==========================================
# 4. Run Demonstration
# ==========================================

if __name__ == "__main__":
    run_evaluation()
