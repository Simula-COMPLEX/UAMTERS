#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/7 下午1:46
# @Author  : Chengjie Lu
# @File    : check.py
# @Software: PyCharm

from pathlib import Path
import pickle

uq_ms = ['mc_dropout', 'mc_dropblock']
d_rates = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
b_sizes = [1, 3, 5, 7, 9]

screw_test_datasets = ['orig']
screw_models = ['best_added_more_noscrews_at_diff_exposure']

sticker_test_datasets = ['orig']
sticker_models = [
    'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2',
    'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2',
    'ssd300_vgg16',
    # 'ssdlite320_mobilenet_v3_large'
]

face_test_datasets = ['orig']
face_models = ['yunet_s', 'yunet_n']


def get_whole_set_screw(t_path):
    test_path = Path('{}/{}'.format(t_path, 'orig'))
    test_images_path = test_path / 'images'
    test_images = list(test_images_path.glob('*.png')) + list(test_images_path.glob('*.jpg'))
    return test_images


def get_whole_set_face(t_path):
    # /home/complexse/workspace/RoboSapiens/Social_Perception/libfacedetection.train/data/widerface/origimg/orig/images
    test_path = Path(t_path) / 'images'
    sub_folders = [x.name for x in test_path.iterdir() if x.is_dir()]
    test_images = []

    for sub_folder in sub_folders:
        sub_folder = test_path / sub_folder
        test_images += list(Path(sub_folder).glob('*.png')) + list(Path(sub_folder).glob('*.jpg'))
    return test_images


def get_whole_set_sticker(t_path):
    # test_path = Path('{}/{}'.format(t_path, 'orig'))
    test_images_path = Path(t_path)
    test_images = list(test_images_path.glob('*.png')) + list(test_images_path.glob('*.jpg'))
    return test_images


def check_folder_contents(folder_path):
    folder = Path(folder_path)
    # 1. Define the expected list of filenames
    expected_files = {f"prediction_{i}.json" for i in range(10)}
    expected_files.add("clustered_predictions.json")

    # print(folder.exists())
    # 2. Check if folder exists
    if not folder.exists():
        # return f"Error: The folder '{folder_path}' does not exist."
        return False, None
        # exit(0)
    # 3. Get actual files in the folder (ignoring subdirectories)
    actual_files = {f.name for f in folder.iterdir() if f.is_file()}
    # 4. Compare sets
    missing_files = expected_files - actual_files
    extra_files = actual_files - expected_files

    # 5. Report results
    if not missing_files and not extra_files:
        return True, "Success: The folder contains exactly the expected 11 files."

    error_msg = []
    if missing_files:
        error_msg.append(f"Missing files: {sorted(list(missing_files))}")
    if extra_files:
        error_msg.append(f"Unexpected extra files: {sorted(list(extra_files))}")

    return False, "\n".join(error_msg)


# 1. Initialize with ALL images ONCE (outside the loop)
# This set will shrink as images fail checks
def get_common_imgs(test_images, path, models, test_datasets, save_file):
    common_imgs = set(test_images)

    print(common_imgs)
    print(f"Starting total images: {len(common_imgs)}")

    for uq in uq_ms:
        print(f"Checking UQ method: {uq}...")

        # Define method-specific parameters
        if uq == "mc_dropout":
            mutation_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        else:
            mutation_rates = [(d, b) for d in d_rates for b in b_sizes]

        for model in models:
            for test_dataset in test_datasets:
                for mutation_rate in mutation_rates:

                    # Format the tag string
                    tag_str = f"{(mutation_rate[0])}_{mutation_rate[1]}" if isinstance(mutation_rate, tuple) else str(
                        mutation_rate)

                    # Optimization: Iterate over a copy (list) of the currently valid images
                    # This automatically skips images that already failed a previous check

                    for test_i in list(common_imgs):
                        # if test_i.stem == 'image_dall_3_52':
                            # print('aaaaaaaaaaaaaaaaaa')
                            # exit(0)

                        p_folder = f'{path}/experiment_results_{uq}/{model}/dataset/{test_dataset}/{uq}_{tag_str}_{test_i.stem}'

                        # print(p_folder)
                        # Check folder validity
                        is_valid, _ = check_folder_contents(p_folder)

                        # Logic: If it fails HERE, it is removed from the global set forever
                        if not is_valid:
                            common_imgs.discard(test_i)

        print(f"  > Valid images remaining after {uq}: {len(common_imgs)}")

    # Final Result
    print("-" * 30)
    print(f"Final Common Images (valid across ALL methods): {len(common_imgs)}")

    # Save the set of Path objects to a binary file
    with open(f'common_images_{save_file}.pkl', 'wb') as f:
        pickle.dump(list(common_imgs), f)

    print(f"Successfully saved {len(common_imgs)} images to 'common_images.pkl'")


if __name__ == '__main__':
    # test_suite_screw = get_whole_set_screw(t_path='/home/complexse/workspace/RoboSapiens/Screw_Detection/'
    #                                               'screw_detection/uq_evaluation/origimg2')
    # screw_results = '/home/complexse/workspace/RoboSapiens/Screw_Detection/screw_detection/experiment_results'
    # for scdm in screw_models:
    #     get_common_imgs(test_suite_screw, path=screw_results, save_file=scdm,
    #                     models=[scdm], test_datasets=screw_test_datasets)

    sticker = '/home/complexse/workspace/RoboSapiens/DTI-Laptop-refubishment/sticker_detector/dataset/origimg/orig'
    sticker_results = '/home/complexse/workspace/RoboSapiens/DTI-Laptop-refubishment/experiment_results'
    test_suite_sticker = get_whole_set_sticker(t_path=sticker)
    # get_common_imgs(test_suite_sticker, path=sticker_results, save_file=f'sticker',
    #                 models=sticker_models, test_datasets=sticker_test_datasets)
    for stdm in sticker_models:
        get_common_imgs(test_suite_sticker, path=sticker_results, save_file=f'{stdm}',
                        models=[stdm], test_datasets=sticker_test_datasets)

    # face = '/home/complexse/workspace/RoboSapiens/Social_Perception/libfacedetection.train/data/widerface/origimg/orig'
    # face_results = '/home/complexse/workspace/RoboSapiens/Social_Perception/experiment_results'
    # test_suite_face = get_whole_set_face(t_path=face)
    # for fdm in face_models:
    #     get_common_imgs(test_suite_face, path=face_results, save_file=f'{fdm}',
    #                     models=[fdm], test_datasets=face_test_datasets)
