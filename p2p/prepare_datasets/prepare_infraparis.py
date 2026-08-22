
# run python p2p/prepare_datasets/prepare_infraparis.py
#
# Infraparis ships with separate RGB and thermal splits, each with its own label map.
# This script only verifies that the dataset has been placed in the expected layout;
# it does not perform any raw-to-processed conversion.

import os
from pathlib import Path

REQUIRED_DIRS = ['images_rgb', 'labels_rgb', 'thermal_mod', 'labels_thermal']


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'Infraparis'

    missing = [d for d in REQUIRED_DIRS if not (ds_path / d).exists()]
    if missing:
        raise FileNotFoundError(
            f'Infraparis not found or incomplete in {ds_path}.\n'
            f'Missing: {missing}\n\n'
            'Download the dataset (see DATASETS.md), then place it as:\n'
            f'  {ds_path}/images_rgb    (RGB images, png)\n'
            f'  {ds_path}/labels_rgb    (single-channel labels, *_labelIds.png, ignore_label=255)\n'
            f'  {ds_path}/thermal_mod   (thermal images, jpg)\n'
            f'  {ds_path}/labels_thermal (single-channel labels, *_labelIds.png, ignore_label=255)\n'
        )

    print(f'Infraparis found in {ds_path} with the expected directory layout.')


if __name__ == '__main__':
    main()
