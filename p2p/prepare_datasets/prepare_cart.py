
# run python aethervision/prepare_datasets/prepare_cart.py
#
# CART (Caltech Aerial RGB-Thermal) is downloaded from the dataset authors' release.
# This script only verifies that the dataset has been placed in the expected layout;
# it does not perform any raw-to-processed conversion.

import os
from pathlib import Path

REQUIRED_DIRS = ['val/color', 'val/thermal8', 'val/annotations']


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'CART'

    missing = [d for d in REQUIRED_DIRS if not (ds_path / d).exists()]
    if missing:
        raise FileNotFoundError(
            f'CART not found or incomplete in {ds_path}.\n'
            f'Missing: {missing}\n\n'
            'Download the dataset (see DATASETS.md), then place the held-out split as:\n'
            f'  {ds_path}/val/color      (RGB images, jpg)\n'
            f'  {ds_path}/val/thermal8   (8-bit thermal images, jpg)\n'
            f'  {ds_path}/val/annotations (single-channel png with class indices, ignore_label=255)\n'
        )

    print(f'CART found in {ds_path} with the expected directory layout.')


if __name__ == '__main__':
    main()
