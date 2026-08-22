
# run python aethervision/prepare_datasets/prepare_msrs.py
#
# MSRS ships with RGB and IR pairs and one shared label map per pair.
# This script only verifies that the dataset has been placed in the expected layout;
# it does not perform any raw-to-processed conversion.

import os
from pathlib import Path

REQUIRED_DIRS = [
    'images/train', 'images/test', 'ir/test',
    'annotations_detectron2/train', 'annotations_detectron2/test',
]


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'msrs'

    missing = [d for d in REQUIRED_DIRS if not (ds_path / d).exists()]
    if missing:
        raise FileNotFoundError(
            f'MSRS not found or incomplete in {ds_path}.\n'
            f'Missing: {missing}\n\n'
            'Download the dataset from https://github.com/Linfeng-Tang/MSRS (see DATASETS.md), '
            'then place it as:\n'
            f'  {ds_path}/images/{{train,test}}         (RGB images, png)\n'
            f'  {ds_path}/ir/test                        (IR images, png)\n'
            f'  {ds_path}/annotations_detectron2/{{train,test}} (single-channel png with class '
            f'indices, ignore_label=255)\n'
        )

    print(f'MSRS found in {ds_path} with the expected directory layout.')


if __name__ == '__main__':
    main()
