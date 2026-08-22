
# run python aethervision/prepare_datasets/prepare_indraeye.py
#
# IndraEye does not ship with an automated download (access is via request to the authors).
# This script only verifies that the dataset has been placed in the expected layout;
# it does not perform any raw-to-processed conversion.

import os
from pathlib import Path

REQUIRED_DIRS = [
    'eo/train/images', 'eo/train/annotations',
    'eo/test/images_original', 'eo/test/annotations',
    'ir/test/images', 'ir/test/annotations',
]


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'indraeye'

    missing = [d for d in REQUIRED_DIRS if not (ds_path / d).exists()]
    if missing:
        raise FileNotFoundError(
            f'IndraEye not found or incomplete in {ds_path}.\n'
            f'Missing: {missing}\n\n'
            'Request access and download the dataset (see DATASETS.md), then place it as:\n'
            f'  {ds_path}/eo/train/images,        {ds_path}/eo/train/annotations\n'
            f'  {ds_path}/eo/test/images_original, {ds_path}/eo/test/annotations\n'
            f'  {ds_path}/ir/test/images,          {ds_path}/ir/test/annotations\n'
            'Images are jpg, annotations are single-channel png with class indices (ignore_label=255).'
        )

    print(f'IndraEye found in {ds_path} with the expected directory layout.')


if __name__ == '__main__':
    main()
