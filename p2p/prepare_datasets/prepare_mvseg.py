
# run python aethervision/prepare_datasets/prepare_mvseg.py
#
# MVSeg ships with RGB and IR frames and one shared label map per co-registered pair.
# This script only verifies that the dataset has been placed in the expected layout;
# it does not perform any raw-to-processed conversion.

import os
from pathlib import Path

REQUIRED_DIRS = ['rgb/test/images', 'rgb/test/labels', 'ir/test/images']


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'MVSeg'

    missing = [d for d in REQUIRED_DIRS if not (ds_path / d).exists()]
    if missing:
        raise FileNotFoundError(
            f'MVSeg not found or incomplete in {ds_path}.\n'
            f'Missing: {missing}\n\n'
            'Download the dataset (see DATASETS.md), then place it as:\n'
            f'  {ds_path}/rgb/test/images  (RGB frames, filenames ending in v.jpg)\n'
            f'  {ds_path}/ir/test/images   (IR frames, filenames ending in i.jpg)\n'
            f'  {ds_path}/rgb/test/labels  (single-channel labels shared by both modalities, '
            f'filenames ending in l.png, ignore_label=255)\n'
        )

    print(f'MVSeg found in {ds_path} with the expected directory layout.')


if __name__ == '__main__':
    main()
