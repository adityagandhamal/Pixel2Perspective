
# run python p2p/prepare_datasets/prepare_openearthmap.py
#
# OpenEarthMap ships as GeoTIFF (.tif) images and labels, with separate RGB and (simulated) SAR
# test splits. This script only verifies that the dataset has been placed in the expected layout;
# it does not perform any raw-to-processed conversion (the ov-seg-main repo's own top-level
# prepare_openearthmap.py was an unfinished draft and was not ported here).

import os
from pathlib import Path

REQUIRED_DIRS = ['train/rgb_images', 'train/labels', 'test/rgb_images', 'test/sar_images', 'test/labels']


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'OpenEarthMap'

    missing = [d for d in REQUIRED_DIRS if not (ds_path / d).exists()]
    if missing:
        raise FileNotFoundError(
            f'OpenEarthMap not found or incomplete in {ds_path}.\n'
            f'Missing: {missing}\n\n'
            'Download the dataset (see DATASETS.md), then place it as:\n'
            f'  {ds_path}/train/rgb_images, {ds_path}/train/labels\n'
            f'  {ds_path}/test/rgb_images,  {ds_path}/test/sar_images, {ds_path}/test/labels\n'
            'All files are GeoTIFF (.tif), ignore_label=255.'
        )

    print(f'OpenEarthMap found in {ds_path} with the expected directory layout.')


if __name__ == '__main__':
    main()
