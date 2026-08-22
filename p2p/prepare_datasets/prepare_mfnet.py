
# run python p2p/prepare_datasets/prepare_mfnet.py
#
# MFNet ships with co-registered RGB/thermal frames sharing one label map per pair.
# This script only verifies that the dataset has been placed in the expected layout;
# it does not perform any raw-to-processed conversion.

import os
from pathlib import Path

REQUIRED_DIRS = ['rgb_val', 'thermal_val', 'gt_val']


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'MFNet'

    missing = [d for d in REQUIRED_DIRS if not (ds_path / d).exists()]
    if missing:
        raise FileNotFoundError(
            f'MFNet not found or incomplete in {ds_path}.\n'
            f'Missing: {missing}\n\n'
            'Download the dataset (see DATASETS.md), then place the validation split as:\n'
            f'  {ds_path}/rgb_val      (RGB frames, e.g. 00498D_rgb.png)\n'
            f'  {ds_path}/thermal_val  (thermal frames, e.g. 00498D_thermal.png)\n'
            f'  {ds_path}/gt_val       (single-channel labels, e.g. 00498D.png, ignore_label=255)\n'
        )

    print(f'MFNet found in {ds_path} with the expected directory layout.')


if __name__ == '__main__':
    main()
