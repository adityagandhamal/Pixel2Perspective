
# run python p2p/prepare_datasets/prepare_semantic_rt.py
#
# SemanticRT ships with co-registered RGB/thermal frames sharing one label map per pair.
# This script only verifies that the dataset has been placed in the expected layout;
# it does not perform any raw-to-processed conversion.

import os
from pathlib import Path

REQUIRED_DIRS = ['rgb', 'thermal', 'labels']


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'SemanticRT'

    missing = [d for d in REQUIRED_DIRS if not (ds_path / d).exists()]
    if missing:
        raise FileNotFoundError(
            f'SemanticRT not found or incomplete in {ds_path}.\n'
            f'Missing: {missing}\n\n'
            'Download the dataset (see DATASETS.md), then place it as:\n'
            f'  {ds_path}/rgb       (RGB frames, jpg)\n'
            f'  {ds_path}/thermal   (thermal frames, jpg)\n'
            f'  {ds_path}/labels    (single-channel labels shared by both modalities, png, '
            f'ignore_label=255)\n'
        )

    print(f'SemanticRT found in {ds_path} with the expected directory layout.')


if __name__ == '__main__':
    main()
