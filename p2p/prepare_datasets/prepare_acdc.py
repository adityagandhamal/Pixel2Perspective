
# run python p2p/prepare_datasets/prepare_acdc.py
#
# This script only checks that the flattened layout register_acdc.py expects is present; it does
# not convert ACDC's official nested rgb_anon/<condition>/<split>/<sequence>/ release structure into
# that layout -- write your own flattening step if starting from the official release.

import os
from pathlib import Path

REQUIRED_DIRS = ['acdc_imgs/train', 'acdc_imgs/val', 'annotations_detectron2/train', 'annotations_detectron2/val']


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'ACDC'

    missing = [d for d in REQUIRED_DIRS if not (ds_path / d).exists()]
    if missing:
        raise FileNotFoundError(
            f'ACDC not found or incomplete in {ds_path}.\n'
            f'Missing: {missing}\n\n'
            'Download ACDC from https://acdc.vision.ee.ethz.ch/ (registration required), flatten its '
            'official rgb_anon/<condition>/<split>/<sequence>/ structure into:\n'
            f'  {ds_path}/acdc_imgs/{{train,val}}          (RGB images, png)\n'
            f'  {ds_path}/annotations_detectron2/{{train,val}} (single-channel labelIds, png, '
            f'ignore_label=255, Cityscapes 19-class convention)\n'
        )

    print(f'ACDC found in {ds_path} with the expected directory layout.')


if __name__ == '__main__':
    main()
