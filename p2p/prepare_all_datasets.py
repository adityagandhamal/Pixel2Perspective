# Description: This script prepares all datasets
# Usage: python -m p2p.prepare_all_datasets --dataset_dir datasets

import os
import argparse
try:
    from detectron2.data import DatasetCatalog
except ImportError:
    from p2p.utils.catalog import DatasetCatalog

from p2p.prepare_datasets import (
    prepare_isprs_potsdam,
    prepare_uavid,
    prepare_indraeye,
    prepare_cart,
    prepare_msrs,
    prepare_mvseg,
    prepare_acdc,
    prepare_semantic_rt,
    prepare_mfnet,
    prepare_infraparis,
    prepare_worldfloods,
    prepare_loveda,
    prepare_openearthmap,
)


def check_datasets(dataset_list):
    print('\nDataset status:')
    missing_datasets = []
    for dataset_name in dataset_list:
        try:
            length = len(DatasetCatalog.get(dataset_name))
            status = f'{length} images' if length > 0 else 'No images found'
            if length == 0:
                missing_datasets.append(dataset_name)
        except (FileNotFoundError, AssertionError):
            status = 'Dataset not found'
            missing_datasets.append(dataset_name)
        print(f'{dataset_name:40s} {status}')
    if len(missing_datasets) == 0:
        print('All datasets processed.')
    else:
        print(f'Found missing datasets or images in: {", ".join(missing_datasets)}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare datasets')
    parser.add_argument('--dataset_dir', type=str, default='datasets', help='dataset directory')
    parser.add_argument('--stats', action='store_true', help='Only show dataset statistics')
    args = parser.parse_args()

    os.environ['DETECTRON2_DATASETS'] = args.dataset_dir
    os.makedirs(args.dataset_dir, exist_ok=True)
    # Register datasets
    import p2p.datasets

    dataset_dict = {
        'isprs_potsdam_sem_seg_test_rgb': prepare_isprs_potsdam,
        'uavid_sem_seg_val': prepare_uavid,
        'indraeye_sem_seg_test_rgb': prepare_indraeye,
        'cart_sem_seg_test_rgb': prepare_cart,
        'msrs_sem_seg_test_rgb': prepare_msrs,
        'mvseg_sem_seg_test_rgb': prepare_mvseg,
        'acdc_sem_seg_val': prepare_acdc,
        'semantic_rt_sem_seg_test_rgb': prepare_semantic_rt,
        'mfnet_sem_seg_test_rgb': prepare_mfnet,
        'infraparis_sem_seg_test_rgb': prepare_infraparis,
        'worldfloods_sem_seg_test_rgb': prepare_worldfloods,
        'loveda_sem_seg_test': prepare_loveda,
        'openearthmap_sem_seg_test_rgb': prepare_openearthmap,

        ### Manual preparation ###
        # ISPRS Potsdam, UAVid, WorldFloods: auto-downloaded/tiled by the scripts above.
        # ACDC, SemanticRT, MFNet, Infraparis, LoveDA, OpenEarthMap, IndraEye, CART, MSRS, MVSeg:
        # request/download per DATASETS.md and place in the documented layout; these scripts only
        # verify the layout (see prepare_datasets/prepare_<name>.py). ACDC additionally has not been
        # validated against real data in this repo -- see the NOTE in register_acdc.py.
    }

    check_datasets(dataset_dict.keys())

    if args.stats:
        exit()

    for dataset_name, prepare_dataset in dataset_dict.items():
        try:
            prepared = len(DatasetCatalog.get(dataset_name)) != 0
        except (FileNotFoundError, AssertionError):
            prepared = False

        if prepared:
            print(f'\n{dataset_name} already prepared')
        else:
            print(f'\nPreparing {dataset_name}')
            try:
                prepare_dataset.main()
            except Exception as e:
                print(f'Error while preparing {dataset_name}: \n{e}')

    print('\nFinished preparing datasets')
    check_datasets(dataset_dict.keys())
