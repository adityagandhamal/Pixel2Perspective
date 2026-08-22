
# Computes MSRIQ between two directories of predicted single-channel label PNGs (one per modality),
# matched by filename (co-registered pairs must share the same basename across the two dirs).
#
# Usage:
#   python -m p2p.evaluation.compute_msriq \
#     --pred_rgb_dir output/model/msrs_sem_seg_test_rgb/pred \
#     --pred_ir_dir output/model/msrs_sem_seg_test_ir/pred \
#     --dataset msrs_sem_seg_test_rgb
#
# Each model/framework saves predicted label maps differently -- this script assumes you've already
# dumped per-pixel argmax class predictions as single-channel PNGs (matching load_sem_seg's file
# pairing convention: same basename, any extension).

import argparse
import numpy as np
from pathlib import Path
from PIL import Image

try:
    from detectron2.data import MetadataCatalog
except ImportError:
    from p2p.utils.catalog import MetadataCatalog

from p2p.evaluation.msriq import MSRIQAccumulator


def _index(directory):
    return {p.stem: p for p in Path(directory).iterdir() if p.is_file()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute MSRIQ between paired RGB/IR predictions')
    parser.add_argument('--pred_rgb_dir', type=str, required=True)
    parser.add_argument('--pred_ir_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True,
                         help='Registered dataset name to read num_classes/classes_of_interest/'
                              'ignore_label from (import p2p.datasets first to register).')
    args = parser.parse_args()

    import p2p.datasets  # noqa: F401
    meta = MetadataCatalog.get(args.dataset)
    num_classes = len(meta.stuff_classes)
    ignore_label = getattr(meta, 'ignore_label', 255)
    classes_of_interest = getattr(meta, 'classes_of_interest', None)

    rgb_files = _index(args.pred_rgb_dir)
    ir_files = _index(args.pred_ir_dir)
    shared_ids = sorted(set(rgb_files) & set(ir_files))
    if len(shared_ids) == 0:
        raise FileNotFoundError('No matching filenames found between the two prediction directories.')
    missing = (set(rgb_files) | set(ir_files)) - set(shared_ids)
    if missing:
        print(f'Warning: {len(missing)} files present in only one directory, skipping them.')

    acc = MSRIQAccumulator(num_classes, ignore_index=ignore_label, classes_of_interest=classes_of_interest)
    for id_ in shared_ids:
        pred_rgb = np.array(Image.open(rgb_files[id_]))
        pred_ir = np.array(Image.open(ir_files[id_]))
        acc.update(pred_rgb, pred_ir)

    msriq, per_class_iou = acc.evaluate()
    print(f'\nMSRIQ ({len(shared_ids)} pairs, dataset={args.dataset}): {100 * msriq:.2f}')
    for i, name in enumerate(meta.stuff_classes):
        print(f'  IoU-{name}: {100 * per_class_iou[i]:.2f}')
