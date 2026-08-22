import os

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_sem_seg
    from detectron2.utils.colormap import colormap
except ImportError:
    from p2p.utils.catalog import DatasetCatalog, MetadataCatalog
    from p2p.utils.data import load_sem_seg
    from p2p.utils.colormap import colormap

# Slant-angle (SA) dataset, adverse-weather driving, RGB only.
# Dataset page: https://acdc.vision.ee.ethz.ch/
#
# Expects a flattened acdc_imgs/{train,val} + annotations_detectron2/{train,val} layout rather than
# ACDC's official nested rgb_anon/<condition>/<split>/<sequence>/ structure -- see
# prepare_datasets/prepare_acdc.py for how to produce that layout from the official release.
CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
    'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle',
)


def register_dataset(root):
    ds_name = 'acdc'
    root = os.path.join(root, 'ACDC')

    for split, image_dirname, sem_seg_dirname in [
        ('train', 'acdc_imgs/train', 'annotations_detectron2/train'),
        ('val', 'acdc_imgs/val', 'annotations_detectron2/val'),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        full_name = f'{ds_name}_sem_seg_{split}'
        DatasetCatalog.register(
            full_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext='png', image_ext='png'
            ),
        )
        MetadataCatalog.get(full_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type='sem_seg',
            ignore_label=255,
            stuff_classes=CLASSES,
            stuff_colors=colormap(rgb=True),
            classes_of_interest=list(range(1, len(CLASSES))),
            background_class=0,
        )


_root = os.getenv('DETECTRON2_DATASETS', 'datasets')
register_dataset(_root)
