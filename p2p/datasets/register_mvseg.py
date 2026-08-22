import os

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_sem_seg
    from detectron2.utils.colormap import colormap
except ImportError:
    from p2p.utils.catalog import DatasetCatalog, MetadataCatalog
    from p2p.utils.data import load_sem_seg
    from p2p.utils.colormap import colormap

# Ground-level (GL) dataset, RGB and IR modalities, co-registered pairs share one label map.
# Dataset page: https://github.com/haolin111/MVSeg
CLASSES = (
    'background(unlabelled)',
    'Car',
    'Bus',
    'Motorcycle',
    'Bicycle',
    'Pedestrian',
    'Motorcycle',
    'Bicyclist',
    'Cart',
    'Bench',
    'Umbrella',
    'Box',
    'Pole',
    'Street_lamp',
    'Traffic_light',
    'Traffic_sign',
    'Car_stop',
    'Color_cone',
    'Sky',
    'Road',
    'Sidewalk',
    'Curb',
    'Vegetation',
    'Terrain',
    'Building',
    'Ground',
)


def register_dataset(root):
    ds_name = 'mvseg'
    root = os.path.join(root, 'MVSeg')

    for split, image_dirname, sem_seg_dirname, image_ext in [
        ('test_rgb', 'rgb/test/images', 'rgb/test/labels', 'v.jpg'),
        ('test_ir', 'ir/test/images', 'rgb/test/labels', 'i.jpg'),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        full_name = f'{ds_name}_sem_seg_{split}'
        DatasetCatalog.register(
            full_name,
            lambda x=image_dir, y=gt_dir, e=image_ext: load_sem_seg(
                y, x, gt_ext='l.png', image_ext=e
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
