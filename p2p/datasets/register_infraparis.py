import os

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_sem_seg
    from detectron2.utils.colormap import colormap
except ImportError:
    from p2p.utils.catalog import DatasetCatalog, MetadataCatalog
    from p2p.utils.data import load_sem_seg
    from p2p.utils.colormap import colormap

# Ground-level (GL) dataset, RGB and IR modalities.
# Dataset page: https://ieee-dataport.org/open-access/infraparis (Franchi et al., WACV 2024)
CLASSES = (
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'traffic_light',
    'traffic_sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
    'bear_deer_cow',
    'garbage_bag_stand_food_trash_can',
)


def register_dataset(root):
    ds_name = 'infraparis'
    root = os.path.join(root, 'Infraparis')

    for split, image_dirname, sem_seg_dirname, image_ext in [
        ('test_rgb', 'images_rgb', 'labels_rgb', 'png'),
        ('test_ir', 'thermal_mod', 'labels_thermal', 'jpg'),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        full_name = f'{ds_name}_sem_seg_{split}'
        DatasetCatalog.register(
            full_name,
            lambda x=image_dir, y=gt_dir, e=image_ext: load_sem_seg(
                y, x, gt_ext='png', image_ext=e
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
