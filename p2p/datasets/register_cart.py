import os

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_sem_seg
    from detectron2.utils.colormap import colormap
except ImportError:
    from p2p.utils.catalog import DatasetCatalog, MetadataCatalog
    from p2p.utils.data import load_sem_seg
    from p2p.utils.colormap import colormap

# Slant-angle (SA) dataset, drone view, RGB (color) and IR (thermal8) modalities.
# Dataset page: https://github.com/aerialpalprivate/CalTech-Aerial-RGB-Thermal (Lee et al., CART)
CLASSES = (
    'unknown',
    'background',
    'bare_ground',
    'boulders_rocky_terrain',
    'human_made_structures',
    'road',
    'shrubs',
    'trees',
    'sky',
    'water',
    'vehicles',
    'person',
)


def register_dataset(root):
    ds_name = 'cart'
    root = os.path.join(root, 'CART')

    # The official val split is used as the held-out test set for this benchmark.
    for split, image_dirname, sem_seg_dirname in [
        ('test_rgb', 'val/color', 'val/annotations'),
        ('test_ir', 'val/thermal8', 'val/annotations'),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        full_name = f'{ds_name}_sem_seg_{split}'
        DatasetCatalog.register(
            full_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext='png', image_ext='jpg'
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
