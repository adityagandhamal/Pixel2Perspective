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
# Dataset page: https://github.com/jiangyu98jiang/SemanticRT (Ji et al., SemanticRT)
CLASSES = (
    'background (unlabeled)',
    'car_stop',
    'bike',
    'bicyclist',
    'motorcycle',
    'motorcyclist',
    'car',
    'tricycle',
    'traffic_light',
    'box',
    'pole',
    'curve',
    'person',
)


def register_dataset(root):
    ds_name = 'semantic_rt'
    root = os.path.join(root, 'SemanticRT')

    for split, image_dirname, sem_seg_dirname, image_ext in [
        ('test_rgb', 'rgb', 'labels', 'jpg'),
        ('test_ir', 'thermal', 'labels', 'jpg'),
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
