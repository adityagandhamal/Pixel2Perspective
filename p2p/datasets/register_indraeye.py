import os

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_sem_seg
    from detectron2.utils.colormap import colormap
except ImportError:
    from p2p.utils.catalog import DatasetCatalog, MetadataCatalog
    from p2p.utils.data import load_sem_seg
    from p2p.utils.colormap import colormap

# Slant-angle (SA) dataset, drone view, RGB (eo) and IR modalities.
# Dataset page: https://github.com/kaskyodia/IndraEye (Sikdar et al., SAGA)
CLASSES = (
    'background(and back-hoe)',
    'bicycle',
    'bus',
    'car',
    'cargo_trike',
    'ignore',
    'motorcycle',
    'person',
    'rickshaw',
    'small_truck',
    'tractor',
    'truck',
    'van',
)


def register_dataset(root):
    ds_name = 'indraeye'
    rgb_root = os.path.join(root, 'indraeye', 'eo')
    ir_root = os.path.join(root, 'indraeye', 'ir')

    for split, base, image_dirname, sem_seg_dirname, image_ext in [
        ('train_rgb', rgb_root, 'train/images', 'train/annotations', 'jpg'),
        ('test_rgb', rgb_root, 'test/images_original', 'test/annotations', 'jpg'),
        ('test_ir', ir_root, 'test/images', 'test/annotations', 'jpg'),
    ]:
        image_dir = os.path.join(base, image_dirname)
        gt_dir = os.path.join(base, sem_seg_dirname)
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
