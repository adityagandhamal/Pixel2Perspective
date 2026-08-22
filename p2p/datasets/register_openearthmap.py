import os

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_sem_seg
    from detectron2.utils.colormap import colormap
except ImportError:
    from p2p.utils.catalog import DatasetCatalog, MetadataCatalog
    from p2p.utils.data import load_sem_seg
    from p2p.utils.colormap import colormap

# Bird's-eye-view (BEV) dataset, RGB and SAR (sim-to-real) modalities.
# Dataset page: https://open-earth-map.org/
#
# NOTE: the source register script's top-level CLASSES tuple has the same missing-comma bug as
# LoveDA ("Bareland" "Rangeland" concatenate into "BarelandRangeland"), but that tuple was dead
# code there -- the actually-registered classes came from a separately (correctly) written list.
# Reproduced correctly here directly.
CLASSES = (
    'Unlabelled',
    'Bareland',
    'Rangeland',
    'Developed_space',
    'Road',
    'Tree',
    'Water',
    'Agriculture_land',
    'Building',
)


def register_dataset(root):
    ds_name = 'openearthmap'
    root = os.path.join(root, 'OpenEarthMap')

    for split, image_dirname, sem_seg_dirname in [
        ('train_rgb', 'train/rgb_images', 'train/labels'),
        ('test_rgb', 'test/rgb_images', 'test/labels'),
        ('test_sar', 'test/sar_images', 'test/labels'),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        full_name = f'{ds_name}_sem_seg_{split}'
        DatasetCatalog.register(
            full_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext='tif', image_ext='tif'
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
