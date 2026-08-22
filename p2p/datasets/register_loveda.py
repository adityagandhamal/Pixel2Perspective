import os

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_sem_seg
    from detectron2.utils.colormap import colormap
except ImportError:
    from p2p.utils.catalog import DatasetCatalog, MetadataCatalog
    from p2p.utils.data import load_sem_seg
    from p2p.utils.colormap import colormap

# Bird's-eye-view (BEV) dataset.
# Dataset page: https://github.com/Junjue-Wang/LoveDA
#
# NOTE: the source register script in ov-seg-main (register_loveda.py) built this class list as
# ["Unlabelled", "Background", "Building", "Road", "Water", "Barren", "Forest" "Agricultural"] --
# missing a comma between "Forest" and "Agricultural", which Python silently concatenates into one
# string "ForestAgricultural". That collapses 8 classes into 7. Fixed here to the correct 8 classes.
CLASSES = (
    'Unlabelled',
    'Background',
    'Building',
    'Road',
    'Water',
    'Barren',
    'Forest',
    'Agricultural',
)


def register_dataset(root):
    ds_name = 'loveda'
    root = os.path.join(root, 'LoveDA')

    for split, image_dirname, sem_seg_dirname in [
        ('test', 'images', 'labels'),
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
