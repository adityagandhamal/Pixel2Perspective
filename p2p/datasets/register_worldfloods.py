import os

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_sem_seg
    from detectron2.utils.colormap import colormap
except ImportError:
    from p2p.utils.catalog import DatasetCatalog, MetadataCatalog
    from p2p.utils.data import load_sem_seg
    from p2p.utils.colormap import colormap

# Bird's-eye-view (BEV) dataset, Sentinel-2 satellite imagery, RGB and IRRG composites.
# Dataset page: https://spaceml-org.github.io/ml4floods/content/worldfloods_dataset.html
CLASSES = (
    'land',
    'water and flood',
    'cloud',
)


def register_dataset(root):
    ds_name = 'worldfloods'
    root = os.path.join(root, 'WorldFloods')

    for split, image_dirname, sem_seg_dirname in [
        ('test_rgb', 'images_detectron2/test/rgb', 'annotations_detectron2/test'),
        ('test_irrg', 'images_detectron2/test/irrg', 'annotations_detectron2/test'),
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
            stuff_colors=colormap(rgb=True)[[4, 0, 8]],
            classes_of_interest=[1],
            background_class=0,
        )


_root = os.getenv('DETECTRON2_DATASETS', 'datasets')
register_dataset(_root)
