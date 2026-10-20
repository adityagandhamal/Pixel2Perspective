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
# Dataset page: https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/default.aspx
CLASSES = (
    'road',
    'building',
    'grass',
    'tree',
    'car',
    'others',
)


def register_dataset(root):
    ds_name = 'isprs_potsdam'
    root = os.path.join(root, 'ISPRS_Potsdam')

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
            stuff_colors=colormap(rgb=True)[[3, 1, 4, 15, 2, 0]],
            classes_of_interest=list(range(len(CLASSES) - 1)),
            background_class=5,
        )


_root = os.getenv('DETECTRON2_DATASETS', 'datasets')
register_dataset(_root)
