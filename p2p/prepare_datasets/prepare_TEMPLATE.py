
# run python p2p/prepare_datasets/prepare_<DATASET NAME>.py

import os
import tqdm
import gdown
import numpy as np
from pathlib import Path
from PIL import Image


def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    # TODO: Add an automated script if possible, otherwise remove code
    ds_path.mkdir(exist_ok=True, parents=True)
    print('Downloading dataset...')
    os.system('wget https://link/to/DATASET.zip')
    os.system('unzip <DATASET>.zip -d ' + str(ds_path))
    os.system('rm <DATASET>.zip')

    # Download from Google Drive
    gdown.download_folder(id='<FOLDER ID>', output=str(ds_path))


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / '<DATASET>'
    # TODO: Remove if an automated download is not possible
    if not ds_path.exists():
        download_dataset(ds_path)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    for split in ['test']:
        # TODO: Change if other directories are required
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        # TODO: Change image directory to the current split. Change the image extension if necessary
        for img_path in tqdm.tqdm((ds_path / '<OLD IMAGE DIR>' / split).glob('*.jpg')):
            img = Image.open(img_path)
            img = img.convert('RGB')
            img.save(img_dir / img_path.name)

            id = img_path.stem
            mask = np.array(Image.open(ds_path / '<OLD IMAGE DIR>' / split / f'{id}.png'))
            # TODO: Map the current annotations to the right format (classes 0, 1, ...; ignore value 255)
            Image.fromarray(mask).save(anno_dir / f'{id}.png')

        print(f'Saved {split} images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
