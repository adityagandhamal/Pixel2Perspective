# Datasets for Pixel2Perspective

## Download datasets

WorldFloods downloads and prepares automatically. The remaining twelve require a manual/requested download
(see their sections below). For ISPRS Potsdam and UAVid, the preparation scripts then unzip and tile the
data; for the other ten they only verify the layout.

- ISPRS Potsdam: https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/default.aspx (Potsdam)
- UAVid: https://uavid.nl (Download -> Semantic Labelling with Images Only)
- WorldFloods: auto-downloaded via `gdown` (Google Drive folder, see `prepare_worldfloods.py`)
- IndraEye: https://github.com/airl-iisc/IndraEye (download links: https://bit.ly/indraeye)
- CART: https://data.caltech.edu/records/cks6g-ps927, code at
  https://github.com/aerorobotics/caltech-aerial-rgbt-dataset
- ACDC: https://acdc.vision.ee.ethz.ch/ (registration required)
- MSRS: https://github.com/Linfeng-Tang/MSRS
- MVSeg: https://jiwei0921.github.io/Multispectral-Video-Semantic-Segmentation/, code at
  https://github.com/jiwei0921/MVSS-Baseline
- SemanticRT: https://github.com/jiwei0921/SemanticRT
- MFNet: https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/
- Infraparis: https://ensta-u2is-ai.github.io/infraParis/, code at
  https://github.com/ENSTA-U2IS-AI/infraParis
- LoveDA: https://github.com/Junjue-Wang/LoveDA
- OpenEarthMap: https://open-earth-map.org/

## Install environment

`pip install -e .` installs the packages required for dataset preparation (including `gdown` and
`rasterio` for the WorldFloods conversion). If you plan to evaluate detectron2-based models, see
[setup_env.sh](setup_env.sh); `detectron2` is not required just to load the datasets.

## Dataset download and preparation

```sh
python -m p2p.prepare_all_datasets --dataset_dir datasets
python -m p2p.prepare_all_datasets --dataset_dir datasets --stats   # check status only
```

WorldFloods is downloaded automatically, and ISPRS Potsdam and UAVid are tiled automatically once
downloaded. The other ten datasets must be downloaded/requested manually and placed in the layout described below; their preparation scripts only
verify that layout (see the per-dataset sections).

If you are using another dataset directory than `datasets`, export it as `DETECTRON2_DATASETS` before
evaluating (see `eval.sh`).

## Dataset overview

Note the different licenses of the datasets, and please cite the corresponding papers if you use them —
BibTeX is provided in [datasets.bib](datasets.bib).

Datasets, by viewpoint cluster:
1. **Bird's-eye view**: [ISPRS Potsdam](#isprs-potsdam), [WorldFloods](#worldfloods), [LoveDA](#loveda),
   [OpenEarthMap](#openearthmap)
2. **Slant-angle**: [UAVid](#uavid), [IndraEye](#indraeye), [CART](#cart), [ACDC](#acdc)
3. **Ground-level**: [MSRS](#msrs), [MVSeg](#mvseg), [SemanticRT](#semanticrt), [MFNet](#mfnet),
   [Infraparis](#infraparis)

### Custom datasets

Use `p2p/datasets/register_TEMPLATE.py` and `p2p/prepare_datasets/prepare_TEMPLATE.py` as a starting point
for adding a new dataset.

### ISPRS Potsdam

Registered names: `isprs_potsdam_sem_seg_test_rgb`, `isprs_potsdam_sem_seg_test_irrg`

Dataset page: https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/default.aspx. Licence: undefined;
citation required, notice of BSF Swissphoto as dataset provider.

```sh
python -m p2p.prepare_datasets.prepare_isprs_potsdam
```
Tiles the 6000x6000 orthophotos into non-overlapping 1024x1024 patches (zero-padded, ignored during eval).

### WorldFloods

Registered names: `worldfloods_sem_seg_test_rgb`, `worldfloods_sem_seg_test_irrg`

Dataset page: https://spaceml-org.github.io/ml4floods/content/worldfloods_dataset.html

Paper: Mateo-Garcia, G. et al. (2021). Towards global flood mapping onboard low cost satellites with
machine learning. Scientific Reports 11(1). Licence: see dataset page; citation required.

```sh
python -m p2p.prepare_datasets.prepare_worldfloods
```
Downloads the labeled test set via `gdown`, converts Sentinel-2 bands to normalized RGB/IRRG composites,
and tiles into 1024x1024 patches.

### LoveDA

Registered name: `loveda_sem_seg_test`

Dataset page: https://github.com/Junjue-Wang/LoveDA

Paper: Wang, J., Zheng, Z., Ma, A., Lu, X., Zhong, Y. (2021). LoveDA: A remote sensing land-cover dataset
for domain adaptive semantic segmentation. arXiv:2110.08733. Licence: see dataset page; citation required.

Download and place as `datasets/LoveDA/images`, `datasets/LoveDA/labels` (png), then:
```sh
python -m p2p.prepare_datasets.prepare_loveda
```
**Note:** the source `register_loveda.py` in `ov-seg-main` had a missing-comma bug that silently merged
the "Forest" and "Agricultural" classes into one ("ForestAgricultural"), registering 7 classes instead of
8. Fixed here — `register_loveda.py` in this repo uses the correct 8-class list.

### OpenEarthMap

Registered names: `openearthmap_sem_seg_train_rgb`, `openearthmap_sem_seg_test_rgb`,
`openearthmap_sem_seg_test_sar`

Dataset page: https://open-earth-map.org/

Paper: Xia, J., Yokoya, N., Adriano, B., Broni-Bediako, C. (2023). OpenEarthMap: A benchmark dataset for
global high-resolution land cover mapping. WACV 2023. Licence: see dataset page; citation required.

Download and place as `datasets/OpenEarthMap/{train,test}/rgb_images`, `.../train/labels`,
`.../test/sar_images`, `.../test/labels` (all GeoTIFF), then:
```sh
python -m p2p.prepare_datasets.prepare_openearthmap
```

### UAVid

Registered name: `uavid_sem_seg_val`. Same as in [AetherVision-Bench](../AetherVision-Bench/DATASETS.md).
```sh
python -m p2p.prepare_datasets.prepare_uavid
```

### IndraEye

Registered names: `indraeye_sem_seg_train_rgb`, `indraeye_sem_seg_test_rgb`, `indraeye_sem_seg_test_ir`.
Same dataset and layout as in AetherVision-Bench — see
[AetherVision-Bench/DATASETS.md](../AetherVision-Bench/DATASETS.md#indraeye).
```sh
python -m p2p.prepare_datasets.prepare_indraeye
```

### CART

Registered names: `cart_sem_seg_test_rgb`, `cart_sem_seg_test_ir`. Same as in AetherVision-Bench — see
[AetherVision-Bench/DATASETS.md](../AetherVision-Bench/DATASETS.md#cart).
```sh
python -m p2p.prepare_datasets.prepare_cart
```

### ACDC

Registered names: `acdc_sem_seg_train`, `acdc_sem_seg_val`

Dataset page: https://acdc.vision.ee.ethz.ch/ (registration required)

Paper: Sakaridis, C., Dai, D., Van Gool, L. (2021). ACDC: The Adverse Conditions Dataset with
Correspondences for semantic driving scene understanding. ICCV 2021. Licence: see dataset page (custom,
research use); citation required.

The official release is organized as `rgb_anon/<condition>/<split>/<sequence>/*_rgb_anon.png` with
`gt/<condition>/<split>/<sequence>/*_gt_labelIds.png` (19 Cityscapes classes) — this repo expects a
flattened `acdc_imgs/{train,val}` + `annotations_detectron2/{train,val}` layout instead. Write your own
flattening step if starting from the official release, or adapt `register_acdc.py` to read the nested
layout directly.

Since `DETECTRON2_DATASETS` is a single shared root, make sure `ACDC/` is placed alongside the other 12
datasets in your unified dataset directory (symlink or copy it in if it lives elsewhere), e.g.
`ln -s /path/to/ACDC datasets/ACDC`.

### MSRS

Registered names: `msrs_sem_seg_train_rgb`, `msrs_sem_seg_test_rgb`, `msrs_sem_seg_test_ir`. Same as in
AetherVision-Bench — see [AetherVision-Bench/DATASETS.md](../AetherVision-Bench/DATASETS.md#msrs).
```sh
python -m p2p.prepare_datasets.prepare_msrs
```

### MVSeg

Registered names: `mvseg_sem_seg_test_rgb`, `mvseg_sem_seg_test_ir`. Same as in AetherVision-Bench — see
[AetherVision-Bench/DATASETS.md](../AetherVision-Bench/DATASETS.md#mvseg).
```sh
python -m p2p.prepare_datasets.prepare_mvseg
```

### SemanticRT

Registered names: `semantic_rt_sem_seg_test_rgb`, `semantic_rt_sem_seg_test_ir`

Dataset page: https://github.com/jiwei0921/SemanticRT

Paper: Ji, W., Li, J., Bian, C., Zhang, Z., Cheng, L. (2023). SemanticRT: A large-scale dataset and method
for robust semantic segmentation in multispectral images. ACM Multimedia 2023. Licence: see dataset page;
citation required.

Download and place as `datasets/SemanticRT/rgb`, `.../thermal` (jpg), `.../labels` (png, shared by both
modalities), then:
```sh
python -m p2p.prepare_datasets.prepare_semantic_rt
```

### MFNet

Registered names: `mfnet_sem_seg_test_rgb`, `mfnet_sem_seg_test_ir`

Dataset page: https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/

Paper: Ha, Q., Watanabe, K., Karasawa, T., Ushiku, Y., Harada, T. (2017). MFNet: Towards real-time semantic
segmentation for autonomous vehicles with multi-spectral scenes. IROS 2017. Licence: see dataset page;
citation required.

Download and place as `datasets/MFNet/rgb_val`, `.../thermal_val`, `.../gt_val` (png), then:
```sh
python -m p2p.prepare_datasets.prepare_mfnet
```

### Infraparis

Registered names: `infraparis_sem_seg_test_rgb`, `infraparis_sem_seg_test_ir`

Dataset page: https://ensta-u2is-ai.github.io/infraParis/, code at
https://github.com/ENSTA-U2IS-AI/infraParis

Paper: Franchi, G., Hariat, M., Yu, X., Belkhir, N., Manzanera, A., Filliat, D. (2024). InfraParis: A
multi-modal and multi-task autonomous driving dataset. WACV 2024. Licence: see dataset page; citation
required.

Download and place as `datasets/Infraparis/images_rgb`, `.../labels_rgb` (png), `.../thermal_mod` (jpg),
`.../labels_thermal` (png), then:
```sh
python -m p2p.prepare_datasets.prepare_infraparis
```
