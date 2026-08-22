# Getting Started with Pixel2Perspective

## Installation

```sh
pip install -e .
```

If you plan to evaluate detectron2-based models, install detectron2 separately — see
[setup_env.sh](setup_env.sh) or https://detectron2.readthedocs.io/en/latest/tutorials/install.html.
`detectron2` is not required just to load the datasets.

## Prepare the datasets

See [DATASETS.md](DATASETS.md) for download links, licenses and expected directory layout, then:
```sh
python -m p2p.prepare_all_datasets --dataset_dir datasets
python -m p2p.prepare_all_datasets --dataset_dir datasets --stats   # check status only
export DETECTRON2_DATASETS=datasets
```

## Register the datasets

```python
import p2p.datasets
```
This registers all 24 dataset/modality/split combinations across the 13 datasets (see
[DATASETS.md](DATASETS.md) for the full list of registered names).

## Evaluation settings used in the paper

Four OVSS models (SAN, Zegformer, OVSeg, CAT-Seg, plus `-L` variants of OVSeg/CAT-Seg) are evaluated under
four training regimes, each validated on all 13 datasets:

1. **COCO-Stuff-trained** — standard zero-shot setting.
2. **IndraEye-trained** — fine-tuned on the slant-angle IndraEye (RGB) training set.
3. **MSRS + ACDC-trained** — fine-tuned on the ground-level MSRS and (adverse-weather) ACDC training sets.
4. **OpenEarthMap-trained** — fine-tuned on the bird's-eye-view OpenEarthMap training set, covering both
   RGB and sim-to-real (SAR) setups.

Run all settings with [eval.sh](eval.sh):
```sh
bash eval.sh
```

## Evaluating with your own model

```sh
export DETECTRON2_DATASETS=datasets
TEST_DATASETS="isprs_potsdam_sem_seg_test_rgb worldfloods_sem_seg_test_rgb loveda_sem_seg_test openearthmap_sem_seg_test_rgb openearthmap_sem_seg_test_sar uavid_sem_seg_val indraeye_sem_seg_test_rgb indraeye_sem_seg_test_ir cart_sem_seg_test_rgb cart_sem_seg_test_ir acdc_sem_seg_val msrs_sem_seg_test_rgb msrs_sem_seg_test_ir mvseg_sem_seg_test_rgb mvseg_sem_seg_test_ir semantic_rt_sem_seg_test_rgb semantic_rt_sem_seg_test_ir mfnet_sem_seg_test_rgb mfnet_sem_seg_test_ir infraparis_sem_seg_test_rgb infraparis_sem_seg_test_ir"

for DATASET in $TEST_DATASETS
do
  python train_net.py --num-gpus 1 --eval-only --config-file <your_config>.yaml \
    DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS <your_weights>.pth OUTPUT_DIR output/<model_name>/$DATASET
done

python p2p/evaluation/pixel2perspective_evaluation.py --model_outputs output/<model_name>
```

## Computing MSRIQ

Once you have saved predicted label-map PNGs for a model's RGB and IR runs on the same dataset (matched by
filename), compute MSRIQ with:
```sh
python -m p2p.evaluation.compute_msriq \
  --pred_rgb_dir output/<model_name>/msrs_sem_seg_test_rgb/pred \
  --pred_ir_dir output/<model_name>/msrs_sem_seg_test_ir/pred \
  --dataset msrs_sem_seg_test_rgb
```
`p2p.evaluation.msriq.MSRIQAccumulator` is also usable directly in Python if your evaluation loop already
holds prediction arrays in memory (no need to round-trip through PNGs).

## Adding a new dataset

Use `p2p/datasets/register_TEMPLATE.py` and `p2p/prepare_datasets/prepare_TEMPLATE.py` as a starting point,
then add a corresponding section in [DATASETS.md](DATASETS.md).
