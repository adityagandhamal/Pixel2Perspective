#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate <env_name>

export DETECTRON2_DATASETS="datasets"
TEST_DATASETS="isprs_potsdam_sem_seg_test_rgb worldfloods_sem_seg_test_rgb loveda_sem_seg_test openearthmap_sem_seg_test_rgb openearthmap_sem_seg_test_sar uavid_sem_seg_val indraeye_sem_seg_test_rgb indraeye_sem_seg_test_ir cart_sem_seg_test_rgb cart_sem_seg_test_ir acdc_sem_seg_val msrs_sem_seg_test_rgb msrs_sem_seg_test_ir mvseg_sem_seg_test_rgb mvseg_sem_seg_test_ir semantic_rt_sem_seg_test_rgb semantic_rt_sem_seg_test_ir mfnet_sem_seg_test_rgb mfnet_sem_seg_test_ir infraparis_sem_seg_test_rgb infraparis_sem_seg_test_ir"

# --- Setting 1: COCO-Stuff-trained (zero-shot) ---
for DATASET in $TEST_DATASETS
do
 python train_net.py --num-gpus 1 --eval-only --config-file configs/<coco_stuff_config>.yaml \
  DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS weights/<model>_coco_stuff.pth \
  OUTPUT_DIR output/<model_name>_coco/$DATASET
done

# --- Setting 2: IndraEye-trained (slant-angle fine-tuning) ---
for DATASET in $TEST_DATASETS
do
 python train_net.py --num-gpus 1 --eval-only --config-file configs/<indraeye_config>.yaml \
  DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS weights/<model>_indraeye.pth \
  OUTPUT_DIR output/<model_name>_indraeye/$DATASET
done

# --- Setting 3: MSRS + ACDC-trained (ground-level fine-tuning) ---
for DATASET in $TEST_DATASETS
do
 python train_net.py --num-gpus 1 --eval-only --config-file configs/<msrs_acdc_config>.yaml \
  DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS weights/<model>_msrs_acdc.pth \
  OUTPUT_DIR output/<model_name>_msrs_acdc/$DATASET
done

# --- Setting 4: OpenEarthMap-trained (bird's-eye-view, RGB + sim-to-real SAR) ---
for DATASET in $TEST_DATASETS
do
 python train_net.py --num-gpus 1 --eval-only --config-file configs/<openearthmap_config>.yaml \
  DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS weights/<model>_openearthmap.pth \
  OUTPUT_DIR output/<model_name>_openearthmap/$DATASET
done

# Combine results
python p2p/evaluation/pixel2perspective_evaluation.py \
  --model_outputs output/<model_name>_coco output/<model_name>_indraeye output/<model_name>_msrs_acdc output/<model_name>_openearthmap

# Run evaluation with:
# nohup bash eval.sh > eval.log &
# tail -f eval.log
