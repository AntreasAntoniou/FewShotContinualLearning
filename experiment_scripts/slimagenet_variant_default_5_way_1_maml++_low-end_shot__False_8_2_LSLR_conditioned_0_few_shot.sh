#!/bin/sh

export GPU_ID=$1

echo $GPU_ID

cd ..
export DATASET_DIR="datasets/"
export CUDA_VISIBLE_DEVICES=$GPU_ID
# Activate the relevant virtual environment:
python train_continual_learning_few_shot_system.py --name_of_args_json_file experiment_config/slimagenet_variant_default_5_way_1_maml++_low-end_shot__False_8_2_LSLR_conditioned_0.json --gpu_to_use $GPU_ID