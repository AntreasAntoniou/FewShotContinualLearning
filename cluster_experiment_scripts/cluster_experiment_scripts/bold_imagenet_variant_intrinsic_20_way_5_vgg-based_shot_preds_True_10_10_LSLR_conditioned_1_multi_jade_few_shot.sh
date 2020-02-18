#!/bin/sh

#SBATCH --partition=big

#SBATCH --time=0-24:00:00

# set number of GPUs
#SBATCH --gres=gpu:4

# set the number of nodes
#SBATCH --nodes=1

module load cuda/9.0
module load python3/3.6.3

export STUDENT_ID=axa_35

mkdir -p /jmain01/home/JAD003/sxr01/axa35-sxr01/${STUDENT_ID}

export TMPDIR=/jmain01/home/JAD003/sxr01/axa35-sxr01/${STUDENT_ID}/
export TMP=/jmain01/home/JAD003/sxr01/axa35-sxr01/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
#export DATASET_DIR=${TMP}/datasets/
export DATASET_DIR=/jmain01/home/JAD003/sxr01/axa35-sxr01/HowToTrainYourMAMLPytorch_research_edition/datasets
# Activate the relevant virtual environment:

source /jmain01/home/JAD003/sxr01/axa35-sxr01/miniconda3/bin/activate mlp
cd ..
python train_continual_learning_few_shot_system.py --name_of_args_json_file experiment_config/bold_imagenet_variant_intrinsic_20_way_5_vgg-based_shot_preds_True_10_10_LSLR_conditioned_1.json --gpu_to_use 0
