#!/bin/sh

#SBATCH --partition=small

#SBATCH --time=0-24:00:00

# set number of GPUs
#SBATCH --gres=gpu:1

# set the number of nodes
#SBATCH --nodes=1

module load cuda/9.0
module load python3/3.6.3

export STUDENT_ID=axa_35

mkdir -p /jmain01/home/JAD003/sxr01/axa35-sxr01/${STUDENT_ID}

export TMPDIR=/jmain01/home/JAD003/sxr01/axa35-sxr01/${STUDENT_ID}/
export TMP=/jmain01/home/JAD003/sxr01/axa35-sxr01/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=/jmain01/home/JAD003/sxr01/axa35-sxr01/HowToTrainYourMAMLPytorch_research_edition/datasets
#export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:

source /jmain01/home/JAD003/sxr01/axa35-sxr01/miniconda3/bin/activate mlp
cd ..
python train_continual_learning_few_shot_system.py --name_of_args_json_file experiment_config/mini-imagenet_embedding_variant_standard_5_way_1_vgg-fine-tune-scratch_shot__True_3_3_LSLR_conditioned_0.json --gpu_to_use 0
