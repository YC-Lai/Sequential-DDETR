#!/bin/bash
#SBATCH --job-name="Sequential-DETR"
#SBATCH --partition=v100-32g
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00
#SBATCH --chdir=.
#SBATCH --output=cout.txt
#SBATCH --error=cerr.txt

# sbatch_pre.sh

module load gcc/7.5.0 cuda/11.1 python/3.8.10-gpu-cuda-11.1 
echo ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
#cd /home/b05501024/Sequential-DDETR/models/msda
#sh ./make.sh
#export NCCL_P2P_DISABLE=1
#CUDA_VISIBLE_DEVICES=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
echo $CUDA_VISIBLE_DEVICES
GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 ./configs/r50_deformable_detr.sh --masks --no_aux_loss --num_workers 8 --num_frames 12
# GPUS_PER_NODE=3 ./tools/run_dist_slurm.sh v100-32g sequential_ddetr 3 ./configs/r50_deformable_detr.sh --masks --no_aux_loss --num_workers 4 --cache_mode

# sbatch_post.sh
