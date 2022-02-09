#!/bin/bash

#SBATCH -A sio134
#SBATCH --job-name="multichannel_vae"
#SBATCH --output="multichannel_vae.%j.%N.out"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=40
#SBATCH --export=ALL
#SBATCH -t 36:00:00
#SBATCH --mem=374G
#SBATCH --no-requeue

module purge
module load gpu
module load slurm
module load openmpi
module load amber

cd ../
source activate GPU2
python3 train_fully_conv_multichannel.py --id 3