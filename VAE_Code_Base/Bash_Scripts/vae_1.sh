#!/bin/bash

#SBATCH -A sio134
#SBATCH --job-name="Singlechannel_vae"
#SBATCH --output="outputs/Singlechannel_vae.%j.%N.out"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=40
#SBATCH --export=ALL
#SBATCH -t 48:00:00
#SBATCH --mem=374G
#SBATCH --no-requeue

module purge
module load gpu
module load slurm
module load openmpi
module load amber

cd ../
source activate GPU2
python3 train_fully_conv.py --id 1