#!/bin/bash

#SBATCH -J multi-modal-gpu-test
#SBATCH -p gpu
#SBATCH -A r00939
#SBATCH -o ./logs/output_%j.txt
#SBATCH -e ./logs/error_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nbangal@iu.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --time=02:00:00
#SBATCH --mem=20GB

#Load any modules that your program needs
module load python/gpu

#Run your program
srun python train.py --data /N/project/SingleCell_Image/Nischal/Dilip/FINAL-FILES/phenotypes_final.csv --batch_size 128 --epochs 50 --ttv_split 0.85 0.10 0.05 --optim adam --optim_params 0.001 0.0001 --model uni_modal.GeneTransformer --workers 1