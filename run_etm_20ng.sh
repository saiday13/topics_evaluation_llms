#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --partition=informatik-mind
#SBATCH --ntasks=5
#SBATCH --job-name=master-project
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=20G
#SBATCH -o /scratch/yusupova/server_dump/output.%j.%N.log
#SBATCH -e /scratch/yusupova/server_dump/error.%j.%N.log

module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate etm	   
python3 soup_nuts/models/etm/main.py \
    --mode train \
    --data_path soup_nuts/models/etm/data/20ng \
    --output_dir results/etm-20ng \
    --num_topics 50 \
    --train_embeddings 1 \
    --epochs 1000 \
    --tc 1

conda deactivate

