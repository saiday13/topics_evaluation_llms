#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=informatik-mind
#SBATCH --ntasks=5
#SBATCH --job-name=master-project
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=20G
#SBATCH -o /scratch/yusupova/server_dump/output.%j.%N.log
#SBATCH -e /scratch/yusupova/server_dump/error.%j.%N.log

module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate dvae
python3 soup_nuts/models/dvae/main.py \
    --input_dir data/wikitext/processed_wiki \
    --output_dir results/dvae-wiki \
    --eval_path train.dtm.npz \
    --num_topics 50

conda deactivate
