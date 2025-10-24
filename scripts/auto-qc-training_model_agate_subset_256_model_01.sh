#!/bin/sh

#SBATCH --job-name=automated-qc-Regressor # job name

#SBATCH --mem=240g        
#SBATCH --time=24:00:00          
#SBATCH -p a100-4,a100-8
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16    

#SBATCH --mail-type=begin       
#SBATCH --mail-type=end          
#SBATCH --mail-user=lundq163@umn.edu
#SBATCH -e logs/automated-qc-Regressor-%j.err
#SBATCH -o logs/automated-qc-Regressor-%j.out
#SBATCH -A feczk001


# 24GB+ GPU memory when using batch size 32
# could also try using --use-weighted-loss flag if needed

cd /users/1/lundq163/projects/automated-qc/src/training || exit

export PYTHONPATH=/users/1/lundq163/projects/automated-qc/src:$PYTHONPATH
export AUTO_QC_CACHE_DIR=/scratch.global/lundq163/auto_qc_model_00_cache/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/users/1/lundq163/projects/automated-qc/.venv/bin/python \
/users/1/lundq163/projects/automated-qc/src/training/training.py \
--model-save-location "/scratch.global/lundq163/auto_qc_model_01/model_01.pt" \
--plot-location "/users/1/lundq163/projects/automated-qc/doc/models/model_01/model_01.png" \
--folder "/scratch.global/lundq163/auto_qc_subset_256/" \
--csv-input-file "/users/1/lundq163/projects/automated-qc/data/anat_qc_t1w_t2w_subset_256.csv" \
--csv-output-file "/users/1/lundq163/projects/automated-qc/doc/models/model_01/model_01.csv" \
--tb-run-dir "/users/1/lundq163/projects/automated-qc/src/training/runs/" \
--split-strategy "stratified" \
--train-split 0.8 \
--model "Regressor" \
--lr 0.001 \
--scheduler "plateau" \
--batch-size 8 \
--epochs 100 \
--optimizer "Adam" \
--num-workers 12
