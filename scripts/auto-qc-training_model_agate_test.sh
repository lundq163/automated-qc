#!/bin/sh

#SBATCH --job-name=automated-qc-Regressor # job name

#SBATCH --mem=180g        
#SBATCH --time=1:00:00          
#SBATCH -p a100-4,a100-8
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=6      

#SBATCH --mail-type=begin       
#SBATCH --mail-type=end          
#SBATCH --mail-user=lundq163@umn.edu
#SBATCH -e output_logs/automated-qc-Regressor-%j.err
#SBATCH -o output_logs/automated-qc-Regressor-%j.out
#SBATCH -A feczk001

cd /users/1/lundq163/projects/automated-qc/src/training || exit
/users/1/lundq163/projects/automated-qc/.venv/bin/python \
/users/1/lundq163/projects/automated-qc/src/training/training.py \
--csv-input-file "/users/1/lundq163/projects/automated-qc/data/anat_qc_t1w_t2w_test_subset.csv" \
--batch-size 1 \
--num-workers 1 \
--epochs 1 \
--model-save-location "/users/1/lundq163/projects/automated-qc/models/model_test.pt" \
--plot-location "/users/1/lundq163/projects/automated-qc/doc/models/model_test/model_test.png" \
--folder "/scratch.global/lundq163/auto_qc_test/" \
--csv-output-file "/users/1/lundq163/projects/automated-qc/doc/models/model_test/model_test.csv" \
--use-train-validation-cols
