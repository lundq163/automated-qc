#!/bin/sh

#SBATCH --job-name=automated-qc-ResNet # job name

#SBATCH --mem=180g        
#SBATCH --time=1:00:00          
#SBATCH -p a100-4,a100-8
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=6      

#SBATCH --mail-type=begin       
#SBATCH --mail-type=end          
#SBATCH --mail-user=lundq163@umn.edu
#SBATCH -e output_logs/automated-qc-ResNet-%j.err
#SBATCH -o output_logs/automated-qc-ResNet-%j.out
#SBATCH -A feczk001

cd /users/1/lundq163/projects/automated-qc/src/training || exit
/users/1/lundq163/projects/automated-qc/.venv/bin/python \
/users/1/lundq163/projects/automated-qc/src/training/training.py \
--csv-input-file "/users/1/lundq163/projects/automated-qc/data/anat_qc_t1w_t2w.csv" \
--batch-size 1 \
--num-workers 1 \
--epochs 1 \
--model-save-location "/users/1/lundq163/projects/automated-qc/models/model_00.pt" \
--plot-location "/users/1/lundq163/projects/automated-qc/doc/models/model_00/model_test.png" \
--folder "/home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag/05-training_ready/" \
--csv-output-file "/users/1/lundq163/projects/automated-qc/doc/models/model_00/model_test.csv" \
--use-train-validation-cols
