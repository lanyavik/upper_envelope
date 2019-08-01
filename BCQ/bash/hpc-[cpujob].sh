#!/bin/sh
#
#SBATCH --verbose
#SBATCH -p aquila
#SBATCH --job-name=500k_s1
#SBATCH --output=test_job_%j.out
#SBATCH --error=test_job_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --mem=16GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xc1305@nyu.edu



module load anaconda3 cuda/9.0 glfw/3.3 gcc/7.3 mesa/19.0.5 llvm/7.0.1
source activate rl

python main_ue_train.py --seed 1 --buffer_size 500K
