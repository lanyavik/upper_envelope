#!/bin/sh
#
#SBATCH --verbose
#SBATCH -p aquila
#SBATCH --job-name=ddpgbuffer_n0.3
#SBATCH --output=test_job_%j.out
#SBATCH --error=test_job_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --mail-type=END
#SBATCH --mail-user=xc1305@nyu.edu


module load anaconda3 cuda/9.0 glfw/3.3 gcc/7.3
source activate rl
python main_ddpg_run.py --env_set Hopper-v2 --seed 0 --expl_noise 0.5 --buffer_size 500000

