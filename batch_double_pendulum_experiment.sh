#!/bin/bash
#SBATCH -J double_pendulum_experiment
#SBATCH -o /gpfs/scratch/pr63so/ge49rev3/shnn-paper/double-pendulum/out.txt
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=mpp3
#SBATCH --partition=mpp3_batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
# 256 is the maximum reasonable value for CooLMUC-3
#SBATCH --mail-type=end
#SBATCH --mail-user=rahma@in.tum.de
#SBATCH --export=NONE
#SBATCH --time=48:00:00

module load slurm_setup

echo $SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /dss/dsshome1/0B/ge49rev3/.conda_init
conda activate s-hnn

python /dss/dsshome1/0B/ge49rev3/shnn-repo/src/double_pendulum_experiment.py --repeat 10 --save-dir /gpfs/scratch/pr63so/ge49rev3/shnn-paper/double-pendulum --resample-duplicates
