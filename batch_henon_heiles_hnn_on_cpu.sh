#!/bin/bash
#SBATCH -J henon_heiles_hnn_on_cpu_experiment
#SBATCH -o /dss/dsshome1/0B/ge49rev3/shnn-repo/henon-heiles-hnn-on-cpu/out.txt
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
#SBATCH --time=10:00:00

module load slurm_setup

echo $SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /dss/dsshome1/0B/ge49rev3/.conda_init
conda activate s-hnn

python /dss/dsshome1/0B/ge49rev3/shnn-repo/src/henon_heiles_hnn_on_cpu_experiment.py --save-dir /dss/dsshome1/0B/ge49rev3/shnn-repo/henon-heiles-hnn-on-cpu --device cpu --batch-size 2048 --total-steps 180000
