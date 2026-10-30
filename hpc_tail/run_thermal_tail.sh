#!/bin/bash
#SBATCH -A nano                  # Project/account
#SBATCH -p ulow                  # Partition
#SBATCH -N 1                     # One node per n
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52       # whole node: the GCG tasks of one n run in parallel inside the job
#SBATCH --mem=64GB               # rho_s is rebuilt from the job records; the 18 GB RDM bank is not loaded
#SBATCH --time=6-00:00:00        # resumable: finished tasks are skipped when the job is started again
#SBATCH --output=logs/tail_%A_n%a.log
#SBATCH --error=logs/tail_%A_n%a.err
#SBATCH -J L8_tail
#SBATCH --array=5-8              # array index = n

# ═══════════════════════════════════════════════════════════════════
#  Rerun of the Fig. 3(b) tail (L = 8, U/t = 10, k_B T/t >= 0.15).
#  Production run: 8 restarts, 200 iterations, q = r..r+4 (every point hit the iteration limit).
#
#  n = 5, 6, 7 : fresh starts (20 restarts, 1000 iterations, q = r..r+10)
#                + warm starts from the stored decompositions (2000 iterations, q = r+10)
#  n = 8       : warm starts for every tail temperature, fresh starts only at k_B T/t = 0.5
#                (T index 19) for q = r, r+5, r+10 as a check
#
#  Results: results/rdm_pkl_results/thermal_runs/tail_rerun_L8/n<n>/   (one small PKL per task)
#  Then:    python hpc_tail/merge_thermal_tail.py
#  If fewer cores are free, lower --cpus-per-task (e.g. 26): the job starts sooner and runs longer.
# ═══════════════════════════════════════════════════════════════════

export TMPDIR=/unimore_home/kshahzad/tmp
mkdir -p $TMPDIR
module purge
module load slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# many single-threaded GCG tasks in parallel: 1 BLAS thread each
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /unimore_home/kshahzad/hubbard_gme
mkdir -p logs

N=$SLURM_ARRAY_TASK_ID
WORKERS=${SLURM_CPUS_PER_TASK:-52}
COMMON="--n $N --T-min 0.15 --restarts 20 --max-iter 1000 --warm-iter 2000 --k-extra 10 --workers $WORKERS"

echo "=========================================================="
echo " L=8 thermal tail, n=$N on $HOSTNAME (job $SLURM_JOB_ID), workers=$WORKERS"
echo " started: $(date)"
echo "=========================================================="

if [ "$N" -le 7 ]; then
    python hpc_tail/thermal_tail_hpc.py $COMMON --mode both
else
    python hpc_tail/thermal_tail_hpc.py $COMMON --mode both --fresh-T 19 --fresh-q 0 5 10
fi

status=$?
echo " finished: $(date)   exit=$status"
exit $status
