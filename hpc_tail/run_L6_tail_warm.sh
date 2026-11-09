#!/bin/bash
#SBATCH -A nano
#SBATCH -p ulow
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --mem=64GB
#SBATCH --time=4-00:00:00        # resumable: finished tasks are skipped
#SBATCH --output=logs/L6_tail_%A_n%a.log
#SBATCH --error=logs/L6_tail_%A_n%a.err
#SBATCH -J L6_tail
#SBATCH --array=4-6              # array index = n

# ═══════════════════════════════════════════════════════════════════
#  Warm-start rerun of the Fig. 3(a) tail (L = 6, k_B T/t >= 0.15).
#  One start per stored q, enlarged to q = r+10.
#  n = 4, 5 : 2000 iterations      n = 6 : 4000 iterations
#  Results: results/rdm_pkl_results/thermal_runs/tail_rerun_L6/n<n>/
#  If a tail is still left: rerun with --mode fresh (20 restarts, 1000 iterations).
# ═══════════════════════════════════════════════════════════════════

export TMPDIR=/unimore_home/kshahzad/tmp
mkdir -p $TMPDIR
module purge
module load slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /unimore_home/kshahzad/hubbard_gme
mkdir -p logs

RUNS=results/rdm_pkl_results/thermal_runs
# the production PKL of step 1: exactly one file must match
PKLS=( $(cd $RUNS && ls thermal_v1_L6_Ne6_U10_t1_*full_restarts10_iter300_kmax5_*_N12-*diag0.pkl 2>/dev/null) )
if [ ${#PKLS[@]} -ne 1 ]; then
    echo "expected one step-1 PKL in $RUNS, found ${#PKLS[@]}: ${PKLS[*]}"; exit 1
fi
PKL=${PKLS[0]}

N=$SLURM_ARRAY_TASK_ID
WORKERS=${SLURM_CPUS_PER_TASK:-52}
if [ "$N" -eq 6 ]; then WARM=4000; else WARM=2000; fi

echo " L=6 tail (warm), n=$N on $HOSTNAME (job $SLURM_JOB_ID), warm-iter=$WARM"
echo " PKL: $PKL"
echo " started: $(date)"

python hpc_tail/thermal_tail_hpc.py --n $N --mode warm \
  --T-min 0.15 --k-extra 10 --warm-iter $WARM --workers $WORKERS \
  --pkl "$PKL" --out $RUNS/tail_rerun_L6

status=$?
echo " finished: $(date)   exit=$status"
exit $status

