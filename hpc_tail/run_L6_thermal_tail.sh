#!/bin/bash
#SBATCH -A nano                  # Project/account
#SBATCH -p ulow                  # Partition
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52       # the (T, n) GCG jobs run in parallel inside the job
#SBATCH --mem=64GB
#SBATCH --time=4-00:00:00        # resumable: finished (T, n) jobs are reused from job_cache
#SBATCH --output=logs/L6_thermal_%j.log
#SBATCH --error=logs/L6_thermal_%j.err
#SBATCH -J L6_thermal

# ═══════════════════════════════════════════════════════════════════
#  Fig. 3(a) on the Fig. 3(b) grid (L = 6, U/t = 10, full Hamiltonian).
#  12 temperatures k_B T/t = 0 .. 0.2894737, the same points as L = 8.
#  Same GCG settings as the old Fig. 3(a): 10 restarts, 300 iterations, q = r..r+5.
#  Writes the decompositions to job_cache/, needed by the warm-start tail rerun.
#  Then: sbatch hpc_tail/run_L6_tail_warm.sh
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

WORKERS=${SLURM_CPUS_PER_TASK:-52}
echo " L=6 thermal on $HOSTNAME (job $SLURM_JOB_ID), workers=$WORKERS, started: $(date)"

python main.py --thermal --L 6 --Ne 6 --t 1.0 --U 10.0 \
  --T-min 0 --T-max 0.2894737 --T-steps 12 --min-n 2 \
  --full-hamiltonian \
  --thermal-solver gcg --n-restarts 10 --max-iter 300 --k-max-extra 5 \
  --workers $WORKERS --use-pkl-cache \
  --output-prefix L6_thermal_T0to0.3

status=$?
echo " finished: $(date)   exit=$status"
exit $status

