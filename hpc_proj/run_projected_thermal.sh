#!/bin/bash
#SBATCH -A nano                  # Project/account
#SBATCH -p ulow                  # Partition
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52       # one worker per (T, n) job and per RDM cluster
#SBATCH --mem=256GB              # L=8: the RDM bank alone is about 18 GB and is shared with the workers
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/proj_thermal_%A_%a.log
#SBATCH --error=logs/proj_thermal_%A_%a.err
#SBATCH -J proj_thermal
#SBATCH --array=1-2

# ═══════════════════════════════════════════════════════════════════
#  Fig. 3 with the doublons on the n detection sites removed, on the 12 temperatures of the published figure.
#    1  L=6   10 restarts, 300 iterations, k_max_extra 5
#    2  L=8    8 restarts, 200 iterations, k_max_extra 4
#  Resumable: finished (T, n) jobs are reused from job_cache when the job is started again.
#
#  Print one command without running it:
#    SLURM_ARRAY_TASK_ID=1 DRY_RUN=1 bash hpc_proj/run_projected_thermal.sh
# ═══════════════════════════════════════════════════════════════════

if [ -z "$DRY_RUN" ]; then
    export TMPDIR=/unimore_home/kshahzad/tmp
    mkdir -p $TMPDIR logs
    module purge
    module load slurm
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate myenv
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
fi

WORKERS=${SLURM_CPUS_PER_TASK:-8}
case $SLURM_ARRAY_TASK_ID in
    1) ARGS="--L 6 --Ne 6 --n-restarts 10 --max-iter 300 --k-max-extra 5 --output-prefix proj_fig3a_L6" ;;
    2) ARGS="--L 8 --Ne 8 --n-restarts 8 --max-iter 200 --k-max-extra 4 --output-prefix proj_fig3b_L8" ;;
    *) echo "unknown array index: $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

CMD="python main.py --thermal $ARGS --t 1.0 --U 10.0 --T-min 0 --T-max 0.2894737 --T-steps 12 --min-n 2 --full-hamiltonian --thermal-solver gcg --workers $WORKERS --use-pkl-cache --project-doublons"
if [ -n "$DRY_RUN" ]; then echo "$CMD"; exit 0; fi

echo "[$(date)] task $SLURM_ARRAY_TASK_ID on $(hostname): $CMD"
$CMD
echo "[$(date)] done"
