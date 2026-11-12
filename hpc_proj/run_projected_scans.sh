#!/bin/bash
#SBATCH -A nano                  # Project/account
#SBATCH -p ulow                  # Partition
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # the T = 0 scans are serial: these cores go to BLAS
#SBATCH --mem=128GB              # L=8 full space: about 10-15 GB peak for the dense diagonalisation
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/proj_scan_%A_%a.log
#SBATCH --error=logs/proj_scan_%A_%a.err
#SBATCH -J proj_scan
#SBATCH --array=1-8

# ═══════════════════════════════════════════════════════════════════
#  T = 0 scans of the paper with the doublons on the n detection sites removed.
#  Each task has the settings of the published PKL plus --project-doublons;
#  the results are saved next to the old ones with a _proj1 tag.
#
#    1  Fig. 1(a)  U/t, L=6, Sz=0          5  Fig. 4(a)  V/t, L=6
#    2  Fig. 1(b)  U/t, L=8                6  Fig. 4(b)  V/t, L=8
#    3  Fig. 2(a)  excited singlet, L=6    7  Fig. 5     U/t, L=8, Ne=6, Sz=0
#    4  Fig. 2(b)  excited singlet, L=8    8  Fig. 6     Ne scan, L=8
#
#  Print one command without running it:
#    SLURM_ARRAY_TASK_ID=1 DRY_RUN=1 bash hpc_proj/run_projected_scans.sh
# ═══════════════════════════════════════════════════════════════════

if [ -z "$DRY_RUN" ]; then
    export TMPDIR=/unimore_home/kshahzad/tmp
    mkdir -p $TMPDIR logs
    module purge
    module load slurm
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate myenv
    NTHREADS=${SLURM_CPUS_PER_TASK:-8}
    export OMP_NUM_THREADS=$NTHREADS
    export OPENBLAS_NUM_THREADS=$NTHREADS
    export MKL_NUM_THREADS=$NTHREADS
fi

USCAN1="--u-scan --u-scan-min 1 --u-scan-max 20 --u-scan-steps 20 --t 1.0"
ES="--excited-singlet --singlet-k 16 --full-hamiltonian --t 0.1 --u-scan --u-scan-min 0.1 --u-scan-max 5.0 --u-scan-steps 21"
VSCAN="--v-scan --v-scan-min 0.0 --v-scan-max 12.0 --v-scan-steps 25 --U 10.0 --t 1.0 --full-hamiltonian"

case $SLURM_ARRAY_TASK_ID in
    1) ARGS="$USCAN1 --L 6 --Ne 6 --n-restarts 10 --max-iter 200 --k-max-extra 4 --output-prefix proj_fig1a_L6" ;;
    2) ARGS="$USCAN1 --L 8 --Ne 8 --full-hamiltonian --n-restarts 8 --max-iter 200 --k-max-extra 4 --output-prefix proj_fig1b_L8" ;;
    3) ARGS="$ES --L 6 --Ne 6 --n-restarts 10 --max-iter 300 --k-max-extra 5 --output-prefix proj_fig2a_L6" ;;
    4) ARGS="$ES --L 8 --Ne 8 --n-restarts 10 --max-iter 300 --k-max-extra 5 --output-prefix proj_fig2b_L8" ;;
    5) ARGS="$VSCAN --L 6 --Ne 6 --n-restarts 10 --max-iter 300 --k-max-extra 5 --output-prefix proj_fig4a_L6" ;;
    6) ARGS="$VSCAN --L 8 --Ne 8 --n-restarts 10 --max-iter 300 --k-max-extra 5 --output-prefix proj_fig4b_L8" ;;
    7) ARGS="$USCAN1 --L 8 --Ne 6 --n-restarts 10 --max-iter 300 --k-max-extra 4 --output-prefix proj_fig5_L8_Ne6" ;;
    8) ARGS="--n-scan --n-scan-values 3 4 5 6 7 8 --L 8 --U 10 --t 1.0 --n-restarts 10 --max-iter 300 --k-max-extra 2 --output-prefix proj_fig6_L8" ;;
    *) echo "unknown array index: $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

CMD="python main.py $ARGS --min-n 2 --mode uniform --thermal-solver gcg --use-pkl-cache --project-doublons"
if [ -n "$DRY_RUN" ]; then echo "$CMD"; exit 0; fi

echo "[$(date)] task $SLURM_ARRAY_TASK_ID on $(hostname): $CMD"
$CMD
echo "[$(date)] done"
