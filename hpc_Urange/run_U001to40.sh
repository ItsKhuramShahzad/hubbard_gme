#!/bin/bash
#SBATCH -A nano                  # Project/account
#SBATCH -p ulow                  # Partition
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/Urange_%A_%a.log
#SBATCH --error=logs/Urange_%A_%a.err
#SBATCH -J U001to40
#SBATCH --array=1-8

# ═══════════════════════════════════════════════════════════════════
#  U/t = 0.01 ... 40 for L = 6 and L = 8, half filling (Ne = L), t = 1, V = 0,
#  full state (no doublon projection), full Hamiltonian (all Sz).
#  Two grids per case, merged afterwards for the plot:
#     grid A : U/t = 0.01 ... 1   12 points (step 0.09)
#     grid B : U/t = 1    ... 40  40 points (step 1)
#
#    task  L  state            grid
#     1    6  ground state      A
#     2    6  ground state      B
#     3    6  excited singlet   A
#     4    6  excited singlet   B
#     5    8  ground state      A
#     6    8  ground state      B
#     7    8  excited singlet   A
#     8    8  excited singlet   B
#
#  Solver for all tasks: GCG, 10 restarts, 300 iterations, k_max_extra 5
#  (the Fig. 2 settings). Excited singlet: re-located at every point by <S^2>;
#  --singlet-k 32 on grid A because near U = 0 many levels lie close together.
#  The scans are serial: BLAS gets the cores.
# ═══════════════════════════════════════════════════════════════════

export TMPDIR=/unimore_home/kshahzad/tmp
mkdir -p $TMPDIR logs
module purge
module load slurm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

NTHREADS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$NTHREADS OPENBLAS_NUM_THREADS=$NTHREADS MKL_NUM_THREADS=$NTHREADS NUMEXPR_NUM_THREADS=$NTHREADS

cd /unimore_home/kshahzad/hubbard_gme

case $SLURM_ARRAY_TASK_ID in
  1) L=6; STATE=gs; GRID=A ;;
  2) L=6; STATE=gs; GRID=B ;;
  3) L=6; STATE=es; GRID=A ;;
  4) L=6; STATE=es; GRID=B ;;
  5) L=8; STATE=gs; GRID=A ;;
  6) L=8; STATE=gs; GRID=B ;;
  7) L=8; STATE=es; GRID=A ;;
  8) L=8; STATE=es; GRID=B ;;
  *) echo "ERROR: unknown array index $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

if [ "$GRID" = "A" ]; then UMIN=0.01; UMAX=1.0;  USTEPS=12; SK=32
else                       UMIN=1.0;  UMAX=40.0; USTEPS=40; SK=16
fi
ES_FLAGS=""
[ "$STATE" = "es" ] && ES_FLAGS="--excited-singlet --singlet-k $SK"

echo "=========================================================="
echo " task $SLURM_ARRAY_TASK_ID on $HOSTNAME (job $SLURM_JOB_ID)"
echo " L=$L Ne=$L state=$STATE grid=$GRID  U/t=$UMIN..$UMAX ($USTEPS points)"
echo " started: $(date)"
echo "=========================================================="

python main.py --L $L --Ne $L --t 1.0 --mode uniform --full-hamiltonian $ES_FLAGS \
  --thermal-solver gcg --n-restarts 10 --max-iter 300 --k-max-extra 5 \
  --u-scan --u-scan-min $UMIN --u-scan-max $UMAX --u-scan-steps $USTEPS \
  --use-pkl-cache --output-dir outputs \
  --output-prefix Urange_${STATE}_L${L}_grid${GRID}
status=$?

echo "=========================================================="
echo " finished: $(date)   exit=$status"
echo "=========================================================="
exit $status
