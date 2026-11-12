#!/bin/bash
#SBATCH -A nano                  # Project/account
#SBATCH -p ulow                  # Partition (ice4hpc / ulow)
#SBATCH -N 1                     # One node
#SBATCH --ntasks=1               # One task per array element
#SBATCH --cpus-per-task=8        # See NOTE ON CORES below
#SBATCH --mem=128GB              # See NOTE ON MEMORY below
#SBATCH --time=14-00:00:00       # Max runtime (14 days)
#SBATCH --output=logs/L6_Ut1to40_%A_%a.log
#SBATCH --error=logs/L6_Ut1to40_%A_%a.err
#SBATCH -J L6_Ut1to40            # Job name
#SBATCH --array=1-2              # 1 = excited singlet, 2 = ground state

# ═══════════════════════════════════════════════════════════════════
#  L=6 GME U-scans (T=0, full Hamiltonian), copy of slurm_L8_excited_singlet_scans.sh
#  on the SAME grid as the L=8 data of paper Fig. 2(b), so L=6 and L=8 match point by point:
#
#    array task 1 : first excited singlet, t=0.1 fixed, U = 0.1 .. 4.02  (17 points)
#    array task 2 : ground state,          t=0.1 fixed, U = 0.1 .. 4.02  (17 points)
#
#  U/t = 1, 3.45, 5.9, ... 37.75, 40.2: exactly the first 17 points of the L=8 grid
#  (U = 0.1 .. 5, 21 points), i.e. the points of the figures cut at U/t = 40.
#  Solver as the Fig. 2(b) run: 10 restarts, 300 iterations, k_max_extra 5, --singlet-k 16.
#
#  Each task covers ALL body orders n = 2..6 in a single run.
#  The first excited singlet is re-located at EVERY grid point by its <S^2>,
#  because its eigenstate index moves with U/t through level crossings.
#
#  NOTE ON CORES
#  -------------
#  run_U_dependence_gme is SERIAL — it takes no n_workers argument, so we give
#  BLAS the cores. main.py sets these with os.environ.setdefault, so exporting
#  them here wins.
#
#  NOTE ON MEMORY
#  --------------
#  L=6, Ne=6 full space is C(12,6) = 924 states: tiny. 128 GB is far more than
#  needed; kept as in the L=8 script.
# ═══════════════════════════════════════════════════════════════════

export TMPDIR=/unimore_home/kshahzad/tmp
mkdir -p $TMPDIR logs
module purge
module load slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv             # <-- env with numpy/scipy/openfermion/matplotlib/tqdm

# Serial scan: let BLAS use the allocated cores (see NOTE ON CORES).
NTHREADS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$NTHREADS
export OPENBLAS_NUM_THREADS=$NTHREADS
export MKL_NUM_THREADS=$NTHREADS
export NUMEXPR_NUM_THREADS=$NTHREADS

cd /unimore_home/kshahzad/hubbard_gme   # <-- repo path on the cluster

# ═══════════════════════════════════════════════════════════════════
#  EDIT THESE ONCE
# ═══════════════════════════════════════════════════════════════════
L=6                  # sites
NE=6                 # electrons (half filling = L)

# SECTOR=""                       # "" = Sz=0 sector
SECTOR="--full-hamiltonian"       # complete Hamiltonian (all Sz)

# U-scan settings: t fixed, U swept
USCAN_T=0.1
USCAN_UMIN=0.1       # NOT 0: at U=0 the excited manifold is degenerate and
USCAN_UMAX=4.02      # the singlet search raises RuntimeError
USCAN_STEPS=17

COMMON="--L $L --Ne $NE --mode uniform $SECTOR \
        --singlet-k 16 \
        --thermal-solver gcg --n-restarts 10 --k-max-extra 5 --max-iter 300 \
        --use-pkl-cache --output-dir outputs"
# ═══════════════════════════════════════════════════════════════════

echo "=========================================================="
echo " task $SLURM_ARRAY_TASK_ID on $HOSTNAME (job $SLURM_JOB_ID)"
echo " L=$L Ne=$NE  sector='$SECTOR'  BLAS threads=$NTHREADS"
echo " started: $(date)"
echo "=========================================================="

case $SLURM_ARRAY_TASK_ID in

  1)  # ---- U-scan: first excited singlet, t fixed ----
      echo "ES U-SCAN  t=$USCAN_T  U=$USCAN_UMIN..$USCAN_UMAX ($USCAN_STEPS points)"
      python main.py $COMMON --excited-singlet --t $USCAN_T \
        --u-scan --u-scan-min $USCAN_UMIN --u-scan-max $USCAN_UMAX \
        --u-scan-steps $USCAN_STEPS \
        --output-prefix ES_uscan_L${L}_t${USCAN_T}
      ;;

  2)  # ---- U-scan: ground state, t fixed ----
      echo "GS U-SCAN  t=$USCAN_T  U=$USCAN_UMIN..$USCAN_UMAX ($USCAN_STEPS points)"
      python main.py $COMMON --t $USCAN_T \
        --u-scan --u-scan-min $USCAN_UMIN --u-scan-max $USCAN_UMAX \
        --u-scan-steps $USCAN_STEPS \
        --output-prefix GS_uscan_L${L}_t${USCAN_T}
      ;;

  *)  echo "ERROR: unknown array index $SLURM_ARRAY_TASK_ID (expected 1 or 2)"
      exit 1
      ;;
esac

status=$?
echo "=========================================================="
echo " finished: $(date)   exit=$status"
echo "=========================================================="
exit $status
