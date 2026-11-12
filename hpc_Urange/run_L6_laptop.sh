#!/bin/bash
# L = 6 part of run_U001to40.sh (tasks 1-4), run on the laptop: ground state and first excited singlet,
# grid A: U/t = 0.01 ... 1 (12 points), grid B: U/t = 1 ... 40 (40 points). Same settings as the HPC script.
cd /home/anjum/PhDWork/hubbard_gme
PY=~/miniconda3/envs/qchem/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
run() {   # state grid
  local STATE=$1 GRID=$2 ES="" UMIN UMAX USTEPS SK
  if [ "$GRID" = "A" ]; then UMIN=0.01; UMAX=1.0; USTEPS=12; SK=32; else UMIN=1.0; UMAX=40.0; USTEPS=40; SK=16; fi
  [ "$STATE" = "es" ] && ES="--excited-singlet --singlet-k $SK"
  $PY main.py --L 6 --Ne 6 --t 1.0 --mode uniform --full-hamiltonian $ES \
    --thermal-solver gcg --n-restarts 10 --max-iter 300 --k-max-extra 5 \
    --u-scan --u-scan-min $UMIN --u-scan-max $UMAX --u-scan-steps $USTEPS \
    --use-pkl-cache --output-dir outputs --output-prefix Urange_${STATE}_L6_grid${GRID} \
    > logs_Urange/L6_${STATE}_grid${GRID}.log 2>&1
  echo "$(date +%H:%M) L6 $STATE grid $GRID exit=$?"
}
run gs A & run gs B & run es A & run es B &
wait
echo "$(date +%H:%M) ALL DONE"
