#!/bin/bash
# qsub -q common20 /home/panov/calc/ver_sign_mu_12_5_types/01_run.sh
# qsub -l nodes=8:ppn=12 -q regular12 /home/panov/calc/ver_sign_mu_12_5_types/01_run.sh
qsub -l nodes=10:ppn=12 -q regular12 /home/panov/calc/python/01_run.sh
# qsub -l nodes=4:ppn=20 -q common20 /home/panov/calc/python/01_run.sh
#qsub -q regular12 /home/panov/calc/ver_sign_mu_12_5_types/01_run.qsub