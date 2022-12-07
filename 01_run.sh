#!/bin/bash
#PBS -d .
# -l nodes=2:ppn=16
#PBS -N two_fluids_test_01
#PBS -j oe
#PBS -l walltime=24:00:00
####export DISPLAY=':0.0'
# mpirun /share/anaconda3/bin/python3 /home/panov/calc/python/main_test_rk45_inf.py
mpirun /share/anaconda3/bin/python3 /home/panov/calc/python/main_test_rk45.py
#runjob -pr openmpi -np 8 uname -n
#mpirun hostname
###mpirun -np 8 -machinefile $PBS_NODEFILE uname -n
#mpirun uname -n

