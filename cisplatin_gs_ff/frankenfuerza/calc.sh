#!/bin/bash

#$ -M mfarrugi@nd.edu
#$ -m a
#$ -pe smp 8         
#$ -q long            
#$ -N cisplatin_eig

source /scratch365/mfarrugi/q2mm/rh-ho/schrodinger.ve/bin/activate

export SCHRODINGER_ALLOW_UNSAFE_MULTIPROCESSING=1

../../q2mm/q2mm/q2mm/calculate.py -mb cisplatin_opt_freq_m06.mae
