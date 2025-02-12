#!/bin/bash

#$ -M mfarrugi@nd.edu
#$ -m a
#$ -pe smp 8         
#$ -q long            
#$ -N cisplatin_q

source /scratch365/mfarrugi/q2mm/rh-ho/schrodinger.ve/bin/activate

export SCHRODINGER_ALLOW_UNSAFE_MULTIPROCESSING=1

../../q2mm/q2mm/q2mm/loop.py loop-q.in
