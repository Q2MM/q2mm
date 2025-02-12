#!/bin/bash

#$ -M mfarrugi@nd.edu
#$ -m a
#$ -pe smp 16         
#$ -q long            
#$ -N ho_fuerza


echo $SGE_TASK_ID

base_dir = /scratch365/mfarrugi/q2mm/rh-ho/localized_gbx_N_100/q_fuerza

# copy over schrodinger.ve
# copy correct egg into schrodinger.ve/bin or just reference the q2mm code?

source /scratch365/mfarrugi/q2mm/rh-ho/schrodinger.ve/bin/activate

for index in {1..10}
do
    echo $index

    /scratch365/mfarrugi/q2mm/q2mm/q2mm/loop.py loop-eig.in
    
    mkdir 500_tapered_HO/$index
    rm *.q2mm.*
    mv hybrid_opt_history.bin 500_tapered_HO/$index/
    mv mm3.hybrid.fld  500_tapered_HO/$index/
    mv mm3_*.fld 500_tapered_HO/$index/
    mv root.log  500_tapered_HO/$index/
    mv rh_qf_ho_end.* 500_tapered_HO/$index/
    scp mm3.old.fld mm3.fld
    
done

