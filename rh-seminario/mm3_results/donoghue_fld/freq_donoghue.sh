#!/bin/bash -l
# NOTE the -l flag!
#
#SBATCH -J donoghue_freq
#SBATCH -o donoghue_freq.output
#SBATCH -e donoghue_freq.err
# Default in slurm
#SBATCH --mail-user mikaela.farrugia@astrazeneca.com
#SBATCH --mail-type=ALL
# Request 5 hours run time
#SBATCH -t 5:0:0
#SBATCH -p core 
#SBATCH -N 4
#
 
module load schrodinger

#home_dir = '/home/kzdq760/rh_hyd_enamide/donoghue_fld'

for index in {1..9}
do
    sed "s/xxxx/$index/g" freq.com  >  donoghue_freq_$index.com

    $SCHRODINGER/bmin donoghue_freq_$index
done

