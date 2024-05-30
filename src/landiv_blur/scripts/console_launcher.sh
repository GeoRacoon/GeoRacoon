#!/bin/bash

#SBATCH -c 32
#SBATCH --time=4:00:0
#SBATCH --mem=120000M

# module load mamba
# source activate landiv
# srun python parallel_filter.py --source=/shares/niklaus.ieu.uzh/first_approach/Europe/landcover/reclass_GLC_FCS30_2015_utm32U.tif --output=/shares/niklaus.ieu.uzh/first_approach/Europe/output/lct_heterogeneity_utm32u.tif --scale=30 --diameter=30000 --truncate=3 --nbrcpu=32 --bwidth=2500 --bheight=2500 --entropy_ubyte=1 --blur_int=1
python parallel_filter.py --source=../../../data/ch.tif --output=../../../results/ch_test.tif --scale=100 --diameter=10000 --truncate=3 --nbrcpu=8 --bwidth=500 --bheight=550 --entropy_ubyte=1 --blur_int=1
