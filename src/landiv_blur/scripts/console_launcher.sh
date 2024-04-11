#!/bin/bash

#SBATCH -c 64
#SBATCH --time=5:00:0
#SBATCH --mem=128000M

module load mamba
source activate landiv
srun python parallel_filter.py --source=/shares/niklaus.ieu.uzh/first_approach/Europe/landcover/reclass_GLC_FCS30_2015_utm32U.tif --output=/shares/niklaus.ieu.uzh/first_approach/Europe/output/lct_heterogeneity_diam_30000_utm32u.tif --scale=30 --diameter=30000 --truncate=3 --nbrcpu=64 --bwidth=2220 --bheight=2960
