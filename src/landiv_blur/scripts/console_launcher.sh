#!/bin/bash

#SBATCH -c 32
#SBATCH --time=4:00:0
#SBATCH --mem=120000M

module load mamba
source activate landiv
srun python parallel_filter.py --source=/shares/niklaus.ieu.uzh/first_approach/Europe/landcover/reclass_GLC_FCS30_2015_utm32U.tif --output=/shares/niklaus.ieu.uzh/first_approach/Europe/output/lct_heterogeneity_diam_30000_utm32u.tif --scale=30 --diameter=30000 --truncate=3 --nbrcpu=32 --bwidth=2500 --bheight=2500
