#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=32000M

module load mamba
source activate landiv
srun python parallel_filter.py --source=/data/jliech/landiv/first_approach/Europe/landcover/reclass_GLC_FCS30_2015_utm32U.tif --output=/data/jliech/landiv/first_approach/Europe/output/lct_heterogeneity_diam_1000_utm32u.tif --scale=100 --diameter=1000 --truncate=3
