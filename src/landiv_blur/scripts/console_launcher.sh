#!/bin/bash

#SBATCH -c 64
#SBATCH --time=1:00:0
#SBATCH --mem=64000M

module load mamba
source activate landiv
srun python parallel_filter.py --source=/data/jliech/landiv/first_approach/Europe/landcover/reclass_GLC_FCS30_2015_utm32U.tif --output=/scratch/jliech/landiv/first_approach/Europe/output/lct_heterogeneity_diam_30000_utm32u.tif --scale=100 --diameter=30000 --truncate=3 --nbrcpu=64
