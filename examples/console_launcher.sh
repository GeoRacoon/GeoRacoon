#!/bin/bash

#SBATCH -c 32
#SBATCH --time=4:00:0
#SBATCH --mem=120000M

module load mamba
source activate landiv
landiv --source=/shares/niklaus.ieu.uzh/first_approach/Europe/landcover/reclass_GLC_FCS30_2015_utm32U.tif --output=/shares/niklaus.ieu.uzh/first_approach/Europe/output/lct_heterogeneity_utm32u.tif --scale=30 --diameter=30000 --truncate=3 --nbrcpu=32 --bwidth=4000 --bheight=4000 --entropy_ubyte=1 --blur_int=1
