#!/bin/bash

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision
python Plot.py $1 $2
conda deactivate
