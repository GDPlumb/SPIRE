#!/bin/bash

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision
python HPS_main.py $1
conda deactivate

find ./HPS -name model.pt | xargs rm
