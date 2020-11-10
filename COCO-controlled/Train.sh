#!/bin/bash

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision

CUDA_VISIBLE_DEVICES=0 python Train.py $1 $2 $3 $4 0,2 &
CUDA_VISIBLE_DEVICES=1 python Train.py $1 $2 $3 $4 1,3 &

wait

conda deactivate
