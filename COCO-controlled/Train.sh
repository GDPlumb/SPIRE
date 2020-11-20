#!/bin/bash

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision

CUDA_VISIBLE_DEVICES=0 python Train.py $1 $2 $3 $4 0 &
CUDA_VISIBLE_DEVICES=1 python Train.py $1 $2 $3 $4 1 &
CUDA_VISIBLE_DEVICES=2 python Train.py $1 $2 $3 $4 2 &
#CUDA_VISIBLE_DEVICES=3 python Train.py $1 $2 $3 $4 3 &

wait

conda deactivate
