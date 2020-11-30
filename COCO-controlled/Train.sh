#!/bin/bash

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision

CUDA_VISIBLE_DEVICES=0 python Train.py $1 $2 $3 $4 0,4 &
CUDA_VISIBLE_DEVICES=1 python Train.py $1 $2 $3 $4 1,5 &
CUDA_VISIBLE_DEVICES=2 python Train.py $1 $2 $3 $4 2,6 &
CUDA_VISIBLE_DEVICES=3 python Train.py $1 $2 $3 $4 3,7 &

wait

conda deactivate
