#!/bin/bash

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision

CUDA_VISIBLE_DEVICES=0 python Evaluate.py $1 $2 $3 $4 0,1,2 &
CUDA_VISIBLE_DEVICES=1 python Evaluate.py $1 $2 $3 $4 3,4,5 &

wait

conda deactivate
