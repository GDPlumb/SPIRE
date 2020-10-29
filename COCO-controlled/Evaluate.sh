
CUDA_VISIBLE_DEVICES=0 python Evaluate.py $1 $2 $3 $4 0,1,2 &
CUDA_VISIBLE_DEVICES=1 python Evaluate.py $1 $2 $3 $4 3,4,5 &

wait
