
source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision

CUDA_VISIBLE_DEVICES=0 python EvaluatePair.py $1 $2 $3 $4 0,1 &
CUDA_VISIBLE_DEVICES=1 python EvaluatePair.py $1 $2 $3 $4 2,3 &

wait

conda deactivate

