
source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision

python EvaluatePair.py $1 $2 $3 $4

conda deactivate

