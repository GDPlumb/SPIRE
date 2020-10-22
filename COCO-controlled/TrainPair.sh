
source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision

python TrainPair.py $1 $2 $3 $4

conda deactivate
