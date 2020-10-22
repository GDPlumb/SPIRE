
source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision

python SetupPair.py $1 $2

conda deactivate
