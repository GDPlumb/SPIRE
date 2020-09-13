
root='/home/gregory/Datasets/COCO'
year='2017'
main_class='bicycle'
spurious_class='person'

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision

python Plot.py $root $year $main_class $spurious_class

conda deactivate
