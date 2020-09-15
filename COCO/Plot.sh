
root='/home/gregory/Datasets/COCO'
year='2017'
spurious_class='person'

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision

for main_class in 'skis' 'frisbee' 'tennis-racket' 'bicycle'
do
    python Plot.py $root $year $main_class $spurious_class
done

conda deactivate
