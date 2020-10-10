
root='/home/gregory/Datasets/COCO'
year='2017'
spurious_class='person'

rm -rf Plot
mkdir Plot

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision

for main_class in 'main' 'handbag' 'sports-ball' 'knife' 'frisbee' 'skateboard' 'snowboard' 'wine-glass' 'tennis-racket' 'remote' 'skis' 'tie' 'toothbrush'
do
    echo $main_class
    python Plot.py $root $year $main_class $spurious_class
    echo ''
done

conda deactivate
