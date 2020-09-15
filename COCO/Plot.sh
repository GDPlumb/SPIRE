
root='/home/gregory/Datasets/COCO'
year='2017'
spurious_class='person'

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision

for main_class in 'snowboard' 'couch' 'tie' 'handbag' 'skis' 'remote' 'toothbrush' 'sports-ball' 'knife' 'cell-phone' 'fork' 'wine-glass' 'skateboard' 'spoon' 'backpack' 'bench' 'frisbee'
do
    echo $main_class
    python Plot.py $root $year $main_class $spurious_class
    echo ''
done

conda deactivate
