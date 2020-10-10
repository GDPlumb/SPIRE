
root='/home/gregory/Datasets/COCO'
year='2017'
spurious_class='person'

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision

for task in 'random-tune-paint'
do
    python Models_evaluate.py $root $year $spurious_class $task
done

conda deactivate
