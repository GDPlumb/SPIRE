
root='/home/gregory/Datasets/COCO'
year='2017'
spurious_class='person'

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision

#python Models_evaluate.py $root $year $spurious_class 'initial-transfer'
python Models_evaluate.py $root $year $spurious_class 'random-transfer'
python Models_evaluate.py $root $year $spurious_class 'augment-transfer'
python Models_evaluate.py $root $year $spurious_class 'both-transfer'

conda deactivate