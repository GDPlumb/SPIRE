
task='augment-transfer'
spurious_class='person'

root='/home/gregory/Datasets/COCO'
year='2017'
num_workers=2

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision

python Configs.py $root $year $num_workers $task $spurious_class

for ((i=0;i<$num_workers;i++))
do
    CUDA_VISIBLE_DEVICES=$i python Models.py $i &
done
wait

conda deactivate
