
root='/home/gregory/Datasets/COCO'
year='2017'
spurious='person'

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision

#python FormatData.py $root 'val' $year 'standard'
#python FormatData.py $root 'train' $year 'standard'

#python FormatData.py $root 'val' $year 'random'
#python FormatData.py $root 'train' $year 'random'

#python FormatData.py $root 'val' $year 'spurious' $spurious
#python FormatData.py $root 'train' $year 'spurious' $spurious

#python FormatData.py $root 'train' $year 'split' $spurious

conda deactivate
