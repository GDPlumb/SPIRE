
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

#python FormatData.py $root 'val' $year 'random-pixel'
#python FormatData.py $root 'train' $year 'random-pixel'

#python FormatData.py $root 'val' $year 'spurious-pixel' $spurious

conda deactivate

conda activate edge-connect

#python FormatData_InPainter.py $root 'val' $year 'random'
#python FormatData_InPainter.py $root 'train' $year 'random'

python FormatData_InPainter.py $root 'val' $year $spurious

conda deactivate
