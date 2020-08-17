
# Spurious Correlation Classes
target='bowl'
confounder='dining-table'

# Conda Configuration
root='/home/gregory/Datasets/COCO/'
year='2017'

# Storage Drive Location
storage='/media/gregory/HDD'

# Edge-Connect Location
ec_source='/home/gregory/Desktop/edge-connect'

# Move old datasets to a storage drive
#mv DataAugmentation/* $storage

# Create new datasets
source /home/gregory/anaconda3/etc/profile.d/conda.sh
#conda activate countervision
#python DataAugmentation.py 'none' $confounder $root $year
#conda deactivate

# In Paint those datasets
#conda activate edge-connect
#python DataAugmentation-InPainter.py 'none' $confounder $year $ec_source
#conda deactivate

# Train models on those datasets
conda activate countervision
python Models-DA.py 'none' $confounder
conda deactivate

# Evaluate the models
conda activate countervision
python CompareModels.py $target $confounder $root $year
conda deactivate
