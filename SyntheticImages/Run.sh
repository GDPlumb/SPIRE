
config_choice='check'
num_workers=2
base_location='/mnt/HDD/CounterVision/SyntheticImages'

source /home/gregory/anaconda3/etc/profile.d/conda.sh
conda activate countervision
python Configs.py $config_choice $num_workers
#python Run_data.py 'generate' $base_location
#./Run_models.sh 'model_original' $num_workers $base_location
#python Run_data.py 'augment' $base_location
#./Run_models.sh 'model_augmented' $num_workers $base_location
./Run_models.sh 'test' $num_workers $base_location
conda deactivate

# Cleaning Up Models
# find -name '*model_test*' | xargs rm
