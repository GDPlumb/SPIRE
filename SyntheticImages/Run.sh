source /home/gregory/anaconda3/etc/profile.d/conda.sh
conda activate countervision
#python Run.py 'generate'
#python Run.py 'model_original'
#python Run.py 'augment'
#python Run.py 'model_augmented'
#python Run.py 'model_mixup'
python Run.py 'model_augmented_paired'
#python Run.py 'model_mixup_paired'
conda deactivate

