#!/bin/bash

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision
python Remove.py
python Add.py
conda deactivate
