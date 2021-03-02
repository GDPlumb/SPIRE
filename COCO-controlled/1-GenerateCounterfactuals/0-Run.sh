#!/bin/bash

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision
python Remove.py
python Add.py
conda deactivate

conda activate edge-connect
python InPaint.py # Depends on the output of Remove.py
conda deactivate
