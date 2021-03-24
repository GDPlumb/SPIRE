#!/bin/bash

source /home/gregory/anaconda3/etc/profile.d/conda.sh

label1='runway'
label2='street'
spurious='airplane'
    
conda activate countervision
python Remove.py $label1 $label2 $spurious
python Add.py $label1 $label2 $spurious
conda deactivate

conda activate edge-connect
python InPaint.py $label1 $label2 $spurious # Depends on the output of Remove.py
conda deactivate
