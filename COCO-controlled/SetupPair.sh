#!/bin/bash

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision
echo 'Masking'
python SetupPair.py $1 $2
conda deactivate

conda activate edge-connect
echo 'Painting'
python SetupPair_InPainter.py $1 $2
conda deactivate
