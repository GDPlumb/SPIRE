#!/bin/bash

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision
python SetupPair.py $1 $2
python SetupPair_Add.py $1 $2
conda deactivate

conda activate edge-connect
python SetupPair_InPainter.py $1 $2
conda deactivate
