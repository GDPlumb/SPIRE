#!/bin/bash

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision
python SetupTuple.py $1 $2 $3
python SetupTuple_Add.py $1 $2 $3
conda deactivate

conda activate edge-connect
python SetupTuple_InPainter.py $1 $2 $3
conda deactivate
