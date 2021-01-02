#!/bin/bash

source /home/gregory/anaconda3/etc/profile.d/conda.sh

conda activate countervision
# This step must be run before training initial-transfer and running FindSCs
#python Setup.py
# All of the following steps require that FindSCs has produced the list of object pairs to run
#python Setup_Remove.py
#python Setup_Add.py
conda deactivate

conda activate edge-connect
#python Setup_InPainter.py # This step depends on the output of Setup_Remove
conda deactivate
