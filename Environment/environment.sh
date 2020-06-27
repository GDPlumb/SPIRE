
conda env remove --name countervision
conda env create -f environment.yml
source activate countervision
conda install -c conda-forge pycocotools=2.0.*
conda list --name countervision
