
conda env remove --name countervision
conda env create -f environment.yml
source activate countervision
yes | conda install  -c pytorch pytorch=1.4.* torchvision=0.5.*
yes | conda install -c conda-forge pycocotools=2.0.* ipywidgets=7.5.*
conda list --name countervision
