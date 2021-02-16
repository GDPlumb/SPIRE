
sudo rm -rf edge-connect
git clone https://github.com/marcotcr/edge-connect.git
cd edge-connect

conda env remove --name edge-connect
yes | conda create --name edge-connect python=3.6
source activate edge-connect
yes | conda install pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch
yes | pip install -r requirements.txt
yes | pip install ipykernel

chmod +x ./scripts/download_model.sh
sudo ./scripts/download_model.sh

sudo chmod -R ugo+rw ../edge-connect

mv config.yml ./checkpoints/places2/
