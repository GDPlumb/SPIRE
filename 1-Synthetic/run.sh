
rm *.pdf
rm -rf train*
python run.py
tensorboard --logdir ./
