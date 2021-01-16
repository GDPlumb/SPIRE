
def get_data_dir():
    return '/home/gregory/Datasets/COCO-test'
    
def get_random_seed():
    return 0

def get_data_fold():
    # -1 indicates that we are going to train on the validation set and use the training set for evaluation
    return 1

# For evaluation, we upper bound the number of samples used (mostly significant for Setup*.py)
def get_max_samples():
    return 1000
