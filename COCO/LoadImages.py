
from glob import glob
import json

def load_images(pairs, base_dir):
    
    # Get the original images and labels
    with open('{}/images.json'.format(base_dir), 'r') as f:
        data = json.load(f)
        
    images = {}
    for id in data:
        images[id] = {}
        images[id]['orig'] = data[id]
        
    # Get the various counterfactual versions that are available
    for pair in pairs:
        for dir in glob('{}/{}/*'.format(base_dir, pair)):
            with open('{}/images.json'.format(dir), 'r') as f:
                data = json.load(f)
                
            name = '{}-{}'.format(pair, dir.split('/')[-1])
            for id in data:
                images[id][name] = data[id]
    
    return images
