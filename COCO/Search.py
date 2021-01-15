
import json
import sys

from LoadImages import load_images

sys.path.insert(0, '../Common/')
from ModelWrapper import ModelWrapper
from ResNet import get_model
from SearchHelper import get_map, get_diff

def search(model_dir, data_dir, coco):

    # Setup the data
    with open('./FindSCs.json', 'r') as f:
        pairs = json.load(f)
            
    # Setup the model
    model = get_model(mode = 'eval', parent = '{}/model.pt'.format(model_dir), out_features = 91)
    model.cuda()
    model.eval()
    
    wrapper = ModelWrapper(model, get_names = True)
    
    # Run the Search
    metrics = {}
    for pair in pairs:
        main = pair.split('-')[0]
        spurious = pair.split('-')[1]
        
        images = load_images([pair], data_dir)
        
        # Get the index that we care about for this pair
        index = coco.get_class_id(main)
        
        # Get the image splits for this pair
        with open('{}/splits/{}-{}.json'.format(data_dir, main, spurious), 'r') as f:
            splits = json.load(f)
        both = splits['both']
        just_main = splits['just_main']
    
        # Test removing Main/Spurious from Both
        ids = [id for id in both]
        map_orig = get_map(wrapper, images, ids, 'orig', index = index)
        
        names = []
        for object in ['main', 'spurious']:
            for mask in ['box', 'pixel-paint']:
                names.append('{}-{}-both-{}-{}'.format(main, spurious, object, mask))
        
        for name in names:
            map_name = get_map(wrapper, images, ids, name, index = index)
            metrics[name] = get_diff(map_name, map_orig)
            
        # Test adding Spurious to Just Main
        ids = [id for id in just_main]
        map_orig = get_map(wrapper, images, ids, 'orig', index = index)
        name = '{}-{}-just_main+spurious'.format(main, spurious)
        map_name = get_map(wrapper, images, ids, name, index = index)
        metrics[name] = get_diff(map_name, map_orig)
        
        # Save the output
        with open('{}/search.json'.format(model_dir), 'w') as f:
            json.dump(metrics, f)
