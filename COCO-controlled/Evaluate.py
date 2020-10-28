
import json
import numpy as np
import sys
import torch
import torchvision.models as models

from Misc import get_pair, process_set

sys.path.insert(0, '../COCO/')
from COCOWrapper import COCOWrapper

def evaluate(mode, main, spurious, p_correct, trial, p_main = 0.5, p_spurious = 0.5):

    base = './Pairs/{}-{}/{}/{}/trial{}'.format(main, spurious, p_correct, mode, trial)
        
    # Get the 'testing' images
    coco = COCOWrapper(mode = 'val')
    both, just_main, just_spurious, neither = get_pair(coco, main, spurious)

    # Setup the model
    model = models.mobilenet_v2(pretrained = True)
    model.classifier[1] = torch.nn.Linear(in_features = 1280, out_features = 1)
    model.cuda()
    
    model.load_state_dict(torch.load('{}/model.pt'.format(base)))
    model.eval()
        
    # Run the evaluation
    acc_both = process_set(model, both, 1)
    acc_main = process_set(model, just_main, 1)
    acc_spur = process_set(model, just_spurious, 0)
    acc_neither = process_set(model, neither, 0)
    
    out = {}
    out['both'] = acc_both
    out['just_main'] = acc_main
    out['just_spurious'] = acc_spur
    out['neither'] = acc_neither
    out['average'] = np.mean([acc_both, acc_main, acc_spur, acc_neither])
    
    with open('{}/results.json'.format(base), 'w') as f:
        json.dump(out, f)

if __name__ == '__main__':

    mode = sys.argv[1]
    main = sys.argv[2]
    spurious = sys.argv[3]
    p_correct = float(sys.argv[4])
    trials = sys.argv[5].split(',')

    for trial in trials:
        evaluate(mode, main, spurious, p_correct, trial)
