
import os

mode = 'initial-transfer'

SETUP = True
TRAIN = True
EVAL = True
SEARCH = True

for pair in [
            #('car', 'person'),
            ('bottle', 'person'),
            #('cup', 'person'),
            #('bowl', 'person'),
            #('chair', 'person'),
            #('dining+table', 'person'),
            #('bottle', 'cup'),
            #('bottle', 'dining+table'),
            #('cup', 'bowl'),
            #('cup', 'chair'),
            #('cup', 'dining+table'),
            #('bowl', 'dining+table'),
            #('chair', 'dining+table')
            ]:
    print()
    print(pair)
    
    main = pair[0]
    spurious = pair[1]
    
    if SETUP:
        print('Setting Up')
        os.system('./SetupPair.sh {} {}'.format(main, spurious))
        
    for p in [0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1]:
        print(p)
        if TRAIN:
            print('Training')
            os.system('./Train.sh {} {} {} {}'.format(mode, main, spurious, p))
        if TRAIN or EVAL:
            print('Evaluating')
            os.system('./Evaluate.sh {} {} {} {}'.format(mode, main, spurious, p))
        if TRAIN or SEARCH:
            print('Searching')
            os.system('./Search.sh {} {} {} {}'.format(mode, main, spurious, p))
    print('Plotting')
    os.system('./Plot.sh {} {}'.format(main, spurious))
    print()
