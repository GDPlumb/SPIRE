
import os

mode = 'initial-transfer'

TRAIN = True
EVAL = True

for pair in [('car', 'person'),
            ('bottle', 'person'),
            ('cup', 'person'),
            ('bowl', 'person'),
            ('chair', 'person'),
            ('dining+table', 'person'),
            ('bottle', 'cup'),
            ('bottle', 'dining+table'),
            ('cup', 'bowl'),
            ('cup', 'chair'),
            ('cup', 'dining+table'),
            ('bowl', 'dining+table'),
            ('chair', 'dining+table')]:
    print()
    print(pair)
    
    main = pair[0]
    spurious = pair[1]
    
    if mode == 'initial-transfer' and TRAIN:
        print('Setting Up')
        os.system('python SetupPair.py {} {}'.format(main, spurious))
        
    for p in [0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1]:
        print(p)
        if TRAIN:
            print('Training')
            os.system('./Train.sh {} {} {} {}'.format(mode, main, spurious, p))
        if EVAL:
            print('Evaluating')
            os.system('./Evaluate.sh {} {} {} {}'.format(mode, main, spurious, p))
    print()

print('Plotting')
os.system('python Plot.py')
print()
