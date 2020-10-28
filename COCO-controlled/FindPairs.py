
import sys

from Misc import get_pair

sys.path.insert(0, '../COCO/')
from COCOWrapper import COCOWrapper

threshold = 100

coco = COCOWrapper(mode = 'val')

names = []
for cat in coco.cats:
    names.append(cat['name'])
n = len(names)

print()
for i in range(n):
    main = names[i]
    for j in range(i + 1, n):
        spurious = names[j]

        both, just_main, just_spurious, neither = get_pair(coco, main, spurious)
        if len(both) > threshold and len(just_main) > threshold and len(just_spurious) > threshold:
            print(main, spurious)
