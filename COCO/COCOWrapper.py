
from pycocotools.coco import COCO

class COCOWrapper():

    def __init__(self, root = '/home/gregory/Datasets/COCO/', mode = 'val', year = '2017'):
    
        coco = COCO('{}/annotations/instances_{}{}.json'.format(root, mode, year))
        cats = coco.loadCats(coco.getCatIds())
        self.coco = coco
        self.cats = cats
        self.root = root
        self.mode = mode
        self.year = year
        
    def get_class_name(self, id):
        cats = self.cats
        for i in range(len(cats)):
            if cats[i]['id'] == id:
                return cats[i]['name']
        return "None"

    def get_class_id(self, name):
        cats = self.cats
        for i in range(len(cats)):
            if cats[i]['name'] == name:
                return cats[i]['id']
        return "None"
        
    def get_images_with_cats(self, cats):
        coco = self.coco
        return coco.loadImgs(coco.getImgIds(catIds = coco.getCatIds(catNms = cats)))
        
    def get_annotations(self, img_obj):
        coco = self.coco
        return coco.loadAnns(coco.getAnnIds(img_obj['id'], iscrowd = None))
        
    def get_cat_ids(self, names):
        coco = self.coco
        return coco.getCatIds(catNms = names)

    def show_metrics(self, precision, recall):
        cats = self.cats
        
        print()
        print("Object Precision Recall")
        MAP = 0.0
        MAR = 0.0
        for cat in cats:
            p = precision[cat['id']]
            MAP += p
            r = recall[cat['id']]
            MAR += r
            print(cat['name'], p, r)
            
        print()
        print("MAP MAR")
        print(MAP / len(cats), MAR / len(cats))
        print()
