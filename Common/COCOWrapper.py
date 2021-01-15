
import io
import numpy as np
from pycocotools.coco import COCO
from sklearn.metrics import precision_score, recall_score
import sys

class COCOWrapper():

    def __init__(self, root = '/home/marcotcr/datasets/COCO', mode = 'val', year = '2017'):

        # sys.stdout = io.StringIO()
        coco = COCO('{}/annotations/instances_{}{}.json'.format(root, mode, year))
        # sys.stdout = sys.__stdout__
        cats = coco.loadCats(coco.getCatIds())
        self.coco = coco
        self.cats = cats
        self.root = root
        self.mode = mode
        self.year = year

    def get_base_dir(self):
        return '{}/{}{}'.format(self.root, self.mode, self.year)

    def get_class_name(self, id):
        cats = self.cats
        for i in range(len(cats)):
            if cats[i]['id'] == id:
                return cats[i]['name']
        return "None"

    def get_class_id(self, name):
        cats = self.cats
        name = name.replace('+', ' ')
        for i in range(len(cats)):
            if cats[i]['name'] == name:
                return cats[i]['id']
        return "None"

    def get_images_with_cats(self, cats):
        coco = self.coco
        return coco.loadImgs(coco.getImgIds(catIds = coco.getCatIds(catNms = cats)))

    def split_images_by_cats(self, cats):
        coco = self.coco

        imgs_all = coco.getImgIds()
        imgs_with = coco.getImgIds(catIds = coco.getCatIds(catNms = cats))
        imgs_without = np.setdiff1d(imgs_all, imgs_with)

        return imgs_with, imgs_without

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

    def get_metrics(self, precision, recall):
        cats = self.cats

        MAP = 0.0
        MAR = 0.0
        for cat in cats:
            p = precision[cat['id']]
            MAP += p
            r = recall[cat['id']]
            MAR += r

        MAP /= len(cats)
        MAR /= len(cats)

        return MAP, MAR

    def get_metrics_classless(self, y_hat, y_true):
        cats = self.cats

        y_hat_all = []
        y_true_all = []
        for cat in cats:
            id = cat['id']

            for y in y_hat[:, id]:
                y_hat_all.append(y)

            for y in y_true[:, id]:
                y_true_all.append(y)


        return precision_score(y_true_all, y_hat_all), recall_score(y_true_all, y_hat_all)
