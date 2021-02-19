
import io
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np
from pycocotools.coco import COCO
from sklearn.metrics import precision_score, recall_score
import sys

def id_from_path(path):
    return path.split('/')[-1].split('.')[0].lstrip('0')

class COCOWrapper():

    def __init__(self, root = '/home/gregory/Datasets/COCO', mode = 'val', year = '2017'):
    
        stdout_orig = sys.stdout
        sys.stdout = io.StringIO()
        coco = COCO('{}/annotations/instances_{}{}.json'.format(root, mode, year))
        sys.stdout = stdout_orig
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
        return 'None'

    def get_class_id(self, name):
        cats = self.cats
        name = name.replace('+', ' ')
        for i in range(len(cats)):
            if cats[i]['name'] == name:
                return cats[i]['id']
        return 'None'
    
    def get_annotations(self, img_obj):
        coco = self.coco
        return coco.loadAnns(coco.getAnnIds(img_obj['id'], iscrowd = None))
        
    def get_cat_ids(self, names):
        coco = self.coco
        return coco.getCatIds(catNms = names)
        
    def get_images_with_cats(self, cats):
        coco = self.coco
        return coco.loadImgs(coco.getImgIds(catIds = coco.getCatIds(catNms = cats)))
        
    def split_images_by_cats(self, cats):
        coco = self.coco
        
        imgs_all = coco.getImgIds()
        imgs_with = coco.getImgIds(catIds = coco.getCatIds(catNms = cats))
        imgs_without = np.setdiff1d(imgs_all, imgs_with)
        
        return imgs_with, imgs_without
    
    def get_splits_pair(self, main, spurious):        
        main = main.replace('+', ' ')
        spurious = spurious.replace('+', ' ')

        ids_main = [img['file_name'] for img in self.get_images_with_cats([main])]
        ids_spurious = [img['file_name'] for img in self.get_images_with_cats([spurious])]

        both = np.intersect1d(ids_main, ids_spurious)
        
        just_main = np.setdiff1d(ids_main, ids_spurious)
        
        just_spurious = np.setdiff1d(ids_spurious, ids_main)

        neither = [img['file_name'] for img in self.get_images_with_cats(None)]
        neither = np.setdiff1d(neither, ids_main)
        neither = np.setdiff1d(neither, ids_spurious)
        
        splits = {}
        splits['both'] = both
        splits['just_main'] = just_main
        splits['just_spurious'] = just_spurious
        splits['neither'] = neither

        base_dir = self.get_base_dir()
        for name in splits:
            splits[name] = ['{}/{}'.format(base_dir, f) for f in splits[name]]
            
        return splits
       
    def construct_id2objs(self):
        coco = self.coco
        
        cats = {x['id']: x['name'] for x in coco.loadCats(coco.getCatIds())}
        
        id2objs = {}
        for img_id, img_obj in coco.anns.items():
            i = img_obj['image_id']
            o = cats[img_obj['category_id']]

            if i not in id2objs:
                id2objs[i] = [o]
            elif o not in id2objs[i]: # We don't care about how many of each object there are
                id2objs[i].append(o)
        
        self.id2objs = id2objs
        
    def construct_captions(self):
        stdout_orig = sys.stdout
        sys.stdout = io.StringIO()
        captions = COCO('{}/annotations/captions_{}{}.json'.format(self.root, self.mode, self.year))
        self.captions = captions
        sys.stdout = stdout_orig  
    
    def construct_id2words(self):
        # Get the words associated with each image via its caption
        stop_words = set(stopwords.words('english'))  
        tokenizer = RegexpTokenizer(r'\w+')

        id2words = {}
        for img_id, img_obj in self.captions.anns.items():
            i = img_obj['image_id']
            c = img_obj['caption']

            c = c.lower()
            words = tokenizer.tokenize(c)  
            words = [w for w in words if not w in stop_words]  

            if i not in id2words:
                id2words[i] = []

            for w in words:
                if w not in id2words[i]:
                    id2words[i].append(w)
                    
        self.id2words = id2words
        
    def clean_id_maps(self):
        id2objs = self.id2objs
        id2words = self.id2words
        
        # Make sure that both of those mappings have the same keys
        k1 = [key for key in id2objs]
        k2 = [key for key in id2words]

        just_obj = np.setdiff1d(k1, k2)
        just_cap = np.setdiff1d(k2, k1)

        for key in just_obj:
            del id2objs[key]

        for key in just_cap:
            del id2words[key] 
        
    def construct_counts(self, threshold_dict = {'train':1000, 'val':0}):
        threshold = threshold_dict[self.mode]
        
        id2words = self.id2words
        
        # Get a list of all of the words used in the caption and their counts
        word2count = {}
        for i in id2words.keys():
            words = id2words[i]
            for w in words:
                if w not in word2count:
                    word2count[w] = 0
                word2count[w] += 1
         
        self.word2count = word2count
            
        # Find the most common words
        common_words = []
        for w in word2count.keys():
            if word2count[w] >= threshold:
                common_words.append(w)
        self.common_words = common_words
        
    def construct_word2ids(self):
        id2words = self.id2words
        # Map the common words to images
        word2ids = {}
        for word in self.common_words:
            word2ids[word] = []
            for img_id in id2words.keys():
                if word in id2words[img_id]:
                    word2ids[word].append(img_id)
                    
        self.word2ids = word2ids
        
    def setup_caption_maps(self):
        self.construct_id2objs()
        self.construct_captions()
        self.construct_id2words()
        self.clean_id_maps()
        self.construct_counts()
        self.construct_word2ids()
        
    def get_obj_counts(self, img_ids):
        out = {}
        num_imgs = len(img_ids)
        for img_id in img_ids:
            objs = self.id2objs[img_id]
            for obj in objs:
                if obj not in out:
                    out[obj] = 0
                out[obj] += 1 / num_imgs
        return out
        
    def compare_words(self, word1, word2):

        counts1 = self.get_obj_counts(self.word2ids[word1])
        counts2 = self.get_obj_counts(self.word2ids[word2])

        k1 = [key for key in counts1]
        k2 = [key for key in counts2]

        keys = list(set(k1).union(set(k2)))

        diff = {}
        for key in keys:
            if key in counts1:
                v1 = counts1[key]
            else:
                v1 = 0.0

            if key in counts2:
                v2 = counts2[key]
            else:
                v2 = 0.0

            diff[key] = v1 - v2

        diff_sorted = sorted(diff.items(), key = lambda x: np.abs(x[1]), reverse = True)

        return diff_sorted        

    def get_splits_words(self, label1, label2, spurious):
        
        imgs = self.get_images_with_cats(None)
        id2filename = {}
        for img in imgs:
            id2filename[str(img['id'])] = img['file_name']

        word2ids = self.word2ids
        id2objs = self.id2objs
        
        ids1 = word2ids[label1]
        ids2 = word2ids[label2]

        just1 = np.setdiff1d(ids1, ids2)
        just2 = np.setdiff1d(ids2, ids1)

        splits = {}
        splits['1s'] = [] # Answer is 1 (eg, label 1) and Spurious is present
        splits['1ns'] = [] # Answer is 1 and no Spurious
        splits['0s'] = [] # Answer is 0 (eg, label 2) and Spurious
        splits['0ns'] = [] # Answer is 0 and no Spurious

        for img_id in just1:
            if spurious in id2objs[img_id]:
                splits['1s'].append(str(img_id))
            else:
                splits['1ns'].append(str(img_id))


        for img_id in just2:
            if spurious in id2objs[img_id]:
                splits['0s'].append(str(img_id))
            else:
                splits['0ns'].append(str(img_id))
                
        base_dir = self.get_base_dir()
        for name in splits:
            splits[name] = ['{}/{}'.format(base_dir, id2filename[id]) for id in splits[name]]

        return splits
        
        