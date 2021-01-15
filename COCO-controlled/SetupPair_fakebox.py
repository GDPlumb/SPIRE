
import numpy as np
import os
import pickle
from PIL import Image
import sys
from torchvision import transforms

from Config import get_data_dir, get_random_seed
from Misc import id_from_path

sys.path.insert(0, '../Common/')
from COCOWrapper import COCOWrapper
from Dataset import MakeSquare
from FormatData import get_mask, merge_images_parallel

def get_custom_resize(d):
    return transforms.Compose([
            MakeSquare(),
            transforms.Resize((d,d))
            ])


def add_fakeboxes(coco, save_dir, ids_background, ids_object, chosen_id, images, name, label, unmask_classes = None ):
    imgs = coco.get_images_with_cats(None)
    id2img = {}
    for img in imgs:
        id2img[id_from_path(img['file_name'])] = img

    base_dir = coco.get_base_dir()

    for i in range(len(ids_background)):
        # Get info for the base image
        id = ids_background[i]

        base_image = np.array(Image.open('{}/{}'.format(base_dir, id2img[id]['file_name'])).convert('RGB'))
        width, height, _ = base_image.shape
        dim_min = min(width, height)

        anns = coco.coco.loadAnns(coco.coco.getAnnIds(imgIds = id2img[id]['id']))
        # Get info for the object image
        id_object = ids_object[i]

        object_image = Image.open('{}/{}'.format(base_dir, id2img[id_object]['file_name'])).convert('RGB')

        anns_object = coco.coco.loadAnns(coco.coco.getAnnIds(imgIds = id2img[id_object]['id']))
        mask = get_mask(anns_object, chosen_id, coco.coco, mode = 'box', unmask = False)

        # Merge the two images
        custom_resize = get_custom_resize(dim_min)

        mask = np.array(custom_resize(Image.fromarray(np.squeeze(mask))))
        object_image = np.array(custom_resize(object_image))
        mask_indices = np.where(mask != 0)


        if unmask_classes is not None:
            save_image = base_image.copy()
        base_image[mask_indices[0], mask_indices[1]] =  [124, 116, 104]
        if unmask_classes is not None:
            mask = get_mask(anns, unmask_classes, coco.coco, mode='pixel', unmask = False)
            mask_indices = np.where(mask != 0)
            if len(mask_indices) == 3:
                base_image[mask_indices[0], mask_indices[1]] =  save_image[mask_indices[0], mask_indices[1]]

        image_new = Image.fromarray(np.uint8(base_image))
        # return image_new
        # Save the output
        file_new = '{}/{}.jpg'.format(save_dir, id)
        image_new.save(file_new)
        images[id][name] = [file_new, label]

if __name__ == '__main__':

    print('Adding')

    main = sys.argv[1]
    spurious = sys.argv[2]


    np.random.seed(get_random_seed())

    pair_dir = '{}/{}-{}'.format(get_data_dir(), main, spurious)
    for mode in ['val', 'train']:
        mode_dir = '{}/{}'.format(pair_dir, mode)

        with open('{}/images.p'.format(mode_dir), 'rb') as f:
            images = pickle.load(f)

        with open('{}/splits.p'.format(mode_dir), 'rb') as f:
            splits = pickle.load(f)

        coco = COCOWrapper(mode = mode)

        imgs = coco.get_images_with_cats(None)
        id2img = {}
        main_id = coco.get_class_id(main)
        spurious_id = coco.get_class_id(spurious)
        with_main = splits['both'] + splits['just_main']
        with_spurious = splits['both'] + splits['just_spurious']
        for img in imgs:
            id2img[id_from_path(img['file_name'])] = img

        unmask_map = {
            'both': [main_id, spurious_id],
            'just_main': [main_id],
            'just_spurious': [spurious_id],
            'neither': None,
            }
        for config in [('both', 1, 1), ('just_main', 1, 0), \
                       ('just_spurious', 0, 1), ('neither', 0, 0)]:
            # ('just_spurious', 'just_main', main, 1), \
            #             ('just_main', 'just_spurious', spurious, 1), \
            #             ('neither', 'just_main', main, 1), \
            #             ('neither', 'just_spurious', spurious, 0)]:

            background_split, main_in, spurious_in = config
            # object_split = config[1]
            # chosen_class = config[2]
            # label = config[3]
            # chosen_id = [coco.get_class_id(chosen_class)]


            ids_background = np.array(splits[background_split])


            n = len(ids_background)
            # MUST FIX: There is some data leakage here
            np.random.shuffle(ids_background)
            indices = list(np.random.choice(with_main, int(n/2))) + list(np.random.choice(with_spurious, int(n/2)))
            for k in range(len(indices)):
                i = ids_background[k]
                j = indices[k]
                while i == j:
                    if k < n/2:
                        j = np.random.choice(with_main)
                    else:
                        j = np.random.choice(with_spurious)
                indices[k] = j
            indices = np.array(indices)
            half = int(n/2)
            save_location = '{}/{}+fake_main_box'.format(mode_dir, background_split)
            os.system('rm -rf {}'.format(save_location))
            os.system('mkdir {}'.format(save_location))
            name = '{}+fake_main_box'.format(background_split)
            add_fakeboxes(coco, save_location, ids_background[:half], indices[:half], [main_id], images, name, main_in,  unmask_classes=unmask_map[background_split])

            save_location = '{}/{}+fake_spurious_box'.format(mode_dir, background_split)
            name = '{}+fake_spurious_box'.format(background_split)
            os.system('rm -rf {}'.format(save_location))
            os.system('mkdir {}'.format(save_location))
            add_fakeboxes(coco, save_location, ids_background[half:len(indices)], indices[half:], [spurious_id], images, name, main_in,  unmask_classes=unmask_map[background_split])

        with open('{}/images.p'.format(mode_dir), 'wb') as f:
            pickle.dump(images, f)
            #
            # indices = np.zeros((n), dtype = np.int)
            # for i in range(n):
            #     if mode == 'train':
            #         # On the training distributions, Just Main and Just Spurious have the same size and we draw the first k samples from the list to use for each split
            #         # As a result, we need to be careful not to leak extra information by seeing objects drawn from images that are not part of those k samples
            #         indices[i] = i
            #     elif mode == 'val':
            #         # On the validation set, Just Main and Just Spurious have different sides
            #         # But we do not need to worry about data leaking, so we pick a random source
            #         indices[i] = np.random.randint(0, len(ids_object))
