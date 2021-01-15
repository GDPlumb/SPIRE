
from multiprocessing.pool import ThreadPool
import numpy as np
import os
import pickle
from PIL import Image
from pycocotools.coco import COCO
import random
import sys
from torchvision import transforms

from COCOHelper import id_from_path
from Dataset import MakeSquare

def get_mask(anns, mask_classes, coco, mode = 'box', unmask = True, unmask_classes = None):

    # Find the mask for each object we want to remove
    mask = []
    for ann in anns:
        if ann['category_id'] in mask_classes:
            # Get the pixelwise mask from COCO
            tmp = coco.annToMask(ann)
            # Process that mask
            if mode == 'pixel':
                mask.append(tmp)
            elif mode == 'box':
                idx = np.where(tmp == 1.0)
                if len(idx[0] > 0):
                    min_0 = np.min(idx[0])
                    max_0 = np.max(idx[0])
                    min_1 = np.min(idx[1])
                    max_1 = np.max(idx[1])
                    tmp_new = np.copy(tmp)
                    tmp_new[min_0:max_0, min_1:max_1] = 1.0
                    mask.append(tmp_new)
                else: # BUG?  This handles the images that have some blank annotations
                    mask.append(tmp)

    # If we found at least one object to mask from the image
    if len(mask) > 0:
        mask = np.expand_dims(1.0 * (np.sum(np.array(mask), axis = 0) >= 1.0), axis = 2)

        # If we want to unmask any objects other than those we explicitly masked
        if unmask:
            unmask = []

            if unmask_classes is None:
                def check(cat):
                    return (cat not in mask_classes)
            else:
                def check(cat):
                    return (cat in unmask_classes)

            for ann in anns:
                if check(ann['category_id']):
                    tmp = coco.annToMask(ann)
                    unmask.append(tmp)
            if len(unmask) > 0:
                unmask = np.expand_dims(1.0 * (np.sum(np.array(unmask), axis = 0) >= 1.0), axis = 2)
                mask = mask - unmask
                mask = np.clip(mask, 0, 1)

        return mask

def apply_mask(img, mask, value = 'default', invert = False):
    # Fill the masked values
    if invert:
        mask = (np.squeeze(mask) != 1)
    else:
        mask = (np.squeeze(mask) == 1)
    img_np = np.array(img)
    if value == 'default':
        img_np[mask] = [124, 116, 104]
    elif value == 'random':
        img_np[mask] =  np.random.randint(low = 0, high = 256, size = (np.sum(mask), 3))
    elif value == 'mean':
        img_np[mask] = np.mean(np.array(img), axis = (0,1)).astype(np.int)
    img = Image.fromarray(img_np)
    return img

def fake_mask_images(images, images_fake, coco, base_location, save_location, chosen_id = None, mode = 'box', invert = False, unmask = True, unmask_classes = None, use_png = False):
    for img_obj, img_fake in zip(images, images_fake):
        base_filename = img_obj['file_name']
        fake_filename = img_fake['file_name']
        img = Image.open('{}/{}'.format(base_location, base_filename)).convert('RGB')

        filename = '{}/{}'.format(save_location, base_filename)
        label = np.zeros((91)) #Note:  this used to set the data type, COCO-controlled and COCO-nuanced were run with that

        anns = coco.loadAnns(coco.getAnnIds(imgIds = img_obj['id']))
        fake_anns = coco.loadAnns(coco.getAnnIds(imgIds = img_fake['id']))

        if len(anns) > 0:

            # Randomly select one of the object categories present in this image to mask
            if chosen_id is None:
                tmp_id = random.choice(list(set([ann['category_id'] for ann in anns])))
            else:
                tmp_id = chosen_id

            # Setup the label
            for ann in anns:
                label[ann['category_id']] = 1.0
            label[tmp_id] = 0.0

            # Mask the image
            if not isinstance(tmp_id, list):
                tmp_id = [tmp_id]
            mask = get_mask(fake_anns, tmp_id, coco, mode = mode, unmask = unmask, unmask_classes = unmask_classes)
            a = Image.fromarray(np.squeeze(mask))
            a = a.resize(img.size)
            mask = np.expand_dims(np.array(a), 2)
            return img, mask
            print(len(mask.nonzero()[0]))

            # new_mask = np.zeros((np.array(img).shape[0], np.array(img).shape[1], 1))
            # a, b, c =np.where(mask==1)
            # a  = np.minimum(a, new_mask.shape[0] - 1)
            # b  = np.minimum(b, new_mask.shape[1] - 1)
            # new_mask[(a, b, c)] = 1
            # mask = new_mask

            # mask = mask.reshape((np.array(img).shape[0], np.array(img).shape[1], 1))
            # print(mask.shape, np.array(img).shape)
            img = apply_mask(img, mask, invert = invert)
            return img

        # Save the output
#         if use_png: # Preserves exact pixel values - used to pass the masked pixels to the inpainter
#             filename = '{}png'.format(filename[:-3])
#         filenames.append(filename)
#         labels.append(label)
#         img.save(filename)

def mask_images(images, coco, base_location, save_location, chosen_id = None, mode = 'box', invert = False, unmask = True, unmask_classes = None, use_png = False):
    filenames = []
    labels = []
    for img_obj in images:

        base_filename = img_obj['file_name']
        img = Image.open('{}/{}'.format(base_location, base_filename)).convert('RGB')

        filename = '{}/{}'.format(save_location, base_filename)
        label = np.zeros((91)) #Note:  this used to set the data type, COCO-controlled and COCO-nuanced were run with that

        anns = coco.loadAnns(coco.getAnnIds(imgIds = img_obj['id']))

        if len(anns) > 0:

            # Randomly select one of the object categories present in this image to mask
            if chosen_id is None:
                tmp_id = random.choice(list(set([ann['category_id'] for ann in anns])))
            else:
                tmp_id = chosen_id

            # Setup the label
            for ann in anns:
                label[ann['category_id']] = 1.0
            label[tmp_id] = 0.0

            # Mask the image
            if not isinstance(tmp_id, list):
                tmp_id = [tmp_id]
            mask = get_mask(anns, tmp_id, coco, mode = mode, unmask = unmask, unmask_classes = unmask_classes)
            img = apply_mask(img, mask, invert = invert)

        # Save the output
        if use_png: # Preserves exact pixel values - used to pass the masked pixels to the inpainter
            filename = '{}png'.format(filename[:-3])
        filenames.append(filename)
        labels.append(label)
        img.save(filename)

    return filenames, labels

def mask_images_parallel(images, coco, base_location, save_location, chosen_id = None, mode = 'box', invert = False, unmask = True, unmask_classes = None, use_png = False, workers = 24):

    # Split the images to pass them to the workers
    images_split = []
    for i in range(workers):
        images_split.append([])

    next_worker = 0
    for image in images:
        images_split[next_worker].append(image)
        next_worker = (next_worker + 1) % workers

    # Define the worker function
    def mask_images_worker(id, images_split = images_split, coco = coco, base_location = base_location, save_location = save_location, chosen_id = chosen_id, mode = mode, invert = invert, unmask = unmask, unmask_classes = unmask_classes, use_png = use_png):
        names, labels = mask_images(images_split[id], coco, base_location, save_location, chosen_id = chosen_id, mode = mode, invert = invert, unmask = unmask, unmask_classes = unmask_classes, use_png = use_png)
        with open('tmp-{}.p'.format(id), 'wb') as f:
            pickle.dump([names, labels], f)

    # Run
    pool = ThreadPool(processes = workers)
    pool.map(mask_images_worker, range(workers))

    # Collect the output
    filenames = []
    labels = []
    for i in range(workers):
        with open('tmp-{}.p'.format(i), 'rb') as f:
            data = pickle.load(f)

        for j in range(len(data[0])):
            filenames.append(data[0][j])
            labels.append(data[1][j])

    # Save the output
    with open('{}-info.p'.format(save_location), 'wb') as f:
        pickle.dump([filenames, labels], f)

    # Clean up
    os.system('rm tmp-*.p')

def get_custom_resize(d):
    return transforms.Compose([
            MakeSquare(),
            transforms.Resize((d,d))
            ])

def merge_images(coco, save_dir, ids_background, ids_object, chosen_id, mask_only=False, unmask_classes = None ):
    imgs = coco.get_images_with_cats(None)
    id2img = {}
    for img in imgs:
        id2img[id_from_path(img['file_name'])] = img

    base_dir = coco.get_base_dir()

    filenames = []
    labels = []
    for i in range(len(ids_background)):
        # Get info for the base image
        id = ids_background[i]

        base_image = np.array(Image.open('{}/{}'.format(base_dir, id2img[id]['file_name'])).convert('RGB'))
        width, height, _ = base_image.shape
        dim_min = min(width, height)

        anns = coco.coco.loadAnns(coco.coco.getAnnIds(imgIds = id2img[id]['id']))
        label = np.zeros((91))
        for ann in anns:
            label[ann['category_id']] = 1.0

        # Get info for the object image
        id_object = ids_object[i]

        object_image = Image.open('{}/{}'.format(base_dir, id2img[id_object]['file_name'])).convert('RGB')

        anns_object = coco.coco.loadAnns(coco.coco.getAnnIds(imgIds = id2img[id_object]['id']))
        mask = get_mask(anns_object, chosen_id, coco.coco, mode = 'box' if mask_only else 'pixel', unmask = False)

        # Merge the two images
        custom_resize = get_custom_resize(dim_min)

        mask = np.array(custom_resize(Image.fromarray(np.squeeze(mask))))
        object_image = np.array(custom_resize(object_image))
        mask_indices = np.where(mask != 0)


        if mask_only:
            if unmask_classes is not None:
                save_image = base_image.copy()
            base_image[mask_indices[0], mask_indices[1]] =  [124, 116, 104]
            if unmask_classes is not None:
                mask = get_mask(anns, unmask_classes, coco.coco, mode='pixel', unmask = False)
                mask_indices = np.where(mask != 0)
                if len(mask_indices) == 3:
                    base_image[mask_indices[0], mask_indices[1]] =  save_image[mask_indices[0], mask_indices[1]]
        else:
            for j in range(3):
                base_image[mask_indices[0], mask_indices[1], j] = object_image[mask_indices[0], mask_indices[1], j]

        image_new = Image.fromarray(np.uint8(base_image))
        # return image_new

        label[chosen_id[0]] = 1.0 #Add the object we pasted on.  Note:  this may cover other objects and so the labels are noisy

        # Save the output
        file_new = '{}/{}.jpg'.format(save_dir, id)
        image_new.save(file_new)

        filenames.append(file_new)
        labels.append(label)
    return filenames, labels

def merge_images_parallel(coco, save_dir, ids_background, ids_object, chosen_id, workers = 24, mask_only=False, unmask_classes=None):

    # Split the images to pass them to the workers
    ids_background_split = []
    ids_object_split = []
    for i in range(workers):
        ids_background_split.append([])
        ids_object_split.append([])

    next_worker = 0
    for i in range(len(ids_background)):
        ids_background_split[next_worker].append(ids_background[i])
        ids_object_split[next_worker].append(ids_object[i])
        next_worker = (next_worker + 1) % workers

    # Define the worker function
    def merge_images_worker(id, ids_background_split = ids_background_split, ids_object_split = ids_object_split, coco = coco, save_dir = save_dir, chosen_id = chosen_id):
        names, labels = merge_images(coco, save_dir, ids_background_split[id], ids_object_split[id], chosen_id, mask_only=mask_only, unmask_classes=unmask_classes)
        with open('tmp-{}.p'.format(id), 'wb') as f:
            pickle.dump([names, labels], f)

    # Run
    pool = ThreadPool(processes = workers)
    pool.map(merge_images_worker, range(workers))

    # Collect the output
    filenames = []
    labels = []
    for i in range(workers):
        with open('tmp-{}.p'.format(i), 'rb') as f:
            data = pickle.load(f)

        for j in range(len(data[0])):
            filenames.append(data[0][j])
            labels.append(data[1][j])

    # Save the output
    with open('{}-info.p'.format(save_dir), 'wb') as f:
        pickle.dump([filenames, labels], f)

    # Clean up
    os.system('rm tmp-*.p')
