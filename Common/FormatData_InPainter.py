
import numpy as np
import os
import pickle
import skimage
import sys

ec_source = '/home/gregory/Desktop/edge-connect'
sys.path.insert(0, ec_source)
import src.edge_connect


## This is to preven a decompression error
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_coords(image, color = [124, 116, 104], tol = 0):
    dif = np.abs(image - color)
    dif = np.mean(dif, axis= 2)
    out =  np.where(dif <= tol)
    return out
    
def get_mask(coords, width, height):
    out = np.zeros((width, height))
    out[coords[0], coords[1]] = 1
    return out
    
def paint(ip, image, mask):
    return ip.inpaint([image], [mask])[0].cpu().numpy()
