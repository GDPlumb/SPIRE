import numpy as np
import cv2

def add_rectangle(im, loc = (0, 0), lengths = (10, 10), color = (255, 255, 255)):
    im_old = np.copy(im)
    cv2.rectangle(im, loc, (loc[0] + lengths[0], loc[1] + lengths[1]), color, cv2.FILLED)
    map = np.any(im_old != im, axis = 2)
    return map
    
def spawn_rectangle(lengths = (10, 10)):
    loc = (np.random.randint(0, 64 - lengths[0]), np.random.randint(0, 64 - lengths[1]))
    return loc
    
def add_circle(im, loc = (0,0), radius = 10, color = (255, 255, 255)):
    im_old = np.copy(im)
    cv2.circle(im, (loc[0] + radius, loc[1] + radius), radius, color, cv2.FILLED)
    map = np.any(im_old != im, axis = 2)
    return map
    
def spawn_circle(radius = 10):
    loc = (np.random.randint(0, 64 - 2 * radius), np.random.randint(0, 64 - 2 * radius))
    return loc
    
def get_N():
    N = np.zeros((10, 10), dtype = np.int)
    N[:, 0:2] = 1
    N[:, 8:10] = 1
    for i in range(1, 9):
        N[i - 1, i] = 1
        N[i, i] = 1
        N[i + 1, i] = 1
    return N
    
def get_Y():
    Y = np.zeros((10, 10), dtype = np.int)
    Y[9, 0:2] = 1
    Y[8, 1] = 1
    for i in range(2, 10):
        Y[9 - i, i] = 1
        Y[10 - i, i] = 1
        Y[11 - i, i] = 1
    for i in range(1, 5):
        Y[i - 1, i] = 1
        Y[i, i] = 1
        Y[i + 1, i] = 1
    Y[0:2, 0] = 1
    return Y

def add_letter(im, char = 'Y', loc = (0, 0), color = (255, 255, 255)):
    map = np.zeros((64, 64), dtype = np.int)
    if char == 'Y':
        c = get_Y()
    elif char == 'N':
        c = get_N()
    map[loc[0]:loc[0] + 10, loc[1]:loc[1] + 10] = c
    map = map.astype('bool')
    im[map] = color
    return map
    
def spawn_letter():
    loc = (np.random.randint(0, 54), np.random.randint(0, 54))
    return loc

def make_im(color_b, include_1, shape_1, color_1, include_2, shape_2, color_2, char):
    out = []
    
    # Create the background color
    im = 125 * color_b * np.ones((64, 64, 3), dtype = 'uint8')
    
    # Add the first object
    if include_1 == 0:
        out.append(None)
    elif include_1 == 1:
        if color_1 == 0:
            c = (255, 255, 255)
        elif color_1 == 1:
            c = (255, 0, 0)
        if shape_1 == 0:
            loc = spawn_rectangle()
            map = add_rectangle(im, loc = loc, color = c)
        elif shape_1 == 1:
            loc = spawn_circle()
            map = add_circle(im, loc = loc, color = c)
        out.append(map)
        
    # Add the first object
    if include_2 == 0:
        out.append(None)
    elif include_2 == 1:
        if color_2 == 0:
            c = (255, 255, 255)
        elif color_2 == 1:
            c = (0, 255, 0)
        if shape_2 == 0:
            loc = spawn_rectangle()
            map = add_rectangle(im, loc = loc, color = c)
        elif shape_2 == 1:
            loc = spawn_circle()
            map = add_circle(im, loc = loc, color = c)
        out.append(map)
    
    # Add the letter
    loc = spawn_letter()
    if char == 0:
        map = add_letter(im, 'N', loc = loc, color = (0, 0, 255))
    elif char == 1:
        map = add_letter(im, 'Y', loc = loc, color = (0, 0, 255))
    out.append(map)
    
    return im, out
