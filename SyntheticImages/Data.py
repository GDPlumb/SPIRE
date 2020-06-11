import numpy as np
import cv2

def add_rectangle(im, loc = (0, 0), lengths = (10, 10), color = (255, 255, 255)):
    cv2.rectangle(im, loc, (loc[0] + lengths[0], loc[1] + lengths[1]), color, cv2.FILLED)
    
def spawn_rectangle(lengths = (10, 10)):
    loc = (np.random.randint(0, 64 - lengths[0]), np.random.randint(0, 64 - lengths[1]))
    return (loc, lengths)
    
def add_circle(im, loc = (0,0), radius = 10, color = (255, 255, 255)):
    cv2.circle(im, (loc[0] + radius, loc[1] + radius), radius, color, cv2.FILLED)
    
def spawn_circle(radius = 10):
    loc = (np.random.randint(0, 64 - 2 * radius), np.random.randint(0, 64 - 2 * radius))
    return (loc, (2 * radius, 2 * radius))
    
def add_letter(im, char = 'Y', loc = (0, 0), color = (255, 255, 255)):
    cv2.putText(im, char, (loc[0], loc[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    
def spawn_letter():
    loc = (np.random.randint(0, 54), np.random.randint(0, 54))
    return (loc, (10, 10))

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
            s = spawn_rectangle()
            loc = s[0]
            add_rectangle(im, loc = loc, color = c)
        elif shape_1 == 1:
            s = spawn_circle()
            loc = s[0]
            add_circle(im, loc = loc, color = c)
        out.append(s)
        
    # Add the first object
    if include_2 == 0:
        out.append(None)
    elif include_2 == 1:
        if color_2 == 0:
            c = (255, 255, 255)
        elif color_2 == 1:
            c = (0, 255, 0)
        if shape_2 == 0:
            s = spawn_rectangle()
            loc = s[0]
            add_rectangle(im, loc = loc, color = c)
        elif shape_2 == 1:
            s = spawn_circle()
            loc = s[0]
            add_circle(im, loc = loc, color = c)
        out.append(s)
    
    # Add the letter
    s = spawn_letter()
    loc = s[0]
    if char == 0:
        add_letter(im, 'N', loc = loc, color = (0, 0, 255))
    elif char == 1:
        add_letter(im, 'Y', loc = loc, color = (0, 0, 255))
    out.append(s)
    
    return im, out


def sample_1(p = 1.0):
    
    y = None
    
    include_1 = np.random.binomial(n = 1, p = 0.95)
    include_2 = np.random.binomial(n = 1, p = 0.95)
    
    # If we don't have two objects, then the answer is no
    if y is None and (include_1 == 0 or include_2 == 0):
        y = 0
    
    shape_1 = np.random.randint(2)
    shape_2 = np.random.randint(2)
    if y is None:
        if shape_1 == shape_2:
            y = 1
        else:
            y = 0
    
    # With probability p, set the background color to match the label
    if np.random.uniform() < p:
        color_b = y
    else:
        color_b = (y + 1) % 2
    
    color_1 = np.random.randint(2)
    color_2 = np.random.randint(2)
    char = np.random.randint(2)
    
    im, out = make_im(color_b, include_1, shape_1, color_1, include_2, shape_2, color_2, char)
    return im, y, out
