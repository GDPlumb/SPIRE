
import numpy as np

from Data import make_im


# Spurious Correlation between the color of the background and the answer
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

# Spurious Correlation between the letter and the answer
def sample_2(p = 1.0):
    
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

    color_b = np.random.randint(2)
    color_1 = np.random.randint(2)
    color_2 = np.random.randint(2)
    
    # With probability p, set the character to match the label
    if np.random.uniform() < p:
        char = y
    else:
        char = (y + 1) % 2
    
    im, out = make_im(color_b, include_1, shape_1, color_1, include_2, shape_2, color_2, char)
    return im, y, out
    
# Spurious Correlation between the color of the first object and the answer
def sample_3(p = 1.0):
    
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

    color_b = np.random.randint(2)
    
    # With probability p, set the color of the first object to match the label
    if np.random.uniform() < p:
        color_1 = y
    else:
        color_1 = (y + 1) % 2
        
    color_2 = np.random.randint(2)
    char = np.random.randint(2)
    

    im, out = make_im(color_b, include_1, shape_1, color_1, include_2, shape_2, color_2, char)
    return im, y, out

