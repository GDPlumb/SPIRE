
import numpy as np
from sklearn.model_selection import train_test_split

from Images import make_im

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

# Sample datasets
def load(sample, n, p, n_neutral):

    # Create a training dataset with the spurious correlation
    X = np.zeros((n, 64, 64, 3))
    Y = np.zeros((n, 1))
    meta = []

    for i in range(n):
        x, y, out = sample(p = p)
        X[i, :] = x
        Y[i] = y
        meta.append(out)

    X = np.float32(X) / 255
    Y = np.float32(Y)

    X_train, X_test, Y_train, Y_test, meta_train, meta_test = train_test_split(X, Y, meta, test_size = 0.25)
    X_val, X_test, Y_val, Y_test, meta_val, meta_test = train_test_split(X_test, Y_test, meta_test, test_size = 0.5)

    # Create a neutral dataset without the spurious correlation for evaluation
  
    X_neutral = np.zeros((n_neutral, 64, 64, 3))
    Y_neutral = np.zeros((n_neutral, 1))

    for i in range(n_neutral):
        x, y, out = sample(p = 0.5)
        X_neutral[i, :] = x
        Y_neutral[i] = y

    X_neutral = np.float32(X_neutral) / 255
    Y_neutral = np.float32(Y_neutral)
    
    return  X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, meta_train, meta_val, meta_test
