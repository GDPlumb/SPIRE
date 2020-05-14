
import numpy as np

# Create a completely neutral distribution
def sample_uniform():
    feature = np.zeros((6))
    feature[0] = np.random.randint(2) #character
    feature[1] = np.random.uniform() #x
    feature[2] = np.random.uniform() #y
    feature[3] = np.random.randint(2) #color of character
    feature[4] = np.random.uniform() # shade
    feature[5] = np.random.randint(2) #sticker
    return feature, feature[0]


# Create a correlation between the character and the sticker
def sample_1(p = 1.0):
    feature = np.zeros((6))
    feature[0] = np.random.randint(2) #character
    feature[1] = np.random.uniform() #x
    feature[2] = np.random.uniform() #y
    feature[3] = np.random.randint(2) #color of character
    feature[4] = np.random.uniform() # shade
    if np.random.uniform() < p:
        feature[5] = feature[0]
    else:
        feature[5] = np.random.randint(2)
    return feature, feature[0]
