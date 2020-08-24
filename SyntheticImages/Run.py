

import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf


from Explore import explore
from Heuristics import augment
from Load import load
from Model import get_model, loss

from Core import acc
from Train import train

def eval(mode, n, p, n_neutral):

    if mode == 1:
        from Heuristics import heuristic_1 as heuristic_base
        from Sample import sample_1 as sample
    elif mode == 2:
        from Heuristics import heuristic_2 as heuristic_base
        from Sample import sample_2 as sample
    elif mode == 3:
        from Heuristics import heuristic_3 as heuristic_base
        from Sample import sample_3 as sample

    def heuristic_weak(im, meta):
        return heuristic_base(im, meta, use_strong = False)
        
    def heuristic_strong(im, meta):
        return heuristic_base(im, meta, use_strong = True)
    
    out = np.zeros((6))

    # Generate the data and train a model on it
    X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, meta_train, meta_val, meta_test = load(sample, n, p, n_neutral)

    model = get_model()
    train(model, loss, X_train, Y_train, X_val, Y_val, "Models/original")
    out[0] = acc(model, X_test, Y_test)
    out[1] = acc(model, X_neutral, Y_neutral)

    # Augment the data with the weak heuristic and fine tune
    X_train_aug, Y_train_aug = augment(X_train, Y_train, meta_train, heuristic_weak)
    X_val_aug, Y_val_aug = augment(X_val, Y_val, meta_val, heuristic_weak)

    model.load_weights("Models/original")
    train(model, loss, X_train_aug, Y_train_aug, X_val_aug, Y_val_aug, "Models/new")
    out[2] = acc(model, X_test, Y_test)
    out[3] = acc(model, X_neutral, Y_neutral)

    # Augment the data with the strong heuristic and fine tune
    if mode != 2:
        X_train_aug, Y_train_aug = augment(X_train, Y_train, meta_train, heuristic_strong)
        X_val_aug, Y_val_aug = augment(X_val, Y_val, meta_val, heuristic_strong)

        model.load_weights("Models/original")
        train(model, loss, X_train_aug, Y_train_aug, X_val_aug, Y_val_aug, "Models/new")
        out[4] = acc(model, X_test, Y_test)
        out[5] = acc(model, X_neutral, Y_neutral)
        
    return out
    
if __name__ == "__main__":

    n_neutral = 200
    n_trials = 5

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

    modes = [1,2,3]
    n_array = [5000, 10000, 15000, 20000]
    p_array = [0.5, 0.8, 0.85, 0.9, 0.95, 1.0]

    n_runs = len(modes) * len(n_array) * len(p_array)

    result = np.zeros((n_runs, 9))
    c = 0
    for mode in modes:
        for n in n_array:
            for p in p_array:
                config_avg = np.zeros((6))
                for trial in range(n_trials):
                    config_avg += eval(mode, n, p, n_neutral)
                config_avg /= n_trials
                result[c, 0] = mode
                result[c, 1] = n
                result[c, 2] = p
                for i in range(6):
                    result[c, 3 + i] = config_avg[i]
                
                c += 1
                np.savetxt("Run.csv", np.round(result, 3), fmt = "%f", delimiter=",", header = "mode, n, p, Original - Test, Original - Neutral, Weak - Test, Weak - Neutral, Strong - Test, Strong - Neutral")
