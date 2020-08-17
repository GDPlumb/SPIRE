

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

def eval(sample, n, p, n_neutral, heuristic):

    out = np.zeros((4))

    X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, meta_train, meta_val, meta_test = \
    load(sample, n, p, n_neutral)

    model = get_model()
    train(model, loss, X_train, Y_train, X_val, Y_val, "Models/original")
    out[0] = acc(model, X_test, Y_test)
    out[1] = acc(model, X_neutral, Y_neutral)


    X_train_aug, Y_train_aug = augment(X_train, Y_train, meta_train, heuristic)
    X_val_aug, Y_val_aug = augment(X_val, Y_val, meta_val, heuristic)

    train(model, loss, X_train_aug, Y_train_aug, X_val_aug, Y_val_aug, "Models/new")
    out[2] = acc(model, X_test, Y_test)
    out[3] = acc(model, X_neutral, Y_neutral)
    
    return out
    
if __name__ == "__main__":
    mode = 3
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

    if mode == 1:
        from Heuristics import heuristic_1 as heuristic
        from Sample import sample_1 as sample
    elif mode == 2:
        from Heuristics import heuristic_2 as heuristic
        from Sample import sample_2 as sample
    elif mode == 3:
        from Heuristics import heuristic_3 as heuristic
        from Sample import sample_3 as sample
    
    n_array = [5000, 10000, 15000, 20000]
    p_array = [0.5, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    n_runs = len(n_array) * len(p_array)
    
    result = np.zeros((n_runs, 6))
    c = 0
    for n in n_array:
        for p in p_array:
            config_avg = np.zeros((4))
            for trial in range(n_trials):
                config_avg += eval(sample, n, p, n_neutral, heuristic)
            config_avg /= n_trials
            result[c, 0] = n
            result[c, 1] = p
            for i in range(4):
                result[c, 2 + i] = config_avg[i]
            
            c += 1
            print()
            print("Finished ", c, " out of ", n_runs)
            print()
            
    np.savetxt("Run-result.csv", np.round(result, 3), fmt = "%f", delimiter=",", header = "n, p, Original - Test, Original - Neutral, New - Test, New - Neutral")
