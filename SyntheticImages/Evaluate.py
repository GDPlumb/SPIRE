import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
import tensorflow as tf

sys.path.insert(0, "../Code/")
from Train import train

def loss(model, inputs, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = model(inputs), labels = labels))

def evaluate(sample, n = 10000, p = 0.9, n_neutral = 200, name = "Experiment"):
    
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
    
    # Train a model
    
    tf.keras.backend.clear_session()

    model = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape=(64,64,3)),
              tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
              tf.keras.layers.MaxPooling2D((2, 2)),
              tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
              tf.keras.layers.MaxPooling2D((2, 2)),
              tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
              tf.keras.layers.MaxPooling2D((2, 2)),
              tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
              tf.keras.layers.MaxPooling2D((2, 2)),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(1)
              ])

    train(model, loss, X_train, Y_train, X_val, Y_val, name)
    
    return model, X_train, X_val, X_test, X_neutral, Y_train, Y_val, Y_test, Y_neutral, meta_train, meta_val, meta_test
