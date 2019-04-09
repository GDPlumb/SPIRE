
# TODO
# Separate data gen and heuristic (put into a class that you pass)
# Create reguglarizer classes (pass the perturbation function)

import math
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate import gen_logic

import sys
sys.path.insert(0, "../Code/")
from models import MLP

import os
# The networks are small enough that training is faster on CPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def run(gen_func, n, problem,
        hidden_layer_sizes, lr,
        reg = 10000.0):

    # Allow multiple sessions on a single GPU.
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # Reset TF graph (avoids issues with repeat experiments)
    tf.reset_default_graph()

    # Generate the data
    x, y = gen_func(n)
    
    n_input = x.shape[1]
    n_out = y.shape[1]
    
    # Graph INputs
    X = tf.placeholder("float", [None, n_input], name = "X_in")
    Y = tf.placeholder("float", [None, n_out], name = "Y_in")

    # Construct the model
    shape = [n_input]
    for size in hidden_layer_sizes:
        shape.append(size)
    shape.append(n_out)

    network = MLP(shape)
    with tf.variable_scope("model", reuse = tf.AUTO_REUSE):
        pred = network.model(X)

    # Define the loss
    if problem == "binary_classification":
        model_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = pred))
        tf.summary.scalar("Loss/CE", model_loss)
        
        pred_binary = tf.round(tf.nn.sigmoid(pred))

        _, perf_op = tf.metrics.accuracy(labels = Y, predictions = pred_binary)
        tf.summary.scalar("Metrics/Acc", perf_op)
      
    # Hueristic
    X_reg = tf.placeholder("float", [None, n_input], name = "X_reg")
    noise = tf.random_uniform([tf.shape(X_reg)[0],2], minval = -0.1, maxval = 0.1)
    
    X_reg_pert = X_reg + tf.pad(noise, [[0,0], [2,0]])
    
    with tf.variable_scope("model", reuse = tf.AUTO_REUSE):
        pred_reg = network.model(X_reg)
    
    with tf.variable_scope("model", reuse = tf.AUTO_REUSE):
        pred_reg_pert = network.model(X_reg_pert)

    reg_loss = reg * tf.losses.mean_squared_error(labels = pred_reg, predictions = pred_reg_pert)
    tf.summary.scalar("Loss/reg", reg_loss)
    
    # Finish Loss
    loss_op = model_loss + reg_loss
    tf.summary.scalar("Loss/final", loss_op)
    summary_op = tf.summary.merge_all()

    # Define the optimization process
    optimizer = tf.train.AdamOptimizer(learning_rate = lr)
    train_op = optimizer.minimize(loss_op)

    # Do stuff
    init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    with tf.Session(config = tf_config) as sess:
        train_writer = tf.summary.FileWriter("train_" + str(reg), sess.graph)
        
        # Train Model
        sess.run(init)
        
        for i in range(100):
            for j in range(500):
                sess.run(train_op, feed_dict = {X: np.reshape(x[j], (1,4)), Y:  np.reshape(y[j], (1,1)), X_reg: np.reshape(x[j], (1,4))})
                if j % 10 == 0:
                    summary = sess.run(summary_op, feed_dict = {X: x, Y: y, X_reg: x})
                    train_writer.add_summary(summary, i)

        # Plot the learned function for dim 0 and 1
        
        n = 1000
        
        for eval in ["zeros", "random", "bad"]:
        
            x = np.random.uniform(size=(n, 2))

            if eval == "zeros":
                x = np.hstack((x, np.zeros((n,2))))
            elif eval == "random":
                x = np.hstack((x, np.random.uniform(size=(n, 2))))
            elif eval == "bad":
                x = np.hstack((x,x))
            y_hat = sess.run(pred_binary, feed_dict = {X: x})

            indices_0 = np.where(y_hat == 0)[0]
            indices_1 = np.where(y_hat == 1)[0]
            
            plt.scatter(x[indices_0, 0], x[indices_0, 1], marker='x')
            plt.scatter(x[indices_1, 0], x[indices_1, 1], marker='+')
            
            plt.savefig(str(reg) + "_" + eval + ".pdf")

            plt.close()


run(gen_logic, 500, "binary_classification", [10,10], 0.01, reg = 0.0)
run(gen_logic, 500, "binary_classification", [10,10], 0.01, reg = 100000.0)
