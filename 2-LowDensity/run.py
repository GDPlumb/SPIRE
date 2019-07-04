
import json
import math
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from problem_density import LowDensity

import sys
sys.path.insert(0, "../Code/")
from Models import MLP
from DataManager import BatchManager
from Regularizers import Invariance, Monotonicity

import os
# The networks are small enough that training is faster on CPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def perturb(x, indices):
    x_new = np.copy(x)
    for i in indices:
        x_new[i] += np.random.uniform(low = -0.1, high = 0.1)
    return x_new

def run(problem, num_data = 100,
        objective = "regression",
        hidden_layer_sizes = [10, 10],
        batch_size = 8, learning_rate = 0.01,
        min_epochs = 100, stopping_epochs = 50,  tol = 0.001,
        heuristics = None, checks = None):
    
    # Setup working directory
    cwd = os.getcwd()
    name = "TB/"
    if heuristics is not None:
        name += str(heuristics)
    else:
        name += "none/"
    os.makedirs(name)
    os.chdir(name)

    # Allow multiple sessions on a single GPU.
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # Reset TF graph (avoids issues with repeat experiments)
    tf.reset_default_graph()

    # Generate the data
    x = problem.gen(num_data)
    y = problem.label(x)
    
    x_t, x_v, y_t, y_v = train_test_split(x, y, test_size = 0.25)

    bm = BatchManager(x_t, y_t, batch_size)
    
    n_input = x.shape[1]
    n_out = y.shape[1]
    
    # Graph Inputs
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
    if objective == "regression":
        model_loss = tf.losses.mean_squared_error(labels = Y, predictions = pred)
        tf.summary.scalar("Loss/MSE", model_loss)

    loss_op = model_loss
      
    # Add the heuristics to the loss
    X_reg = tf.placeholder("float", [None, n_input], name = "X_reg")
    if heuristics is not None:

        with tf.variable_scope("model", reuse = tf.AUTO_REUSE):
            pred_reg = network.model(X_reg)

        c = 0
        for item in heuristics:
            with tf.name_scope("heuristic_" + str(c)) as scope:
                if item[0] == "inv":
                    reg_loss = Invariance(network, X_reg, item[1], item[2], pred_reg, item[3])
                elif item[0] == "mon":
                    reg_loss = Monotonicity(network, X_reg, item[1], item[2], pred_reg, item[3], item[4])

            tf.summary.scalar("Loss/reg_" + str(c), reg_loss)
                
            loss_op += reg_loss
                
            c += 1
    
    # Finish Loss
    tf.summary.scalar("Loss/final", loss_op)
    summary_op = tf.summary.merge_all()

    # Define the optimization process
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Run the experiment
    init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    saver = tf.train.Saver(max_to_keep = 1)

    with tf.Session(config = tf_config) as sess:
        train_writer = tf.summary.FileWriter("train", sess.graph)
        val_writer = tf.summary.FileWriter("val")

        # Train Model
        sess.run(init)
        
        best_epoch = 0.0
        best_perf = np.inf
        epoch = 0
        while True:

            # Stopping condition
            if epoch - best_epoch > stopping_epochs and epoch > min_epochs:
                break

            # Run a training epoch
            total_batch = int(num_data / batch_size)
            for i in range(total_batch):
                x_b, y_b = bm.next_batch()
                sess.run(train_op, feed_dict = {X: x_b, Y: y_b, X_reg: x_b})
                
            summary_train = sess.run(summary_op, feed_dict = {X: x_t, Y: y_t, X_reg: x_t})
            train_writer.add_summary(summary_train, epoch)
            
            summary_val, val_perf = sess.run([summary_op, model_loss], feed_dict = {X: x_v, Y: y_v, X_reg: x_v})
            val_writer.add_summary(summary_val, epoch)
            
            if val_perf < best_perf - tol:
                best_epoch = epoch
                best_perf = val_perf
                saver.save(sess, "./model.cpkt")
                
            epoch += 1

        # Restore the chosen model
        saver.restore(sess, "./model.cpkt")

        # Visualize the MSE across the two dimensions
        values = np.linspace(0, 1, 100)
        grid = np.zeros((10000, 2))
        c = 0
        for i in range(100):
            for j in range(100):
                grid[c, 0] = values[i]
                grid[c, 1] = values[j]
                c += 1

        grid_pred = sess.run(pred, feed_dict = {X: grid})
        grid_y = problem.label(grid, noise = 0.0)
        grid_error = np.transpose(np.reshape(grid_pred - grid_y, (100, 100)))

        plt.imshow(grid_error)
        plt.colorbar()
            
        plt.savefig("out.pdf")

        plt.close()

        # Evaluate whether or not the heuristic was actually enforced
        diffs = np.zeros((3))
        for i in range(10000):
            x = grid[i, :]
            x_pred = sess.run(pred, feed_dict = {X: np.reshape(x, (1,2))})
            
            c = 0
            
            # Evaluate invariance
            x_new = perturb(x, [1])
            x_pred_pert = sess.run(pred, feed_dict = {X: np.reshape(x_new, (1,2))})

            diffs[c] += (x_pred - x_pred_pert)**2
            c += 1
            
            # Evaluate monotonicity
            for index in [0, 1]:
                x_new = np.copy(x)
                x_new[index] += 0.05
                
                x_pred_pert = sess.run(pred, feed_dict = {X: np.reshape(x_new, (1,2))})
                
                diffs[c] += x_pred_pert - x_pred
                c += 1

        diffs /= x_v.shape[0]
        with open("tests.txt", "w") as outfile:
            json.dump(diffs.tolist(), outfile)

        os.chdir(cwd)

problem = LowDensity()
run(problem)

run(problem, heuristics = [["inv", 1, 0.1, 1000.0]])

run(problem, heuristics = [["mon", 0, 0.1, 0.1, 1.0]])

run(problem, heuristics = [["mon", 0, 0.1, 0.1, 1.0], ["inv", 1, 0.1, 1000.0]])

run(problem, heuristics = [["mon", 0, 0.1, 1.0, 1.0], ["inv", 1, 0.1, 1000.0]])

run(problem, heuristics = [["mon", 0, 0.1, 10.0, 1.0], ["inv", 1, 0.1, 1000.0]])


