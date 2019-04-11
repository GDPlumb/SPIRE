
# TODO
# Separate data gen and heuristic (put into a class that you pass)
# Create reguglarizer classes (pass the perturbation function)

import math
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from problem_logical import Logical

import sys
sys.path.insert(0, "../Code/")
from Models import MLP
from DataManager import BatchManager
from Regularizers import Invariance

import os
# The networks are small enough that training is faster on CPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def run(problem, n, objective,
        hidden_layer_sizes,
        min_epochs = 100, stopping_epochs = 10, batch_size = 8, learning_rate = 0.01, tol = 0.0005,
        reg = False):
    
    if not reg:
        os.chdir("unreg")
    else:
        os.chdir("reg")

    # Allow multiple sessions on a single GPU.
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # Reset TF graph (avoids issues with repeat experiments)
    tf.reset_default_graph()

    # Generate the data
    x = problem.gen_bad(n)
    y = problem.label(x)
    
    x_t, x_v, y_t, y_v = train_test_split(x, y, test_size = 0.2)
    
    bm = BatchManager(x_t, y_t, batch_size)
    
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
    if objective == "binary_classification":
        model_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = pred))
        tf.summary.scalar("Loss/CE", model_loss)
        
        pred_binary = tf.round(tf.nn.sigmoid(pred))

        _, perf_op = tf.metrics.accuracy(labels = Y, predictions = pred_binary)
        tf.summary.scalar("Metrics/Acc", perf_op)

    loss_op = model_loss
      
    # Hueristics
    X_reg = tf.placeholder("float", [None, n_input], name = "X_reg") #TODO: Don't need to pass this for unregularized model
    if reg:

        with tf.variable_scope("model", reuse = tf.AUTO_REUSE):
            pred_reg = network.model(X_reg)

        c = 0
        for item in problem.heuristics:
            with tf.name_scope("heuristic_" + str(c)) as scope:
                if item[0] == "inv":
                    pred_reg_pert = Invariance(network, X_reg, item[1], item[2])

                reg_loss = item[3] * tf.losses.mean_squared_error(labels = pred_reg, predictions = pred_reg_pert)
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
        best_perf = 0.0
        epoch = 0
        while True:

            # Stopping condition
            if epoch - best_epoch > stopping_epochs and epoch > min_epochs:
                break

            # Run a training epoch
            total_batch = int(n / batch_size)
            for i in range(total_batch):
                x_b, y_b = bm.next_batch()
                sess.run(train_op, feed_dict = {X: x_b, Y: y_b, X_reg: x_b})
            
            summary = sess.run(summary_op, feed_dict = {X: x_t, Y: y_t, X_reg: x_t})
            train_writer.add_summary(summary, epoch)
            
            summary, val_perf = sess.run([summary_op, perf_op], feed_dict = {X: x_v, Y: y_v, X_reg: x_v})
            val_writer.add_summary(summary, epoch)
            
            if val_perf > best_perf + tol:
                best_epoch = epoch
                best_perf = val_perf
                saver.save(sess, "./model.cpkt")
                
            epoch += 1

        # Restore the chosen model
        saver.restore(sess, "./model.cpkt")

        # Plot the learned function for dim 0 and 1
        n = 500
        c = 1
        for eval in ["bad", "random", "zeros"]:
        
            if eval == "bad":
                x = problem.gen_bad(n)
            elif eval == "random":
                x = problem.gen_random(n)
            elif eval == "zeros":
                x = problem.gen_zeros(n)

            y_hat = sess.run(pred_binary, feed_dict = {X: x})

            indices_0 = np.where(y_hat == 0)[0]
            indices_1 = np.where(y_hat == 1)[0]

            plt.subplot(3, 1, c)
            c += 1
            plt.scatter(x[indices_0, 0], x[indices_0, 1], marker='x')
            plt.scatter(x[indices_1, 0], x[indices_1, 1], marker='+')
            plt.title(eval)
            
        plt.savefig("out.pdf")

        plt.close()

        os.chdir("..")

problem = Logical([["inv", 2, 0.1, 100.0], ["inv", 3, 0.1, 100.0]])
run(problem, 100, "binary_classification", [5,5])
run(problem, 100, "binary_classification", [5,5], reg = True)

