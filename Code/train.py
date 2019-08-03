
# TODO:
#  -  modify batch manager and setup so that X_reg is not always needed

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from Models import MLP
from DataManager import BatchManager
from Regularizers import Invariance, Monotonicity

import os
# The networks are small enough that training is faster on CPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def train_eval(x, y, objective, test_size = 0.25,
        hidden_layer_sizes = [10, 10],
        batch_size = 8, learning_rate = 0.01,
        min_epochs = 100, stopping_epochs = 50,  tol = 0.001,
        heuristics = None,
        eval_func = None, x_test = None, y_test = None, flag = None,
        name = None):

    # Setup working directory
    cwd = os.getcwd()
    if name is None:
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

    # Setup batch manager
    x_t, x_v, y_t, y_v = train_test_split(x, y, test_size = test_size)
    bm = BatchManager(x_t, y_t, batch_size)

    # Graph Inputs
    n_input = x.shape[1]
    n_out = y.shape[1]

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

    # Define the model loss
    if objective == "binary_classification":
        model_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = pred))
        tf.summary.scalar("Loss/CE", model_loss)
        
        pred_binary = tf.round(tf.nn.sigmoid(pred))
        perf_op = tf.reduce_mean(tf.cast(tf.equal(Y, pred_binary), tf.float32))
        tf.summary.scalar("Metrics/Acc", perf_op)

        lower_is_better = False

    elif objective == "regression":
        model_loss = tf.losses.mean_squared_error(labels = Y, predictions = pred)
        tf.summary.scalar("Loss/MSE", model_loss)

        perf_op = model_loss

        lower_is_better = True

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

    # Finish Summaries
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
        if not lower_is_better:
            best_perf = -1.0 * np.inf
        else:
            best_perf = np.inf

        epoch = 0
        while True:

            # Stopping condition
            if epoch - best_epoch > stopping_epochs and epoch > min_epochs:
                break

            # Run a training epoch
            total_batch = int(x.shape[0] / batch_size)
            for i in range(total_batch):
                x_b, y_b = bm.next_batch()
                sess.run(train_op, feed_dict = {X: x_b, Y: y_b, X_reg: x_b})
            
   
            summary_train = sess.run(summary_op, feed_dict = {X: x_t, Y: y_t, X_reg: x_t})
            train_writer.add_summary(summary_train, epoch)
            
            summary_val, val_perf = sess.run([summary_op, perf_op], feed_dict = {X: x_v, Y: y_v, X_reg: x_v})
            val_writer.add_summary(summary_val, epoch)
            
            if not lower_is_better and val_perf > best_perf + tol:
                best_epoch = epoch
                best_perf = val_perf
                saver.save(sess, "./model.cpkt")
            elif lower_is_better and val_perf < best_perf - tol:
                best_epoch = epoch
                best_perf = val_perf
                saver.save(sess, "./model.cpkt")
            
            epoch += 1

        # Restore the chosen model
        saver.restore(sess, "./model.cpkt")

        # Run the model evaluation
        out = None
        if eval_func is not None:
            if flag is None:
                eval_func(sess, pred, X)
            elif flag == "UCI":
                out = eval_func(sess, pred, pred_binary, perf_op, X, Y, x_test, y_test)
        else:
            print(sess.run(perf_op, {X: x_test, Y: y_test}))

        # Reset
        os.chdir(cwd)

        return out
