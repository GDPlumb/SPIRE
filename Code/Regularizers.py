
import tensorflow as tf

def Invariance(network, X_reg, d, v):
    with tf.variable_scope("invariance_" + str(d) + "_" + str(v)):
        dim = tf.shape(X_reg)[1]
        noise = tf.random_uniform([tf.shape(X_reg)[0],1], minval = -1.0 * v, maxval = -1.0 * v)
        if d == 0:
            noise_padded = tf.pad(noise, [[0,0], [0, dim - 1]])
        else:
            noise_padded = tf.pad(noise, [[0,0], [d, dim - d - 1]])
        X_reg_pert = X_reg + noise_padded

    with tf.variable_scope("model", reuse = tf.AUTO_REUSE):
        pred_reg_pert = network.model(X_reg_pert)

        return pred_reg_pert
