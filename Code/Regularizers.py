
import tensorflow as tf

# TODO:  add a parameter for the noise distribution
def PredPert(network, X_reg, d, v):
    with tf.variable_scope("PredPert_" + str(d) + "_" + str(v)):
        dim = tf.shape(X_reg)[1]
        noise = tf.random_uniform([tf.shape(X_reg)[0],1], minval = -1.0 * v, maxval = v)
        if d == 0:
            noise_padded = tf.pad(noise, [[0,0], [0, dim - 1]])
        else:
            noise_padded = tf.pad(noise, [[0,0], [d, dim - d - 1]])
        X_reg_pert = X_reg + noise_padded

    with tf.variable_scope("model", reuse = tf.AUTO_REUSE):
        pred_reg_pert = network.model(X_reg_pert)

        return pred_reg_pert, noise

def Invariance(network, X_reg, d, v, pred_reg, w):
    pred_reg_pert,_ = PredPert(network, X_reg, d, v)
    return w * tf.losses.mean_squared_error(labels = pred_reg, predictions = pred_reg_pert)

# TODO:  does this work for multidimensional outputs?
def Monotonicity(network, X_reg, d, v, pred_reg, w, dir):
    pred_reg_pert, noise = PredPert(network, X_reg, d, v)
    return w * tf.reduce_mean(tf.nn.relu(-1.0 * dir * tf.sign(noise) * (pred_reg_pert - pred_reg)))

