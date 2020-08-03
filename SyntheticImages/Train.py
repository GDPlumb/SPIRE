
import math
import numpy as np
import tensorflow as tf

from BatchManager import BatchManager

def train(model, loss, X_train, y_train, X_val, y_val, model_path,
        learning_rate = 0.001, learning_rate_decay = 0.3, learning_rate_drops = 5,
        batch_size = 64, min_epochs = 3, stopping_epochs = 3, stopping_tol = 0.0001):
             
    # Define the gradient
    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)
    
    # Setup the batch manager
    n = X_train.shape[0]
    bm = BatchManager(X_train, y_train)
    batches_per_epoch = math.floor(n / batch_size)
    
    bm_val = BatchManager(X_val, y_val)
    n_val = X_val.shape[0]
    batches_per_epoch_val = math.floor(n_val / batch_size)
    
    # Setup the initial optimizer (it will be re-initialized when the learning rate gets dropped)
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    # Basic counters for the training process
    epoch = 0
    best_epoch = 0
    best_loss = np.inf
    drops = 0

    # Run the training loop
    while True:

        # Check the stopping condition
        if epoch - best_epoch > stopping_epochs and epoch > min_epochs:
            # We have finished only if we have fully decayed the learning_rate
            if drops == learning_rate_drops:
                break
            else:
                print("Dropping learning_rate")
                learning_rate *= learning_rate_decay
                optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
                drops += 1
            
        # Train for an epoch
        epoch_loss_avg = tf.keras.metrics.Mean()
        for i in range(batches_per_epoch):
            x_batch, y_batch = bm.next_batch(batch_size = batch_size)
            loss_value, grads = grad(model, x_batch, y_batch)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_avg.update_state(loss_value)
        epoch_loss = epoch_loss_avg.result().numpy()
        
        # Calculate the validation loss
        epoch_loss_avg_val = tf.keras.metrics.Mean()
        for i in range(batches_per_epoch_val):
            x_batch, y_batch = bm_val.next_batch(batch_size = batch_size)
            loss_value = loss(model, x_batch, y_batch)
            epoch_loss_avg_val.update_state(loss_value)
        value = epoch_loss_avg_val.result().numpy()
        
        # Check if we have made progress
        if value < best_loss - stopping_tol:
            print("Epoch / Epoch Train Loss / Val Loss: " + str(epoch) + " " + str(epoch_loss) + " " + str(value) + " -> saving")
            best_loss = value
            best_epoch = epoch
            model.save_weights(model_path)
        else:
            print("Epoch / Epoch Train Loss / Val Loss: " + str(epoch) + " " + str(epoch_loss) + " " + str(value))
        
        # Update counters
        epoch += 1
        
    model.load_weights(model_path)
