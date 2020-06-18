
import tensorflow as tf

def get_model():
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
              
    return model

def loss(model, inputs, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = model(inputs), labels = labels))

