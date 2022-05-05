from tensorflow.keras import layers
import tensorflow as tf

tf.config.run_functions_eagerly(True)

def resblock(x, filters, kernelsize, bn=False):
    for i in range(4):
        fx = layers.Conv2D(filters, kernelsize, activation='relu', padding='same')(x)
        if bn:
            fx = layers.BatchNormalization()(fx)
    if bn:
        fx = layers.Conv2D(64, 3, padding='same')(fx)
    else:
        fx = layers.Conv2D(1, kernelsize, padding='same')(fx)
    out = layers.Add()([x,fx])
#     out = layers.ReLU()(out)
#     out = layers.BatchNormalization()(out)
    return out
