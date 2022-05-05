import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers, losses 
from tensorflow.keras.layers import Input, Add, subtract, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from scipy.io import loadmat
import pandas as pd
tf.config.run_functions_eagerly(True)

def RicianNet(input_shape = (10, 51, 2), feature_map=128):
    X_input= Input(shape=input_shape)
    X_shortcut=X_input
    
    X = resblock(X_input, feature_map, 7)
    X_two = Conv2D(64, 3, padding='same')(X)
    X = resblock(X_two, feature_map, 7, bn=True)
    X = Conv2D(64, (3, 3), padding='same')(X)
    X = layers.Add()([X_two, X])
    out = Conv2D(1, 7, padding='same')(X)
        
    model = Model(inputs = X_input, outputs = out, name='RicianNet')

    return model