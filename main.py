import tensorflow as tf
import numpy as np

from model import RicianNet

X_train = np.zeros()
Y_train = np.zeros()
X_test = np.zeros()
Y_test = np.zeros()
X_valid = np.zeros()
Y_valid = np.zeros()

tf.config.run_functions_eagerly(True)

if __name__ == "__main__":
    model = RicianNet()
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    EPOCHS=2
    # training
    batch_sz = 100
    import time
    start_time=time.time()
    model.fit(X_train,Y_train, epochs=EPOCHS, verbose=1,validation_data=(X_valid,Y_valid),batch_size = batch_sz)
    print("--- %s Training time in minutes ---" % ((time.time() - start_time)/60))


    # testing
    import math;
    scores = model.evaluate(X_test, Y_test, batch_size=80, verbose=1)
    print('\nTest MSE dB: %.5f loss: %.5f' % (10*math.log(scores[0], 10),scores[1])) 