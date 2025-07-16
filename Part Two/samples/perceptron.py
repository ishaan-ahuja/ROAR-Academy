## This is course material for Introduction to Modern Artificial Intelligence
## Example code: perceptron.py
## Author: Allen Y. Yang,  Intelligent Racing Inc.
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

# Please make sure to install tensorflow and keras modules by running pip commands:
# python -m pip install --upgrade pip
# python -m pip install tensorflow==2.10

import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from keras.activations import sigmoid

linearSeparableFlag = True
x_bias = 0

def toy_2D_samples(x_bias ,linearSeparableFlag):
    if linearSeparableFlag:
        samples1 = np.random.multivariate_normal([5+x_bias, 0], [[1, 0],[0, 1]], 100)
        samples2 = np.random.multivariate_normal([-5+x_bias, 0], [[1, 0],[0, 1]], 100)

        samples = np.concatenate((samples1, samples2 ), axis =0)
    
        # Plot the data
        plt.plot(samples1[:, 0], samples1[:, 1], 'bo')
        plt.plot(samples2[:, 0], samples2[:, 1], 'rx')
        plt.show()

    else:
        samples1 = np.random.multivariate_normal([5+x_bias, 5], [[1, 0],[0, 1]], 50)
        samples2 = np.random.multivariate_normal([-5+x_bias, -5], [[1, 0],[0, 1]], 50)
        samples3 = np.random.multivariate_normal([-5+x_bias, 5], [[1, 0],[0, 1]], 50)
        samples4 = np.random.multivariate_normal([5+x_bias, -5], [[1, 0],[0, 1]], 50)

        samples = np.concatenate((samples1, samples2, samples3, samples4 ), axis =0)
    
        # Plot the data
        plt.plot(samples1[:, 0], samples1[:, 1], 'bo')
        plt.plot(samples2[:, 0], samples2[:, 1], 'bo')
        plt.plot(samples3[:, 0], samples3[:, 1], 'rx')
        plt.plot(samples4[:, 0], samples4[:, 1], 'rx')
        plt.show()


    labels1 = np.zeros(100)
    labels2 = np.ones(100)
    labels = np.concatenate((labels1, labels2 ), axis =0)
    return samples, labels

samples, labels = toy_2D_samples(x_bias ,linearSeparableFlag)

# Split training and testing set

randomOrder = np.random.permutation(200)
trainingX = samples[randomOrder[0:100],:]
trainingY = labels[randomOrder[0:100]]
testingX = samples[randomOrder[100:200],:]
testingY = labels[randomOrder[100:200]]

model = Sequential()
model.add(Dense(1, input_shape=(2,), activation='sigmoid', use_bias=False))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['binary_accuracy'])
model.fit(trainingX, trainingY, epochs=100, batch_size=10, verbose=1, validation_split=0.2)

# score = model.evaluate(testingX, testingY, verbose=0)
score = 0
for i in range(100):
    output = model.predict(np.array([testingX[i,:]]))
    if output<0.5:
        estimate = 0
        plt.plot(testingX[i, 0], testingX[i, 1], 'bo')
    else: 
        estimate = 1
        plt.plot(testingX[i, 0], testingX[i, 1], 'rx')

    if estimate == testingY[i]:
        score = score  + 1

# Draw decision boundary line
weights = model.get_weights()[0].flatten()  # shape (2,)
w1, w2 = weights[0], weights[1]
# For sigmoid, decision boundary is w1*x1 + w2*x2 = 0
x_vals = np.linspace(np.min(samples[:,0])-1, np.max(samples[:,0])+1, 100)
if w2 != 0:
    y_vals = -w1/w2 * x_vals
    plt.plot(x_vals, y_vals, 'k--', label='Decision boundary')
    plt.legend()

plt.show()
print('Test accuracy:', score)