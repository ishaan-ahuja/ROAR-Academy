## This is course material for Introduction to Modern Artificial Intelligence
## Example code: mnist_cnn.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 500


# input image dimensions
img_rows, img_cols = 28, 28

# load the data built in Keras, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Display some examples from first 20 images
print(y_train[0:20])
plt.figure(1)
for i in range(20):
    plt.subplot(2,10,i+1)
    plt.imshow(x_train[i], cmap = plt.cm.binary)
plt.show()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# When calculating image data, convert from uint8 to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Reduce the element range from [0, 255] to [0, 1]
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# CNN is created also using Sequential model.   There are 32 channels in the first convolutional layer, and 64 channels in the second convolutional layer.
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='mean_squared_error',
              optimizer='adagrad',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.2)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

## Save the model
# model.save('MNIST_CNN.h5')