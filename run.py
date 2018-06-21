from imp import reload
from random import shuffle

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import hickle as hkl
import os
np.random.seed(123)
from PIL import Image
import theano
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.datasets import mnist
from keras import backend as K
K.set_image_dim_ordering('th')

def set_keras_backend(backend):
    if K.backend() != backend:
       os.environ['KERAS_BACKEND'] = backend
       reload(K)
       assert K.backend() == backend

# call the function with "theano"
set_keras_backend("theano")
print(K.backend())
# load train_x
train_x = np.load('train_x.npy')
print('loaded train_x')
# load train_y
xls = pd.ExcelFile("train_labels.xlsx")
df1 = xls.parse(0)
train_y = df1.values
print('loaded train_y')
#load test_x
test_x = np.load('test_x.npy')
print('loaded test_x')
model = Sequential()

model.add(Convolution2D(128, (3, 3), activation='relu', input_shape=(3, 256,256), data_format='channels_first'))
print('conv1')
model.add(Convolution2D(64, (3, 3), activation='relu'))
print('conv2')
model.add(MaxPooling2D(pool_size=(2, 2)))
print('mp1')
model.add(Dropout(0.25))
print('drop')
model.add(Flatten())
print('flatten')
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

print('model')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('model compiled')
model.fit(train_x, train_y, batch_size=32, nb_epoch=10, verbose=1)
predicts = model.predict(test_x)
predicts = np.argmax(predicts, axis=1)
print(predicts)
# predicts = [label_index[p] for p in predicts]
#
# df = pd.DataFrame(columns=['fname', 'camera'])
# df['fname'] = index
# df['camera'] = predicts
# df.to_csv("sub.csv", index=False)
# score = model.evaluate(X_test, Y_test, verbose=0)
# plt.imshow(X_train[1])
# plt.show()
