from imp import reload
import pandas as pd
import numpy as np
import os
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')


def set_keras_backend(backend):
    """
    Function that sets backend of cnn
    :param backend: backend parameter to set
    :return: nothing
    """

    if K.backend() != backend:
       os.environ['KERAS_BACKEND'] = backend
       reload(K)
       assert K.backend() == backend

# call the function with "theano" backend
set_keras_backend("theano")

# load train_x
train_x = np.load('train_x.npy')

# load train_y
xls = pd.ExcelFile("train_labels.xlsx")
df1 = xls.parse(0)
train_y = df1.values

# load test_x
test_x = np.load('test_x.npy')

# setup model
model = Sequential()

model.add(Convolution2D(128, (3, 3), activation='relu', input_shape=(256, 256, 3), data_format='channels_first'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# compile and train the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=32, epochs=10, verbose=1)

# test the model
predicts = model.predict(test_x)
predicts = np.argmax(predicts, axis=1)
print(predicts)

