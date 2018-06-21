import os

import numpy as np

#
# train_x = []
# counter = 0
# for subdir, dirs, files in os.walk("train_data"):
#     for file in files:
#         file_path = 'train_data/' + file
#         array = np.load(file_path)
#         train_x.append(array)
#         print('counter', counter, file_path)
#         counter += 1
#
# np.save('train_x.npy', train_x)

array = np.load('train_x.npy')
print(array.shape)