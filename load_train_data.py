from random import shuffle

import pandas as pd
import numpy as np
import os
np.random.seed(123)
from PIL import Image

def read_file(file_path, counter, new_file_paths):
    print("counter", counter)

    im_array = np.array(Image.open(file_path), dtype="uint8")
    pil_im = Image.fromarray(im_array)
    new_array = np.array(pil_im.resize((256, 256)))
    npy_file_name = 'train_data' + os.sep + get_class_from_path(file_path) + os.sep + os.path.basename(file_path)[
                                                                                      :-4] + '.npy'
    new_file_paths.append(npy_file_name)
    np.save(npy_file_name, new_array / 255)
    return new_array / 255


def concat_to_frame(data_frame, concat_data_frame):
    return pd.concat([data_frame, concat_data_frame], ignore_index=True)


def label_transform(labels):
    labels = pd.get_dummies(pd.Series(labels))
    label_index = labels.columns.values
    labels.to_excel('train_lables.xlsx', sheet_name='labels')
    return labels, label_index


def get_class_from_path(filepath):
    str = os.path.dirname(filepath).split(os.sep)[-1]
    return os.path.dirname(filepath).split(os.sep)[-1]


list_paths = []
list_classes = []
# loading and preparing train data
for subdir, dirs, files in os.walk("train"):
    for folder in dirs:
        folder_path = "train" + os.sep + folder
        list_classes.append(folder)
        print("folder_path ", folder_path)
        for subdir, dirs, files in os.walk(folder_path):
            for file in files:
                filepath = folder_path + os.sep + file
                list_paths.append(filepath)

counter = 0
new_file_paths = []
for filepath in list_paths:
    read_file(filepath, counter, new_file_paths)
    counter += 1
    # load files and put them in x_train
new_file_paths = []
for subdir, dirs, files in os.walk("train_data"):
    for folder in dirs:
        folder_path = "train_data" + os.sep + folder
        list_classes.append(folder)
        print("folder_path ", folder_path)
        for subdir, dirs, files in os.walk(folder_path):
            for file in files:
                filepath = folder_path + os.sep + file
                new_file_paths.append(filepath)
shuffle(new_file_paths)
labels = [get_class_from_path(filepath) for filepath in new_file_paths]
y, label_index = label_transform(labels)

train_x = []
counter = 0
print('loading numpy arrays')
for file in new_file_paths:
    array = np.load(file)
    train_x.append(array)
    print('counter', counter, file)
    counter += 1

np.save('train_x.npy', train_x)