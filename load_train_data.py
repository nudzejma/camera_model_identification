from random import shuffle
import pandas as pd
import numpy as np
import os
from PIL import Image

np.random.seed(123)


def read_file(file_path, counter, new_file_paths):
    """
    Reads the image and constructs numpy array of her pixels values
    :param file_path: path of file
    :param counter: current number of file to be processed
    :param new_file_paths: array of new file paths
    :return: numpy array of image file
    """
    print("counter", counter)

    im_array = np.array(Image.open(file_path), dtype="uint8")
    pil_im = Image.fromarray(im_array)
    new_array = np.array(pil_im.resize((256, 256)))
    npy_file_name = 'train_data' + os.sep + get_class_from_path(file_path) + os.sep + os.path.basename(file_path)[
                                                                                      :-4] + '.npy'
    new_file_paths.append(npy_file_name)
    np.save(npy_file_name, new_array / 255)
    return new_array / 255


def label_transform(labels):
    """
    Transforms labels into indexes
    :param labels: labels to transform
    :return: tuple of labels and indexes
    """
    labels = pd.get_dummies(pd.Series(labels))
    label_index = labels.columns.values
    labels.to_excel('train_lables.xlsx', sheet_name='labels')
    return labels, label_index


def get_class_from_path(filepath):
    """
    From file path gets directory name to know which class image file belongs to
    :param filepath: path of image file
    :return: class name
    """
    return os.path.dirname(filepath).split(os.sep)[-1]


list_paths = []
list_classes = []
# loading and preparing train data
for subdir, dirs, files in os.walk("train"):
    for folder in dirs:
        folder_path = "train" + os.sep + folder
        list_classes.append(folder)
        for subdir, dirs, files in os.walk(folder_path):
            for file in files:
                filepath = folder_path + os.sep + file
                list_paths.append(filepath)

counter = 0
new_file_paths = []
for filepath in list_paths:
    read_file(filepath, counter, new_file_paths)
    counter += 1

#   load files and put them in x_test
#   save the file paths of numpy files so we can load them later
new_file_paths = []
for subdir, dirs, files in os.walk("train_data"):
    for folder in dirs:
        folder_path = "train_data" + os.sep + folder
        list_classes.append(folder)
        for subdir, dirs, files in os.walk(folder_path):
            for file in files:
                filepath = folder_path + os.sep + file
                new_file_paths.append(filepath)
shuffle(new_file_paths)
labels = [get_class_from_path(filepath) for filepath in new_file_paths]
y, label_index = label_transform(labels)

#   load numpy arrays and append them in one train_x.npy file
train_x = []
counter = 0
for file in new_file_paths:
    array = np.load(file)
    train_x.append(array)
    print('counter', counter, file)
    counter += 1

np.save('train_x.npy', train_x)