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
    npy_file_name = 'test_data' + os.sep + os.path.basename(file_path)[:-4] + '.npy'
    new_file_paths.append(npy_file_name)
    np.save(npy_file_name, new_array / 255)
    return new_array / 255


list_paths = []
list_classes = []
for subdir, dirs, files in os.walk("test"):
    for file in files:
        filepath = "test" + os.sep + file
        list_paths.append(filepath)

counter = 0
new_file_paths = []
for filepath in list_paths:
    read_file(filepath, counter, new_file_paths)
    counter += 1

# load files and put them in x_test
new_file_paths = []
for subdir, dirs, files in os.walk("test_data"):
    for file in files:
        filepath = "test_data" + os.sep + file
        new_file_paths.append(filepath)

#   load numpy arrays and append them in one test_x.npy file
test_x = []
counter = 0
for file in new_file_paths:
    array = np.load(file)
    test_x.append(array)
    print('counter', counter, file)
    counter += 1
test_x = np.array(test_x)
np.save('test_x.npy', test_x)