import numpy as np

arr = np.array([[0, 1], [1, 0]])
arr2 = np.array([[1, 1], [1, 1]])
# Dump data to file
# hkl.dump( arr, 'new_data_file1.hkl' )
# hkl.dump( arr2, 'new_data_file1.hkl' )
#
# # Load data from file
# data2 = hkl.load( 'new_data_file1.hkl' )
#
# print( arr == data2 )

from tempfile import TemporaryFile
# outfile = TemporaryFile()
# x = np.arange(10)
# np.save(outfile, arr)
# np.save(outfile, arr2)
# outfile.seek(0) # Only needed here to simulate closing & reopening file
# b = np.load(outfile)
# print(b)

# outfile = TemporaryFile()
# np.savez('test', arr2, arr2)
# # outfile.seek(0) # Only needed here to simulate closing & reopening file
# npzfile = np.load('test.npz')
# print(npzfile.files)
# np.savez('test.npz', arr2, arr2)
# npzfile = np.load('test.npz')
# print(npzfile.files)
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#
# np.savetxt('test.txt', arr, delimiter=',')
# np.savetxt('test.txt', arr2, delimiter=',')
# with open('data.txt', 'a') as f:
#     np.savetxt(f, arr)
# np.loadtxt('test.txt')
# with open('train_data.txt', 'a') as input:
#     input.write('[')
#     for a in arr:
#         input.write(a)
#         input.write(',')
#     input.write(']')

# with open('test.txt', 'r') as output:
#     for line in output:
#         print(line)

import pandas as pd
from openpyxl import load_workbook

import os

directory = os.fsencode("excel_files")

data = pd.DataFrame(arr)
# data2 = pd.DataFrame(arr2)
# data.append(data2, ignore_index=True)
# works with 2d
# data = pd.concat([data, pd.DataFrame(arr2)], ignore_index=True)
# data.to_excel('excel_files\\train_data.xlsx', sheet_name='data')
# print(data)

np.save('3dsave.npy', [[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]])
b = np.load('3dsave.npy')
print(';asl;ad')