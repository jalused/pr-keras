#!/usr/bin/env python
import os
import subprocess
import numpy as np
from keras.utils import np_utils

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data_path = "dataset/"
train_file = "train.data"
test_file = "test.data"
preprocessed_data_file = "data.npz"
data_url = "http://staff.ustc.edu.cn/~ketang/PPT/dataset.zip"

if ~os.path.isfile(data_path + preprocessed_data_file):
    child = subprocess.Popen("wget " + data_url, shell = True) 
    child.wait()
    child = subprocess.Popen("unzip dataset.zip", shell = True)
    child.wait()
    subprocess.Popen("rm dataset.zip", shell = True)
classes = 2

train_data = np.genfromtxt(data_path + train_file)
test_data = np.genfromtxt(data_path + test_file)

train_x = train_data[:, 0 : train_data.shape[1] - 1]
train_y = train_data[:, train_data.shape[1] - 1 : train_data.shape[1]]
test_x = test_data[:, 0 : test_data.shape[1] - 1]
test_y = test_data[:, test_data.shape[1] - 1 : test_data.shape[1]]
temp = train_x[:, 0]


train_x = train_x.astype("double")
test_x = test_x.astype("double")

train_x = (train_x - np.min(train_x, 0)) / (np.max(train_x, 0) - np.min(train_x, 0))
test_x = (test_x - np.min(test_x, 0)) / (np.max(test_x, 0) - np.min(test_x, 0))

positive_train_x = train_x[0 : 466, :]
positive_train_y = train_y[0 : 466, :]
negative_train_x = train_x[466 : train_x.shape[0], :]
negative_train_y = train_y[466 : train_x.shape[0], :]

#index = np.arange(negative_train_y.shape[0])
#np.random.shuffle(index)
#index = index[0 : 466]
#negative_train_x = negative_train_x[index, :]
#negative_train_y = negative_train_y[index, :]

#positive_train_x = np.tile(positive_train_x, (17, 1))
#positive_train_y = np.tile(positive_train_y, (17, 1))

negative_train_y = negative_train_y.reshape(negative_train_y.shape[0], 1)

train_x = np.row_stack((positive_train_x, negative_train_x))
train_y = np.row_stack((positive_train_y, negative_train_y))

#pca = PCA(n_components = 2)
#train_x = pca.fit_transform(train_x)
#test_x = pca.transform(test_x)

np.savez(data_path + preprocessed_data_file, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
