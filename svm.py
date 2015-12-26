#!/usr/bin/env python
from sklearn import svm
import numpy as np
data_path = "dataset/data_svm.npz"
data = np.load(data_path)
train_x = data["train_x"]
train_y = data["train_y"]
train_y = train_y.reshape(train_y.shape[0])
test_x = data["test_x"]
test_y = data["test_y"]
test_y = test_y.reshape(test_y.shape[0])

print("train_x.shape: " + str(train_x.shape))
print("test_x.shape: " + str(test_x.shape))
print("train_y.shape: " + str(train_y.shape))
print("test_y.shape: " + str(test_y.shape))

clf = svm.SVC()
print("train_y:")
print(str(train_y))
clf.fit(train_x, train_y)
score = clf.score(test_x, test_y)
label = clf.predict(test_x)
print("test_y:")
print(str(test_y))
print("label:")
print("label.shape: " + str(label.shape))
print("test_y.shape: " + str(test_y.shape))
bad = label == test_y
right = label.sum(0)
print("right: " + str(right))
print("acc: " + str(np.double(right) / np.double(test_x.shape[0])))
