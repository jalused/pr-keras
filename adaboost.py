#!/usr/bin/env python
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
data_path = "dataset/data_adaboost.npz"

data = np.load(data_path)
train_x = data["train_x"]
train_y = data["train_y"]
test_x = data["test_x"]
test_y = data["test_y"]

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1), algorithm = "SAMME", n_estimators = 200)
clf.fit(train_x, train_y)
score = clf.score(test_x, test_y)
print("score: " + str(score))
