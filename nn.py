#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import theano
theano.config.openmp = True
import sys
import  os
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.regularizers import l2
from keras.utils import np_utils
#from keras.utils.visualize_util import plot
#import pdb

'''
    Train a simple deep NN on the MNIST dataset.

    Get to 98.30% test accuracy after 20 epochs (there is *a lot* of margin for parameter tuning).
    2 seconds per epoch on a GRID K520 GPU.
'''

data_path = "dataset/data.npz"

np.random.seed(1337)  # for reproducibility
nb_classes = 2
nb_epoch = 100
batch_size = 128

data = np.load(data_path)
train_x = data["train_x"]
train_y = data["train_y"]
test_x = data["test_x"]
test_y = data["test_y"]

train_y = (train_y + 1) / 2
test_y = (test_y + 1) / 2

train_y = np_utils.to_categorical(train_y, nb_classes)
true_labels = test_y
test_y = np_utils.to_categorical(test_y, nb_classes)
print("train_x.shape: " + str(train_x.shape))
print("test_x.shape: " + str(test_x.shape))
print("train_y.shape: " + str(train_y.shape))
print("test_y.shape: " + str(test_y.shape))

print(train_x.shape[0], ' train samples')
print(test_x.shape[0], ' test samples')

model = Sequential()
model.add(Dense(128, activation = 'relu', input_shape=(train_x.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(128,  activation = 'relu'))
model.add(Dropout(0.5))
#model.add(Dense(32, W_regularizer = l2(0.0001), b_regularizer = l2(0.0001)))
#model.add(Activation('relu'))
#model.add(Dense(64, W_regularizer = l2(0.001), b_regularizer = l2(0.001)))
#model.add(Activation('relu'))
#model.add(Dense(128, W_regularizer = l2(0.0001), b_regularizer = l2(0.0001)))
#model.add(Activation('relu'))
#model.add(Dense(256, W_regularizer = l2(0.0001), b_regularizer = l2(0.0001)))
#model.add(Activation('relu'))
#model.add(Dense(512, W_regularizer = l2(0.001), b_regularizer = l2(0.001)))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(train_y.shape[1]))
model.add(Activation('softmax'))
#model.add(Dropout(0.5))

rms = RMSprop()
adagrad = Adagrad()
adadelta = Adadelta()
adam = Adam()
sgd = SGD(lr = 10e-3, momentum = 0.0, decay = 10e-4)

model.compile(loss='binary_crossentropy', optimizer = adadelta)
#plot(model, to_file='model.png')
class_weight = {0 : 1, 1 : 16}
model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, class_weight = class_weight, verbose=1, shuffle = True, validation_data = (test_x, test_y))
[score, acc] = model.evaluate(test_x, test_y, show_accuracy=True, verbose=0)
print("score: " + str(score) + ", acc: " + str(acc))
label = model.predict_classes(test_x, batch_size = 128, verbose=1)
prob = model.predict(test_x, batch_size = 128, verbose = 1)
print("prob: " + str(prob))
#label = (label == 1)
print("label.shape = " + str(label.shape))
label = label.reshape(label.shape[0], 1)
label = label > 0
num_true = label.sum(0)
right = (label == true_labels)
right = right.sum(0)
acc = np.double(right) / np.double(test_y.shape[0])
pos_samples = true_labels.sum(0)
neg_samples = test_y.shape[0] - pos_samples
print("label.shape: " + str(label.shape))
print(str(label))
print("true_labels.shape: " + str(true_labels.shape))
print(str(true_labels))

pos_right = np.logical_and(label, true_labels)
pos_right = pos_right.sum(0)
print("pos_right:" + str(pos_right) + ", pos_samples: " + str(pos_samples))
neg_right = right - pos_right;
print("neg_right:" + str(neg_right) + ", neg_samples: " + str(neg_samples))

print("right = " + str(right) + ", acc = " + str(acc))
if np.any(label):
    print("any True")
if np.all(label):
    print("all True")
print("num_true: " + str(num_true))
print(str(label))
#print('Test score:', score[0])
#print('Test accuracy:', score[1])
