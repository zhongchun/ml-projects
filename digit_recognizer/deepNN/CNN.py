#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @FileName: undefined
 @Desc:
 @Author: yuzhongchun
 @Date: 2018-08-04 19:33:57
 @Last Modified by: yuzhongchun
 @Last Modified time: 2018-08-04 19:33:57
"""

# Libraries and settings
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.preprocessing.image
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.tree
import sklearn.ensemble
import os
import datetime
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#display parent directory and working directory
print(
    os.path.dirname(os.getcwd()) + ':', os.listdir(
        os.path.dirname(os.getcwd())))
print(os.getcwd() + ':', os.listdir(os.getcwd()))

# Analyze data
data_path = "./digit_recognizer/dataset/"
train_url = f'{data_path}train.csv'
test_url = f'{data_path}test.csv'

data_df = pd.read_csv(train_url)
print('train.csv loaded: data_df({0[0]},{0[1]})'.format(data_df.shape))

# normalize data and split into training and validation sets


# function to normalize data
def normalize_data(data):
    # scale features using statistics that are robust to outliers
    # rs = sklearn.preprocessing.RobustScaler()
    # rs.fit(data)
    # data = rs.transform(data)
    # data = (data-data.mean())/(data.std()) # standardisation
    data = data / data.max()  # convert from [0:255] to [0.:1.]
    # data = ((data / 255.)-0.5)*2. # convert from [0:255] to [-1.:+1.]
    return data


# convert class labels from scalars to one-hot vectors e.g. 1 => [0 1 0 0 0 0 0 0 0 0]
def dense_to_ont_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# convert one-hot encodings into labels
def one_hot_to_dense(labels_one_hot):
    return np.argmax(labels_one_hot, 1)


# computet the accuracy of label predictions
def accuracy_from_dense_labels(y_target, y_pred):
    y_target = y_target.reshape(-1, )
    y_pred = y_pred.reshape(-1, )
    return np.mean(y_target == y_pred)


# computet the accuracy of one-hot encoded predictions
def accuracy_from_one_hot_labels(y_target, y_pred):
    y_target = one_hot_to_dense(y_target).reshape(-1, )
    y_pred = one_hot_to_dense(y_pred).reshape(-1, )
    return np.mean(y_target == y_pred)


# extract and normalize images
# (42000,28,28,1) array
x_train_valid = data_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
# convert from int64 to float32
x_train_valid = x_train_valid.astype(np.float)
x_train_valid = normalize_data(x_train_valid)
image_width = 28
image_height = 28
image_size = 784

# extract image labels
# (42000,1) array
y_train_valid_labels = data_df.iloc[:, 0].values
labels_count = np.unique(y_train_valid_labels).shape[0]
# number of different labels = 10

# plot some images and labels
plt.figure(figsize=(15, 9))
for i in range(50):
    plt.subplot(5, 10, 1 + i)
    plt.title(y_train_valid_labels[i])
    plt.imshow(x_train_valid[i].reshape(28, 28), cmap=cm.binary)
plt.show()
