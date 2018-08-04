#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @FileName: undefined
 @Desc:
 @Author: yuzhongchun
 @Date: 2018-08-04 16:35:19
 @Last Modified by: yuzhongchun
 @Last Modified time: 2018-08-04 16:35:19
"""

import numpy as np
import pandas as pd

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense, Dropout, Flatten

# settings
LEARNING_RATE = 1e-4
# set to 20000 on local environment to get 0.99 accuracy
TRAINING_ITERATIONS = 2500

DROPOUT = 0.5
BATCH_SIZE = 50

# set to 0 to train on all available data
VALIDATION_SIZE = 2000

# image number to output
IMAGE_TO_DISPLAY = 10

# read training data
data_path = "./digit_recognizer/dataset/"
train_url = f'{data_path}train.csv'
test_url = f'{data_path}test.csv'

data = pd.read_csv(train_url)

print('data({0[0]},{0[1]})'.format(data.shape))
print(data.head())

images = data.iloc[:, 1:].values
images = images.astype(np.float)

images = np.multiply(images, 1.0 / 255.0)

print('images({0[0]},{0[1]})'.format(images.shape))

image_size = images.shape[1]
print('image_size => {}'.format(image_size))
image_width = np.ceil(np.sqrt(image_size)).astype(np.uint8)
image_height = image_width
print('image_width => {0} \nimage_height => {1}'.format(
    image_width, image_height))


def display(img):
    one_image = img.reshape(image_width, image_height)

    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    # plt.imshow(one_image, cmap=cm.gray)
    plt.show()


display(images[IMAGE_TO_DISPLAY])

labels_flat = data.iloc[:, 0].values.ravel()

print('labels_flat({0})'.format(len(labels_flat)))
print('labels_flat[{0}] => {1}'.format(IMAGE_TO_DISPLAY,
                                       labels_flat[IMAGE_TO_DISPLAY]))

labels_count = np.unique(labels_flat).shape[0]
print('labels_count => {0}'.format(labels_count))


# convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

print('labels({0[0]},{0[1]})'.format(labels.shape))
print('labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY, labels[IMAGE_TO_DISPLAY]))

# Split data into training & validation
validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = images[VALIDATION_SIZE:]

print('train_images({0[0]},{0[1]})'.format(train_images.shape))
print('validation_images({0[0]},{0[1]})'.format(validation_images.shape))

# Model building and Training
# Reshape the images in 2D as Keras is taking them as inputs
train_img = train_images.reshape(
    [train_images.shape[0], image_width, image_height, 1])
val_img = validation_images.reshape(
    [validation_images.shape[0], image_width, image_height, 1])

model = Sequential()

# First layer, 5x5 conv and 2x2 maxpooling
model.add(
    Conv2D(
        32,
        input_shape=(28, 28, 1),
        kernel_size=(5, 5),
        kernel_initializer="truncated_normal",
        activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

# Second layer
model.add(
    Conv2D(
        64,
        kernel_size=(5, 5),
        kernel_initializer="truncated_normal",
        activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

# Dense layer
model.add(
    Dense(1024, kernel_initializer="truncated_normal", activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())

# Softmax layer
model.add(
    Dense(10, kernel_initializer="truncated_normal", activation='softmax'))

# Compile
model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fit the model with the validation score as well (5 epochs should get you to 0.99 on the val set)
hist = model.fit(
    train_img,
    train_labels,
    batch_size=50,
    epochs=5,
    validation_data=(val_img, validation_labels))
