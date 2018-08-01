#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @FileName: undefined
 @Desc:
 @Author: yuzhongchun
 @Date: 2018-08-01 22:52:36
 @Last Modified by: yuzhongchun
 @Last Modified time: 2018-08-01 22:52:36
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf

# setting
LEARNING_RATE = 1e-4
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
print('image_width => {0} \n image_height => {1}'.format(
    image_width, image_height))

def display(img):
    one_image = img.reshape(image_width, image_height)

    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    plt.show()

display(images[IMAGE_TO_DISPLAY])
