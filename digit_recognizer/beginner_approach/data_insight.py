#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @FileName: undefined
 @Desc:
 @Author: yuzhongchun
 @Date: 2018-07-29 21:42:21
 @Last Modified by: yuzhongchun
 @Last Modified time: 2018-07-29 21:42:21
"""

import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

data_path = "./digit_recognizer/dataset/"
train_url = f'{data_path}train.csv'
test_url = f'{data_path}test.csv'

labeled_images = pd.read_csv(train_url)
images = labeled_images.iloc[0:5000, 1:]
labels = labeled_images.iloc[0:5000, :1]

train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, train_size=0.8, test_size=0.2, random_state=0)

# View an image
i = 1
img = train_images.iloc[i].values
img = img.reshape((28, 28))
plt.imshow(img, cmap='gray')
plt.title(train_labels.iloc[i])
plt.show()

hist_values = plt.hist(train_images.iloc[i])
plt.show()
print(hist_values)
print('=' * 120)

# Training a model
# clf = svm.SVC()
# clf.fit(train_images, train_labels.values.ravel())
# print(clf.score(test_images, test_labels))

# Improve
train_images[train_images > 0] = 1
test_images[test_images > 0] = 1

img = train_images.iloc[i].values.reshape((28, 28))
# plt.imshow(img, cmap='gray')
plt.imshow(img, cmap='binary')
plt.title(train_labels.iloc[i])
plt.show()

hist_values = plt.hist(train_images.iloc[i])
plt.show()
print(hist_values)
print('=' * 120)

# Retraining the model
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
print(clf.score(test_images, test_labels))

test_data = pd.read_csv(test_url)
test_data[test_data > 0] = 1
results = clf.predict(test_data)

df = pd.DataFrame(results)
df.index.name = 'ImageId'
df.index += 1
df.columns = ['Label']
df.to_csv(f'{data_path}results.csv', header=True)
print("Done")
