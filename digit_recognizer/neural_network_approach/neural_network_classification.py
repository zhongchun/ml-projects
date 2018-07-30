#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @FileName: undefined
 @Desc:
 @Author: yuzhongchun
 @Date: 2018-07-29 22:46:37
 @Last Modified by: yuzhongchun
 @Last Modified time: 2018-07-29 22:46:37
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn import decomposition
# from sklearn import datasets

# from mpl_toolkits.mplot3d import Axes3D

from sklearn.neural_network import MLPClassifier
from sklearn import metrics

np.random.seed(1)

data_path = "./digit_recognizer/dataset/"
train_url = f'{data_path}train.csv'
test_url = f'{data_path}test.csv'

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)

# Exploratory analysis
print(train.shape)
print(test.shape)
print(train.head())
print('=' * 180)

plt.hist(train['label'])
plt.title("Frequency Histogram of Numbers in Training Data")
plt.xlabel("Number Value")
plt.ylabel("Frequency")
plt.show()

# plot the first 25 digits in the training set
f, ax = plt.subplots(5, 5)
for i in range(1, 26):
    data = train.iloc[i, 1:785].values
    # label = train.iloc[i, :1].values
    nrows, ncols = 28, 28
    grid = data.reshape((nrows, ncols))
    n = math.ceil(i / 5) - 1
    m = [0, 1, 2, 3, 4] * 5
    ax[m[i - 1], n].imshow(grid, cmap='gray')
plt.show()

# PCA
# normalize data
label_train = train['label']
train = train.drop('label', axis=1)

train = train / 255
test = test / 255

train['label'] = label_train

# PCA decomposition
# pca = decomposition.PCA(n_components=200)
# pca.fit(train.drop('label', axis=1))
# plt.plot(pca.explained_variance_ratio_)
# plt.ylabel('% of variance explained')
# plt.show()
# plot reaches asymptote at around 50, which is optimal number of PCs to use.

# PCA decomposition with optimal number of PCs
# decompose train data
pca = decomposition.PCA(n_components=50)
pca.fit(train.drop('label', axis=1))
PCtrain = pd.DataFrame(pca.transform(train.drop('label', axis=1)))
PCtrain['label'] = train['label']

# decompose test data
# pca.fit(test)
PCtest = pd.DataFrame(pca.transform(test))

# 3d scatter figure
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x = PCtrain[0]
# y = PCtrain[1]
# z = PCtrain[2]

# colors = [int(i % 9) for i in PCtrain['label']]
# ax.scatter(x, y, z, c=colors, marker='o', label=colors)

# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')

# plt.show()

# Neural Network
X = PCtrain.drop('label', axis=1)[0:20000]
y = PCtrain['label'][0:20000]
clf = MLPClassifier(
    solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3500, ), random_state=1)
model = clf.fit(X, y)
print('=' * 120)
print(model)

# accuracy and confusion matrix
predicted = model.predict(PCtrain.drop('label', axis=1)[20001:42000])
expected = PCtrain['label'][20001:42000]
print("Classification report for classifier %s:\n%s\n" %
      (model, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

# Output results to file
output = pd.DataFrame(clf.predict(PCtest), columns=['Label'])
output.reset_index(inplace=True)
output.rename(columns={'index': 'ImageId'}, inplace=True)
output['ImageId'] = output['ImageId'] + 1
output.to_csv(f'{data_path}results_nn.csv', index=False)
print("Done")
