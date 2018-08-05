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

# 1. Libraries and settings
import numpy as np
import pandas as pd
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
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Display parent directory and working directory
print('=' * 120)
print(
    os.path.dirname(os.getcwd()) + ':', os.listdir(
        os.path.dirname(os.getcwd())))
print(os.getcwd() + ':', os.listdir(os.getcwd()))

# 2. Analyze data
# 2.1 Load and check data
data_path = "./digit_recognizer/dataset/"
train_url = f'{data_path}train.csv'
test_url = f'{data_path}test.csv'

data_df = pd.read_csv(train_url)

# basic info about data
print('=' * 120)
print(data_df.info())
print('=' * 120)
print('train.csv loaded: data_df({0[0]},{0[1]})'.format(data_df.shape))

# no missing values
print('=' * 120)
print(data_df.isnull().any().describe())

# 10 different labels ranging from 0 to 9
print('=' * 120)
print('distinct labels: ', data_df['label'].unique())

# data are approximately balanced
print('=' * 120)
print(data_df['label'].value_counts())

# 2.2 Normalize data and split into training and validation sets


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
def dense_to_one_hot(labels_dense, num_classes):
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

# labels in one hot representation
y_train_valid = dense_to_one_hot(y_train_valid_labels,
                                 labels_count).astype(np.uint8)

# dictionaries for saving results
y_valid_pred = {}
y_train_pred = {}
y_test_pred = {}
train_loss = {}
valid_loss = {}
train_acc = {}
valid_acc = {}

print('=' * 120)
print('x_train_valid.shape = ', x_train_valid.shape)
print('y_train_valid_lables.shape = ', y_train_valid_labels.shape)
print('image size = ', image_size)
print('image_width = ', image_width)
print('image_height = ', image_height)
print('labels_count = ', labels_count)
print('=' * 120)

# print(x_train_valid[1])


# 3. Manipulate data
# generate new images via rotations, translations and zooming using keras
def generate_images(imgs):
    # rotations, translations, zoom
    image_generator = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1)
    # get transformed images
    imgs = image_generator.flow(
        imgs.copy(), np.zeros(len(imgs)), batch_size=len(imgs),
        shuffle=False).next()
    return imgs[0]


# check image generation
fig, axs = plt.subplots(5, 10, figsize=(15, 9))
for i in range(5):
    n = np.random.randint(0, x_train_valid.shape[0] - 2)
    axs[i, 0].imshow(x_train_valid[n:n + 1].reshape(28, 28), cmap=cm.binary)
    axs[i, 1].imshow(
        generate_images(x_train_valid[n:n + 1]).reshape(28, 28),
        cmap=cm.binary)
    axs[i, 2].imshow(
        generate_images(x_train_valid[n:n + 1]).reshape(28, 28),
        cmap=cm.binary)
    axs[i, 3].imshow(
        generate_images(x_train_valid[n:n + 1]).reshape(28, 28),
        cmap=cm.binary)
    axs[i, 4].imshow(
        generate_images(x_train_valid[n:n + 1]).reshape(28, 28),
        cmap=cm.binary)
    axs[i, 5].imshow(
        generate_images(x_train_valid[n:n + 1]).reshape(28, 28),
        cmap=cm.binary)
    axs[i, 6].imshow(
        generate_images(x_train_valid[n:n + 1]).reshape(28, 28),
        cmap=cm.binary)
    axs[i, 7].imshow(
        generate_images(x_train_valid[n:n + 1]).reshape(28, 28),
        cmap=cm.binary)
    axs[i, 8].imshow(
        generate_images(x_train_valid[n:n + 1]).reshape(28, 28),
        cmap=cm.binary)
    axs[i, 9].imshow(
        generate_images(x_train_valid[n:n + 1]).reshape(28, 28),
        cmap=cm.binary)
plt.show()

# 4. Try out some basic models with sklearn
# First try out some basic sklean models
logreg = sklearn.linear_model.LogisticRegression(
    verbose=0, solver='lbfgs', multi_class='multinomial')
decision_tree = sklearn.tree.DecisionTreeClassifier()
extra_trees = sklearn.ensemble.ExtraTreesClassifier(verbose=0)
gradient_boost = sklearn.ensemble.GradientBoostingClassifier(verbose=0)
random_forest = sklearn.ensemble.RandomForestClassifier(verbose=0)
gaussianNB = sklearn.naive_bayes.GaussianNB()

# store models in dictionary
base_models = {
    'logreg': logreg,
    'extra_trees': extra_trees,
    'gradient_boost': gradient_boost,
    'random_forest': random_forest,
    'decision_tree': decision_tree,
    'gaussianNB': gaussianNB
}

# choose models for out-of-folds predictions
take_models = ['logreg', 'random_forest', 'extra_trees']

for mn in take_models:
    train_acc[mn] = []
    valid_acc[mn] = []

# cross validations
cv_num = 10
kfold = sklearn.model_selection.KFold(cv_num, shuffle=True, random_state=123)

for i, (train_index, valid_index) in enumerate(kfold.split(x_train_valid)):
    # start timer
    start = datetime.datetime.now()

    # train and validation data of original images
    x_train = x_train_valid[train_index].reshape(-1, 784)
    y_train = y_train_valid[train_index]
    x_valid = x_train_valid[valid_index].reshape(-1, 784)
    y_valid = y_train_valid[valid_index]

    for mn in take_models:
        # create cloned model from base models
        model = sklearn.base.clone(base_models[mn])
        model.fit(x_train, one_hot_to_dense(y_train))

        # predictions
        y_train_pred[mn] = model.predict_proba(x_train)
        y_valid_pred[mn] = model.predict_proba(x_valid)
        train_acc[mn].append(
            accuracy_from_one_hot_labels(y_train_pred[mn], y_train))
        valid_acc[mn].append(
            accuracy_from_one_hot_labels(y_valid_pred[mn], y_valid))

        print(
            i, ': ' + mn + ' train/valid accuracy = %.3f/%.3f' %
            (train_acc[mn][-1], valid_acc[mn][-1]))

    # only one iteration:
    if False:
        break

print(mn + ': averaged train/valid accuracy = %.3f/%.3f' %
      (np.mean(train_acc[mn]), np.mean(valid_acc[mn])))

# Compare accuracies of base models
# boxplot algorithm comparison
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(1, 2, 1)
plt.title('Train accuracy')
plt.boxplot([train_acc[mn] for mn in train_acc.keys()])
ax.set_xticklabels([mn for mn in train_acc.keys()])
ax.set_ylabel('Accuracy')
ax.set_ylim([0.90, 1.0])

ax = fig.add_subplot(1, 2, 2)
plt.title('Valid accuracy')
plt.boxplot([valid_acc[mn] for mn in train_acc.keys()])
ax.set_xticklabels([mn for mn in train_acc.keys()])
ax.set_ylabel('Accuracy')
ax.set_ylim([0.90, 1.0])

plt.show()

for mn in train_acc.keys():
    print(mn + ' averaged train/valid accuracy = %.3f/%.3f' %
          (np.mean(train_acc[mn]), np.mean(valid_acc[mn])))
