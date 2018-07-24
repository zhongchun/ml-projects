#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @FileName: undefined
 @Desc:
 @Author: yuzhongchun
 @Date: 2018-07-22 20:55:00
 @Last Modified by: yuzhongchun
 @Last Modified time: 2018-07-22 20:55:00
"""

import pandas as pd
import utils
from sklearn import linear_model
from sklearn import preprocessing

train_url = "./titanic_survivor_predictor/datasets/train.csv"
test_url = "./titanic_survivor_predictor/datasets/test.csv"

train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

utils.clean_data(train_df)

target = train_df["Survived"].values
features = train_df[["Pclass", "Age", "Sex", "SibSp", "Parch"]].values
classifier = linear_model.LogisticRegression()
classifier_ = classifier.fit(features, target)
print(classifier_.score(features, target))

poly = preprocessing.PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(features)
classifier_ = classifier.fit(poly_features, target)
print(classifier_.score(poly_features, target))
