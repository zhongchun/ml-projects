#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @FileName: undefined
 @Desc:
 @Author: yuzhongchun
 @Date: 2018-07-23 22:29:48
 @Last Modified by: yuzhongchun
 @Last Modified time: 2018-07-23 22:29:48
"""

import pandas as pd
import utils
from sklearn import tree
from sklearn import model_selection

train_url = "./titanic_survivor_predictor/datasets/train.csv"
test_url = "./titanic_survivor_predictor/datasets/test.csv"

train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

# print(train_df.head(8))
utils.clean_data(train_df)
# print(train_df.head(8))

target = train_df["Survived"].values
feature_names = ["Pclass", "Age", "Fare", "Embarked", "Sex", "SibSp", "Parch"]
features = train_df[feature_names].values

decision_tree = tree.DecisionTreeClassifier(random_state=1)
decision_tree_ = decision_tree.fit(features, target)

print(decision_tree_.score(features, target))

scores = model_selection.cross_val_score(
    decision_tree, features, target, scoring='accuracy', cv=50)
print(scores)
print(scores.mean())

generalized_tree = tree.DecisionTreeClassifier(
    random_state=1, max_depth=7, min_samples_split=2)
generalized_tree_ = generalized_tree.fit(features, target)

print(generalized_tree_.score(features, target))

scores = model_selection.cross_val_score(
    generalized_tree, features, target, scoring='accuracy', cv=50)
print(scores)
print(scores.mean())

tree.export_graphviz(generalized_tree_, feature_names=feature_names, out_file="tree.dot")
