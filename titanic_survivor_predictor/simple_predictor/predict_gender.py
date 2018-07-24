#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @FileName: undefined
 @Desc:
 @Author: yuzhongchun
 @Date: 2018-07-22 20:35:17
 @Last Modified by: yuzhongchun
 @Last Modified time: 2018-07-22 20:35:17
"""

import pandas as pd
# from matplotlib import pyplot as plt

train_url = "./titanic_survivor_predictor/datasets/train.csv"
test_url = "./titanic_survivor_predictor/datasets/test.csv"

train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

# print(train_df.describe())

train_df["Hyp"] = 0
train_df.loc[train_df.Sex == "female", "Hyp"] = 1

train_df["Result"] = 0
train_df.loc[train_df.Survived == train_df["Hyp"], "Result"] = 1

print(train_df["Sex"].value_counts())
print(train_df["Hyp"].value_counts())
print(train_df["Result"].value_counts())

print(train_df["Sex"].value_counts(normalize=True))
print(train_df["Hyp"].value_counts(normalize=True))
print(train_df["Result"].value_counts(normalize=True))
