#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @FileName: undefined
 @Desc:
 @Author: yuzhongchun
 @Date: 2018-07-24 23:04:30
 @Last Modified by: yuzhongchun
 @Last Modified time: 2018-07-24 23:04:30
"""

import pandas as pd

train_url = "./house_price_predictor/datasets/train.csv"
train_df = pd.read_csv(train_url)

print(train_df.head(10))
print(train_df.shape)
print(train_df.describe())
# show the descriptive statistics of object data types
print(train_df.describe(include=['O']))
train_df.info()
print(train_df.isnull().sum())
