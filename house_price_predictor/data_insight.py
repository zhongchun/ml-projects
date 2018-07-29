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

path = "./house_price_predictor/datasets/"
train_url = f'{path}train.csv'
test_url = f'{path}test.csv'

# train_df = pd.read_csv(train_url)
train_df = pd.read_csv(train_url, index_col='Id')
test_df = pd.read_csv(test_url, index_col='Id')

# 1. Preview the data and take a peek
print("=" * 120)
print(train_df.head(10))
print("=" * 120)
print(train_df.tail(10))
print("=" * 120)
print(train_df.columns.values)
print("=" * 120)
print(train_df.shape)
print("=" * 120)
train_df.info()
print("=" * 120)
print(train_df.describe())
print("=" * 120)
# show the descriptive statistics of object data types
print(train_df.describe(include=['O']))
print("=" * 120)
print(train_df.isnull().sum())
print("=" * 120)

# proportion of null values, i.e. missing values
total = train_df.isnull().sum().sort_values(ascending=False)
percent_1 = train_df.isnull().sum() / train_df.isnull().count() * 100
percent_2 = (round(percent_1, 2)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
print(total)
print("=" * 40)
print(missing_data.head(20))
print("=" * 40)
