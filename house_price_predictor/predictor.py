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
from sklearn.ensemble import RandomForestRegressor

path = "./house_price_predictor/datasets/"
train_url = f'{path}train.csv'
test_url = f'{path}test.csv'

# train_df = pd.read_csv(train_url)
train_df = pd.read_csv(train_url, index_col='Id')
test_df = pd.read_csv(test_url, index_col='Id')

# 2. Preprocess the data
target = train_df['SalePrice']
train_df = train_df.drop('SalePrice', axis=1)
train_df['training_set'] = True
test_df['training_set'] = False

full_df = pd.concat([train_df, test_df])
# print(full_df.shape)

# fill in the missing values
full_df = full_df.interpolate()
# Convert categorical variable into dummy/indicator variables
full_df = pd.get_dummies(full_df)

# train_df = full_df[full_df['training_set'] == True]
train_df = full_df[full_df['training_set'].__eq__(True)]
train_df = train_df.drop('training_set', axis=1)

test_df = full_df[full_df['training_set'].__eq__(False)]
test_df = test_df.drop('training_set', axis=1)

# 3. Train a model
# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(train_df, target)

preds = rf.predict(test_df)
submission = pd.DataFrame({'Id': test_df.index, 'SalePrice': preds})
submission.to_csv(f'{path}submission.csv', index=False)
print("Done")
