#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @FileName: undefined
 @Desc:
 @Author: yuzhongchun
 @Date: 2018-07-29 16:23:24
 @Last Modified by: yuzhongchun
 @Last Modified time: 2018-07-29 16:23:24
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

train_data_path = "./departure_predictor/dataset/train.csv"
sample_data = pd.read_csv(train_data_path)

# 1. Data Preprocess
# numeric attribute
num_cols = [
    'Age', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

# categorical attribute, i.e. nominal attribute
# cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole',
# 'MaritalStatus', 'Over18', 'OverTime']
cat_cols = ['Gender', 'MaritalStatus', 'OverTime']

# ordinal attribute
ord_cols = [
    'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance'
]

# target column
target_col = ['Attrition']

total_cols = num_cols + cat_cols + ord_cols
used_data = sample_data[total_cols + target_col]

print('Number of features: {}'.format(len(total_cols)))
print("=" * 180)
print(used_data.head())

# Split the train dataset and test dataset
pos_data = used_data[used_data['Attrition'] == 1].reindex()
train_pos_data = pos_data.iloc[:int(len(pos_data) * 0.8)].copy()
test_pos_data = pos_data.iloc[int(len(pos_data) * 0.8):].copy()

neg_data = used_data[used_data['Attrition'] == 0].reindex()
train_neg_data = neg_data.iloc[:int(len(neg_data) * 0.8)].copy()
test_neg_data = neg_data.iloc[int(len(neg_data) * 0.8):].copy()

train_data = pd.concat([train_pos_data, train_neg_data])
test_data = pd.concat([test_pos_data, test_neg_data])

print('=' * 180)
print('Number of train dataset: ', len(train_data))
print('Propotion of positive and negative train sample: ',
      len(train_pos_data) / len(train_neg_data))
print('=' * 180)
print(train_data.head())

print('=' * 180)
print('Number of test dataset: ', len(test_data))
print('Propotion of positive and negative test sample: ',
      len(test_pos_data) / len(test_neg_data))
print('=' * 180)
print(test_data.head())

# 2. Feature extraction
# One-Hot encoding on categorical data
# Label encoding
# Gender
gender_label_enc = preprocessing.LabelEncoder()
train_data['Gender_Label'] = gender_label_enc.fit_transform(
    train_data['Gender'])
# MaritalStatus
marital_label_enc = preprocessing.LabelEncoder()
train_data['Marital_Label'] = marital_label_enc.fit_transform(
    train_data['MaritalStatus'])
# OverTime
ot_label_enc = preprocessing.LabelEncoder()
train_data['OT_Label'] = ot_label_enc.fit_transform(train_data['OverTime'])

print('=' * 40)
print('Gender:')
print(train_data.groupby('Gender_Label').size())

print('=' * 40)
print('MaritalStatus:')
print(train_data.groupby('Marital_Label').size())

print('=' * 40)
print('OverTime:')
print(train_data.groupby('OT_Label').size())

# One-Hot Encoding
one_hot_enc = preprocessing.OneHotEncoder()
train_cat_feats = one_hot_enc.fit_transform(
    train_data[['Gender_Label', 'Marital_Label', 'OT_Label']]).toarray()
print('=' * 40)
print(train_cat_feats[:5, :])

# same operation on test dataset
# Gender
test_data['Gender_Label'] = gender_label_enc.transform(test_data['Gender'])
# MaritalStatus
test_data['Marital_Label'] = marital_label_enc.transform(
    test_data['MaritalStatus'])
# OverTime
test_data['OT_Label'] = ot_label_enc.transform(test_data['OverTime'])
# One-Hot Encoding
test_cat_feats = one_hot_enc.transform(
    test_data[['Gender_Label', 'Marital_Label', 'OT_Label']]).toarray()
print('=' * 40)
print(test_cat_feats[:5, :])

# Assemble all the features of train dataset
train_num_feats = train_data[num_cols].values
train_ord_feats = train_data[ord_cols].values
train_feats = np.hstack((train_num_feats, train_ord_feats, train_cat_feats))
train_targets = train_data[target_col].values

# Assemble all the features of test dataset
test_num_feats = test_data[num_cols].values
test_ord_feats = test_data[ord_cols].values
test_feats = np.hstack((test_num_feats, test_ord_feats, test_cat_feats))
test_targets = test_data[target_col].values

print('=' * 40)
print('train data: ', train_feats.shape)
print('test data: ', test_feats.shape)
print('=' * 40)

# Process unbalanced data
print('Before resample: ')
print('Number of postive sample: ', len(train_targets[train_targets == 1]))
print('Number of negative sample: ', len(train_targets[train_targets == 0]))

sm = SMOTE(random_state=0)
train_resampled_feats, train_resampled_targets = sm.fit_sample(
    train_feats, train_targets)
print('=' * 40)
print('After resample: ')
print('Number of postive sample: ',
      len(train_resampled_targets[train_resampled_targets == 1]))
print('Number of negative sample: ',
      len(train_resampled_targets[train_resampled_targets == 0]))

# 3. Train a model
# Logistic Regression
lr_clf = LogisticRegression()
lr_clf.fit(train_resampled_feats, train_resampled_targets)

# Random Forest
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(train_resampled_feats, train_resampled_targets)

# 4. Evaluation
print('=' * 40)
print('Number of positive sample: ', len(test_pos_data))
print('Number of negative sample: ', len(test_neg_data))

# RF
print('=' * 40)
print('Random Forest: ')
test_pred = rf_clf.predict(test_feats)
print(metrics.confusion_matrix(test_targets, test_pred, labels=[1, 0]))
print('Accuracy score: ', metrics.accuracy_score(test_targets, test_pred))

# LR
print('=' * 40)
print('Logistic Regression: ')
test_pred = lr_clf.predict(test_feats)
print(metrics.confusion_matrix(test_targets, test_pred, labels=[1, 0]))
print('Accuracy score: ', metrics.accuracy_score(test_targets, test_pred))
