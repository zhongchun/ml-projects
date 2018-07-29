#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @FileName: undefined
 @Desc:
 @Author: yuzhongchun
 @Date: 2018-07-29 15:50:19
 @Last Modified by: yuzhongchun
 @Last Modified time: 2018-07-29 15:50:19
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_data_path = "./departure_predictor/dataset/train.csv"
sample_data = pd.read_csv(train_data_path)

# 1. take a peek into the data
print(sample_data.head())
print("=" * 80)
sample_data.info()
print("=" * 170)
print(sample_data.describe())

# positive and negative sample statistics
pos_data = sample_data[sample_data['Attrition'] == 1]
neg_data = sample_data[sample_data['Attrition'] == 0]

print('Positive sample count: {}, propotion: {}'.format(
    len(pos_data),
    len(pos_data) / len(sample_data)))
print('Negative sample count: {}, propotion: {}'.format(
    len(neg_data),
    len(neg_data) / len(sample_data)))

plt.figure()
sns.countplot(x='Attrition', data=sample_data, hue='Education')
plt.show()

# 2. univariate relationship with Attrition
plt.figure()
# Age vs. Attrition
plt.subplot(2, 2, 1)
sns.boxplot(x='Attrition', y='Age', data=sample_data)

# DistanceFromHome vs. Attrition
plt.subplot(2, 2, 2)
sns.boxplot(x='Attrition', y='DistanceFromHome', data=sample_data)

# MonthlyIncome vs. Attrition
plt.subplot(2, 2, 3)
sns.boxplot(x='Attrition', y='MonthlyIncome', data=sample_data)

# NumCompaniesWorked vs. Attrition
plt.subplot(2, 2, 4)
sns.boxplot(x='Attrition', y='NumCompaniesWorked', data=sample_data)

plt.show()

plt.figure()
# MaritalStatus vs. Attrition
plt.subplot(2, 1, 1)
sns.countplot(x='Attrition', hue='MaritalStatus', data=sample_data)

# Gender vs. Attrition
plt.subplot(2, 1, 2)
sns.countplot(x='Attrition', hue='Gender', data=sample_data)
plt.show()

# 3. multivariable relationship
# plt.figure()
sns.pairplot(
    sample_data,
    hue='Attrition',
    vars=[
        'Age', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
        'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
    ])
plt.show()
