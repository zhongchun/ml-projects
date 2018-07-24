#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @FileName: undefined
 @Desc:
 @Author: yuzhongchun
 @Date: 2018-07-17 22:23:05
 @Last Modified by: yuzhongchun
 @Last Modified time: 2018-07-17 22:23:05
"""

# Import the libraries
# data processing
import pandas as pd

# data visualization
import seaborn as sns
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

# 1. Load data
train_url = "./titanic_survivor_predictor/datasets/train.csv"
test_url = "./titanic_survivor_predictor/datasets/test.csv"
train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

# 2. Data Exploration and Analysis
# 2.1 a simple look into the data
print(train_df.head(8))
print(train_df.shape)
train_df.info()
print(train_df.describe())

total = train_df.isnull().sum().sort_values(ascending=False)
percent_1 = train_df.isnull().sum() / train_df.isnull().count() * 100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
print(total)
print(missing_data.head(5))
print(train_df.columns.values)

# 2.2 Age and Sex
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
women = train_df[train_df['Sex'] == 'female']
men = train_df[train_df['Sex'] == 'male']
ax = sns.distplot(
    women[women['Survived'] == 1].Age.dropna(),
    bins=18,
    label=survived,
    ax=axes[0],
    kde=False)
ax = sns.distplot(
    women[women['Survived'] == 0].Age.dropna(),
    bins=40,
    label=not_survived,
    ax=axes[0],
    kde=False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(
    men[men['Survived'] == 1].Age.dropna(),
    bins=18,
    label=survived,
    ax=axes[1],
    kde=False)
ax = sns.distplot(
    men[men['Survived'] == 0].Age.dropna(),
    bins=40,
    label=not_survived,
    ax=axes[1],
    kde=False)
ax.legend()
_ = ax.set_title('Male')
plt.show()

# 2.3 Embarked, Pclass and Sex
FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=2.4, aspect=2)
FacetGrid.map(
    sns.pointplot,
    'Pclass',
    'Survived',
    'Sex',
    palette=None,
    order=None,
    hue_order=None)
FacetGrid.add_legend()
plt.show()

# Pclass
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.show()

grid = sns.FacetGrid(
    train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
grid.add_legend()
plt.show()

# 2.4 SibSp and Parch
data = [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)

print(train_df['not_alone'].value_counts())

axes = sns.factorplot('relatives', 'Survived', data=train_df, aspect=2.5)
plt.show()
