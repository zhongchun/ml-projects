#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @FileName: undefined
 @Desc: A beginner's guide
        1. Exploratory Data Analysis (EDA) with Visualization
        2. Feature Extraction
        3. Data Modelling
        4. Model Evaluation
 @Author: yuzhongchun
 @Date: 2018-07-25 22:20:03
 @Last Modified by: yuzhongchun
 @Last Modified time: 2018-07-25 22:20:03
"""

# 1. Load Modules
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# setting seaborn default for plots
sns.set()

# 2. Load Datasets
train_url = "./titanic_survivor_predictor/datasets/train.csv"
test_url = "./titanic_survivor_predictor/datasets/test.csv"
train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

# 3.1 Looking into the training dataset
# print first 5 rows of the train dataset
print(train_df.head())
# total rows and columns
print(train_df.shape)
# describing training dataset: show different values of numeric data types
print(train_df.describe())
# show the descriptive statistics of object data types
print(train_df.describe(include=['O']))
print(train_df.info())
print(train_df.isnull().sum())

# 3.2 Looking into the testing dataset
print(test_df.head())
print(test_df.shape)
print(test_df.info())
print(test_df.isnull().sum())

# Relationship between Features and Survival
# survived proportion
survived = train_df[train_df["Survived"] == 1]
not_survived = train_df[train_df["Survived"] == 0]
print("Survived: %i (%.1f%%)" % (len(survived),
                                 float(len(survived)) / len(train_df) * 100.0))
print("Not Survived: %i (%.1f%%)" %
      (len(not_survived), float(len(not_survived)) / len(train_df) * 100.0))
print("Total: %i" % len(train_df))

# univariate with Survival
# Pclass vs. Survival
print(train_df.Pclass.value_counts())
print(train_df.groupby('Pclass').Survived.value_counts())
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'],
                                               as_index=False).mean())
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.show()

# Sex vs. Survival
print(train_df.Sex.value_counts())
print(train_df.groupby('Sex').Survived.value_counts())
print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
sns.barplot(x='Sex', y='Survived', data=train_df)
plt.show()

# Embarked vs. Survived
print(train_df.Embarked.value_counts())
print(train_df.groupby('Embarked').Survived.value_counts())
print(train_df[['Embarked', 'Survived']].groupby(['Embarked'],
                                                 as_index=False).mean())
sns.barplot(x='Embarked', y='Survived', data=train_df)
plt.show()

# Parch vs. Survival
print(train_df.Parch.value_counts())
print(train_df.groupby('Parch').Survived.value_counts())
print(train_df[['Parch', 'Survived']].groupby(['Parch'],
                                              as_index=False).mean())
sns.barplot(x='Parch', y='Survived', data=train_df)
plt.show()

# SibSp vs. Survival
print(train_df.SibSp.value_counts())
print(train_df.groupby('SibSp').Survived.value_counts())
print(train_df[['SibSp', 'Survived']].groupby(['SibSp'],
                                              as_index=False).mean())
sns.barplot(x='SibSp', y='Survived', data=train_df)
plt.show()

# Age vs. Survival
# this way as before is not ok
print(train_df.Age.value_counts())
print(train_df.groupby('Age').Survived.value_counts())
print(train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean())
# sns.barplot(x='Age', y='Survived', data=train_df)
# plt.show()

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
sns.violinplot(
    x='Embarked', y='Age', hue='Survived', data=train_df, split=True, ax=ax1)
sns.violinplot(
    x='Pclass', y='Age', hue='Survived', data=train_df, split=True, ax=ax2)
sns.violinplot(
    x='Sex', y='Age', hue='Survived', data=train_df, split=True, ax=ax3)
plt.show()

total_survived = train_df[train_df['Survived'] == 1]
total_not_survived = train_df[train_df['Survived'] == 0]
male_survived = train_df[(train_df['Survived'] == 1)
                         & (train_df['Sex'] == 'male')]
female_survived = train_df[(train_df['Survived'] == 1)
                           & (train_df['Sex'] == 'female')]
male_not_survived = train_df[(train_df['Survived'] == 0)
                             & (train_df['Sex'] == 'male')]
female_not_survived = train_df[(train_df['Survived'] == 0)
                               & (train_df['Sex'] == 'female')]

plt.figure(figsize=[15, 5])
plt.subplot(111)
sns.distplot(
    total_survived['Age'].dropna().values,
    bins=range(0, 81, 1),
    kde=False,
    color='blue')
sns.distplot(
    total_not_survived['Age'].dropna().values,
    bins=range(0, 81, 1),
    kde=False,
    color='red',
    axlabel='Age')

plt.figure(figsize=[15, 5])
plt.subplot(121)
sns.distplot(
    female_survived['Age'].dropna().values,
    bins=range(0, 81, 1),
    kde=False,
    color='blue')
sns.distplot(
    female_not_survived['Age'].dropna().values,
    bins=range(0, 81, 1),
    kde=False,
    color='red',
    axlabel='Female Age')

plt.subplot(122)
sns.distplot(
    male_survived['Age'].dropna().values,
    bins=range(0, 81, 1),
    kde=False,
    color='blue')
sns.distplot(
    male_not_survived['Age'].dropna().values,
    bins=range(0, 81, 1),
    kde=False,
    color='red',
    axlabel='Male Age')

plt.show()

# multiple variable
# Pclass & Sex vs. Survival
tab = pd.crosstab(train_df['Pclass'], train_df['Sex'])
print(tab)
tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Pclass')
plt.ylabel('Percentage')
plt.show()
sns.factorplot(
    'Sex', 'Survived', hue='Pclass', size=4, aspect=2, data=train_df)
plt.show()

# Pclass, Sex & Embarked vs Survival
sns.factorplot(
    x='Pclass', y='Survived', hue='Sex', col='Embarked', size=5, data=train_df)
plt.show()

# Correlating Features
# Heatmap of correlation between different features
# Positive numbers
# Negative numbers
plt.figure()
sns.heatmap(
    train_df.drop('PassengerId', axis=1).corr(),
    vmax=0.6,
    square=True,
    annot=True)
plt.show()

# Feature extraction
# Name feature
train_test_data = [train_df, test_df]
for dataset in train_test_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')
print(train_df.head())
print(pd.crosstab(train_df['Title'], train_df['Sex']))

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].replace([
        'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir',
        'Jonkheer', 'Dona'
    ], 'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
print(train_df.groupby('Title').Survived.value_counts())
print(train_df[['Title', 'Survived']].groupby(['Title'],
                                              as_index=False).mean())
# convert the categorical Title values into numeric form
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
print(train_df.head())
