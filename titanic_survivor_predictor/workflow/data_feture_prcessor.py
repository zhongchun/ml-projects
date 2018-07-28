#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @FileName: undefined
 @Desc:
 @Author: yuzhongchun
 @Date: 2018-07-28 19:25:47
 @Last Modified by: yuzhongchun
 @Last Modified time: 2018-07-28 19:25:47
"""

# data analysis and wrangling
import pandas as pd
import numpy as np
# import random as rnd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Acquire data
train_url = "./titanic_survivor_predictor/datasets/train.csv"
test_url = "./titanic_survivor_predictor/datasets/test.csv"
train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)
combine = [train_df, test_df]

# 2. Analyze by describing data
"""
    Which features are available in the dataset?
    Which features are categorical?
        Categorical: Survived, Sex, and Embarked
        Oridinal: Pclass
    Which featureas are numerical?
        Continous: Age, Fare
        Discrete: SibSp, Parch
    Which features are mixed data types?
        Ticket is a mix of numeric and alphanumeric data types. Cabin is alphanumeric.
    Which featurea may contain errors or typos?
        Name feature may contain errors or typos as there are several ways used to describe a name including titles
        round brackets, and quotes used for alternative or short names.
    Which features contain blank, null or empty values?
        Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.
        Cabin > Age are incomplete in case of test dataset.
    What are the data types for various features?
        Seven features are integer or floats. Six in case of test dataset.
        Five features are strings (object).
    What is the distribution of numerical feature values across the samples?
    What is the distribution of categorical features?
    Preview the data
    "
"""

print(train_df.columns.values)
print(train_df.head())
print(train_df.tail())
train_df.info()
print('=' * 40)
test_df.info()
print('=' * 90)
print(train_df.describe())
print('=' * 90)
print(train_df.describe(include=['O']))
"""
Assumtions based on data analysis
    Correlating
    Completing
    Correcting
    Creating
    Classifying
"""

# Analyze by pivoting features
print('=' * 40)
print(train_df[['Pclass', 'Survived']].groupby(
    ['Pclass'], as_index=False).mean().sort_values(
        by='Survived', ascending=False))
print('=' * 40)
print(train_df[["Sex", "Survived"]].groupby(
    ['Sex'], as_index=False).mean().sort_values(
        by='Survived', ascending=False))
print('=' * 40)
print(train_df[["SibSp", "Survived"]].groupby(
    ['SibSp'], as_index=False).mean().sort_values(
        by='Survived', ascending=False))
print('=' * 40)
print(train_df[["Parch", "Survived"]].groupby(
    ['Parch'], as_index=False).mean().sort_values(
        by='Survived', ascending=False))
print('=' * 40)

# Analyze by visualizing data
# Correlating numerical features
# Age vs. Survived
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()

# Correlating numerical and ordinal features
# Pclass, Age vs Survived
# grid = sns.FacetGrid(train_df, col='Survived', hue='Survived')
grid = sns.FacetGrid(
    train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
grid.add_legend()
plt.show()

# Correlating categorical features
# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.show()

# Correlating categorical and numerical features
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(
    train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=0.5, ci=None)
grid.add_legend()
plt.show()

# Wrangle data
# Correcting by dropping features
print("Before", train_df.shape, test_df.shape, combine[0].shape,
      combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine[0].shape,
      combine[1].shape)

# Creating new feature extracting from existing
# retain the new Title feature from Name
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(pd.crosstab(train_df['Title'], train_df['Sex']))

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace([
        'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir',
        'Jonkheer', 'Dona'
    ], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print(train_df[['Title', 'Survived']].groupby(['Title'],
                                              as_index=False).mean())

# convert the categorical titles to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

print("=" * 150)
print(train_df.head())

# drop the Name feature
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
print("=" * 150)
print(train_df.shape, test_df.shape)

# Converting a categorical feature
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

print("=" * 70)
print(train_df.head())

# Completing a numerical continuous feature
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()

guess_ages = np.zeros((2, 3))
print(guess_ages)
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i)
                               & (dataset['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) &
                        (dataset.Pclass == j + 1), 'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

print("=" * 70)
print(train_df.head())

# create Age bands and determine correlations with Survived
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
print("=" * 40)
print(train_df[['AgeBand', 'Survived']].groupby(
    ['AgeBand'], as_index=False).mean().sort_values(
        by='AgeBand', ascending=True))

for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age']
print("=" * 80)
print(train_df.head())

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
print("=" * 80)
print(train_df.head())

# Create new feature combining existing features
# create a new feature for FamilySize which combines Parch and SibSp
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print("=" * 60)
print(train_df[['FamilySize', 'Survived']].groupby(
    ['FamilySize'], as_index=False).mean().sort_values(
        by='Survived', ascending=False))

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print("=" * 60)
print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'],
                                                as_index=False).mean())

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

print("=" * 60)
print(train_df.head())

# create an artificial feature combining Pclass and Age.
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

print("=" * 60)
print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))

# Completing a categorical feature
freq_port = train_df.Embarked.dropna().mode()[0]
print("Frequence port: ", freq_port)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

print(train_df[['Embarked', 'Survived']].groupby(
    ['Embarked'], as_index=False).mean().sort_values(
        by='Survived', ascending=False))

# Converting categorical feature to numeric
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({
        'S': 0,
        'C': 1,
        'Q': 2
    }).astype(int)

print("=" * 80)
print(train_df.head())

# Quick completing and converting a numeric feature

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
print("=" * 80)
print(test_df.head())

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
print("=" * 80)
print(train_df[['FareBand', 'Survived']].groupby(
    ['FareBand'], as_index=False).mean().sort_values(
        by='FareBand', ascending=True))

for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454),
                'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31),
                'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

print("=" * 80)
print(train_df.head(10))
print("=" * 80)
print(test_df.head(10))

train_df.to_csv(
    "./titanic_survivor_predictor/workflow/train_df_fatures.csv", index=False)
test_df.to_csv(
    "./titanic_survivor_predictor/workflow/test_df_fatures.csv", index=False)
print("Done")
