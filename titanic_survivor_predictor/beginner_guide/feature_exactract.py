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
import numpy as np

# 2. Load Datasets
train_url = "./titanic_survivor_predictor/datasets/train.csv"
test_url = "./titanic_survivor_predictor/datasets/test.csv"
train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

# 3. Feature extraction
# 3.1 Name feature
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

# 3.2 Sex feature
# convert the categorical value of Sex into numeric
sex_mapping = {"female": 0, "male": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping).astype(int)
    # dataset['Sex'] = dataset['Sex'].fillna(0)
print(train_df.head())

# 3.3 Embarked feature
print(train_df.Embarked.unique())
print(train_df.Embarked.value_counts())
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
print(train_df.head())
print(train_df.Embarked.unique())
print(train_df.Embarked.value_counts())
# convert the categorical value of Embarked into numeric
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping).astype(int)
print(train_df.head())

# 3.3 Age feature
# fill the null value of Age with a random number between (mean_age - std_age) and (mean_age + std_age)
# and then categorizes age into 5 different age range
# train_df.info()
for dataset in train_test_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(
        age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'],
                                                as_index=False).mean())
print(train_df.head())
for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

print(train_df.head())

# 3.4 Fare feature
# replace missing Fare values with median of Fare
for dataset in train_test_data:
    dataset['Fare'] = dataset['Fare'].fillna(train_df['Fare'].median())
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
print(train_df[['FareBand', 'Survived']].groupby(['FareBand'],
                                                 as_index=False).mean())
print(train_df.head())
for dataset in train_test_data:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454),
                'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31),
                'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
print(train_df.head())

# 3.5 SibSp & Parch feature
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print(train_df[['FamilySize', 'Survived']].groupby(
    ['FamilySize'], as_index=False).mean())

for dataset in train_test_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'],
                                                as_index=False).mean())
print(train_df.head())
print(test_df.head())

# 4. Feature selection / Engineering
features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
train_df = train_df.drop(features_drop, axis=1)
test_df = test_df.drop(features_drop, axis=1)
train_df = train_df.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)
print(train_df.head())
print(test_df.head())

# 5. Classification & Accuracy
X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
X_test = test_df.drop("PassengerId", axis=1).copy()
print(X_train.shape, y_train.shape, X_test.shape)

"""
Classification Algorithms
    - Logistic Regression
    - Support Vector Machines (SVC)
    - Linear SVC
    - k-Nearest Neighbor (kNN)
    - Decision Tree
    - Random Forest
    - Naive Bayes (GaussianNB)
    - Stochastic Gradient Descent (SGD)
Training and testing procedure
    1. train some classifiers with the training data
    2. use the trained classifier to predict the Survival outcome of test data
    3. calculate the accuracy score (in percentange) of the trained classifier
"""
