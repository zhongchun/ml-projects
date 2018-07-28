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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import confusion_matrix
# import itertools

# setting seaborn default for plots
sns.set()

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

# Logistic Regression
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round(clf.score(X_train, y_train) * 100, 2)
print(str(acc_log_reg) + ' percent')

# Support Vector Machine
clf = SVC()
clf.fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)
acc_svc = round(clf.score(X_train, y_train) * 100, 2)
print(acc_svc)

# Linear SVM: is a SVM model with linear kernel
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred_linear_svc = clf.predict(X_test)
acc_linear_svc = round(clf.score(X_train, y_train) * 100, 2)
print(acc_linear_svc)

# k-Nearest Neighbors
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
y_pred_knn = clf.predict(X_test)
acc_knn = round(clf.score(X_train, y_train) * 100, 2)
print(acc_knn)

# Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree = clf.predict(X_test)
acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
print(acc_decision_tree)

# Random Forest
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest = clf.predict(X_test)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print(acc_random_forest)

# Gaussian Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred_gnb = clf.predict(X_test)
acc_gnb = round(clf.score(X_train, y_train) * 100, 2)
print(acc_gnb)

# Perceptron
clf = Perceptron(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_perceptron = clf.predict(X_test)
acc_perceptron = round(clf.score(X_train, y_train) * 100, 2)
print(acc_perceptron)

# Stochastic Gradient Descent
clf = SGDClassifier(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_sgd = clf.predict(X_test)
acc_sgd = round(clf.score(X_train, y_train) * 100, 2)
print(acc_sgd)

# 6. Confusion Matrix
"""
                 PredictedPositive       PredictedNegative
ActualPositive          TP                     FN
ActualNegative          FP                     TN

    True Positive: The classifier predicted Survived and the passenger actually Survived.
    True Negative: The classifier predicted Not Survived and the passenger actually Not Survived.
    False Postiive: The classifier predicted Survived but the passenger actually Not Survived.
    False Negative: The classifier predicted Not Survived but the passenger actually Survived.
"""
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest_training_set = clf.predict(X_train)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print("Accuracy: %i %% \n" % acc_random_forest)

class_names = ['Survived', 'Not Survived']

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_train, y_pred_random_forest_training_set)
np.set_printoptions(precision=2)

print('Confusion Matrix in Numbers')
print(cnf_matrix)
print('')

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(
    axis=1)[:, np.newaxis]

print('Confusion Matrix in Percentage')
print(cnf_matrix_percent)
print('')

true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

df_cnf_matrix = pd.DataFrame(
    cnf_matrix, index=true_class_names, columns=predicted_class_names)

df_cnf_matrix_precent = pd.DataFrame(
    cnf_matrix_percent, index=true_class_names, columns=predicted_class_names)

plt.figure(figsize=(15, 5))
plt.subplot(121)
sns.heatmap(df_cnf_matrix, annot=True, fmt='d')

plt.subplot(122)
sns.heatmap(df_cnf_matrix_precent, annot=True)

plt.show()

# 7. Comparing Models
models = pd.DataFrame({
    'Model': [
        'Logistic Regression', 'Support Vector Machines', 'Linear SVC', 'KNN',
        'Decision Tree', 'Random Forest', 'Naive Bayes', 'Perceptron',
        'Stochastic Gradient Decent'
    ],
    'Score': [
        acc_log_reg, acc_svc, acc_linear_svc, acc_knn, acc_decision_tree,
        acc_random_forest, acc_gnb, acc_perceptron, acc_sgd
    ]
})

print(models.sort_values(by='Score', ascending=False))

# 8. Create Submission File to Kaggle
print(test_df.head())
submission = pd.DataFrame({
    "PassengerId": test_df['PassengerId'],
    "Survived": y_pred_random_forest
})

print(submission)
submission.to_csv(
    './titanic_survivor_predictor/beginner_guide/submission.csv', index=False)
