#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @FileName: undefined
 @Desc:
 @Author: yuzhongchun
 @Date: 2018-07-21 19:05:15
 @Last Modified by: yuzhongchun
 @Last Modified time: 2018-07-21 19:05:15
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Algorithms
# from sklearn import linear_model
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict

# metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

train_process_url = "./titanic_survivor_predictor/datasets/train_process.csv"
test_process_url = "./titanic_survivor_predictor/datasets/test_process.csv"

train_df = pd.read_csv(train_process_url)
test_df = pd.read_csv(test_process_url)

# train_df.info()
# print(train_df.describe())

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()

# 1. Perceptron
perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, Y_train)
Y_pred_perceptron = perceptron.predict(X_test)
score_perceptron = perceptron.score(X_train, Y_train)
acc_perceptron = round(score_perceptron * 100, 2)
# print("Perceptron Score: ")
# print(score_gaussian)
# print(acc_gaussian)

# 2. SGD: Stochastic Gradient Descent
sgd = SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred_sgd = sgd.predict(X_test)
score_sgd = sgd.score(X_train, Y_train)
acc_sgd = round(score_sgd * 100, 2)
# print("SGD Score: ")
# print(score_sgd)
# print(acc_sgd)

# 3. Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_lr = logreg.predict(X_test)
score_lr = logreg.score(X_train, Y_train)
acc_lr = round(score_lr * 100, 2)
# print("Logistic Regression Score: ")
# print(score_lr)
# print(acc_lr)

# 4. K Nearest Neighbor
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
score_knn = knn.score(X_train, Y_train)
acc_knn = round(score_knn * 100, 2)
# print("KNN Score: ")
# print(score_knn)
# print(acc_knn)

# 5. Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred_gaussian = gaussian.predict(X_test)
score_gaussian = gaussian.score(X_train, Y_train)
acc_gaussian = round(score_gaussian * 100, 2)
# print("Gaussian Naive Bayes Score: ")
# print(score_gaussian)
# print(acc_gaussian)

# 6. Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
Y_pred_lda = lda.predict(X_test)
score_lda = lda.score(X_train, Y_train)
acc_lda = round(score_lda * 100, 2)

# 7. Linear Support Vector Machine
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred_linear_svc = linear_svc.predict(X_test)
score_linear_svc = linear_svc.score(X_train, Y_train)
acc_linear_svc = round(score_linear_svc * 100, 2)
# print("Linear Support Vector Machine Score: ")
# print(score_linear_svc)
# print(acc_linear_svc)

# 8. Support Vector Machine
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svc = svc.predict(X_test)
score_svc = svc.score(X_train, Y_train)
acc_svc = round(score_svc * 100, 2)

# 9. Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_dt = decision_tree.predict(X_test)
score_dt = decision_tree.score(X_train, Y_train)
acc_dt = round(score_dt * 100, 2)
# print("Decision Tree Score: ")
# print(score_dt)
# print(acc_dt)

# 10. Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_rf = random_forest.predict(X_test)
score_rf = random_forest.score(X_train, Y_train)
acc_rf = round(score_rf * 100, 2)
# print("Random Forest Score: ")
# print(score_rf)
# print(acc_rf)

results = pd.DataFrame({
    'Model': [
        'Perceptron', 'Stochastic Gradient Decent', 'Logistic Regression',
        'KNN', 'Gausssian Naive Bayes', 'Linear Discriminant Analysis',
        'Linear Support Vector Machine', 'Support Vector Machine',
        'Decision Tree', 'Random Forest'
    ],
    'Score': [
        acc_perceptron, acc_sgd, acc_lr, acc_knn, acc_gaussian, acc_lda,
        acc_linear_svc, acc_svc, acc_dt, acc_rf
    ]
})
result_df = results.sort_values(by='Score', ascending=False)
result_df = results.set_index('Score')
print(result_df.head(10))

# K-Fold Cross Validation
rf = RandomForestClassifier(n_estimators=100)
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
scores = cross_val_score(rf, X_train, Y_train, cv=kfold, scoring="accuracy")
print("Scores: ", scores)
print("Mean: ", scores.mean())
print("Standard Deviation: ", scores.std())

# Feature Importance
importances = pd.DataFrame({
    'feature':
    X_train.columns,
    'importance':
    np.round(random_forest.feature_importances_, 3)
})
importances = importances.sort_values(
    'importance', ascending=False).set_index('feature')

print(importances.head(15))

# importances.plot.bar()
# plt.show()

# drop alone and Parch
train_df = train_df.drop('not_alone', axis=1)
test_df = test_df.drop('not_alone', axis=1)
train_df = train_df.drop("Parch", axis=1)
test_df = test_df.drop("Parch", axis=1)

random_forest = RandomForestClassifier(n_estimators=100, oob_score=True)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
score = random_forest.score(X_train, Y_train)
acc = round(score * 100, 2)
print(round(
    acc,
    2,
), "%")
print("oob score: ", round(random_forest.oob_score_, 4) * 100, "%")

# Hyperparameter Tuning
# param_grid = {
#     "criterion": ["gini", "entropy"],
#     "min_samples_leaf": [1, 5, 10, 25, 50, 70],
#     "min_samples_split": [2, 4, 10, 12, 16, 18, 25, 35],
#     "n_estimators": [100, 400, 700, 1000, 1500]
# }
# rf = RandomForestClassifier(
#     n_estimators=100,
#     max_features='auto',
#     oob_score=True,
#     random_state=1,
#     n_jobs=-1)
# clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=1)
# clf.fit(X_train, Y_train)
# print("Best Params: ")
# print(clf.best_params_)

# Test new parameters
random_forest = RandomForestClassifier(
    criterion='gini',
    min_samples_leaf=1,
    min_samples_split=10,
    n_estimators=100,
    max_features='auto',
    oob_score=True,
    random_state=1,
    n_jobs=-1)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
score = random_forest.score(X_train, Y_train)
acc = round(score * 100, 2)
print(round(
    acc,
    2,
), "%")
print("oob score: ", round(random_forest.oob_score_, 4) * 100, "%")

# Further Evaluation
# Confusion Matrix
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
con_matrix = confusion_matrix(Y_train, predictions)
print(con_matrix)

# Precision and Recall
print("Precision: ", precision_score(Y_train, predictions))
print("Recall: ", recall_score(Y_train, predictions))

# F-Score
print("F-Score: ", f1_score(Y_train, predictions))

# Precision Recall Curve
# getting the probabilities of our predictions
y_scores = random_forest.predict_proba(X_train)
y_scores = y_scores[:, 1]

precision, recall, threshold = precision_recall_curve(Y_train, y_scores)


def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])


plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()


def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g--", linewidth=2.5)
    plt.ylabel("recall", fontsize=19)
    plt.xlabel("precision", fontsize=19)
    plt.axis([0, 1.5, 0, 1.5])


plt.figure(figsize=(14, 7))
plot_precision_vs_recall(precision, recall)
plt.show()

# ROC AUC Curve
# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(
    Y_train, y_scores)

# plotting them against each other


def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)


plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()

# ROC AUC Score
r_a_score = roc_auc_score(Y_train, y_scores)
print("ROC-AUC-Score: ", r_a_score)
