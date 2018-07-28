#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @FileName: undefined
 @Desc:
 @Author: yuzhongchun
 @Date: 2018-07-28 23:04:08
 @Last Modified by: yuzhongchun
 @Last Modified time: 2018-07-28 23:04:08
"""

import pandas as pd

# machine learning
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
"""
Now we are ready to train a model and predict the required solution.
There are 60+ predictive modelling algorithms to choose from.
We must understand the type of problem and solution requirement to narrow down
to a select few models which we can evaluate. Our problem is a classification and
regression problem. We want to identify relationship between output (Survived or not)
with other variables or features (Gender, Age, Port...). We are also perfoming a
category of machine learning which is called supervised learning as we are training
our model with a given dataset. With these two criteria - Supervised Learning plus
Classification and Regression, we can narrow down our choice of models to a few. These include:

    -Perceptron
    -Logistic Regression
    -KNN or k-Nearest Neighbors
    -Naive Bayes classifier
    -Support Vector Machines
    -Decision Tree
    -Random Forrest
    -Artificial neural network
    -RVM or Relevance Vector Machine
"""

train_url = "./titanic_survivor_predictor/workflow/train_df_fatures.csv"
test_url = "./titanic_survivor_predictor/workflow/test_df_fatures.csv"
train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()
print(X_train.shape, Y_train.shape, X_test.shape)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print("Logistic Regression:", acc_log)

# We can use Logistic Regression to validate our assumptions and decisions for
# feature creating and completing goals. This can be done by calculating the
# coefficient of the features in the decision function.
# Positive coefficients increase the log-odds of the response
# (and thus increase the probability), and negative coefficients
# decrease the log-odds of the response (and thus decrease the probability).
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
print(coeff_df.sort_values(by='Correlation', ascending=False))

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print("SVC: ", acc_svc)

# kNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print("kNN: ", acc_knn)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print("Gaussian Naive Bayes: ", acc_gaussian)

# Perceptron
perceptron = Perceptron(max_iter=10, tol=None)
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print("Perceptron: ", acc_perceptron)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print("Linear SVC: ", acc_linear_svc)

# Stochastic Gradient Descent
sgd = SGDClassifier(max_iter=10, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print("SGD: ", acc_sgd)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print("Decision Tree: ", acc_decision_tree)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print("Random Forest: ", acc_random_forest)

# Model evaluation
models = pd.DataFrame({
    'Model': [
        'Support Vector Machines', 'KNN', 'Logistic Regression',
        'Random Forest', 'Naive Bayes', 'Perceptron',
        'Stochastic Gradient Decent', 'Linear SVC', 'Decision Tree'
    ],
    'Score': [
        acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian,
        acc_perceptron, acc_sgd, acc_linear_svc, acc_decision_tree
    ]
})
print("=" * 60)
print(models.sort_values(by='Score', ascending=False))
print("=" * 60)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": Y_pred
})
submission.to_csv(
    './titanic_survivor_predictor/workflow/submission.csv', index=False)
