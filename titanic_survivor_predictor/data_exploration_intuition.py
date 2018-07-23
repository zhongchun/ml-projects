#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @FileName: undefined
 @Desc:
 @Author: yuzhongchun
 @Date: 2018-07-22 18:26:49
 @Last Modified by: yuzhongchun
 @Last Modified time: 2018-07-22 18:26:49
"""

import pandas as pd
from matplotlib import pyplot as plt

train_url = "./titanic_survivor_predictor/datasets/train.csv"
test_url = "./titanic_survivor_predictor/datasets/test.csv"

train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

# data properties
train_df.info()
print(train_df.head(10))

# data visualization
fit = plt.figure(figsize=(18, 6))

plt.subplot2grid((2, 3), (0, 0))
train_df.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Survived")

plt.subplot2grid((2, 3), (0, 1))
plt.scatter(train_df.Survived, train_df.Age, alpha=0.1)
plt.title("Age wrt Survived")

plt.subplot2grid((2, 3), (0, 2))
train_df.Pclass.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Class")

plt.subplot2grid((2, 3), (1, 0), colspan=2)
for x in [1, 2, 3]:
    train_df.Age[train_df.Pclass == x].plot(kind='kde')
plt.title("Class wrt Age")
plt.legend(("1st", "2nd", "3rd"))

plt.subplot2grid((2, 3), (1, 2))
train_df.Embarked.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Embarked")

plt.show()

fit = plt.figure(figsize=(18, 6))
plt.subplot2grid((3, 4), (0, 0))
train_df.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Survived")

plt.subplot2grid((3, 4), (0, 1))
train_df.Survived[train_df.Sex == 'male'].value_counts(normalize=True).plot(
    kind="bar", alpha=0.5)
plt.title("Men Survived")

plt.subplot2grid((3, 4), (0, 2))
train_df.Survived[train_df.Sex == 'female'].value_counts(normalize=True).plot(
    kind="bar", alpha=0.5, color="#FA0000")
plt.title("Women Survived")

plt.subplot2grid((3, 4), (0, 3))
train_df.Sex[train_df.Survived == 1].value_counts(normalize=True).plot(
    kind="bar", alpha=0.5, color=["b", "r"])
plt.title("Sex of Survived")

plt.subplot2grid((3, 4), (1, 0), colspan=4)
for x in [1, 2, 3]:
    train_df.Survived[train_df.Pclass == x].plot(kind='kde')
plt.title("Class wrt Survived")
plt.legend(("1st", "2nd", "3rd"))

plt.subplot2grid((3, 4), (2, 0))
train_df.Survived[(train_df.Sex == 'male')
                  & (train_df.Pclass == 1)].value_counts(normalize=True).plot(
                      kind="bar", alpha=0.5)
plt.title("Rich Men Survived")

plt.subplot2grid((3, 4), (2, 1))
train_df.Survived[(train_df.Sex == 'male')
                  & (train_df.Pclass == 3)].value_counts(normalize=True).plot(
                      kind="bar", alpha=0.5)
plt.title("Poor Men Survived")

plt.subplot2grid((3, 4), (2, 2))
train_df.Survived[(train_df.Sex == 'female')
                  & (train_df.Pclass == 1)].value_counts(normalize=True).plot(
                      kind="bar", alpha=0.5)
plt.title("Rich Women Survived")

plt.subplot2grid((3, 4), (2, 3))
train_df.Survived[(train_df.Sex == 'female')
                  & (train_df.Pclass == 3)].value_counts(normalize=True).plot(
                      kind="bar", alpha=0.5)
plt.title("Poor Women Survived")

plt.show()
