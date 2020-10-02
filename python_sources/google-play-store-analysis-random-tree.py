# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import random
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

dataset = pd.read_csv("../input/googleplaystore.csv").dropna()
print (dataset.head(5))

dataset['Installs'].head(2)
dataset.Installs = dataset.Installs.astype(str).str.replace(",","").astype(str)
dataset.Installs = dataset.Installs.apply(lambda x: x.strip("+"))
dataset.Installs = dataset.Installs.replace("Free", 0)
dataset.Installs = dataset.Installs.astype(float)

dataset.Reviews = dataset.Reviews.astype(str).str.replace(".0M","000.0").astype(str)

dataset.Reviews = dataset.Reviews.astype(float)
dataset.Rating = dataset.Rating.fillna(0)

coll_list = dataset.columns
corr = dataset[coll_list].corr()
sns.heatmap(corr)

rows = [random.choice(dataset.index.values) for i in range(10000)]
x = dataset[['Reviews', 'Rating']].loc[rows]
y = dataset['Installs'].loc[rows]

#x = dataset[['Reviews', 'Rating']]
#y = dataset['Installs']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
print (x_train.shape)
print (x_test.shape)
print (y_train.shape)
print (y_test.shape)

dr = tree.DecisionTreeClassifier()
dr.fit(x_train,y_train)
y_pred = dr.predict(x_test)
print ("Decision Tree Classifier accuracy is: ",metrics.accuracy_score( y_test, y_pred)*100, '%')


# Any results you write to the current directory are saved as output.