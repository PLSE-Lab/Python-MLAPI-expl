#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
from sklearn import linear_model, preprocessing, svm, tree, neighbors, model_selection
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost import plot_importance
import seaborn as sns
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


# In[ ]:


steam=pd.read_csv("../input/steam-store-games/steam.csv")
steam.head()


# Removing special characters from categorical columns to see if there's anything I can do, ended up tokenizing them. It was no use lol (at least they look good)

# In[ ]:


steam['platforms']=steam['platforms'].str.replace(r'\W',' ')
steam['categories']=steam['categories'].str.replace(r'\W',' ')
steam['steamspy_tags']=steam['steamspy_tags'].str.replace(r'\W',' ')
steam.head()


# Decided to do exploratory data analysis, let's dive in. Let's see average prices of publishers.

# In[ ]:


steam.publisher.nunique()


# In[ ]:


price_averages=steam.groupby('publisher')['price'].mean()
price_averages.head()


# Let's see if there's any correlation between the price and the number of positive ratings a game has.

# In[ ]:


steam.positive_ratings.corr(steam.price)


# I don't think so. Let's edit owners column into something decent. 

# In[ ]:


steam.owners.unique()


# I'm going to take the upper side of the intervals by simply removing the number and the line. The owners column is a datatype of object, so we have to convert it into numeric before moving on.

# In[ ]:


steam.owners=steam.owners.str.replace('\d+-', '')
steam.owners=pd.to_numeric(steam.owners)


# In[ ]:


steam.owners.dtype


# Let's see which game genre is most trending.

# In[ ]:


genreprices=steam.groupby("genres")['owners'].mean()
genreprices=pd.DataFrame(genreprices)
genreprices.head()
genreprices.sort_values("genres", ascending=False)


# Violence sells. Let's preprocess the data for regression and classification.

# In[ ]:


steam=steam.drop(columns=["release_date","name","developer","appid","english","achievements"])
steam.head()


# Label encoding the data.

# In[ ]:


le = preprocessing.LabelEncoder()
steam.genres=le.fit_transform(steam.genres)
steam.steamspy_tags=le.fit_transform(steam.steamspy_tags)
steam.categories=le.fit_transform(steam.categories)
steam.publisher=le.fit_transform(steam.publisher)
steam.platforms=le.fit_transform(steam.platforms)
steam.head()


# Correlations seem to be weak.

# In[ ]:


corr = steam.corr()
corr


# In[ ]:


sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# Let's predict the prices from rest of the features with linear regression.

# In[ ]:


X=steam.loc[:, steam.columns != 'price']
y=steam.price


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[ ]:


linearreg = LinearRegression(fit_intercept=True, normalize=False)
linearreg.fit(X_train, y_train)
predlinear = linearreg.predict(X_test)


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test, predlinear)


# I want to try to predict the number of owners since there's a little bit of correlation comparing to price. Taking the upper side of the interval might yield a high error though, let's dive in.

# In[ ]:


X=steam.loc[:, steam.columns != 'owners']
y=steam.owners
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=56)
r2_score(y_test, predlinear)


# Negative R2 score means that model really underperformed comparing to the case where I'd use the mean. Let's see the other metrics of this model. 

# In[ ]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, predlinear)


# As I have guessed using the upper side of the interval has failed me. 

# Let's try to predict the genres with several classifiers. There are 1552 genres so I'm not expecting a high accuracy rate or anything.

# In[ ]:


steam.genres.nunique()


# In[ ]:


X=steam.loc[:, steam.columns != 'genres']
y=steam.genres
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[ ]:


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
predknn = neigh.predict(X_test)


# In[ ]:


print(classification_report(y_test, predknn, digits=5))


# KNN yields 19 percent of accuracy. At least you tried. Let's see what decision trees have to offer.

# In[ ]:


dtclassifier=DecisionTreeClassifier(criterion="entropy", max_depth=None)
dtclassifier.fit(X_train,y_train)
preddt = dtclassifier.predict(X_test)


# In[ ]:


print(classification_report(y_test, preddt, digits=5))


# Using decision trees I've reached an accuracy of 47 percent. I will now use XGBoost. 

# In[ ]:


xgbclassifier=XGBClassifier(silent=0, learning_rate=0.01,  objective='reg:logistic',n_estimators=100, max_depth=3, gamma=3)
xgbclassifier.fit(X_train, y_train)
predxgb = xgbclassifier.predict(X_test)


# I've kept the parameters as greedy as possible so it took me half an hour to train the model, yet yielding not that much of a difference comparing to decision trees. My null hypothesis in this case is not that strong anyway, I didn't really expect the XGBoost to give me a 80 percent accuracy or anything, nothing can, but at least I know that with feature randomization, bagging and boosting, my model doesn't overfit or underfit and it's a comfort to know that I have a good bias-variance trade-off.

# In[ ]:


print(classification_report(y_test, predxgb, digits=3))

