#!/usr/bin/env python
# coding: utf-8

# ## Comparing Classification Methods.
# 
# In this notebook, I'm going to compare two different classification methods, logistic regression and a random forest.
# I know this would probably be more informative if the data wasn't so clean and the fits so good, but I'm just getting
# my feet wet.  I'll just use the default setting on each of the methods.  I did some very light feature engineering by
# adding in two columns for the area of the sepal and petal.  The random tree should find this anyway, so it's more for
# the logistic regression.

# In[ ]:


#Seaborn throws a lot of warnings, so we're going to ignore them.
import warnings
warnings.filterwarnings("ignore");

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


df = pd.read_csv('../input/Iris.csv', index_col = 'Id')
df['SepalAreaCm'] = df.SepalLengthCm * df.SepalWidthCm
df['PetalAreaCm'] = df.PetalLengthCm * df.PetalWidthCm
df.Species = df.Species.astype('category')


# Lets take a look at what our interaction features look like.

# In[ ]:


sns.lmplot(x = 'SepalAreaCm', y = 'PetalAreaCm', data = df, hue = 'Species', fit_reg=False);
df[['SepalAreaCm', 'PetalAreaCm', 'Species']].groupby('Species').mean()


# ### Logistic Regression

# In[ ]:


X = df.iloc[:,[0,1,2,3,5,6]]
Y = df.Species.cat.codes

logreg = LogisticRegression(C=1e5).fit(X,Y)

kfold = cross_validation.KFold(len(Y), n_folds=30)
lrScores = cross_validation.cross_val_score(logreg, X, Y, cv=kfold)
print('Score for cross validation: {:.2%}'.format(lrScores.mean()))


# ### Logistic Regression on Sepal length and Width

# In[ ]:


X_sep = df.iloc[:,[0,1]]

logregS = LogisticRegression(C=1e5).fit(X_sep,Y)

lrsScores = cross_validation.cross_val_score(logregS, X_sep, Y, cv=kfold)
print('Score for cross validation: {:.2%}'.format(lrsScores.mean()))


# ### Logistic Regression on Petal Length and Width

# In[ ]:


X_ped = df.iloc[:,[2,3]]

logregP = LogisticRegression(C=1e5).fit(X_ped,Y)

lrpScores = cross_validation.cross_val_score(logregP, X_ped, Y, cv=kfold)
print('Score for cross validation: {:.2%}'.format(lrpScores.mean()))


# ### Logistic Regression on Sepal and Petal Area

# In[ ]:


X_are = df.iloc[:,[5,6]]

logregA = LogisticRegression(C=1e5).fit(X_are,Y)

lraScores = cross_validation.cross_val_score(logregA, X_sep, Y, cv=kfold)
print('Score for cross validation: {:.2%}'.format(lraScores.mean()))


# The regression for all of the features predicts right 96% of the time.  That's pretty good.
# On the next three cells, I fit with just the sepal lengths, the petal lengths, and the areas respectively.  It seems from the above, that the most significant
# feature would be the length and width of the petals.  The other two sets only increase the accuracy by one percent.
# 
# 
# ### Random Forest

# In[ ]:


rf = RandomForestClassifier().fit(X,Y)

rfScores = cross_validation.cross_val_score(rf, X, Y, cv=kfold)
print('Score for cross validation where: {:.2%}'.format(rfScores.mean()))


# At 95% the random forest is slightly worse than the logistic regression.  I imagine that has to do
# with the default settings of the random forest.  As it stands though, the logistic regression is very slightly better for this data out of the box.
# I could do the same split as I did above, but the forest would perform for the worse.  It will perform better the more features you 
# use, so when fitting for a random forest you should use as many features as you can.  As I said above, the interaction terms I put in 
# probalby didn't help that much, because the forest will find those interactions by design.
# 
# As I said, this is my first script on Kaggle, save a few submissions on the tutorial data sets so in the future I should be digging in
# to more of these datasets in more depth.
