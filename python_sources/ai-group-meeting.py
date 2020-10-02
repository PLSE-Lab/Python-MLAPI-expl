#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Workflow
# 1. Problem.
# 2. Acquire data.
# 3. Prepare, cleanse the data.
# 4. Analyze and explore the data.
# 5. Model the problem.
# 6. Supply or submit the results.
# 
# Just a remainder of titanic dataset:
# Knowing from a training set of samples listing passengers who survived or did not survive the Titanic disaster, can our model determine based on a given test dataset not containing the survival information, if these passengers in the test dataset survived or not.
# 
# * On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. Translated 32% survival rate.
# * One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew.
# * Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# All given by kaggle's description. 

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]


# In[ ]:


print(train_df.columns.values)


# In[ ]:


train_df.head()


# In[ ]:


train_df.tail()


# In[ ]:


train_df.info()
print('_'*40)
test_df.info()


# In[ ]:


train_df.describe()


# ### Assumtions based on data analysis
# 
# We arrive at following assumptions based on data analysis done so far. We may validate these assumptions further before taking appropriate actions.
# 
# **Correlating.**
# 
# We want to know how well does each feature correlate with Survival. We want to do this early in our project and match these quick correlations with modelled correlations later in the project.
# 
# **Completing.**
# 
# 1. We may want to complete Age feature as it is definitely correlated to survival.
# 2. We may want to complete the Embarked feature as it may also correlate with survival or another important feature.
# 
# **Correcting.**
# 
# 1. Ticket feature may be dropped from our analysis as it contains high ratio of duplicates (22%) and there may not be a correlation between Ticket and survival.
# 2. Cabin feature may be dropped as it is highly incomplete or contains many null values both in training and test dataset.
# 3. PassengerId may be dropped from training dataset as it does not contribute to survival.
# 4. Name feature is relatively non-standard, may not contribute directly to survival, so maybe dropped.
# 
# **Creating.**
# 
# 1. We may want to create a new feature called Family based on Parch and SibSp to get total count of family members on board.
# 2. We may want to engineer the Name feature to extract Title as a new feature.
# 3. We may want to create new feature for Age bands. This turns a continous numerical feature into an ordinal categorical feature.
# 4. We may also want to create a Fare range feature if it helps our analysis.
# 
# **Classifying.**
# 
# We may also add to our assumptions based on the problem description noted earlier.
# 
# 1. Women (Sex=female) were more likely to have survived.
# 2. Children (Age<?) were more likely to have survived. 
# 3. The upper-class passengers (Pclass=1) were more likely to have survived.

# In[ ]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )
plot_correlation_map( train_df )


# In[ ]:


print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin',"Name","Embarked"], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin',"Name","Embarked"], axis=1)

combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape


# In[ ]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()


# ## Model, predict and solve
# 
# Now we are ready to train a model and predict the required solution. There are 60+ predictive modelling algorithms to choose from. We must understand the type of problem and solution requirement to narrow down to a select few models which we can evaluate. Our problem is a classification and regression problem. We want to identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port...). We are also perfoming a category of machine learning which is called supervised learning as we are training our model with a given dataset. With these two criteria - Supervised Learning plus Classification and Regression, we can narrow down our choice of models to a few. These include:
# 
# - Logistic Regression
# - KNN or k-Nearest Neighbors
# - Support Vector Machines
# - Naive Bayes classifier
# - Decision Tree
# - Random Forrest
# - Perceptron
# - Artificial neural network
# - RVM or Relevance Vector Machine

# 

# In[ ]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
X_train = train_df.drop(["PassengerId","Survived","Fare","Age"], axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop(["PassengerId","Fare","Age"],axis=1).copy()
X_train.head()
#X_train.info()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# In[ ]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[ ]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

