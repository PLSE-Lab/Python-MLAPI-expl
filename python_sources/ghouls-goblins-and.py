#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# load packages 
import pandas as pd
import numpy as np
# visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
# scikitlearn packages
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# PERFORMANCE PARAMETERS
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# load data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


# missing values
train.isnull().sum().sum()


# In[ ]:


# visualization
sns.countplot(x='color', data=train) 


# In[ ]:


sns.countplot(x='type', data=train)  # data is fairly balanced


# In[ ]:


# shuffle the datasets to randomized the 
train = train.sample(frac=1)
train.head()


# In[ ]:


# feature selection
y = train.type
# drop irrelevant columns
train = train.drop(['type','id'], axis=1)
train.head()


# In[ ]:


train.describe()


# In[ ]:


# scatter plot matrix
scatter_matrix(train)
plt.show() # dataset is fairly normalized 


# In[ ]:


# one hot encoding
train = pd.get_dummies(train, prefix_sep='_', drop_first=True)
train.head()


# In[ ]:


# label encoder for the outputs
from sklearn.preprocessing import LabelEncoder

le_y = LabelEncoder()
le_y = le_y.fit(y)

y = le_y.transform(y)
y


# In[ ]:


# adjusted r-squared
# import statsmodels. formula.api
import statsmodels.formula.api as smf
# regression formular
model = smf.ols(formula='y ~  bone_length + rotting_flesh + hair_length + has_soul + color_blood + color_blue + color_clear + color_green + color_white', data=train)
# fit the regression
model_fit = model.fit()
# extract and readjust r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)


# In[ ]:


# no ensembling techniques
# desicion tree

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=.1, random_state=42)


# In[ ]:


from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state=42)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
# performance
trees= accuracy_score(y_test,y_pred)


# In[ ]:


# logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=42)
lr.fit(X_train,y_train)

# performance
y_pred= lr.predict(X_test)
# performance
log =accuracy_score(y_test,y_pred)


# In[ ]:


# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

# perofrmance
y_pred = knn.predict(X_test)
# performance
knn = accuracy_score(y_test,y_pred)


# In[ ]:


# naive bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train,y_train)

# perofrmance
y_pred = gnb.predict(X_test)
# performance
bayes = accuracy_score(y_test,y_pred)


# In[ ]:


plt.figure(figsize=(16, 6))
s = sns.barplot(x=["Naive Bayes", "Logistic Regression","KNN","Decision Trees"], y=[bayes,knn,log,trees],color="seagreen")
for p in s.patches:
    s.annotate(format(p.get_height(), '.2f'), 
              (p.get_x() + p.get_width() / 2., 

               p.get_height()), ha = 'center', va = 'center', 
              xytext = (0, 10), textcoords = 'offset points')


# In[ ]:


# ensemble models
# BAGGING
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_features='auto',criterion='entropy', class_weight='balanced')

grid = {"n_estimators":[1,200],"max_depth":[1,200]}

clf_cv = GridSearchCV(clf,grid,cv=5)
clf_cv.fit(X_train,y_train)

clf_cv.best_params_



clf = accuracy_score(y_test,y_pred)


# In[ ]:


# BOOSTING
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=3)
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)

gbc =accuracy_score(y_test,y_pred)


# In[ ]:


# stacking
from mlxtend.classifier import StackingClassifier
knn = KNeighborsClassifier(n_neighbors=4)
lr = LogisticRegression()
gnb = GaussianNB()

sclf = StackingClassifier(classifiers=[knn,lr,gnb], meta_classifier=gnb)
sclf.fit(X_train,y_train)
y_pred = sclf.predict(X_test)

sclf = accuracy_score(y_test,y_pred)


# In[ ]:


plt.figure(figsize=(16, 6))
s = sns.barplot(x=["Random Forest Classifier","Gradient Boosting","Stacking"], y=[clf,gbc,sclf],color="maroon")
for p in s.patches:
    s.annotate(format(p.get_height(), '.2f'), 
              (p.get_x() + p.get_width() / 2., 

               p.get_height()), ha = 'center', va = 'center', 
              xytext = (0, 10), textcoords = 'offset points')

