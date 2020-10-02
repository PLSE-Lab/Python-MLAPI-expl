#!/usr/bin/env python
# coding: utf-8

# # The dataset

# 
# Content
# candy-data.csv includes attributes for each candy along with its ranking. For binary variables, 1 means yes, 0 means no. The data contains the following fields:
# 
# chocolate: Does it contain chocolate?
# 
# fruity: Is it fruit flavored?
# 
# caramel: Is there caramel in the candy?
# 
# peanutalmondy: Does it contain peanuts, peanut butter or almonds?
# nougat: Does it contain nougat?
# 
# crispedricewafer: Does it contain crisped rice, wafers, or a cookie component?
# 
# hard: Is it a hard candy?
# 
# bar: Is it a candy bar?
# 
# pluribus: Is it one of many candies in a bag or box?
# 
# sugarpercent: The percentile of sugar it falls under within the data set.
# 
# pricepercent: The unit price percentile compared to the rest of the set.
# 
# winpercent: The overall win percentage according to 269,000 matchups.

# import the necessary libraries.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas_profiling import ProfileReport


# Let's load the file and see what the data looks like.

# In[ ]:


candy = pd.read_csv('../input/the-ultimate-halloween-candy-power-ranking/candy-data.csv')


# In[ ]:


candy.head()


# In[ ]:


candy.shape


# In[ ]:


candy.info()


# Reporting data to know more details about it

# # EDA

# In[ ]:


def count(feature):
    
    # Show the counts of observations in each categorical bin using bars
    sns.countplot(x=feature,data=candy)
    


# In[ ]:


candy.head()


# In[ ]:


fig, ax = plt.subplots(3, 3,figsize=(15,20))
plt.subplot(3,3,1)
count('chocolate')
plt.subplot(3,3,2)
count('fruity')
plt.subplot(3,3,3)
count('caramel')
plt.subplot(3,3,4)
count('peanutyalmondy')
plt.subplot(3,3,5)
count('nougat')
plt.subplot(3,3,6)
count('crispedricewafer')
plt.subplot(3,3,7)
count('bar')
plt.subplot(3,3,8)
count('pluribus')
plt.subplot(3,3,9)
count('hard')


# I can notice that most candies do not have a filling inside
# 
# The most common filling is chocolate and fruit

# ### Let's find out the percentages

# In[ ]:


def box(var):
    # this function take the variable and return a boxplot for each type of fish
    sns.boxplot(x="chocolate", y=var, data=candy,palette='rainbow')


# In[ ]:


fig, ax = plt.subplots(3, 1,figsize=(15,20))
plt.subplot(3,1,1)
box('sugarpercent')
plt.subplot(3,1,2)
box('pricepercent')
plt.subplot(3,1,3)
box('winpercent')


# We can see that the dessert that contains chocolate has an average sugar level,
# 
# more expensive than the rest,
# 
# and the percentage of winning it according to the larger matching cases

# In[ ]:


candy.head()


# # Feature Selection

# Feature importance is an inbuilt class that comes with Tree Based Classifiers, we will be using Extra Tree Classifier for extracting the top 8 features for the dataset.

# In[ ]:


X = candy.drop(['chocolate','competitorname'],axis=1) #independent columns
y = candy['chocolate']   #target column i.e price range
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10,10))
feat_importances.nlargest(8).plot(kind='barh')
plt.show()


# Now we can see the 8 most important variables in this data

# # Correlation Matrix with Heatmap

# Correlation states how the features are related to each other or the target variable.
# Correlation can be positive (increase in one value of feature increases the value of the target variable) or negative (increase in one value of feature decreases the value of the target variable)
# Heatmap makes it easy to identify which features are most related to the target variable, we will plot heatmap of correlated features 

# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(candy.corr(),cmap='coolwarm',annot=True,linecolor='white',linewidths=4)


# ## Converting Categorical Features 

# We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

# In[ ]:


candy.info()


# In[ ]:


competitorname = pd.get_dummies(candy['competitorname'],drop_first=True)


# In[ ]:


candy.drop('competitorname',axis=1,inplace=True)


# In[ ]:


candy=pd.concat([candy,competitorname],axis=1)


# In[ ]:


candy.head()


# # Building a Logistic Regression model

# ### Training and Testing Data

# In[ ]:


X = candy.drop('chocolate',axis=1)
y = candy['chocolate']


# In[ ]:


#spliting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train ,X_test , y_train , y_test =train_test_split(X,y, test_size = 0.2 , random_state=4)


# # Training and Predicting

# In[ ]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()


# ### fit the model

# In[ ]:


log.fit(X_train,y_train)


# In[ ]:


predictions = log.predict(X_test)


# Let's move on to evaluate our model!

# # Evaluation

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


confusion_matrix(y_test,predictions)


# In[ ]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, predictions))


# # model is ready

# I hope you enjoy this study and I would appreciate if you add your comments below.
# 
# Seif Mohamed
