#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


# get homesite & test csv files as a DataFrame
homesite_df = pd.read_csv('../input/train.csv')
test_df     = pd.read_csv('../input/test.csv')

# preview the data
homesite_df.head()


# In[ ]:


homesite_df.info()
print("----------------------------")
test_df.info()


# In[ ]:


# drop unnecessary columns, these columns won't be useful in analysis and prediction
homesite_df = homesite_df.drop(['QuoteNumber'], axis=1)


# In[ ]:


# date

homesite_df['Year']  = homesite_df['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
homesite_df['Month'] = homesite_df['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
homesite_df['Week']  = homesite_df['Original_Quote_Date'].apply(lambda x: int(str(x)[8:10]))

test_df['Year']  = test_df['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
test_df['Month'] = test_df['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
test_df['Week']  = test_df['Original_Quote_Date'].apply(lambda x: int(str(x)[8:10]))

homesite_df.drop(['Original_Quote_Date'], axis=1,inplace=True)
test_df.drop(['Original_Quote_Date'], axis=1,inplace=True)


# In[ ]:


# customers purchased 

sns.countplot(x="QuoteConversion_Flag", data=homesite_df)


# In[ ]:


# year

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,5))

sns.countplot(x="QuoteConversion_Flag",hue="Year", data=homesite_df, ax=axis1)
sns.countplot(x=homesite_df["Year"].loc[homesite_df["QuoteConversion_Flag"] == 1], order=[2013,2014,2015], ax=axis2)


# In[ ]:


# month

sns.countplot(x=homesite_df["Month"].loc[homesite_df["QuoteConversion_Flag"] == 1], order=[1,2,3,4,5,6,7,8,9,10,11,12])


# In[ ]:


# fill NaN values

homesite_df.fillna(-1, inplace=True)
test_df.fillna(-1, inplace=True)


# In[ ]:


from sklearn import preprocessing

for f in homesite_df.columns:
    if homesite_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(homesite_df[f].values) + list(test_df[f].values))
        homesite_df[f] = lbl.transform(list(homesite_df[f].values))
        test_df[f] = lbl.transform(list(test_df[f].values))


# In[ ]:


# define training and testing sets

X_train = homesite_df.drop("QuoteConversion_Flag",axis=1)
Y_train = homesite_df["QuoteConversion_Flag"]
X_test  = test_df.drop("QuoteNumber",axis=1).copy()


# In[ ]:


# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict_proba(X_test)

random_forest.score(X_train, Y_train)


# In[ ]:


# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)


# In[ ]:


Y_pred[:,1][0:10]


# In[ ]:





# In[ ]:




