#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[ ]:


#Import the dataset
churn_data = pd.read_csv('../input/churn_data.csv')
churn_data.head() #Printing first 5 rows of the dataset


# In[ ]:


#Logistic Regression
#Tell the model thear are some categorical variables, using C() before the categorical variables
mod1 = smf.glm(formula='returnCustomer~ C(title)+C(paymentMethod)+C(couponDiscount)+C(purchaseValue)+               C(giftwrapping)+C(throughAffiliate)+C(shippingFees)+C(dvd)+C(blueray)+C(vinyl)+C(videogame)+C(videogameDownload)               +C(tvEquiment)+C(prodOthers)+C(prodSecondHand)',
               
               data=churn_data, family=sm.families.Binomial()).fit()
mod1.summary()


# In[ ]:


# Sklearn is a very popular package in maching learning, we will often use this
X=churn_data[['title','paymentMethod','couponDiscount','purchaseValue','giftwrapping','throughAffiliate','shippingFees','dvd','blueray','vinyl','videogame','videogameDownload','tvEquiment','prodOthers','prodSecondHand']]
y=churn_data['returnCustomer']


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

#Tell the model thear are some categorical variables
X=pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(X_train, y_train)


# In[ ]:


y_pred = logreg.predict(X_test)
#test your model on new test data and show accuracy
print(logreg.score(X_test, y_test))


# In[ ]:


# See accuracy in a table
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[ ]:


from sklearn.metrics import roc_curve, auc, roc_auc_score
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
auc(false_positive_rate, true_positive_rate)


# In[ ]:




