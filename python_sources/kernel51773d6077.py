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


#                                              **Sinking of Titanic**
# Image of Titanic
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# Do a complete analysis on what sorts of people were likely to survive.

# In[ ]:


import pandas as pd
import pandas_profiling 

titanic_df = pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


# import plotting libraries
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set(font_scale=1.5)

# import libraries for model validation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# import libraries for metrics and reporting
from sklearn.metrics import confusion_matrix


# In[ ]:



 df = titanic_df 


# In[ ]:



df.shape


# In[ ]:


df.head()


# **Dropping These Variables**
# 
# Fare - Does the fare a person paid effect his survivability? Will be related to Pclass
# What about a person's name, ticket number, and passenger ID number? # Identity columns
# 
# **Retaining These Variables**
# 
# Survived - This variable is obviously relevant.
# Pclass - Does a passenger's class on the boat affect their survivability?
# Sex - Could a passenger's gender impact their survival rate?
# Age - Does a person's age impact their survival rate?
# SibSp - Does the number of relatives on the boat (that are siblings or a spouse) affect a person survivability? Probably
# Parch - Does the number of relatives on the boat (that are children or parents) affect a person survivability? Probably
# Embarked - Does a person's point of embarkation matter?

# In[ ]:


# see distinct values in the Survived column
df.Survived.value_counts()


# In[ ]:


# see distinct values in the Sex column
df.Sex.value_counts()


# In[ ]:


df.columns


# In[ ]:


#create the new dataframe abd assign the varibale to it
X = pd.DataFrame()
X['sex'] = df['Sex']
X['age'] = df['Age']
X['pclass'] = df['Pclass']
X['sibsp'] = df['SibSp']
X['parch'] = df['Parch']
X['Embarked'] = df['Embarked']


# In[ ]:


X.head()


# In[ ]:


y = df['Survived']
y[:5]


# **Check Missing Values**

# In[ ]:


df.isnull().sum()


# * There are only 891 rows in the titanic data frame. Cabin is almost all missing values, so we can drop that variable completely
# * Age seems like a relevant predictor for survival right? We'd want to keep the variables, but it has 177 missing values.

# **Treating Missing Values in Age**

# In[ ]:


X.hist('age')


# In[ ]:


X['age'] = X['age'].fillna(X.age.median()) # because the hist is skewed
print (X.age.isnull().sum())


# **Treating Missing Values in Embarked**

# In[ ]:


print(X.Embarked.mode()[0])
X['Embarked'] = X['Embarked'].fillna(X.Embarked.mode()[0])
print (X.Embarked.isnull().sum())


# **Sex column**

# In[ ]:


print (X.sex[:5])
X['sex'] = pd.get_dummies(X.sex)['female']
print (X.sex[:5])


# **Creating dummies for varibles Pclass**

# In[ ]:


# see distinct values in the pclass column
X.pclass.value_counts()


# In[ ]:


X1 = X.join(pd.get_dummies(df.Pclass))#, prefix='pclass'))
X1[:5]


# In[ ]:


display (X[:5])
X = X.join(pd.get_dummies(df.Pclass, prefix ='pclass'))
display (X[:5])


# In[ ]:


X = X.drop(['pclass_1', 'pclass'], axis=1)
display (X[:5])


# **Creating dummies for varibles Embarked**

# In[ ]:


X = X.join(pd.get_dummies(df.Embarked, prefix ='Embarked'))
display (X[:5])


# In[ ]:


X = X.drop(['Embarked_C', 'Embarked'], axis=1)
display (X[:5])


# **Standardizing Age variable**

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
display (X[:5])
X.age = scaler.fit_transform(X[['age']])
display (X[:5])


# In[ ]:


X.hist('age')


# **Encoding the categorical variables and Standardizing the Continuous Variables are Hygiene Steps**

# **Model Building**

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


# In[ ]:


print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)


# In[ ]:


from sklearn.linear_model import LogisticRegression

# fit the model to the training data
model=LogisticRegression()
model.fit(X_train,y_train)

print (model.intercept_)
print (model.coef_)
print (X_train.columns)


# Interpretation of Model Equation
# Log_odds (Survival)
# Sex - +ve - prob of female survival is higher
# Age - -ve - lower the age survival is higher
# Sibsp - -ve
# Parch - -ve
# Pclass_2 - -ve
# Pclass_3 - -ve
# Embarked_Q - -ve
# Embarked_S - -ve

# log_odds = 0.16895636 + 2.56939259*sex - 0.36361616*age - 0.26361411*sibsp - 0.07520945*parch - 0.57435637*pclass_2 - 1.84009514*pclass_3 - 0.1786976*Embarked_Q  - 0.5393869*Embarked_S

# In[ ]:


# A female in Pclass_1 with Age 0.2 and Embarked_C
log_odds = 0.16895636 + 2.56939259*1 - 0.36361616*(0.2) - 0.26361411*0 - 0.07520945*0 - 0.57435637*0 - 1.84009514*0 - 0.1786976*0  - 0.5393869*0
log_odds


# In[ ]:


import numpy as np
p = np.exp(log_odds)/(1+np.exp(log_odds))
p


# prob for a female in Pclass_1 with Age 0.2 and Embarked_C is 0.9349

# In[ ]:


# A male in Pclass_1 with Age 0.2 and Embarked_C
log_odds = 0.16895636 + 2.56939259*0 - 0.36361616*(0.2) - 0.26361411*0 - 0.07520945*0 - 0.57435637*0 - 1.84009514*0 - 0.1786976*0  - 0.5393869*0
# log_odds
p = np.exp(log_odds)/(1+np.exp(log_odds))
p


# prob for a male in Pclass_1 with Age 0.2 and Embarked_C is 0.5240

# **Prediction**

# In[ ]:


display (X_test[:10])
print ()
display (model.predict_proba(X_test)[:10]) # prob
print ()
display (model.predict(X_test)[:10]) # classification


# In[ ]:


# compute the accuracy of our predictions
from sklearn.metrics import accuracy_score
print ("Logistic testing accuracy is %2.2f" % accuracy_score(y_test,model.predict(X_test)))


# In[ ]:


print ("Logistic training accuracy is %2.2f" % accuracy_score(y_train,model.predict(X_train)))


# **ROC Curve**

# In[ ]:


from sklearn.metrics import roc_auc_score
# lets measure the logistic model AUC
logistic_roc_auc = roc_auc_score(y_test, model.predict(X_test)) 
logistic_roc_auc


# In[ ]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])


# In[ ]:


model.predict_proba(X_test)[:,1][:5] # P(Y=1)


# In[ ]:


display (thresholds[:10])
display (fpr[:10])
display (tpr[:10])


# In[ ]:


# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label = 'ROC curve (area = %0.2f)' % logistic_roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc ="lower right")
plt.show()
print("Logistic AUC = %2.2f " % logistic_roc_auc )


# > **The overall Logistic AUC for this model is 0.79**
