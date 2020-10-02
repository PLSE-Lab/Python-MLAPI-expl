#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# display an image
from IPython.display import Image
Image(url='https://miro.medium.com/max/844/1*MyKDLRda6yHGR_8kgVvckg.png') # Thanks manan-bedi2908 


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns' , None)


# In[ ]:


data = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')
data


# In[ ]:


data.shape


# In[ ]:


data.dtypes.value_counts()


# In[ ]:


data.isnull().sum()


# In[ ]:


df = data.drop(['RowNumber' , 'CustomerId' , 'Surname'] , axis=1)


# In[ ]:


# target 
df.Exited.value_counts(normalize=True)


# In[ ]:


# continuous and categorical variables

var_continuous = df.drop(['Exited' , 'Geography' , 'Gender' , 'HasCrCard' , 'IsActiveMember'] , axis = 1 )
var_categ = df[['Geography' , 'Gender' , 'HasCrCard' , 'IsActiveMember']]


# In[ ]:


var_continuous


# In[ ]:


var_categ


# In[ ]:


# Distributions continuous variables
for col in var_continuous:
    plt.figure()
    sns.distplot(df[col])


# In[ ]:


#viz categorical variables
for col in var_categ:
    plt.figure()
    df[col].value_counts().plot.pie()


# In[ ]:


# Dist target/variables

no_churn = df[df['Exited']==0]
churn = df[df['Exited'] == 1]


# In[ ]:


for col in var_continuous:
    plt.figure()
    sns.distplot(no_churn[col] , label = "negative")
    sns.distplot(churn[col] , label = "positive")
    plt.legend()


# In[ ]:


# Target / Age
plt.figure(figsize=(20,10))
sns.countplot(x='Age' , hue ='Exited' , data = df)


# In[ ]:


# Target / categorical variables
pd.crosstab(df['Exited'] , df.Geography)


# In[ ]:


for col in var_categ:
    plt.figure()
    sns.heatmap(pd.crosstab(df['Exited'] , df[col]) , annot=True)


# In[ ]:


# Preprocessing - Encoding

df = pd.get_dummies(df , drop_first=True)


# In[ ]:


plt.figure(figsize=(15,15))
sns.pairplot(df)


# In[ ]:


df


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr() , annot=True)


# In[ ]:


# Target and features 
X = df.drop(['Exited'] ,  axis=1)
y = df['Exited']


# In[ ]:


X


# In[ ]:


y


# In[ ]:


# train - test - split


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.2 , random_state = 5)


# In[ ]:


# Standardizing the Dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


X_train


# In[ ]:


# Features importance
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)


# In[ ]:


print(model.feature_importances_)


# In[ ]:


feat_importance = pd.Series(model.feature_importances_ , index=X.columns)
feat_importance.nlargest(5).plot(kind='barh') # 
plt.show()


# ### Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)


# In[ ]:


y_pred = rf.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix , f1_score , classification_report
cm = confusion_matrix(y_test,y_pred)
#f1 = f1_score(y_test , y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred)) #1
#print(f1)


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




