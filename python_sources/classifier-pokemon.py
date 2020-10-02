#!/usr/bin/env python
# coding: utf-8

# # Pokemon-Classification
# 
# ### Well everyone know's pokemon,Dont you? But the task of identifying the pokemon into legendary or not can be subtle one.Let's try to figure it out,Through using Machine learning and Data Science.
# 
# Models to be used :
# 
# 1.`RandomForest Classifier`
# 
# 2.`GradientBoostingClassifier`
# 
# 3.`LogisticRegression`
# 
# Workflow :
# 
# 1.Importing data tools.
# 
# 2.Studying data.
# 
# 3.Visualizing data.
# 
# 4.Modelling
# 
# 5.Evaluating model
# 
# 6.Experimenting with data.
# 

# ## 1.Importing tools :

# In[ ]:


# Boiler plate tools :
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## For modelling :
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Modelling tools :
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score,RandomizedSearchCV,GridSearchCV 


# ## 2.Data study :

# In[ ]:


data = pd.read_csv('/kaggle/input/pokemon/pokemon.csv')
data.head()


# In[ ]:


# Let's find out whether there is missing data or not...
data.isna().sum()


# In[ ]:


## The data looks quiet missing,Lets fill it.But before lets check the main metrics...
fig,axes = plt.subplots(figsize=(10,10))
axes.bar(data['name'][:10],data['attack'][:10],color='salmon');
plt.title('Pokemon attack');
plt.xlabel('Pokemon');
plt.ylabel('Attack');


# In[ ]:


data.plot(kind='scatter',x='name',y='sp_attack');


# In[ ]:


# Lets Check the missing values...
for label,content in data.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[ ]:


# Fill them..
for label,content in data.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            data[label] = content.fillna(content.median())
            


# In[ ]:


data.dtypes


# In[ ]:


for label,content in data.items():
    if pd.api.types.is_float_dtype(content):
        data[label] = data[label].astype('int')


# In[ ]:


data.dtypes


# In[ ]:


for label,content in data.items():
    if not pd.api.types.is_numeric_dtype(content):
        data[label] = data[label].astype('category')


# In[ ]:


data.dtypes


# In[ ]:


for label,content in data.items():
    if pd.api.types.is_categorical_dtype(content):
        data[label] = pd.Categorical(content).codes + 1


# In[ ]:


X = data.drop('is_legendary',axis=1)
y = data['is_legendary']


# In[ ]:


model_a = RandomForestClassifier()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model_a.fit(X_train,y_train)
model_a.score(X_test,y_test)


# In[ ]:


model_b = GradientBoostingClassifier()
model_b.fit(X_train,y_train)
model_b.score(X_test,y_test)


# In[ ]:


model_c = LogisticRegression()
model_c.fit(X_train,y_train)
model_c.score(X_test,y_test)


# ## Clearly , RandomForestClassifier has highest score,Lets use it.
# 
# `RFC score = 1.0`
# 
# `GBC score = 0.9937888198757764`
# 
# `LR score = 0.9937888198757764 `

# In[ ]:


# Lets check the cross val score
y_preds = model_a.predict_proba(X_test)
cvm = cross_val_score(model_a,X,y,cv=10)
np.mean(cvm)


# In[ ]:


# Classification metrics :
y_preds = model_a.predict(X_test)

precision = precision_score(y_test,y_preds)
recall = recall_score(y_test,y_preds)
accuracy = accuracy_score(y_test,y_preds)
accuracy,recall,precision


# In[ ]:


## Lets get the legendary predictions : 
Pokemon = pd.DataFrame()
y_preds = model_a.predict(X)
Pokemon['Default values'] = y
Pokemon['Predictions'] = y_preds


# In[ ]:


Pokemon


# In[ ]:


fig,axes = plt.subplots()
axes.stackplot(Pokemon['Default values'],Pokemon['Predictions'],color=['red','blue']);

