#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

print(os.listdir('../input'))
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


df = pd.read_csv('../input/pokemon_alopez247.csv')


# In[10]:


df.head()


# In[ ]:


df.info()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ## Logistic Regression
# I will have the machine predict what constitutes as a legendary pokemon.
# > Steps:
# - EDA to determine useful parameters 
# - Data cleaning
# - Feed into the machine and examine accuracy
# 

# In[ ]:


df.info()


# In[ ]:


leg = df[df['isLegendary'] == True]


# In[ ]:


sns.countplot(x='hasGender', data=leg)


# Most legendary pokes don't have gender. We're gonna throw that parameter into the machine

# In[ ]:


leg_total = leg['Total'].mean()
non_leg_total = df[df['isLegendary'] != True]['Total'].mean()


# In[ ]:


pd.DataFrame([leg_total, non_leg_total], index=['Legendary', 'non-Legendary'], columns=['Average Total'])


# Legendary pokes have a far greater total. It should be noted that this total contains pre-evolutions, who are far inferior to full evolutions

# In[ ]:


plt.figure(figsize=(15,6))
plt.title("Catch rate of Poekemons")
sns.scatterplot(x='Number', y='Catch_Rate', data=df, hue='isLegendary')


# It is shown that legendary pokes generally have a lower catch rate

# In[ ]:


isLegendary = pd.get_dummies(df['isLegendary'], drop_first=True)
hasGender = pd.get_dummies(df['hasGender'], drop_first=True)


# In[ ]:


lr_df = df[['Total', 'Catch_Rate']]


# In[ ]:


lr_df = pd.concat([lr_df, isLegendary, hasGender], axis=1)


# In[ ]:


lr_df.columns = ['Total', 'Catch_Rate', 'isLegendary', 'hasGender']


# In[ ]:


lr_df.head()


# In[ ]:


X = lr_df.drop('isLegendary', axis=1)
y = lr_df['isLegendary']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) 


# In[ ]:


LR = LogisticRegression()
LR.fit(X_train, y_train)


# In[ ]:


pred = LR.predict(X_test)


# In[ ]:


print(classification_report(y_test,pred))


# In[ ]:


print(confusion_matrix(y_test,pred))


# ## Take Away:
# I gave the most important data based on my knowledge of the game so finding useful parameters isn't difficult. Also, given the very segregated dfferences between a legendary and a non-legendary, a machine shouldn't have too much trouble determining the class.
# 
# 
