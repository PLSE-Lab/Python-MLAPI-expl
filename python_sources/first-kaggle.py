#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import time


# In[ ]:


np.random.seed(1988)


# In[ ]:


get_ipython().system(' dir')


# ### Data exploration

# In[ ]:


data = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")
data.head()


# In[ ]:


#there is no NaNs
data[pd.isna(data['ingredients'])]


# In[ ]:


data.shape


# In[ ]:


def extract_ingredients(serie):
    list_ingredients=[]
    for lista in serie:
        for element in lista:
            if element in list_ingredients:
                pass
            elif element not in list_ingredients:
                list_ingredients.append(element)
            else:
                pass
        
    return list_ingredients      


# In[ ]:


ingredients = extract_ingredients(data['ingredients'])


# In[ ]:


len (ingredients)


# In[ ]:


#Types of differents cuisines:
data['cuisine'].unique().shape


# In[ ]:


cuisines = data['cuisine'].unique()
cuisines


# We will try two feature engineering methods, one hot encoding and feature hashing to see what is more effective.

# ## One hot encoding method

# In[ ]:


#Create columns
t = time.time()
for ingredient in ingredients:
    data[ingredient]=np.zeros(len(data["ingredients"]))

print("It took %i seg" %(time.time()-t))


# In[ ]:


def ohe(serie, dtframe):    
    ind=0
    for lista in serie:
        
        for ingredient in lista:
            if ingredient in ingredients:
                dtframe.loc[ind,ingredient]=1
            else:
                pass
        ind +=1


# In[ ]:


t = time.time()
ohe(data['ingredients'], data)
print('it took %i segs' % (time.time()-t))


# #### Train / test split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


predictors = ingredients
response = 'cuisine'


# In[ ]:


X = data[predictors]
Y = data[response]


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)


# In[ ]:


del(data)


# #### Log regression:
# 
# This basic model has given the best accuracy on the previous test

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *


# In[ ]:


log_reg = LogisticRegression(C=1)


# In[ ]:


log_reg.fit(x_train, y_train)


# In[ ]:


y_predicted = log_reg.predict(x_test)
y_predicted


# In[ ]:


accuracy_score(y_test, y_predicted)


# In[ ]:


pd.DataFrame(confusion_matrix(y_test, y_predicted, labels=cuisines), index=cuisines, columns=cuisines)


# ### Preparing the output

# In[ ]:


# Create columns on test
t = time.time()
for ingredient in ingredients:
    test[ingredient]=np.zeros(len(test["ingredients"]))

print('Takes %i seconds' %(time.time()-t))


# In[ ]:


t = time.time()
ohe(test['ingredients'], test)
print('Takes %i seconds' %(time.time()-t))


# In[ ]:


test.head()


# ### Predict

# In[ ]:


y_final_prediction = log_reg.predict(test[predictors])


# In[ ]:


output = test['id']
output = pd.DataFrame(output)
output['cuisine'] = pd.Series(y_final_prediction)


# In[ ]:


output.head()


# In[ ]:


output.to_csv('output.csv', index=False)

