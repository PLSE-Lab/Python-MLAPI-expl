#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as clf

get_ipython().run_line_magic('matplotlib', 'inline')
clf.go_offline()
#import os
#print(os.listdir("../input"))


# In[ ]:


data = pd.read_json('../input/train.json')
print(data.shape)
data.iloc[:1,:]


# In[ ]:


test = pd.read_json('../input/test.json')
print(test.shape)
test.iloc[:1,:]


# In[ ]:


#check is there any null values ??
data[data['ingredients'].isnull()]


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


print(len(ingredients))


# ## How many type of CUISINE Present ????

# In[ ]:


cuisines = data['cuisine'].unique()
print(cuisines.shape)


# ## One Hot Encoding

# In[ ]:


for ingredient in ingredients:
    data[ingredient]=np.zeros(len(data["ingredients"]))


# In[ ]:


for ingredient in ingredients:
    test[ingredient]=np.zeros(len(test["ingredients"]))


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


ohe(data['ingredients'], data)


# In[ ]:


ohe(test['ingredients'], test)


# ## Logistic Regression

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *


# In[ ]:


predict = ingredients
feature = data['cuisine']


# In[ ]:


X = data[predict]
y = feature


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[ ]:


log_reg = LogisticRegression(C=1)


# In[ ]:


log_reg.fit(X_train, y_train)


# In[ ]:


y_predicted = log_reg.predict(X_test)
y_predicted


# ## Accuracy of Model

# In[ ]:


accuracy_score(y_test, y_predicted)


# In[ ]:


pd.DataFrame(confusion_matrix(y_test, y_predicted, labels=cuisines), index=cuisines, columns=cuisines)


# In[ ]:


y_final_prediction = log_reg.predict(test[predict])


# In[ ]:


output = test['id']
output = pd.DataFrame(output)
output['cuisine'] = pd.Series(y_final_prediction)


# ## Submission File

# In[ ]:


output.to_csv('sample_submission.csv', index=False)


# In[ ]:




