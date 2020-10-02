#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


train=pd.read_csv("../input/thidsc/Train.csv")
test=pd.read_csv("../input/thidsc/Test.csv")
sample=pd.read_csv("../input/thidscsub/submit.csv",index_col=False)


# ### Understanding the Data

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sample


# In[ ]:


test.isnull().sum()


# In[ ]:


train['Vendor_Code'].unique().size


# In[ ]:


test['Vendor_Code'].unique().size


# In[ ]:


train['GL_Code'].unique()


# ### Separating

# In[ ]:


y_train=train["Product_Category"]
x_train=train.iloc[:,:-1]


# In[ ]:


train['Product_Category'].shape


# In[ ]:


y_train.shape


# In[ ]:


x_train.shape


# In[ ]:


x_train.drop("Inv_Id",axis=1,inplace=True)


# In[ ]:


test.head()


# In[ ]:


test_ID=test['Inv_Id']


# In[ ]:


test.drop('Inv_Id',axis=1,inplace=True)


# # So our data now is :
# 

# ## Modelling
# 

# ### Required Imports

# In[ ]:


from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


#vectorizer = TfidfVectorizer(ngram_range=(1, 3), tokenizer=tokenizer.tokenize)
#full_text = list(x_train['Item_Description'].values) + list(test['Item_Description'].values)
#vectorizer.fit(full_text)
#train_vectorized = vectorizer.transform(x_train['Item_Description'])
#test_vectorized = vectorizer.transform(test['Item_Description'])


# ### So our Product Description is vectorized now

# In[ ]:


#Vendor_Code=pd.get_dummies(x_train['Vendor_Code'],drop_first=True)
#GL_Code=pd.get_dummies(x_train['GL_Code'],drop_first=True)
#tVendor_Code=pd.get_dummies(x_train['Vendor_Code'],drop_first=True)
#tVendor_Code1=pd.get_dummies(test['Vendor_Code'],drop_first=True)
#tGL_Code=pd.get_dummies(test['GL_Code'],drop_first=True)


# In[ ]:


train_objs_num = len(x_train)
dataset = pd.concat(objs=[x_train, test], axis=0)
dataset = pd.get_dummies(dataset)
x_train = dataset[:train_objs_num]
test = dataset[train_objs_num:]


# In[ ]:


#x_train=pd.concat([x_train,Vendor_Code,GL_Code],axis=1)
#test=pd.concat([test,tVendor_Code1,tGL_Code],axis=1)


# In[ ]:


x_train.shape


# In[ ]:


test.shape


# In[ ]:


x_train.head()


# In[ ]:


x_train = x_train.astype(int)
test= test.astype(int)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, Y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=2)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test,predictions)


# In[ ]:


x_train.shape


# In[ ]:


test.shape


# In[ ]:


pred=logmodel.predict(test)


# In[ ]:


pred


# In[ ]:


pred.shape


# In[ ]:


sample.head()


# In[ ]:


submit = pd.DataFrame(columns=['Inv_Id','Product_Category'])


# In[ ]:


submit['Inv_Id'] = test_ID


# In[ ]:


submit['Product_Category'] = pred


# In[ ]:


submit


# In[ ]:


submit.to_csv("submission.csv",index=False,header=True)


# In[ ]:




