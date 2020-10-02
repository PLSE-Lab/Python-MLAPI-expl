#!/usr/bin/env python
# coding: utf-8

# # Load Libraries

# In[ ]:


import pandas as pd
from sklearn.svm import SVC


# # Load Dataset 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')


# # Check Data for any missing values

# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


#Get Target data 
y = data['default.payment.next.month']

#Load X Variables into a Pandas Dataframe with columns 
X = data.drop(['ID','default.payment.next.month'], axis = 1)


# # Check X Variables

# In[ ]:


X.head()


# In[ ]:


#Check size of data
X.shape


# # Build SVM Model

# In[ ]:


SVM_Model = SVC(gamma='auto')


# In[ ]:


SVM_Model.fit(X,y)


# # Check Accuracy

# In[ ]:


print (f'Accuracy - : {SVM_Model.score(X,y):.3f}')


# # END
