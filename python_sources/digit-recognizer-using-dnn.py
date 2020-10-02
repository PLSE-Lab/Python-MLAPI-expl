#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# # Data Acquisition

# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# In[ ]:


X=train.iloc[:,1:].values


# In[ ]:


y=train.iloc[:,0].values


# # Preparing a Deep Learning Model

# In[ ]:


from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.neural_network import MLPClassifier


# In[ ]:


X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.33,random_state=62)


# In[ ]:


#Here we have used 512 neurons for our hidden layer.
mlp=MLPClassifier([512],activation='relu',solver='adam',learning_rate_init=0.001,)


# # Training the model
# 

# In[ ]:


mlp.fit(X_train,y_train)


# # Evaluation

# In[ ]:


mlp.score(X_train,y_train)


# In[ ]:


mlp.score(X_val,y_val)


# # Training the model on whole dataset

# In[ ]:


mlp.fit(X,y)


# # Submission for Competition

# In[ ]:


pred=mlp.predict(test.values).reshape(-1,1)


# In[ ]:


output=np.concatenate((np.arange(1,test.shape[0]+1).reshape(-1,1),pred),axis=1)


# In[ ]:


submission=pd.DataFrame(output,columns=['ImageId','Label'])


# In[ ]:


submission.to_csv('Submission.csv',index=False)    #Generating submission file.


# In[ ]:




