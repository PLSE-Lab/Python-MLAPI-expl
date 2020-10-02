#!/usr/bin/env python
# coding: utf-8

# Lets look a data we have.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
data=pd.read_csv('../input/creditcard.csv')
data.head()


# This data is well organised and We had clean data so we dont need cleaning of the data.
# 
# Lets find out correlation between our  features and target.

# In[ ]:


corr=np.array(data.corr())
corr=np.around(corr[-1],decimals=2)
corr=pd.DataFrame(corr,index=data.columns)
corr


# As you can see all features are correlated to target with some correlation value. Now We will pick those features only which has more correlation value than 0.10.
# 
# Lets select features those features and see what will be outcome.

# Lets split the dataset in train and test set so I can evaluate model accuracy.

# In[ ]:


x=data[['V4','V3','V10','V12','V14','V16','V17']]
y=data[['Class']]
from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(x,y,test_size =0.33)


# Here we have to predict class, Whether will be fraud(1) or not(0). So It is binary classification.
#  
#  The best algorithm for this is naive bayes with bernoulli theorem.
#  
#  So Lets see outcome of this allgorithm.
#  
#  Lets train the  dataset.

# In[ ]:


from sklearn.naive_bayes import BernoulliNB
model=BernoulliNB()
model.fit(xtr,ytr)


# Lets predict the outcome and evalute the result.

# In[ ]:


predict=model.predict(xte)
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(yte,predict)


# Accuracy score is nearly 0.998 and which is pretty good. 
# 
# For banking it is necessary that it has no true negatives.
# 
# Lets plot the correlation matrix and check  How values are predicted.

# In[ ]:


confusion_matrix(yte,predict)


# Values are well predicted but We have to minimise the error.
# 
# Let me know If you have any suggetion or comment.
# 
# Thanks for watching and upvote if you like.
