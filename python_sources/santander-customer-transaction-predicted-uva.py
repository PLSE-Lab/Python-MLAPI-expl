#!/usr/bin/env python
# coding: utf-8

# **SUMMARY**   
# 
#                     **1. IMPORTED LIBRARIES AND READ THE CSV FILES  **
#                     **2. SPLITTED THE TRAIN DATASET ,BUILT THE MODEL AND   GOT THE ACCURACY OF 90%**
#                     **3. IDENTIFIED THE WRONG PREDICTED ROWS**
#                     **4. BUILT THE MODEL USING TRAIN DATASET AND PREDICTED FOR THE TEST DATASET**

# 1.  **IMPORT THE NECESSARY LIBRARIES**

# In[ ]:



import numpy as np 
import pandas as pd


import os
print(os.listdir("../input"))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score 


# **READ THE CSV FILES**

# In[ ]:


train=pd.read_csv("../input/train.csv")


# In[ ]:


test=pd.read_csv("../input/test.csv")


# **GETTING THE COLUMNS IN X USING iloc**

# In[ ]:


X=train.iloc[ : ,2:202].values


# **GETTING THE TARGET COLUMN IN Y **

# In[ ]:


y=train.iloc[ :,1:2].values
y=y.astype(float)


# **VIEW THE HEAD OF TRAIN DATASET**

# In[ ]:


train.head()


# **CHECK THE CORRELATION BETWEEN THE TARGET AND OTHER COLUMNS**

# In[ ]:


train[train.columns[1:]].corr()['target'][:]


# **TRAIN_TEST_SPLIT THE DATA FOR CHECKING THE ACCURACY  **

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# **BUILDING THE MODEL LINEAR REGRESSION AND FIT X_TRAIN & Y_TRAIN**

# In[ ]:


reg= LinearRegression()


# In[ ]:


reg.fit(X_train,y_train)


# **PREDICTING TARGET VALUES FOR THE ROWS IN X_TEST & ROUND THE DECIMAL VALUES**

# In[ ]:


y_pred=reg.predict(X_test)


# In[ ]:


y_pred=y_pred.round()


# **CHECKING THE ACCUCARCY SCORE FOR OUR PREDICTIONS **

# In[ ]:


accuracy_score(y_pred,y_test)


# **TAKING THE ID AND OTHER VALUES IN y1 TO KNOW WHERE IT HAS PREDICTED WRONG **

# In[ ]:


y1=train.iloc[ :,0:202]


# **SPLITTING WITH THE SAME TEST SIZE**

# In[ ]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size = 0.2, random_state = 0)


# In[ ]:


y_test1=y_test1.reset_index()


# **CONVERTING THE y_pred TO DATAFRAME**

# In[ ]:


y_pred=pd.DataFrame(y_pred)


# **CONCAT THE y_pred AND y_test1 TO FIND WHERE IT PREDICTED ACCURATE AND WHERE IT DONE WRONG PREDICTION**

# In[ ]:


res=pd.concat([y_pred,y_test1],axis=1)


# **RES HAS THE PREDICTIONS,TARGET,id AND OTHER VARIABLES**

# In[ ]:


res=res.rename(columns={0:"predicted"})


# **FINDING THE DIFFERENCE BETWEEN PREDICTED AND THE ACTUAL TARGET VALUE FOR GETTING THE COUNT OF WRONG PREDICTIONS**

# In[ ]:


res['diff']=res['predicted']-res['target']


# **CONVERT THE NEGATIVE NUMBER TO  POSITIVE BY abs()**

# In[ ]:


res["diff"]=res["diff"].abs()


# **wr_pred HAS THE ROWS WHERE OUR MODEL PREDICTED WRONG ,IF LOGIC OR INDIVIDUALITY BEHIND THIS IS FOUND WE MAY MAKE CHANGES IN TEST AFTER THE RESULT , i.e., UNDER RESEARCH**

# In[ ]:


wr_pred=res[res["diff"]==1]


# In[ ]:


wr_pred.head()


# **FINALLY ,PREDICT TO THE TEST DATASET**

# **GET THE VALUES FOR X_TRAIN & Y_TRAIN FROM THE TRAIN DATASET**

# In[ ]:


X_TRAIN=train.iloc[ : ,2:202].values


# In[ ]:


Y_TRAIN=train.iloc[ :,1:2].values
Y_TRAIN=Y_TRAIN.astype(float)


# **GET THE VALUES FOR X_TEST FROM TEST DATASET**

# In[ ]:


X_TEST=test.iloc[ :,1:201].values


# **FIT THE X&Y TRAIN IN THE MODEL**

# In[ ]:


reg.fit(X_TRAIN,Y_TRAIN)


# **PREDICT THE TARGET VALUES FOR THE X_TEST i.e., TEST DATASET**

# In[ ]:


TARGET=reg.predict(X_TEST)


# **ROUND THE DECIMAL VALUES**

# In[ ]:


TARGET=TARGET.round()


# **CONVERT IT TO DATAFRAME**

# In[ ]:


TARGET=pd.DataFrame(TARGET)


# In[ ]:


TARGET.shape


# **GET THE TEST ID FROM THE TEST DATASET**

# In[ ]:


ID_CODE=test.iloc[ :,0:1]


# **CONCAT THE TARGET AND ID_CODE    FINALLY GOT THE RESULTS....**

# In[ ]:


RESULT=pd.concat([ID_CODE,TARGET],axis=1)


# **RENAME 0 AS target **

# In[ ]:


RESULT=RESULT.rename(columns={0:'target'})


# **CONVERT NEGATIVE VALUES TO POSITIVE VALUES**

# In[ ]:


RESULT["target"]=RESULT["target"].abs()


# In[ ]:


RESULT.head()


# In[ ]:


RESULT.to_csv("test_target_pred.csv")


# In[ ]:




