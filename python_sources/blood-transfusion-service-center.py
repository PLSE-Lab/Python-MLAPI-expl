#!/usr/bin/env python
# coding: utf-8

# # Blood transfusion service center

# ## This dataset was retrieved from: https://www.openml.org/d/1464
# ### Author: Prof. I-Cheng Yeh
# Source: UCI <br>
# To cite: Yeh, I-Cheng, Yang, King-Jang, and Ting, Tao-Ming, "Knowledge discovery on RFM model using Bernoulli sequence", Expert Systems with Applications, 2008. <br>
# 
# Data taken from the Blood Transfusion Service Center in Hsin-Chu City in Taiwan -- this is a classification problem. <br>
# 
# To demonstrate the RFMTC marketing model (a modified version of RFM), this study adopted the donor database of Blood  Transfusion Service Center in Hsin-Chu City in Taiwan. The center passes their blood transfusion service bus to one university in Hsin-Chu City to gather blood donated about every three months. To build an FRMTC model, we selected 748 donors at random from the donor database.

# ### Features
# #### V1: Recency - months since last donation <br>
# #### V2: Frequency - total number of donation <br>
# #### V3: Monetary - total blood donated in c.c. <br>
# #### V4: Time - months since first donation), and a binary variable representing whether he/she donated blood in March 2007 (1 stand for donating blood; 0 stands for not donating blood).

# ### Label
# #### The target attribute is a binary variable representing whether he/she donated blood in March 2007 (2 stands for donating blood; 1 stands for not donating blood).

# In[ ]:


# https://www.openml.org/d/1464
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/blood-transfusion-service-center.csv")
df.head()


# In[ ]:


df = df.dropna(how='all')


# In[ ]:


df["Class"].value_counts()


# In[ ]:


from sklearn.utils import resample

df_majority = df[df.Class==2]
df_minority = df[df.Class==1]

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=178,    # to match majority class
                                 random_state=42) # reproducible results

# Combine majority class with upsampled minority class
df = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
df.Class.value_counts()


# In[ ]:


df["Class"].value_counts()


# In[ ]:


X = df.drop(['Class'], axis=1).values
#X = StandardScaler().fit_transform(X)
Y = df['Class']


# In[ ]:


X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.30, random_state = 101)


# In[ ]:


trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Train,Y_Train)
predictionforest = trainedforest.predict(X_Test)
trainedforest.score(X_Train, Y_Train)


# In[ ]:


trainedforest = RandomForestClassifier(n_estimators=700).fit(X,Y)


# In[ ]:


# Saving model to disk
pickle.dump(trainedforest, open('model.pkl','wb'))


# In[ ]:


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


# In[ ]:


print(model.predict([[2,  430, 10350,  86]]))


# In[ ]:


p = model.predict(X_Test)
#print(X_Test)
print(list(p).count(1))
print(list(p).count(2))

