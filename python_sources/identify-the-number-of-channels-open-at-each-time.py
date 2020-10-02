#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


samplesubmission = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')
train = pd.read_csv("../input/liverpool-ion-switching/train.csv")
test = pd.read_csv("../input/liverpool-ion-switching/test.csv")


# In[ ]:


train.info()
train[0:10]


# In[ ]:


test.info()
test[0:10]


# In[ ]:


print("Any missing sample in training set:",train.isnull().values.any())
print("Any missing sample in test set:",test.isnull().values.any(), "\n")


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
def FunLabelEncoder(df):
    for c in df.columns:
        if df.dtypes[c] == object:
            le.fit(df[c].astype(str))
            df[c] = le.transform(df[c].astype(str))
    return df


# In[ ]:


test = FunLabelEncoder(test)
test.info()
test.iloc[235:300,:]


# In[ ]:


train = FunLabelEncoder(train)
train.info()
train.iloc[235:300,:]


# In[ ]:


#Frequency distribution of classes"
train_outcome = pd.crosstab(index=train["open_channels"],  # Make a crosstab
                              columns="count")      # Name the count column

train_outcome


# In[ ]:


#Select feature column names and target variable we are going to use for training
features=['signal']
          
target = "open_channels"


# In[ ]:


#This is input which our classifier will use as an input.
train[features].head(10)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

# We define the model
dtcla = DecisionTreeClassifier(random_state=None)


# We train model
dtcla.fit(train[features],train[target])


# In[ ]:


#Make predictions using the features from the test data set
predictions =dtcla.predict(test[features])

#Display our predictions
predictions


# In[ ]:


#Create a  DataFrame
submission = pd.DataFrame({'time':samplesubmission['time'],'open_channels':predictions})

#Visualize the first 5 rows
submission.head()


# In[ ]:


#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'submission.csv'

submission.to_csv(filename,index=False, float_format='%.4f')

print('Saved file: ' + filename)

