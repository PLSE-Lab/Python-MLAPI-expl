#!/usr/bin/env python
# coding: utf-8

# **Gesture Recognition - Random Forest Classifier Model **
# 
# Here I have used a simple version of the Random Forest Classifier Model. First I trained the model and then randomly selected data from the original dataset and did the prediction. 

# **Importing Libraries**

# In[ ]:


#Importing Libraries
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **Loading Data**

# In[ ]:


data = pd.read_csv('../input/gesture-recognition/ASL_DATA.csv', header='infer')  


# **Data Exploration & Dataset Split**

# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


#Creating a seperate dataset for Letter A, B & C
#data_ABC = data[data['Letter'].isin(['A', 'B','C']) ]


# In[ ]:


#Dropping the ID column
#data_ABC = data_ABC.drop(columns='Id',axis=1)
data = data.drop(columns='Id',axis=1)


# In[ ]:


#data_ABC.head()
data.head()


# **Feature Selection & dataset Split**

# In[ ]:


#Defining the feature & target
feature_col = ['Thumb_Pitch', 'Thumb_Roll', 'Index_Pitch', 'Index_Roll', 'Middle_Pitch', 'Middle_Roll', 'Ring_Pitch', 'Ring_Roll', 'Pinky_Pitch', 'Pinky_Roll', 'Wrist_Pitch', 'Wrist_Roll']
target_col =  ['Letter']

#Applying to the dataset
X = data[feature_col]
y = data[target_col]


# In[ ]:


#spliting the dataset [training & test]
size = 0.1   #10% of the dataset will be used for validation

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=size, random_state=0)


# **Model Generation, Training & Prediction**

# In[ ]:


rfc = RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=None, bootstrap=True)


# In[ ]:


#training the model with training data
rfc.fit(X_train, y_train)


# In[ ]:


#making prediction on test data
y_pred = rfc.predict(X_test)  

#Finding the accuracy & precision of the model
print("Model Accuracy : ",'{:.2%}'.format(metrics.accuracy_score(y_test, y_pred)))
print("Model Precision :", '{:.2%}'.format(metrics.precision_score(y_test,y_pred, average='weighted')))


# **I'm also surprised as you are to see the accuracy & precision**** 
# 
# !["Surprised"](https://image.shutterstock.com/image-photo/surprised-male-hipster-keeps-mouth-260nw-1109619839.jpg)

# **Prediction on Randomly Selected Data**

# In[ ]:


#Randomly selecting 10 records from the main dataset
test_data = data.sample(n=10)


# In[ ]:


test_data.head()


# In[ ]:


#prediction on the randomly selected data
testdata_pred = rfc.predict(test_data.iloc[:,0:12])


# In[ ]:


# Merging the prediction with the test-data
test_data['Prediction'] = testdata_pred


# In[ ]:


test_data.head(10)


# Well, it certainly did a 100% accurate prediction on the randomly selected data. Open to hearing about any suggestion about the model/accuracy :)
