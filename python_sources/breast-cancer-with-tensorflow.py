#!/usr/bin/env python
# coding: utf-8

# # BREAST CANCER PREDICTIONS WITH TENSORFLOW

# In this notebook, we will go use wisconsin (diagnostic) dataset to predict the breast cancer. We will be predicting wether the people has cancer or not(benign or malign), or technically classifying given data into these two classes. 
# 
# We will have the following parts:
# 
# * Importing relevant libraries and data
# * Performing Exploratory Data Analysis (EDA)
# * Creating the Model 
# * Performing Model Evaluation
# 
# 
# Let's get started! 

# # Importing Relevant libraries and data

# In[ ]:


import pandas as pd #Manipulating data
import numpy as np #For Math operations 
import seaborn as sns #For visualization purpose
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split # For Spliting data to training and testing data
from sklearn.preprocessing import MinMaxScaler # MinMaxScaler For fitting the data to the model, it optimize model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from sklearn.metrics import classification_report,confusion_matrix #For model evaluation metrics

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        


# In[ ]:


#Importing the data
cancer_data=pd.read_csv('/kaggle/input/breast-cancer-dataset-uci-ml/cancer_classification.csv')


# A quick look out to the structure of the data that we are going to work with by the following methods. As you may see, our main target feature is benign_0__mal_1.

# In[ ]:


cancer_data.head(5)


# In[ ]:


cancer_data.info()


# In[ ]:


cancer_data.describe().transpose()


# # Performing Exploratory Data Analysis

# Let's plot the feature under prediction to count its classes, 0 or 1. 

# In[ ]:


sns.countplot(x='benign_0__mal_1',data=cancer_data)


# In[ ]:


cancer_data['benign_0__mal_1'].value_counts()


# The above plot reveals that 357 values are for malignant and 212 for benign.

# In[ ]:


#Let's see how features correlates to another
sns.heatmap(cancer_data.corr())


# In[ ]:


cancer_data.corr()['benign_0__mal_1'].sort_values()


# In[ ]:


cancer_data.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')


# From the correlation above, we see that most features are in negative to the begign_0__mal_1. Let's cutt off features 

# In[ ]:


cancer_data.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')


# It's now time to split the data into training and testing by Train test Split

# In[ ]:


X = cancer_data.drop('benign_0__mal_1',axis=1).values
y = cancer_data['benign_0__mal_1'].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)


# As we splited the data, let's now scale them to optimize model training

# In[ ]:


scaler=MinMaxScaler()


# In[ ]:


scaler.fit(X_train)


# In[ ]:


X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


# Note: We do not fit testing data. This would result in error as we can not fit the data that the model has not seen yet. We just only transform all for consistency purpose, but only fitting to the training data only. 

# # Creating the Model

# Few Notes on the model:
# 
# * For best practice, consider having input layers units that are nearly equal to columns or shape of training data. In our case it is 30.
# * We will use early stopping to avoid overfitting on the model and Dropout to optimize our performance. Dropout is for sleeping half of neurons after each layer. Half if for our case, you can choose any value you want, best between 0.4 to 0.8.
# * For classification problems, as always, we often use sigmoid function in last layer.

# In[ ]:


model=Sequential()

model.add(Dense(units=30, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=15, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')


# In[ ]:


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)


# In[ ]:


model.fit(X_train,
          y=y_train,
          epochs=600,
          validation_data=(X_test,y_test),
          verbose=1,
          callbacks=[early_stop]
         
         )


# Thanks to Early Stopping, model stopped training on epochs of 128.

# In[ ]:


model.summary()


# In[ ]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# As seen on plot above, our model worked well on training data. Let's now evaluate the model on testing data

# # Performing Model Evaluation

# In[ ]:


predictions = model.predict_classes(X_test)


# In[ ]:


print('Model Classification Report')
print(classification_report(y_test,predictions))

print('*'*57)
print('Confusion Matrix')
print(confusion_matrix(y_test,predictions))


# The model achieved accuracy of 0.98 percent. From Confusion matrix, the model predicted True Positives:54, True Negative:86, False Positive: 1, False Negative:2

# *Thanks for going with me to the end of this notebook!!*

# In[ ]:




