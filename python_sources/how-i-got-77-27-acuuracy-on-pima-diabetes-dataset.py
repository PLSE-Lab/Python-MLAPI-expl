#!/usr/bin/env python
# coding: utf-8

# # My First Kaggle InClass Competition

# **Importing The Libraries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras import regularizers


# **Loading The Data**

# In[ ]:


df = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')
df.head()


# **Visualizing The Important Features**

# In[ ]:


df.glucose_concentration.hist()
plt.show()


# In[ ]:


df.serum_insulin.hist()
plt.show()


# **Since there were a lot features with high magnitude and it was causing high variance/Overfitting on the training data, I decided to Scale the features in the range of 0 to 1 using Min-Max_scaler.**

# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()
#Scaling The Training Data
x = df.values
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
#Scaling The Testing Data
y = test.values
y_scaled = min_max_scaler.fit_transform(y)
test = pd.DataFrame(y_scaled)
df.head()


# **Visualizing The Important Features After Scaling**

# In[ ]:


df[2].hist() #Glucose Concentration
plt.show()


# In[ ]:


df[5].hist() #Serum Insulin
plt.show()


# **Spliting the training data into Traning and Developement Data**

# In[ ]:


train, dev = train_test_split(df, test_size=0.2)


# **Setting Up the Hyperparameters**

# In[ ]:


hidden_units=300
learning_rate=0.005 #Learning rate was quite optimal
hidden_layer_act='tanh'
output_layer_act='sigmoid'
no_epochs=100 #Increasing The epochs would overfit
bsize = 128 #Batch Size Of 128 


# **Model Architechture**

# In[ ]:


model = Sequential()

model.add(Dense(hidden_units, input_dim=8, activation=hidden_layer_act))
model.add(Dense(hidden_units, activation=hidden_layer_act))
model.add(Dense(1, activation=output_layer_act))


# **Setting Up loss function, Optimizer, Metrics**

# In[ ]:


adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy',optimizer=adam, metrics=['acc'])


# **Training the Model**

# In[ ]:


train_x=train.iloc[:,1:9]
train_y=train.iloc[:,9]

model.fit(train_x, train_y, epochs=no_epochs, batch_size= bsize,  verbose=2)


# **Validation Loss and Accuracy**

# In[ ]:


val_loss, val_acc = model.evaluate(dev.iloc[:,1:9], dev.iloc[:,9])
print("Validation Loss : ", val_loss)
print("Validation Acc : ",val_acc)


# **Low Bias and Low Variance ** : 
# Compared to the highest accuracy this model performed well on training data(Low Bias) and gave a equivalent accuracy on the validation set (Low Variance)

# **Predicting The outputs for the Training Data**

# In[ ]:


test_x=test.iloc[:,1:9]
predictions = model.predict(test_x)
print(predictions)


# **Submission File**

# In[ ]:


rounded = [int(round(x[0])) for x in predictions]
print(rounded)
sample.diabetes = rounded
sample.to_csv('submission.csv',index = False)

