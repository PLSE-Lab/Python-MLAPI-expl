#!/usr/bin/env python
# coding: utf-8

# <font size="20">Discount Prediction</font>

# The objective of this "Discount Prediction" Competition was to build a model to Predict Medical Wholesales Discount to their customers. In this notebook, we will walk through a complete machine learning solution, try out two deep learning models, select a model and , inspect the outputs of the model and draw conclusions. We would like to thank everyone for this hackathon.

# <h1>Importing the Libraries</h1>

# In[ ]:


import pandas as pd #Data Analysis
import numpy as np #Linear Algebra
import seaborn as sns #Data Visualization
import matplotlib.pyplot as plt #Data Visualization


# In[ ]:


import os
print(os.listdir("../input"))


# <h1>Importing the datasets</h1>

# In[ ]:


#This is the Product_sales_train_and_test dataset but without the "[]" in the Customer Basket.
df1=pd.read_csv("../input/remove/data.csv")


# In[ ]:


df2=pd.read_csv("../input/discount-prediction/Train.csv")
df3=pd.read_csv("../input/discount-prediction/test.csv")


# In[ ]:


df1.fillna(float(0.0),inplace=True)
df2.fillna(float(0.0),inplace=True)


# Since to differentiate the Customer Basket is an NLP Problem we will be using CountVectoriser. It converts a collection of text documents to a matrix of token counts. 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv1 = CountVectorizer(max_features=500)
y = cv1.fit_transform(df1["Customer_Basket"]).toarray()


# In[ ]:


thirty= list(y)
thirty1=pd.DataFrame(thirty)


# In[ ]:


final=pd.concat([df1,thirty1],axis=1)


# In[ ]:


df2=df2[df2["BillNo"]!=float(0.0)]


# In[ ]:


finaltrain=pd.merge(final,df2,on="BillNo",how="inner")
finaltest=pd.merge(final,df3,on="BillNo",how="inner")


# In[ ]:


finaltrain.drop(["BillNo","Customer_Basket","Customer","Date"],axis=1,inplace=True)
finaltest.drop(["BillNo","Customer_Basket","Customer","Date"],axis=1,inplace=True)


# In[ ]:


X=finaltrain.drop(["Discount 5%","Discount 12%","Discount 18%","Discount 28%"],axis=1)
y=finaltrain[["Discount 5%","Discount 12%","Discount 18%","Discount 28%"]]


# In[ ]:


X1, y2 = np.array(X), np.array(y)


# In[ ]:


var = np.reshape(X1, (X1.shape[0], X1.shape[1], 1))


# <h1>Modeling</h1>

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# <h1>1. Artificial Neural Networks (ANN)</h1>

# In[ ]:


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', input_dim = 500))
classifier.add(Dropout(0.2))

# Adding the second hidden layer
classifier.add(Dense(units =32 , kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units =16 , kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))

# Adding the output layer
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X, y, batch_size = 10, epochs = 50)


# In[ ]:


annpredictions=classifier.predict(finaltest)


# In[ ]:


discountann=list(annpredictions)


# In[ ]:


abbasann=pd.DataFrame(discountann)


# In[ ]:


abbasann=(abbasann> 0.4)


# <h1>2. LSTM</h1><br>
# First we used ANN but the results were poor and as seen ini our previous kernel we could not see any Customer getting preference for Discounts. Therefore we tried to capture the pattern of discounts been given using an LSTM approach.

# Importing the necessary libraries for an LSTM model.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[ ]:


# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (var.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units=4, activation='softmax'))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'categorical_crossentropy')

# Fitting the RNN to the Training set
regressor.fit(var, y2, epochs = 1, batch_size = 32)


# We have purposely set the epoch time to 1 as it takes a long time for the kernel to commit.

# In[ ]:


finaltest1=np.array(finaltest)
baas=np.reshape(finaltest1, (finaltest1.shape[0], finaltest1.shape[1], 1))


# In[ ]:


discountclass=regressor.predict(baas)


# In[ ]:


discountbaas=list(discountclass)


# In[ ]:


abbas=pd.DataFrame(discountbaas)


# In[ ]:


abbas= (abbas > 0.3)


# <h1>Result</h1>

# At the end we were able to discern that an <b>LSTM</b> gave us the best result. One major change that we noticed was that the Bill Numbers were change from ZA's to A's from July 1st, 2017 and that in this transition the majority of Discounts also drastically changed from 28% to 12%.
