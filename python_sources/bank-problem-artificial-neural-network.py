#!/usr/bin/env python
# coding: utf-8

# ## Problem description:
# **The bank manager wants an accurate analysis of the clients who will be leaving the bank, based on the data of all customers over the past three months. All he wants is a simple analysis of the people who may be leaving so that he can burn the marketing team in their direction until they change their mind in the bank.**
# You as a data analyst will be the reason for the success or failure of the bank and therefore you have to work hard to be able to create the model able to classify them
# If the accuracy of the model of 70-80%, this is the achievement of no one may reach 100% only in very rare events
# > So imagine yourself saying to your boss: Sir these customers may leave the bank at any time by 80%

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
from keras.models import Sequential
from keras.layers import Dense


# ## We will work on two types of strategies in this kernel:
# * Artificial Neural Network (ANN)
# 
# * We will also use the KERAS library to implement  ANN . We will clean and separate the data, compare results  We will address the problem and eventually report to the Director General for decision. We will save the bank and be the champions of the month, It's fun to face a real problem, is not it?**

# We have a statement of accounts for all customers in three different countries and some columns to describe the status of customers such as, is this client active or not, average customer income, client balance, does the customer have a credit card, Any moment or not, and after the columns containing personal data such as account number, customer name and customer gender, we will use this data to predict whether this customer will leave the bank or stay
# 
# **Also note that we will not need three initial columns, it's personal data will not be useful in the analysis, and note that all other columns are independent except the last column where it will be the target**

# In[ ]:


# Read Data From and Reprocessing it
dataset = pd.read_csv('../input/Churn_Modelling.csv')
x=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values


# In[ ]:


dataset.head()


# Now we will work on converting the Categorical data into Numbers so that it is easy to deal with them. We can't make equations for these data and there are non-numerical inputs. This is impossible. We will convert all Categorical data into numeric data  1,2, 3,4 and so on
# We will process this on two columns of the total column, the first column will be the gender, the second column will be the country, we note that there are 3 countries, Spain, France, Germany ..

# In[ ]:


# Label Encoder Categorical Data [1:2]
Label_x1=LabelEncoder()
Label_x2=LabelEncoder()

x[:,1]=Label_x1.fit_transform(x[:,1])
x[:,2]=Label_x2.fit_transform(x[:,2])


# We will now work on converting each numeric value converted into a column and we will scan the dummy variables that may affect the output. To explain the dummy variable, we will simply do so. You have three countries. If the country is France, it is impossible to be Germany or Spain,If the country is France, it is impossible to be Germany or Spain, if not France or Germany is expected to be ... ? Of course, Spain. This is what we will do. 
# you have to do with the Bank's branches in different countries. Therefore, we expect a variable result in changing the country, so we can easily delete the third column or the first column because it will not affect the accounts permanently.

# In[ ]:


#Let's Switch This Label to Coulmn
OneHotEncoder=OneHotEncoder(categorical_features=[1])
x=OneHotEncoder.fit_transform(x).toarray()

#removing Dummy Variable
x=x[:,1:]


# **Unfortunately, it looks like I'm a bit old, it does not matter how you write the code or keep you informed about what's new, all that matters is the effectiveness of your code in the real world. Will your code add value or not? **

# In this process we will work on dividing the data into four sections. The first section is to train the model. The second section is to test the model and its effectiveness on the second part of the data. In fact, you need hundreds of thousands of rows to produce a strong model.

# In[ ]:


#Splitting Data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# We see that we have many values, such as the age of 1-100 and the salary ranges from 1-10000 and cridetScore  ranges from 1-1000, we note that we have many weights for each Feature , in this process we will work to unify those weights to range from -1 to 1 so that we can manufacture an equation to predict

# In[ ]:


#StandardScaler Variable
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# I will explain the neurons in a very simplified way and I hope you like it, you can return to [Wikipedia](https://en.wikipedia.org/wiki/Artificial_neural_network) if you want more details about it Let's start
# The human brain contains billions of neurons, which work together to think, test and analyze, each with millions of functions. We will add one neuron in these data and we will see test results
# 
# Simply enter the data from the input layer, connect to the hidden layer, and then work on output results in the output layer, for example I want to buy a house should care about the size of the house and the location of the house, the number of rooms, utilities, to predict the price of the right home for me **( output Layer )** , All I have to do is find out what I want **(Hidden Layer)** . For example, I want an average house, a house in a quiet area, a house with at least three rooms,I will now make a list of available houses on the ads page **(Input Layer)** to choose the average value for the house I want

# ![](https://upload.wikimedia.org/wikipedia/commons/4/43/Neural_Network.gif)

# **We will add two hidden layers inside our first cell consists of 11 input will work together to make 6 outputs and then the 6 output  will work together to output one result and this result will represent the percentage of survival or exit of the client from the bank**

# You can see some of the terms that we use in this model to predict via Wikipedia [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation),[Sigmoid_function](https://en.wikipedia.org/wiki/Sigmoid_function) ,[Gradient-Descent](https://www.bogotobogo.com/python/scikit-learn/Artificial-Neural-Network-ANN-3-Gradient-Descent.php) , but simply all I will mention are the strategies we will deal with within ANN , [sequential Model Keras ](https://keras.io/getting-started/sequential-model-guide/)

# In[ ]:


#Let's Create our CNN
model = Sequential()
# Adding the first hidden layer
model.add(Dense(input_dim=11,output_dim=6,init='uniform',activation='relu'))
# Adding the second hidden layer
model.add(Dense(output_dim=6,init='uniform',activation='relu'))
# Adding the output hidden layer
model.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

# Compiling The Model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, batch_size = 10, nb_epoch = 10)


# The model has been successfully trained and we are currently working on predicting some values. You can check the values of y_pred that will be between 0 and 1 and this is illogical. We want to put an end to this and we will put a condition if the value is greater than 0.5 the client wil left the bank else will stay with us , after this point we will 

# In[ ]:


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.50)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test,y_pred)*100,'%')


# In the final step, we will calculate the percentage of confirmation of our predict , which reached 81 - 84%. This is not a small number. You can inform your manager of the percentage of Confidence  of  who will leave us at any moment and he will give the marketing team special attention. The last step you will take is to extract personal data and account numbers for people who may leave the bank at any moment. All you have to do is return the separated
# 
# I hope that this simple explanation will help you solve a problem in detail in a simplified way. Please vote and thank you

# In[ ]:




