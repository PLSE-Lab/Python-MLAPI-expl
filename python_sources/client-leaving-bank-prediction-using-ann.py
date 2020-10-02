#!/usr/bin/env python
# coding: utf-8

# The following data set contain the data  of people leaving the bank which contain 10,000 rows and 14 columns.
# The following information is present in columns of dataset.
# 
# Column 1: RowNumber
# Column 2: CustomerID
# Column 3: Surname
# Column 4: CreditScore
# Column 5: Geography(Country)
# Column 6: Gender
# Column 7: Age
# Column 8: Tenure
# Column 9: Balance
# Column 10: NumofProducts
# Column 11: HasCrCard
# Column 12:IsActiveMember
# Column 13: EstimatedSalary
# Column 14: Exited
# 

# In[ ]:


import numpy as np
import pandas as pd
dataset = pd.read_csv("../input/Churn_Modelling.csv")#importing the dataset
X = dataset.iloc[:, 3:13].values #taking independent variables i.e.columns. We have neglected C0, C1 and C3 as model won't depend on these factors
y = dataset.iloc[:, 13].values #taking the dependent variable i.e "Exited" column
X


# Now we are converting our text values to the numerical value as algorithm requires numerical values by using LabelEncoder. Label Encoder assigns
# **France: 0, Spain: 2, Germany: 1** and assigns **Female: 0 and Male:1**

# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
l_e_X1 = LabelEncoder()
X[:, 1] = l_e_X1.fit_transform(X[:,1])# For "Geography" feature
l_e_X2 = LabelEncoder()
X[:,2] = l_e_X2.fit_transform(X[:,2]) #For "Gender" feature
X


# Now we have replaced text data with numerical value but, since variables in "Geography" and "Gender" are nominal variable so we don't want our algorithm to assign special preference to numerical value with higher magnitude.  We use OneHotEncoder to overcome thisby creating dummy variables.
# 

# In[ ]:


ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()
X = X[:, 1:]#taken only two rows of "Geography" feature to avoid dummy variable trap.
X


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #Dividing the dataset into training and testing set. 


# **Feature Scaling**

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train
X_test


# NOW OUR ARTIFICIAL NEURAL NETWORK

# In[ ]:


from keras.models import Sequential #Used for initializing our artificial neural network classifier .
from keras.layers import Dense #Used to create layers to the network.
classifier = Sequential()
#Initializing our model and creating first hidden layer with number of nurons =8 of first hidden layer,init is used to initialize weights,
#which is selected uniform. Rectifier function is taken as activation function and the last parameter "input_dim" is the number  of input features.
classifier.add(Dense(output_dim = 8, init = "uniform", activation = "relu", input_dim = 11))
#Second Hidden Layer
classifier.add(Dense(output_dim = 8, init = "uniform", activation = "relu"))
#Third Hidden Layer
classifier.add(Dense(output_dim = 8, init = "uniform", activation = "relu"))
#Output layer contain only one neuron i.e either "0" or "1"with activation taken as sigmoid function.
classifier.add(Dense(output_dim = 1, init = "uniform", activation = "sigmoid"))
#Now we are going to compile our neural network. Optimizer is used to find the best weights and for that we are using stochastic gradient descent.
#Loss function is logrithmic class. Metrics is the creteria used to evaluate our model and we have choosen accuracy as the criteria.
classifier.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics = ["accuracy"])


# Now we are fitting the model to the training set. "batch_size" is number of obnservations after we choose to update our weights and epochs are number of times we want to traiin our model.

# In[ ]:


classifier.fit(X_train,y_train,batch_size = 1000, epochs = 200)


# In[ ]:


y_pred = classifier.predict(X_test) #Gives us the probability of employee leaving the bank
print(y_pred)
y_pred = (y_pred>0.5) #We use  t hreshold as 0.5
print(y_pred)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

