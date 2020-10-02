#!/usr/bin/env python
# coding: utf-8

# # Bank Customer Classification
# ## Given a dataset consisiting of Bank Customer information, we are asked to build a classifier which will tell us if a customer will exit the bank or not.

# In[ ]:


get_ipython().system('pip install scikit-learn==0.22.0')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

import os
print(os.listdir("../input"))


# # Data Preprocessing 

# ### In this dataset, we have to consider which of the factors may play a role in someone exiting a bank. To do that we must look at all  the column and infer whether it will matter in classifying a new customer or not.  The information about a customer is entailed in columns 0 through 12 (RowNumber-EstimatedSalary), while the output (whether the customer exited or not) is stored in the 13th row (Exited). 
# 
# For as much as we care, neither the customer ID, nor the surname should matter in classification. 
# Therefore, we will use columns 3 (CreditScore) inclusive through the 13th column (exclusive).
# 

# In[ ]:


#importing the dataset
dataset = pd.read_csv('../input/Churn_Modelling.csv', index_col='RowNumber')
dataset.head()


# In[ ]:


X_columns = dataset.columns.tolist()[2:12]
y_columns = dataset.columns.tolist()[-1:]
print(f'All columns: {dataset.columns.tolist()}')
print()
print(f'X values: {X_columns}')
print()
print(f'y values: {y_columns}')


# In[ ]:


X = dataset[X_columns].values # Credit Score through Estimated Salary
y = dataset[y_columns].values # Exited


# In[ ]:


# Encoding categorical (string based) data. Country: there are 3 options: France, Spain and Germany
# This will convert those strings into scalar values for analysis
print(X[:8,1], '... will now become: ')
from sklearn.preprocessing import LabelEncoder
label_X_country_encoder = LabelEncoder()
X[:,1] = label_X_country_encoder.fit_transform(X[:,1])
print(X[:8,1])


# In[ ]:


# We will do the same thing for gender. this will be binary in this dataset
print(X[:6,2], '... will now become: ')
from sklearn.preprocessing import LabelEncoder
label_X_gender_encoder = LabelEncoder()
X[:,2] = label_X_gender_encoder.fit_transform(X[:,2])
print(X[:6,2])


# The Problem here is that we are treating the countries as one variable with ordinal values (0 < 1 <  2). Therefore, one way to get rid of that problem is to split the countries into respective dimensions. that is,
# 
# | Country |  -> | Country|-> |Spain|France|Germany|
# |------|      |------|  |------|    |------|    |------|
# |   Spain |   -> |0| -> |1|0|0|
# |   France | -> |1| -> |0|1|0|
# |   France | ->  |1| -> |0|1|0|
# |   Germany | -> |2| -> |0|0|1|

# You can now see that the first three columns represent the three countries that constituted the "country" category. We can now observe that  we essentially only need two columns: a 0 on two countries means that the country has to be the one variable which wasn't included. This will save us from the problem of using too many dimensions.
# 
# |Spain|France|Germany|-> |France|Germany|
#  |------|    |------|    |------|     |------|     |------|
#  |1|0|0|-> |0|0|
# |0|1|0|-> |1|0|
# |0|1|0|-> |1|0|
# |0|0|1|-> |0|1|
# 
# We have achieved this using the `drop='first'` option in the OneHotEncoder

# Feature scaling is a method used to standardize the range of independent variables or features of data. It is basically scaling all the dimensions to be even so that one independent variable does not dominate another. For example, bank account balance ranges from millions to 0, whereas gender is either 0 or 1. If one of the features has a broad range of values, the distance will be governed by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.

# In[ ]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


pipeline = Pipeline(
    [('Categorizer', ColumnTransformer(
         [ # Gender
          ("Gender Label encoder", OneHotEncoder(categories='auto', drop='first'), [2]),
           # Geography
          ("Geography One Hot", OneHotEncoder(categories='auto', drop='first'), [1])
         ], remainder='passthrough', n_jobs=1)),
     # Standard Scaler for the classifier
    ('Normalizer', StandardScaler())
    ])


# In[ ]:


X = pipeline.fit_transform(X)


# In[ ]:


# Splitting the dataset into the Training and Testing set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


# In[ ]:


print(f'training shapes: {X_train.shape}, {y_train.shape}')
print(f'testing shapes: {X_test.shape}, {y_test.shape}')


# ## END OF PREPROCESSING

# ## Making the NN

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout


# In[ ]:


# Initializing the ANN
classifier = Sequential()


# A hurestic tip is that the amount of nodes (dimensions) in your hidden layer should be the average of your input and output layers, which means that since we have 11 dimensions (representing **Independent variables** Note: Countries still compose only  **one** dimension) and we are looking for a binary output, we calculate this to be $(11+1)\div 2 = 6 $.
# 
# #### The breakdown of the inputs for the first layer is as follows:
# 
# `units`: `6` nodes (number of nodes in hidden layer). Can think of this as number of nodes are in the next layer.
# 
# `activiation`: `relu` becasue we are in an input layer. uses the ReLu activation function for the layer. This is equivalent to $max(0, W \times x^T + b)$
# 
# `input_dim`: `11` because we span 11 dimensions in our input layer. This is needed for the first added layer. The subsequent layers's input dimensions can be inferred using the previously added layer's output dimension. The next hidden layer will know what to expect.
# 
# 
# 

# In[ ]:


# This adds the input layer (by specifying input dimension) AND the first hidden layer (units)
classifier.add(Dense(6, activation = 'relu', input_shape = (X_train.shape[1], )))
classifier.add(Dropout(rate=0.1)) 


# ### We are going to add another layer to this model because we want to implement Deep Learning, which is an artificial Neural network with many layers.
# We will make our second hidden layer also have 6 nodes, just playing with the same arithmetic we used to determine the dimensions of the first hidden layer (average of your input and output layers) $(11+1)\div 2 = 6 $.

# In[ ]:


# Adding the second hidden layer
# Notice that we do not need to specify input dim. 
classifier.add(Dense(6, activation = 'relu')) 
classifier.add(Dropout(rate=0.1)) 


# ### Adding the output layer
# #### The breakdown of the inputs for the output layer is as follows:
# 
# *activiation*: **sigmoid** becasue we are in an output layer. uses the Sigmoid activation function for $\phi$. This is used instead of the ReLu function becasue it generates probabilities for the outcome. We want the probability that each customer leaves the bank.  
# 
# `units`: `6` nodes (number of nodes in hidden layer). Can think of this as number of nodes are in the next layer.
# 
# `input_dim`: `11` because we span 11 dimensions in our input layer. This is needed for the first added layer. The subsequent layers's input dimensions can be inferred using the previously added layer's output dimension. The next hidden layer will know what to expect.
# 

# In[ ]:


# Adding the output layer
# Notice that we do not need to specify input dim. 
# we have an output of 1 node, which is the the desired dimensions of our output (stay with the bank or not)
# We use the sigmoid because we want probability outcomes
classifier.add(Dense(1, activation = 'sigmoid')) 


# 
# ### If we want more than two categories, then we will need to change 
# 
#  1) the *units* parameter to match the desired category count
#  
#  2) the *activation* field to **softmax**.  Basically a sigmoid function but applied to a dependent variable that has more than 2 categories.

# In[ ]:


classifier.summary()


# ## Compiling the Neural Network
# Basically applying Stochastic Gradient descent on the whole Neural Network. We are Tuning the individual weights on each neuron.
# 
# #### The breakdown of the inputs for compiling is as follows:
# 
# `optimizer`: `adam` The algorithm we want to use to find the optimal set of weights in the neural networks.  Adam is a very efficeint variation of Stochastic Gradient Descent.
# 
# `loss`: `binary_crossentropy` This is the loss function used within adam. This should be the logarthmic loss. If our dependent (output variable) is `Binary`, it is `binary_crossentropy`. If `Categorical`, then it is called `categorical_crossentropy`
# 
# `metrics`: `[accuracy]` The accuracy metrics which will be evaluated(minimized) by the model. Used as accuracy criteria to imporve model performance. 

# In[ ]:


classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])


# ## Fitting the Neural Network
# This is where we will be fitting the NN to our training set.
# 
# #### The breakdown of the inputs for compiling is as follows:
# 
# `X_train` The independent variable portion of the data which needs to be fitted with the model.
# 
# `Y_train` The output portion of the data which the model needs to produce after fitting.
# 
# `batch_size`:  How often we want to back-propogate the error values so that individual node weights can be adjusted. 
# 
# `epochs`: The number of times we want to run the entire test data over again to tune the weights. This is like the fuel of the algorithm. 
# 
# 
# `validation_split`: `0.2` The fraction of data to use for validation data. 
# 

# In[ ]:


history = classifier.fit(X_train, y_train, batch_size=32, epochs=200, validation_split=0.1, verbose=2)


# The output network should converge to an accuracy of around 85%
# ## Testing the NN
# ### Predicting the Test set results
# 
# This shows the probability of a customer leaving given the testing data. Each row in X_test corresponds to a row in Y_test

# In[ ]:


plt.plot(np.array(history.history['acc']) * 100)
plt.plot(np.array(history.history['val_acc']) * 100)
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'validation'])
plt.title('Accuracy over epochs')
plt.show()


# In[ ]:


y_pred = classifier.predict(X_test)
print(y_pred[:5])


# To use the confusion Matrix, we need to convert the probabilities that a customer will leave the bank into the form true or false. So we will use the cutoff value 0.5 to indicate whether they are likely to exit or not.

# In[ ]:


y_pred = (y_pred > 0.5).astype(int)
print(y_pred[:5])


# ### Making the Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# ### Significance of the confusion matrix value:
# 
# The output should be close to the table below:
# 
# ||Predicted: No |Predicted: Yes|
#  |------| |------|   |------|  
#  |Actual: No|1504|91|
# |Actual: Yes|184|221
# 

# This means that we should have about $(1504 + 221) = 1726$ correct classifications out of our total testing data size of $2000$.
# This means that our accuracy for this trial was $1726 \div 2000 = 86.3\%$, which matches the classifier's prediction

# In[ ]:


print (((cm[0][0]+cm[1][1])*100)/(len(y_test)), '% of testing data was classified correctly')

