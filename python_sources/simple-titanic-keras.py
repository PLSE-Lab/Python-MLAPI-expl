#!/usr/bin/env python
# coding: utf-8

# # A Very Basic Keras Implementation
# The objective of this notebook is to show how to set up a basic neural network using Keras. It only looks at three columns of the data, `Sex, Age, Fare`. The results would definitely get better with more careful data/feature engineering. However, I didn't want to distract from the neural network implementation.

# ### Import Data Libraries
# 
# > Pandas is used for the initial data prep. Then is transformed into a numpy array for processing within the neural network.

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# ### Make Data Usable
# Neural networks need number values, so we'll replace the categorical data with numeric data by simply using a binary represation where `male = 0, femal = 1`.

# In[4]:


train_df = train_df.replace(["male", "female"], [0, 1]).fillna(0)
test_df = test_df.replace(["male", "female"], [0, 1]).fillna(0)


# ### Select Columns of Interest
# Again, we're only using three columns of data for this network.

# In[5]:


train_y = train_df[["Survived"]]
interest_columns = ["Sex", "Age", "Fare"]
col_num = len(interest_columns)
train_x = train_df[interest_columns]
test_x = test_df[interest_columns]


# ### Convert the Pandas DFs to Numpy Arrays
# Numpy arrays are required for the neural network to process the data.

# In[6]:


y = train_y.astype(np.float32).values
x = train_x.astype(np.float32).values

x_test = test_x.astype(np.float32).values


# ### Check Dimensions of Data 
# When working with neural networks, the dimensionality of your data is crucial to the design. We'll check the shape (size of each dimension) of our numpy arrays (numpy matrices).

# In[7]:


print(x.shape)
print(y.shape)
print(x_test.shape)


# ### Import Keras
# `Sequential` is the type of NN we'll be using. It's also the most basic -- meaning data is flowing from the start to the finish. Nothing tricky.
# <br>`Dense` is the type of layer we want. It's a fully connected layer of neurons. You can play around with densely connected sequential NN models at [here.](https://www.playground.tensorflow.org)
# <br>`Activation` is the function we want to use to simulate the biological activation of neurons. There are lots of them, but some of the standard include `relu`, `sigmoid`, `tanh`. You can read more about them [here.](https://en.wikipedia.org/wiki/Activation_function)
# <br> `train_test_split` is just a simple way to split up our numpy arrays so we can train on one part of the data, and test on another.

# In[8]:


from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split


# In[9]:


nn_in_train, nn_in_test, nn_out_train, nn_out_test = train_test_split(x, y, test_size=.5)


# In[10]:


print(nn_in_train.shape)
print(nn_in_test.shape)


# ### The Model
# This model takes in the 3 column input, passes it through two dense layers with each layer containing 2 neurons. It then goes through a final layer with a single neuron. This final layer uses a `sigmoid` activation function to transform the values between 0 and 1 -- just what we need for our output.
# 
# <br><br> If we want consistency in our results, we need to tell the random weight generator to start at the same values each time. We do this with `np.random.seed(some_starting_int)`. 

# In[27]:


np.random.seed(2)
model = Sequential()

model.add(Dense(2, input_shape=(col_num,)))
model.add(Activation("linear"))

model.add(Dense(2))
model.add(Activation("relu"))

output_num = 1 # One value representing if the passenger survived
model.add(Dense(output_num))
model.add(Activation("sigmoid"))


# ### Compiling the Model
# `loss` is how we measure the error in our model and tell the model how to become better.
# <br>`optimizer` is the function used to adjust the values in the nn and make our loss lower.

# In[28]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# ### Fit the Model
# Now we give the neural network our data.
# <br>`nn_in_train` is the three columns we're interested in using.
# <br>`nn_out_train` is the output of whether or not the data we are training on survived. 
# <br> The network will run through the data `nb_epoch` times, and process `batch_size` amount of passengers at a time.
# 

# In[32]:


model.fit(nn_in_train, nn_out_train, nb_epoch=100, batch_size=50)


# ### Predict on the Test Data
# Now that the model is trained, we can test it on data it hasn't seen before.

# In[33]:


prediction = np.round(model.predict(nn_in_test))


# ### Check Accuracy on Test/Train Split
# This test comes from the splitting of the data we did earlier. We can be limited on the number of competition submissions, so we'll test with data that we know the results, but that the model has never seen.

# In[34]:


np.sum(nn_out_test == prediction)/nn_out_test.shape[0]


# ### Prep for Competition Submission
# If we want to submit the test data for the competition, we can predict on that too. We'll need to do a little more prep work before we're able to save it as a CSV file in the format required for the competition.
# <br> We'll also need to round the output of the neural network so that it's either a 0 or 1 in the survived column.
# <br> *I would also recommended going back and training on the full training data set prior to submission*

# In[16]:


to_kaggle = pd.DataFrame(np.round(model.predict(x_test)))


# We'll add the `PassengerId` column, and make sure the values are the type we want. Then we'll save it as a CSV file.

# In[17]:


result = pd.concat([test_df[['PassengerId']], to_kaggle], axis=1)
result.columns = ["PassengerId", "Survived"]
result.Survived = result.Survived.astype(int)


# In[18]:


result.to_csv("result.csv", index=False) # If we save the index, it adds an additional column


# In[ ]:




