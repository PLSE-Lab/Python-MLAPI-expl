#!/usr/bin/env python
# coding: utf-8

# # <div align="center"> Hello people, everything all right?  </div>
# #### ** This is a tutorial of how to create a simple neural network with Keras for predict video games sales per continent. I'm NOT focusing in data analysis here, only on the simple preprocess the data and create the model.**
# ![](https://media.giphy.com/media/d2Z7NqwF3yImFNHW/giphy-downsized.gif)

# In[ ]:


# Import the necessary libs
from keras.models import Model
from keras.layers import Input, Dense, Activation
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd 


# In[ ]:


# Load the dataset
games = pd.read_csv('../input/video-game-sales-with-ratings/Video_Games_Sales_as_at_22_Dec_2016.csv')
games.head()


# ### 1. In this part, i will preprocess the data, removing unecessary columns. It's a didatic model, so I drop the Name, Yar_of_Release, Other_Sales and Global_Sales (because we just want the continent sales).

# In[ ]:


games = games.drop(['Name', 'Year_of_Release', 'Other_Sales', 'Global_Sales', 'Developer'], 1)
games.head()


# In[ ]:


# Drop some outliers. I'm not will remove the Japan Sales for the dataset not be too small
games = games.loc[games['NA_Sales'] > 1]
games = games.loc[games['EU_Sales'] > 1]
games.shape


# In[ ]:


# Remove NA values
games = games.dropna(axis=0)
games.shape


# ## 2. Now, select the features and the targets

# In[ ]:


games.head()


# In[ ]:


# The features are 'Platform', 'Genre', 'Publisher','Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Rating'
X = games.iloc[:, [0, 1, 2, 6, 7, 8, 9, 10]].values

# In this case, there are 3 targets, one for each continent
y_na = games['NA_Sales'].values
y_eu = games['EU_Sales'].values
y_jp = games['JP_Sales'].values


# ## 3. Encoding the features

# In[ ]:


# Using OneHotEncoder to transform the features into dummy variables
ohencoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(),
                                            [0, 1, 2, 7])], remainder='passthrough')
X = ohencoder.fit_transform(X).toarray()


# ## Preprocessing phase completed. Now, we will construct the neural network

# In[ ]:


# Creating the input layer
# 60 is the number of columns in the X.
my_input = Input(shape=(60, ))


# In[ ]:


# Creating first hidden layer
# (input shape + number of outputs) / 2 = (60 + 3) / 2 = 32 rounded
# It's necessary to say which layer the hidden layer is connected to. In that case, the input layer
my_hidden_layer_1 = Dense(units=32, activation='sigmoid')(my_input)


# In[ ]:


# Creating second hidden layer
my_hidden_layer_2 = Dense(units=32, activation='sigmoid')(my_hidden_layer_1)


# In[ ]:


# Create the output layers, one for each continent
# The activation function "linear" keeps the output values
my_output_layer_1 = Dense(units=1, activation='linear')(my_hidden_layer_2)
my_output_layer_2 = Dense(units=1, activation='linear')(my_hidden_layer_2)
my_output_layer_3 = Dense(units=1, activation='linear')(my_hidden_layer_2)


# In[ ]:


# Create the model with the inputs and outputs
my_model = Model(inputs=my_input,
                outputs=[my_output_layer_1, my_output_layer_2, my_output_layer_3])


# In[ ]:


#Compile the model
my_model.compile(optimizer='adam', loss='mse')


# ## The process takes a little time and the code output is great; so, feel free to skip to the end result.

# 

# In[ ]:


# Fit the model
my_model.fit(X, [y_na, y_eu, y_jp], epochs=7000, batch_size=100)


# ### Neural network done.
# ### For more info, consult the Keras Documentation clicking [here](https://keras.io/)
# ### I hope this tutorial is useful. Reviews are welcome

# In[ ]:




