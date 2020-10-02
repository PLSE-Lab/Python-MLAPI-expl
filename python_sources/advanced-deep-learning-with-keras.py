#!/usr/bin/env python
# coding: utf-8

# # Advanced Deep Learning with Keras in Python

# ## The Keras Functional API
# 

# ### Input layers

# In[ ]:


# Import Input from keras.layers
from keras.layers import Input

# Create an input layer of shape 1
input_tensor = Input(shape=(1,))


# ### Dense layers

# In[ ]:


# Load layers
from keras.layers import Input, Dense

# Input layer
input_tensor = Input(shape=(1,))

# Dense layer
output_layer = Dense(1)

# Connect the dense layer to the input_tensor
output_tensor = output_layer(input_tensor)


# This network will take the input, apply a linear coefficient to it, and return the result.

# ### Output layers

# In[ ]:


# Load layers
from keras.layers import Input, Dense

# Input layer
input_tensor = Input(shape=(1,))

# Create a dense layer and connect the dense layer to the input_tensor in one step
# Note that we did this in 2 steps in the previous exercise, but are doing it in one step now
output_tensor = Dense(1)(input_tensor)


# The output layer allows your model to make predictions.

# ### Build a model

# In[ ]:


# Input/dense/output layers
from keras.layers import Input, Dense
input_tensor = Input(shape=(1,))
output_tensor = Dense(1)(input_tensor)

# Build the model
from keras.models import Model
model = Model(input_tensor, output_tensor)


# ### Compile a model

# This finalizes your model, freezes all its settings, and prepares it to meet some data!

# In[ ]:


# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')


# ### Visualize a model

# In[ ]:


# Import the plotting function
from keras.utils import plot_model
import matplotlib.pyplot as plt

# Summarize the model
model.summary()

# # Plot the model
plot_model(model, to_file='model.png')

# # Display the image
data = plt.imread('model.png')
plt.imshow(data)
plt.show()


# ### Fit the model to the tournament basketball data

# In[ ]:


import pandas as pd

games_tourney_train = pd.read_csv('../input/games-tourney/games_tourney.csv')
games_tourney_train.head()


# In[ ]:


games_tourney_train.shape


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(games_tourney_train['seed_diff'], games_tourney_train['score_diff'])


# In[ ]:


# Now fit the model
model.fit(X_train, y_train,
          epochs=1,
          batch_size=128,
          validation_split=.10,
          verbose=True)


# ### Evaluate the model on a test set

# In[ ]:


# Evaluate the model on the test data
model.evaluate(X_test, y_test)


# ## Two Input Networks Using Categorical Embeddings, Shared Layers, and Merge Layers

# ### Define team lookup

# In[ ]:


games_season = pd.read_csv('../input/games-season/games_season.csv')
games_season.head()


# In[ ]:


# Imports
from keras.layers import Embedding
from numpy import unique

# Count the unique number of teams
n_teams = unique(games_season['team_1']).shape[0]

# Create an embedding layer
team_lookup = Embedding(input_dim=n_teams,
                        output_dim=1,
                        input_length=1,
                        name='Team-Strength')


# The embedding layer is a lot like a dictionary, but your model learns the values for each key.

# ### Define team model

# In[ ]:


# Imports
from keras.layers import Input, Embedding, Flatten
from keras.models import Model

# Create an input layer for the team ID
teamid_in = Input(shape=(1,))

# Lookup the input in the team strength embedding layer
strength_lookup = team_lookup(teamid_in)

# Flatten the output
strength_lookup_flat = Flatten()(strength_lookup)

# Combine the operations into a single, re-usable model
team_strength_model = Model(teamid_in, strength_lookup_flat, name='Team-Strength-Model')


# ### Defining two inputs

# In[ ]:


# Load the input layer from keras.layers
from keras.layers import Input

# Input layer for team 1
team_in_1 = Input((1,), name='Team-1-In')

# Separate input layer for team 2
team_in_2 = Input((1,), name='Team-2-In')


# These two inputs will be used later for the shared layer.

# ### Lookup both inputs in the same model

# In[ ]:


# Lookup team 1 in the team strength model
team_1_strength = team_strength_model(team_in_1)

# Lookup team 2 in the team strength model
team_2_strength = team_strength_model(team_in_2)


# Now your model knows how strong each team is.

# ### Output layer using shared layer

# In[ ]:


# Import the Subtract layer from keras
from keras.layers import Subtract

# Create a subtract layer using the inputs from the previous exercise
score_diff = Subtract()([team_1_strength, team_2_strength])


# This setup subracts the team strength ratings to determine a winner.

# ### Model using two inputs and one output

# ![](basketball_model_2.png)

# In[ ]:


# Imports
from keras.layers import Subtract
from keras.models import Model

# Subtraction layer from previous exercise
score_diff = Subtract()([team_1_strength, team_2_strength])

# Create the model
model = Model([team_in_1, team_in_2], score_diff)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')


# In[ ]:


model.summary()


# ### Fit the model to the regular season training data

# In[ ]:


# Get the team_1 column from the regular season data
input_1 = games_season['team_1']

# Get the team_2 column from the regular season data
input_2 = games_season['team_2']

# Fit the model to input 1 and 2, using score diff as a target
model.fit([input_1, input_2],
          games_season['score_diff'],
          epochs=1,
          batch_size=2048,
          validation_split=.1,
          verbose=True)


# Now our model has learned a strength rating for every team.

# ### Evaluate the model on the tournament test data

# In[ ]:


games_tourney = pd.read_csv('../input/games-tourney/games_tourney.csv')
games_tourney.head()


# In[ ]:


# Get team_1 from the tournament data
input_1 = games_tourney['team_1']

# Get team_2 from the tournament data
input_2 = games_tourney['team_2']

# Evaluate the model using these inputs
model.evaluate([input_1, input_2], games_tourney['score_diff'])


# ## Multiple Inputs: 3 Inputs (and Beyond!)

# ### Make an input layer for home vs. away

# You know there is a well-documented home-team advantage in basketball, so you will add a new input to your model to capture this effect.
# 
# 

# In[ ]:


from keras.layers.merge import Concatenate


# In[ ]:


# Create an Input for each team
team_in_1 = Input(shape=(1,), name='Team-1-In')
team_in_2 = Input(shape=(1,), name='Team-2-In')

# Create an input for home vs away
home_in = Input(shape=(1,), name='Home-In')

# Lookup the team inputs in the team strength model
team_1_strength = team_strength_model(team_in_1)
team_2_strength = team_strength_model(team_in_2)

# Combine the team strengths with the home input using a Concatenate layer, then add a Dense layer
out = Concatenate()([team_1_strength, team_2_strength, home_in])
out = Dense(1)(out)


# Now you have a model with 3 inputs!

# ### Make a model and compile it

# In[ ]:


# Import the model class
from keras.models import Model

# Make a Model
model = Model([team_in_1, team_in_2, home_in], out)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')


# In[ ]:


model.summary()


# ### Fit the model and evaluate

# In[ ]:


# Fit the model to the games_season dataset
model.fit([games_season['team_1'], games_season['team_2'], games_season['home']],
          games_season['score_diff'],
          epochs=1,
          verbose=True,
          validation_split=.1,
          batch_size=2048)

# Evaluate the model on the games_tourney dataset
model.evaluate([games_tourney['team_1'], games_tourney['team_2'], games_tourney['home']], games_tourney['score_diff'])


# ### Plotting models

# In[ ]:


# Imports
import matplotlib.pyplot as plt
from keras.utils import plot_model

# Plot the model
plot_model(model, to_file='model.png')

# Display the image
data = plt.imread('model.png')
plt.imshow(data)
plt.show()


# ### Add the model predictions to the tournament data

# In[ ]:


# Predict
games_tourney['pred'] = model.predict([games_tourney['team_1'],games_tourney['team_2'],games_tourney['home']])


# In[ ]:


games_tourney.head()


# In[ ]:


games_tourney.score_diff.unique()


# Now you can try building a model for the tournament data based on your regular season predictions

# ### Create an input layer with multiple columns

# In[ ]:


# Create an input layer with 3 columns
input_tensor = Input((3,))

# Pass it to a Dense layer with 1 unit
output_tensor = Dense(1)(input_tensor)

# Create a model
model = Model(input_tensor, output_tensor)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')


# ### Fit the model

# In[ ]:


games_tourney_train = games_tourney.query('season < 2010')


# In[ ]:


games_tourney_train.shape


# In[ ]:


# Fit the model
model.fit(games_tourney_train[['home', 'seed_diff', 'pred']],
          games_tourney_train['score_diff'],
          epochs=1,
          verbose=True)


# ### Evaluate the model

# In[ ]:


games_tourney_test = games_tourney.query('season >= 2010')


# In[ ]:


games_tourney_test.shape


# ### Evaluate the model

# In[ ]:


# Evaluate the model on the games_tourney_test dataset
model.evaluate(games_tourney_test[['home', 'seed_diff', 'pred']], 
               games_tourney_test['score_diff'])


# ## Multiple Outputs

# ### Simple two-output model

# "multiple target regression": one model making more than one prediction.

# In[ ]:


# Define the input
input_tensor = Input((2,))

# Define the output
output_tensor = Dense(2)(input_tensor)

# Create a model
model = Model(input_tensor, output_tensor)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')


# ### Fit a model with two outputs

#  this model will predict the scores of both teams.

# In[ ]:


games_tourney_train.shape


# In[ ]:


# Fit the model
model.fit(games_tourney_train[['seed_diff', 'pred']],
  		  games_tourney_train[['score_1', 'score_2']],
  		  verbose=False,
  		  epochs=1000,
  		  batch_size=64)


# In[ ]:


import numpy as np


# In[ ]:


np.mean(model.history.history['loss'])


# ### Inspect the model (I)

# The input layer will have 4 weights: 2 for each input times 2 for each output.
# 
# The output layer will have 2 weights, one for each output.

# In[ ]:


# Print the model's weights
print(model.get_weights())

# Print the column means of the training data
print(games_tourney_train.mean())


# Did you notice that both output weights are about ~53? This is because, on average, a team will score about 53 points in the tournament.

# ### Evaluate the model

# In[ ]:


# Evaluate the model on the tournament test data
model.evaluate(games_tourney_test[['seed_diff', 'pred']],games_tourney_test[['score_1', 'score_2']])


# ### Classification and regression in one model

# In[ ]:


# Create an input layer with 2 columns
input_tensor = Input((2,))

# Create the first output
output_tensor_1 = Dense(1, activation='linear', use_bias=False)(input_tensor)

# Create the second output (use the first output as input here)
output_tensor_2 = Dense(1, activation='sigmoid', use_bias=False)(output_tensor_1)

# Create a model with 2 outputs
model = Model(input_tensor, [output_tensor_1, output_tensor_2])


# ### Compile and fit the model

# In[ ]:


# Import the Adam optimizer
from keras.optimizers import Adam

# Compile the model with 2 losses and the Adam optimzer with a higher learning rate
model.compile(loss=['mean_absolute_error', 'binary_crossentropy'], optimizer=Adam(lr=0.01))

# Fit the model to the tournament training data, with 2 inputs and 2 outputs
model.fit(games_tourney_train[['seed_diff', 'pred']],
          [games_tourney_train[['score_diff']], games_tourney_train[['won']]],
          epochs=10,
          verbose=True,
          batch_size=16384)


# ### Inspect the model (II)

# In[ ]:


# Print the model weights
print(model.get_weights())

# Print the training data means
print(games_tourney_train.mean())


# In[ ]:


# Import the sigmoid function from scipy
from scipy.special import expit as sigmoid

# Weight from the model
weight = 0.14

# Print the approximate win probability predicted close game
print(sigmoid(1 * weight))

# Print the approximate win probability predicted blowout game
print(sigmoid(10 * weight))


# So `sigmoid(1 * 0.14)` is `0.53`, which represents a pretty close game and `sigmoid(10 * 0.14)` is `0.80`, which represents a pretty likely win. In other words, if the model predicts a win of 1 point, it is less sure of the win than if it predicts 10 points. Who says neural networks are black boxes?

# ### Evaluate on new data with two metrics

# Keras will return 3 numbers: 
# 
# the first number will be the sum of both the loss functions, 
# 
# the next 2 numbers will be the loss functions you used when defining the model.

# In[ ]:


# Evaluate the model on new data
model.evaluate(games_tourney_test[['seed_diff', 'pred']],
               [games_tourney_test[['score_diff']], games_tourney_test[['won']]])


# model plays a role as a regressor and a good classifier
