#!/usr/bin/env python
# coding: utf-8

# So, this is based off a simple example from the book "Deep Learning with Python," by Francois Chollet. Chollet himself is a scientist at Google Brain, and the creator of Keras. This is a fantastic book on getting started with using Keras for deep learning purposes, if you haven't read it yet.
# 
# Recently, however, I have been using PyTorch. Anytime I have to build something more complicated from scratch, I find PyTorch provides a much easier framework to work with. It also is a much more Pythonic library, making it feel like a much more natural extension of Python.

# # EDA
# 
# We will be using the famous Boston housing pricing data. Having 506 data points, the dataset is famously small. I will be constructing both a linear model, using Scikit-Learn; and a neural network using PyTorch. Chollet constructs a neural network using Keras. My model has comparable performance.
# 
# But first, let us have a quick look at the data.

# In[ ]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
    'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 
    'LSTAT', 'MEDV'
]

boston_data = pd.read_csv('../input/boston-house-prices/housing.csv', 
                          header=None, 
                          delimiter=r"\s+", 
                          names=column_names)


# First, let us have a look at the

# In[ ]:


#creating a correlation matrix
correlations = boston_data.corr()
sns.heatmap(correlations, square=True, cmap="YlGnBu")

plt.yticks(rotation=0)
plt.xticks(rotation=90)

plt.show()


# Also, taking a look at various histograms.

# In[ ]:


boston_data.hist(bins=10, figsize=(9,7), grid=False);


# # Linear Model

# First, let us build a linear model. Due to the large number of parameters relative to number of training examples, I will be using Ridge regression.

# In[ ]:


import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Translating our data into arrays for processing.
x = np.array(boston_data.drop(['MEDV'], axis=1))
y = boston_data['MEDV'].values

# Train/test split for validation.
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

# Our Model
lr = Ridge(alpha=0.5)
lr.fit(x_train, y_train)


# In[ ]:


from sklearn.metrics import r2_score

r2_score(lr.predict(x_test), y_test)


# # Building Our Neural Network

# Now, let us build our neural network. I will be using two latent layers of neurons, as you can see below from the code. Some tweaking was done on my end to achieve good performance.

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

batch_size = 50
num_epochs = 250
learning_rate = 0.001
hidden_size = 64
batch_no = len(x_train) // batch_size
input_dim = x.shape[1]

# Use a single hidden layer NN.
model = nn.Sequential(
    nn.Linear(input_dim, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, 1)
)

# Use mean squared error loss.
loss = nn.MSELoss(reduce='mean')

# Use Adam to optimize our NN.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# In[ ]:


running_loss = 0

for epoch in range(num_epochs):
    for i in range(402):
        start = i
        end = start + 1
        
        x_batch = autograd.Variable(torch.FloatTensor(x_train[start:end]))
        y_batch = autograd.Variable(torch.FloatTensor(y_train[start:end]))
                
        y_pred = model(x_batch)
        
        loss_step = loss(y_pred, torch.unsqueeze(y_batch, dim=1))
        optimizer.zero_grad()
        loss_step.backward()
        optimizer.step()
        running_loss += loss_step.item()
    
    print("Epoch {}, Loss: {}. Validation R2: {}".format(
        epoch + 1, running_loss, r2_score(model(torch.Tensor(x_test)).detach().numpy(), y_test)))
    running_loss = 0.0


# It should be noted that even though I'm getting, fairly significant improvements, there are two downsides to this approach.
# 
# 1. Some more training time was needed. However, this was not a huge issue in this case. On a CPU this was trained in about 30 seconds.
# 2. More time was taken on my end to tweak all the addition parameters.
