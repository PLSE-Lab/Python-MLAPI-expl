#!/usr/bin/env python
# coding: utf-8

# # Linear Regression in PyTorch
# 
# In this notebook, I'll work on finding the **linear relation between the data points** using **PyTorch**. (Linear Regression)

# In the upcoming post, Ill discuss **Logistic Regression using PyTorch**. Stay tuned!<br><br>
# Link to the previous notebook, https://www.kaggle.com/superficiallybot/pytorch-starter-notebook-series-nb-1

# ### Problem and Dataset Description

# Say x data points are [1,2,3,4,5,6,7,8,9,10] <br>
# and <br>the corresponding y data points are [3,6,9,12,15,18,21,24,27,30]<br>
# Clearly, there is a linear relationship, which we can visually inspect <br> and formulate as y = 3x
# <br><br>
# Our ML model which be trained on this data for multiple epochs and try to approximate the relation as close as possible to y = 3x

# In[ ]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


# In[ ]:


# defining x and y values
x_train = [i for i in range(11)]
y_train = [3*i for i in x_train]


x_train = np.array(x_train, dtype = np.float32)
x_train = x_train.reshape(-1, 1)

y_train = np.array(y_train, dtype = np.float32)
y_train = y_train.reshape(-1, 1)


# 1. Create a Model Class

# In[ ]:


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = self.linear(x)
        return out


# 2. Instantiate Model Class
# 3. Instantiate Loss Class
# 4. Instantiate Optimizer Class

# In[ ]:


# hyper-parameters
input_dim = 1
output_dim = 1
epochs = 300
learning_rate = 0.01


# In[ ]:


model = LinearRegressionModel(input_dim, output_dim)
# if gpu, model.cuda()


# In[ ]:


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


# In[ ]:


# MODEL TRAINING
loss_list = []
for epoch in range(1, epochs + 1):
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))
    
    optimizer.zero_grad()
    
    outputs = model(inputs)
    
    loss = criterion(outputs, labels)
    
    loss.backward()
    loss_list.append(loss.data)
    optimizer.step()
    
    print('Epoch: {}, loss: {}'.format(epoch, loss.item()))


# Test the model on a single variable

# In[ ]:


test_var = Variable(torch.Tensor([[11.0]]))
pred_y = model(test_var)


# In[ ]:


print('prediction on x = 11 (after training) : ', pred_y.item())


# In[ ]:


print('Real Prediction as per the relation y = 3x: x = 11 => y = 33')


# Testing the model on a tensor vector

# In[ ]:


y_test = np.arange(20,30, dtype = np.float32)
y_test = y_test.reshape(-1, 1)


# In[ ]:


y_test = Variable(torch.from_numpy(y_test))


# In[ ]:


preds = model(y_test)


# In[ ]:


for i in zip(preds, np.arange(20,30)):
    print(f'Prediction when x = {i[1]} -> y = {i[0]}')
    


# In[ ]:





# # Visualizations

# In[ ]:


#import the necessary libraries

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
import matplotlib.pyplot as plt


# **Visualizing the Loss plot**

# In[ ]:


sns.lineplot(x = range(10), y = np.array(loss_list[:10]))
plt.show()


# You can observe, by 2 epochs, the model was able to establish the relationship between the X values & y values. <br>
# This was a small dataset, where fortunately overfitting didn't happen. But training for excessive number of epochs usually leads to model overfitting. By plotting loss vs epochs graph, you would be able to find the right number of epochs for your model to train on.

# **Visualizing the Predictions v/s Actual Scatter Plot**

# In[ ]:


sns.scatterplot(x = np.arange(20,30), y = preds.data.numpy().squeeze(), color = 'red', label = 'predicted')
sns.lineplot(x = np.arange(20, 30), y = 3 * np.arange(20,30), color = 'green', label = 'Actual')
plt.show()


# In[ ]:





# In the upcoming post, I'll discuss about **Logistic Regression using PyTorch**. Stay tuned!

# ## If you liked my kernel, please appreciate my effort by giving an UPVOTE to it. Your comments and valued suggestions are warmly welcomed.
