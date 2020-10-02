#!/usr/bin/env python
# coding: utf-8

# **What is Neural Network**
# 
# They are multi layer peceptrons. By stacking many linear units we get neural network.![NN](https://www.researchgate.net/profile/Mohamed_Zahran6/publication/303875065/figure/fig4/AS:371118507610123@1465492955561/A-hypothetical-example-of-Multilayer-Perceptron-Network.png)

# **Why are Neural Networks popular**
# 
# Neural Networks are remarkably good at figuring out functions from X to Y. ![MLP](https://cdn-images-1.medium.com/max/800/1*xj5Y_UrUONLvQhgXNWZSlw.png)
# 
# In general all input features are connected to hidden units and NN's are capable of drawing hidden features out of them.
# 
# The above neural netowork is called 2 layer neural netowork, We don't count Input layer.

# **Computation of NN**
# 
# Computation of NN is done by forward propagation for computing outputs(price in above NN) and Backward pass for computing gradients.
# 
# **What if I remove hidden layer in 2 layer network?**

# **Forward propagation:**
# 
# **Z=W<sup>T</sup>x+b**
# 
# Here Z is the weighted sum of inputs with the inclusion of bias
# 
# **y<sup>^</sup>=a(For simplification)=activation_function(z)** 
# 
# Predicted Output is activation function applied on weighted sum(Z)

# **Activation Functions:** The following activation functions helps in transforming linear inputs to nonlinear outputs. If we apply linear activation function we will get linear seperable line for classifying the outputs.
# 
# 1. *Sigmoid*: 
# 
# Sigmoid(x) = ** 1/(1+exp(-x))**
# 
# ![Sigmoid](https://cdn-images-1.medium.com/max/1600/1*Xu7B5y9gp0iL5ooBj7LtWw.png)
# 
# The main reason why we use sigmoid function is because it exists between (0 to 1). Therefore, it is especially used for models where we have to predict the probability as an output.Since probability of anything exists only between the range of 0 and 1, sigmoid is the right choice.
#     
# 2. *Tanh*: 
# 
# Tanh(x): **(exp(x)-exp(-x))/(exp(x)+exp(-x))**
# 
# ![Tanh](https://i2.wp.com/sefiks.com/wp-content/uploads/2017/01/tanh.png?resize=456%2C300&ssl=1)
# 
# The advantage is that the negative inputs will be mapped strongly negative and the zero inputs will be mapped near zero in the tanh graph.
# 
# 3. *Relu*:
# 
# Relu(x)=max(0,x)
# 
# ![Relu](https://cdn-images-1.medium.com/max/1600/1*XxxiA0jJvPrHEJHD4z893g.png)
# 
# The ReLU is the most used activation function in the world right now. Since, it is used in almost all the convolutional neural networks or deep learning.
#     
# 4. *Softmax*:
# 
# Softmax(y<sub>i</sub>)=exp(y<sub>i</sub>)/sigma(exp(y<sub>j</sub>))
# 
# ![Softmax](https://cdn-images-1.medium.com/max/800/1*670CdxchunD-yAuUWdI7Bw.png)
# 
# In general we use softmax activation function when we have multiple ouput units. For example for predicting hand written digits we have 10 possibilities. We have 10 output units, for getting the 10 probabilities of a given digit we use softmax.
# 
# 
# 
# **Activation functions can be different for hidden and output layers.**

# What is categorical variable? What is numeric variable?
# 
# **Loss Functions**
# 
# *Regression*: When actual Y values are numeric. Eg: Price of house as output variable, range of price of a house can vary within certain range.
#     
# For regression problems: For regression problems we generally use RMSE as loss function.
#     
# *Classification(binary)*: When the given y takes only two values. i.e 0 or 1 Eg: Whether the person will buy the house and each class is mutually exclusive.
#     
# For binary Classification problems: For binary classification proble we generally use **binary cross entropy** as loss function.
# 
# ![BCE](data:image/png;base64,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)
# 

# **Neural Network representation:**
# 
# ![NN](https://cdn-images-1.medium.com/max/400/1*QfLbbPYCLdEj4Pl34CSB9A.png)
# 
# 3 input cells , 4 hidden cells and one output cell
# 
# 
# **W<sup>[i]</sup> --- represent the weight matrix at the i<sup>th</sup> layer**
# 
# **b<sup>[i]</sup> --- represent the bias vector at the i<sup>th</sup> layer**
# 
# Shape of W<sup>[1]</sup> is (3,4) then shape of W<sup>[1]T</sup> is (4,3)
# 
# Shape of b<sup>[1]</sup> is (4,1)
# 
# Shape of W<sup>[2]</sup> is (4,1) then shape of W<sup>[2]T</sup> is (1,4)
# 
# Shape of b<sup>[2]</sup> is (1,1)
# 
# For single input there are three features and the shape of x is (3,1) i.e [ [x1] , [x2] , [x3] ]
# 
# **Forword Propagation of 2 Layer NN for single input**
# 
# * Z<sup>[1]</sup>      =    W<sup>[1]T</sup> * x + b<sup>[1]</sup>------------------shape of Z<sup>[1]</sup> is (4,1)
# 
# * a<sup>[1]</sup>=sigmoid(Z<sup>[1]</sup>)------------------------------------------------shape of a<sup>[1]</sup> is also (4,1)
# 
# * Z<sup>[2]</sup>= W<sup>[2]T</sup> * a<sup>[1]</sup> + b<sup>[2]</sup>-----shape of Z<sup>[2]</sup> is (1,1)
# 
# * a<sup>[2]</sup>=sigmoid(Z<sup>[2]</sup>)------------------------------------------------shape of a<sup>[2]</sup> is also (1,1)
# 
# * yhat=a<sup>[2]</sup>
# 
# **For mutiple inputs:**
# 
# We stack the features of each input vertically, let us say if we have N inputs.
# 
# X=[ [x<sub>1</sub><sup>(1)</sup>,x<sub>1</sub><sup>(2)</sup>........x<sub>1</sub><sup>(n)</sup>] , [x<sub>2</sub><sup>(1)</sup>,x<sub>2</sub><sup>(2)</sup>........x<sub>2</sub><sup>(n)</sup>] , [x<sub>3</sub><sup>(1)</sup>,x<sub>3</sub><sup>(2)</sup>........x<sub>3</sub><sup>(n)</sup>] ]
# 
# shape of x is (3,n)
# 
# the above equations will be same as follows:
# 
# * Z<sup>[1]</sup>      =    W<sup>[1]T</sup> * X + b<sup>[1]</sup>------------------shape of Z<sup>[1]</sup> is (4,n)
# 
# * a<sup>[1]</sup>=sigmoid(Z<sup>[1]</sup>)------------------------------------------------shape of a<sup>[1]</sup> is also (4,n)
# 
# * Z<sup>[2]</sup>= W<sup>[2]T</sup> * a<sup>[1]</sup> + b<sup>[2]</sup>-----shape of Z<sup>[2]</sup> is (1,n)
# 
# * a<sup>[2]</sup>=sigmoid(Z<sup>[2]</sup>)------------------------------------------------shape of a<sup>[2]</sup> is also (1,n)
# 
# We got predicted value for each input. **This concept of stacking multiple column vectors of inputs is called vectorization.**
# 
# Vectorization is faster than writing a for loop over all inputs............

# **Derivative of sigmoid activation function**
# 
# --------IN CLASS BOARD-----------

# **Backward Propagation** (Without hidden layer)
# 
# ![Back_Prop](https://cdn-images-1.medium.com/max/800/1*7lFDfrTNjy156cOeH4oD1g.png)
# 
# BCE Loss: ```-yloga - (1-y)log(1-a) where a = y^```
# 
# ```
# dL(a,y)/da = da = -y/a + (1-y)/(1-a)
# 
# dL(a,y)/dz = dz = a-y
# 
# dL(a,y)/dw = dw = (a-y)*x
# 
# dL(a,y)/db = db = (a-y)
# 
# ```
# 
# Finally updating w and b
# 
# ```
# w = w - learning_rate*dw
# 
# b = b - learning_rate*db
# ```
# 
# 
# 
# 
# 
# 
# 

# **Homework:** **Backward Propagation** (With hidden layer)
# 
# ![2NN](https://cdn-images-1.medium.com/max/800/1*X92YUXvzJIVKQpEkBqLjDw.png)
# 
# 
# * da<sup>[2]</sup>=
# 
# 
# 
# * dz<sup>[2]</sup>=
# 
# 
# 
# * dw<sup>[2]</sup>=
# 
# 
# 
# * db<sup>[2]</sup>=
# 
# 
# 
# * W<sup>[2]</sup>= W<sup>[2]</sup>     - learning_rate *  dw<sup>[2]</sup>
# 
# 
# 
# * b<sup>[2]</sup>=b<sup>[2]</sup>        - learning_rate *  db<sup>[2]</sup>
# 
# 
# 
# * da<sup>[1]</sup>=
# 
# 
# 
# * dz<sup>[1]</sup>=
# 
# 
# 
# * dw<sup>[1]</sup>=
# 
# 
# 
# * db<sup>[1]</sup>=
# 
# 
# 
# * W<sup>[1]</sup>=W<sup>[1]</sup> - learning_rate *  dw<sup>[1]</sup>
# 
# 
# 
# * b<sup>[1]</sup>=b<sup>[1]</sup> - learning_rate *  db<sup>[1]</sup>

# **Introduction to keras**
# 
# * Keras is a minimalist Python library for deep learning that can run on top of TensorFlow.
# 
# * It was developed to make implementing deep learning models as fast and easy as possible for research and development.
# 
# * The main type of model used in keras is called a Sequence which is a linear stack of layers.
# 
# * You create a sequence and add layers to it in the order that you wish for the computation to be performed.
# 
# * Once defined, you compile the model which makes use of the underlying framework to optimize the computation to be performed by your model. In this you can specify the loss function and the **optimizer** to be used.
# 
# * Once compiled, the model must be fit to data. 
# 
# * Once trained, you can use your model to make predictions on new data.
# 
# * We can summarize the construction of deep learning models in Keras as follows:
# 
#     1. Define your model. Create a sequence and add layers.
#     
#     2. Compile your model. Specify loss functions and optimizers.
#     
#     3. Fit your model. Execute the model using data.
#     
#     4. Make predictions. Use the model to generate predictions on new data.
#     
# ![keras](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)
#     
# [Click here to see Keras Documentation](https://keras.io/)
# 
# 

# **Creating your first Deep Neural Network using keras** (Hands on)
# 
# [Competition Link](https://www.kaggle.com/c/diabetes-classification)

# ## Import dependencies

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


# ## Read the data

# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
subm=pd.read_csv('../input/sample_submission.csv')


# ## Stats on Data

# In[ ]:


train.diabetes.value_counts()


# In[ ]:


train.age.hist()
plt.show()


# In[ ]:


train.glucose_concentration.hist()
plt.show()


# In[ ]:


train.bmi.hist()
plt.show()


# ## Hyper parameters
# 
# Parameters which are external to the model. Weights and bias are the parameters which are internal to the model generation.
# 
# * No of Hidden Units.
# * Learning rate.
# * Activation function.
# * No of epochs

# In[ ]:


hidden_units=100
learning_rate=0.01
hidden_layer_act='tanh'
output_layer_act='sigmoid'
no_epochs=100


# ## Model creation

# In[ ]:


model = Sequential()


# ## Layers Addition

# In[ ]:


model.add(Dense(hidden_units, input_dim=8, activation=hidden_layer_act))
model.add(Dense(hidden_units, activation=hidden_layer_act))
model.add(Dense(1, activation=output_layer_act))


# ## Compilation

# In[ ]:


sgd=optimizers.SGD(lr=learning_rate)
model.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['acc'])


# ## Fitting the model
# 

# In[ ]:


train.head()


# In[ ]:


train_x=train.iloc[:,1:9]
train_x.head()


# In[ ]:


train_y=train.iloc[:,9]
train_y.head()


# In[ ]:


model.fit(train_x, train_y, epochs=no_epochs, batch_size=len(train),  verbose=2)


# ## Predicting the model

# In[ ]:


test_x=test.iloc[:,1:]
predictions = model.predict(test_x)


# In[ ]:


predictions


# In[ ]:


rounded = [int(round(x[0])) for x in predictions]
print(rounded)


# ## Submission

# In[ ]:


subm.diabetes=rounded
subm.to_csv('submission.csv',index=False)


# In[ ]:




