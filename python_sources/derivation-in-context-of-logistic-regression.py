#!/usr/bin/env python
# coding: utf-8

# # Understanding Derivation in Context of Logistic Regression:
# This notebook is made for the purpose of refreshining or learning the concept  of derviaton/partial derviation and use these concepts to apply at Regression model, and demonstrate simple approach to understand how excatly the Regression learns to predict the actual output. The work contains three steps as follows:
# 1. Concepts of Derivation
# 2. Walking through step by step the Regression w.r.t its parameters update with Gradient Update
# 3. Discuss convergance of Model.

# ## Derivation:
# We may say derivation is the "fancy" word for the concept slope of the line. The difference in between is that in derivation you are considering the change in slope values when there is infinitestimal change(infintiy small change). 
# To understand how derivatives work, we have provided two examples of addition and multiplication of function that have tow variables(x,y). Let's consider addition first. Below is solved example of addition:
# 
# ![Partial Derivate for Addition](https://github.com/hamzafar/deep_learning_toys/blob/master/images/Derivation%20in%20context%20of%20Logistic%20Regression/i.jpg?raw=true)
# 
# When we are dealing with two vairables, then we take derivatives in flow like we consider one variable as constant and take derivative w.r.t to other variable (Known as partial derivative). In the below figure we first take partial derivate w.r.t x and then y and given intuation how equation is solved:
# 
# ![Description of Addition](https://github.com/hamzafar/deep_learning_toys/blob/master/images/Derivation%20in%20context%20of%20Logistic%20Regression/1.jpg?raw=true)
# 
# Now, let's take example of Multiplication. The same process is followed as in the case of addition above i.e. we first consider x to take partial derivative and y.  The equation and its explanation is shown in figure below:
# ![Partial Derivate for Multiplication](https://github.com/hamzafar/deep_learning_toys/blob/master/images/Derivation%20in%20context%20of%20Logistic%20Regression/j.jpg?raw=true)
# 
# ![Description of Addition](https://github.com/hamzafar/deep_learning_toys/blob/master/images/Derivation%20in%20context%20of%20Logistic%20Regression/2.jpg?raw=true)
# 

# ### Computational Graph and Chain Rule:
# So far we have considered simple equations with only two variables, what if we do have equation of more than two vaiables? 
# \begin{equation*}
# h(a, b, c) = (a * b) + c\\
# \end{equation*}
# To proceed with equations, we use computation graph. The structure of graph is simple, we place variables at input positions and operators in between the graph. And the function output is achieved at the end of graph. This will be more clear viewing below figure:
# ![Forward Pass](https://github.com/hamzafar/deep_learning_toys/blob/master/images/Derivation%20in%20context%20of%20Logistic%20Regression/3.jpg?raw=true)
# In figure above, it can easily be seen that how mathematical operations are perfromed in sequence steps; like a and b are  first multiplied and the result of them is added to c. We may give any name to intermidate state, that might be helpful in derivation steps. Below is the sort equation we can form:
# \begin{equation*}
# h(a, b, c) = (a * b) + c\\
# (a * b) = tmp \\
# h(a, b, c) = tmp + c\\
# \end{equation*}
# 
# 
# Now let's take derivate; but this time we will start operation from right to left, considering each operation step by step. The first step would be working on '+' operation and then 'multipy'.  The following is calculation for partial deriates of function 'h' w.r.t. variables 'a, b, and c'.
# 
# 
# \begin{equation*}
# \frac{\partial h}{\partial c} = \frac{\partial h}{\partial c}(tmp +c)\\
# \frac{\partial h}{\partial c} =1\\
# \frac{\partial h}{\partial tmp} = \frac{\partial h}{\partial tmp}(tmp +c)\\
# \frac{\partial h}{\partial tmp} =1
# \end{equation*}
# 
# 
# For determining partial derivative of a and b we will be using chain rule. And it states that multiply all previous derivates to new. Thus the partial dervatives of a and b will be computed as follows
# \begin{equation*}
# \frac{\partial h}{\partial a} = \frac{\partial h}{\partial tmp}* \frac {\partial tmp}{\partial a}\\
# \frac{\partial h}{\partial b} = \frac{\partial h}{\partial tmp} * \frac{\partial tmp}{\partial b}\\
# \end{equation*}
# 
# The values of paritial derivate of h w.r.t 'a' and 'b' are computed as follows:
# 
# \begin{equation*}
# \frac{\partial h}{\partial a} = \frac{\partial h}{\partial tmp}* \frac {\partial tmp}{\partial a}\\
# \frac{\partial tmp}{\partial a} = \frac{\partial tmp}{\partial a}(a * b)\\
# \frac{\partial tmp}{\partial a} = b\\
# \frac{\partial h}{\partial a} = 3 * 1 = 3 :[{\partial a} = 3; {\partial tmp} = 1]\\
# \end{equation*}
# 
# \begin{equation*}
# \frac{\partial h}{\partial b} = \frac{\partial h}{\partial tmp} * \frac{\partial tmp}{\partial b}\\
# \frac{\partial tmp}{\partial b} = \frac{\partial tmp}{\partial a}(a * b)\\
# \frac{\partial tmp}{\partial b} = a\\
# \frac{\partial h}{\partial b} = 2 * 1 = 2 :[{\partial a} = 2; {\partial tmp} = 1]\\
# \end{equation*}
# 
# The diagram of derivatives below can validate the above concepts.
# ![](https://github.com/hamzafar/deep_learning_toys/blob/master/images/Derivation%20in%20context%20of%20Logistic%20Regression/4.jpg?raw=true)
# 
# You may find further resources about derivation containing rules, calculator of derivator and more examples of derivation  in links below:
# 
# * [CS231n Backpropagation](http://cs231n.github.io/optimization-2/)
# * [Derivate Rules](https://www.mathsisfun.com/calculus/derivatives-rules.html)
# * [Derivation Calculator](https://www.derivative-calculator.net/)
# 
# **I would highly recommend to have a look on the resources and also try to understand sigmoid function presented in the CS231n Backpropagation.**

# ## Regression ##
# In this section, the regression is shown in Computational Graph to described how to compute loss and using that loss calculate gradient and update its wight.  For sake of simplicity we have considered only one training example with two feature set i.e. two inputs x1, x2. These input values are then used in xor function to generate output  (target value 'y'). The generated output value 'y' is used to calculate the loss by comparing this value to the activation generated through regression computation graph.
# 

# In the following computational graph, the two feature set i.e. *x1* and *x2* are multiplied by weights *w1* and *w2* and their product is shown in *d1* and *d2* states. The bias *b* is  added to both *d1* and *d2* state yielding the z state. To get the activation of *z*; the sigmoid function is applied and finally the loss is computed with actual value of xor function (using same values of x1 and x2) *y* and the activation value *a*. We may say this whole process as **Forward Pass**.
# ![](https://github.com/hamzafar/deep_learning_toys/blob/master/images/Derivation%20in%20context%20of%20Logistic%20Regression/5.png?raw=true)

# let's implement the above graph with using random generated weights and bias and see how activation is different from the actual value.
# 

# In[12]:


# This is Python 3 environment 
# Call packages for onward use

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # ploting graph


# In[13]:


def sigmoid(z):
    # Takes input as z and return sogmoid of value
    s = 1 / (1 + np.exp(-z))
    return s


# In[14]:


# gnerate data with two feature set and get label of xor function
x1, x2 = 1, 1
y = int(np.logical_xor(x1,x2))
print('actual value(y): ', y)

# define paramters i.e. weights and bias to random
w1, w2, b = 0.1, 0.5, 0.005

# print('Parameters Before update')
# print('w1: ', w1, 'w2: ', w2, 'b: ', b)

z = w1*x1 + w2*x2 + b

#activation of values 
a = sigmoid(z)
print('activation value: ', a)

# compute the loss of the function(since we have training exmple equal to 1, so cost==loss )
cost = -1 * (y * np.log(a) + (1 - y) * (np.log(1 - a)))  # compute cost
print('loss of function: ', cost)

# Store cost, activation, weights and bias to dictionary
my_dic = {}
my_dic['w1'] = w1
my_dic['w2'] = w2
my_dic['b'] = b
my_dic['activation'] = a
my_dic['cost'] = cost


# We can see that the actual value is *0* is far away from the activation value *0.65* and cost of this function is much higher. Now let's see how gradient upadate the weight and affects the loss(reduce) to get more closer value of activation to the actual value.  Before implementing code, let's do some maths stuff to see how greadients of weights & bias accroding to loss can be calcalulated. 
# 
# In this step, we will go from right to left opposite to Forward pass and we may say it **Backward Pass**

# ![](https://github.com/hamzafar/deep_learning_toys/blob/master/images/Derivation%20in%20context%20of%20Logistic%20Regression/6.jpg?raw=true)

# \begin{equation*}
# \frac{\partial loss}{\partial a} = \frac{-y}{\ a} - \frac{1-y}{1- a}   :>[da] \\
# \frac{\partial loss}{\partial z} = \frac{\partial loss}{\partial a} * \frac{\partial a}{\partial z} :>[dz]\\
# \frac{\partial a}{\partial z} = (1-a)*a <refer-to: cs231n>\\
# \frac{\partial loss}{\partial z} = (a-y) :>[dz]\\
# \frac{\partial loss}{\partial b} = \frac{\partial loss}{\partial z} * \frac{\partial z}{\partial b} :>[db]\\
# \frac{\partial z}{\partial b} =1<sum-rule>\\
# \frac{\partial loss}{\partial b} = (a-y) :>[db] ...(1)\\
# \frac{\partial loss}{\partial d1} = (a-y) :>[dd1]\\
# \frac{\partial loss}{\partial d2} = (a-y) :>[dd2]\\
# \frac{\partial loss}{\partial w1} = \frac{\partial loss}{\partial d1} * \frac{\partial d1}{\partial w1} :>[dw1]\\
# \frac{\partial d1}{\partial w1} = x1<product-rule> \\
# \frac{\partial loss}{\partial w1} = x1 * (a-y) :>[dw1]...(2)\\
# \frac{\partial loss}{\partial w2} = x1 * (a-y) :>[dw2]...(3)\\
# \end{equation*}

# The Equation no 1, 2 and 3 are equations above; determines the gradients of weights and bias w.r.t loss. Their code implementation is given below:

# In[15]:


# BACKWARD PROPAGATION (TO FIND GRAD)
dw1 = x1*(a-y)
dw2 = x2*(a-y)
db = a - y 


# The gradients of weights and bias are used to update the initial weights and bias. In this step we multiply gradient with a learning rate(this controls the graident upadate contribution in the parameters weights and bias).  Why we subtract the product of gradient and learning rate from the previous parameters? following video of Prof. Andrew Ng will give you more intuation about the calculations:

# In[16]:


from IPython.display import HTML

# Youtube
HTML('<iframe width="800" height="400" src="https://www.youtube.com/embed/8mS1DlibKbI" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>')


# In[17]:


# update parameter w1, w2 and b by the equation

lr =0.07 # learning rate

w1 = w1 - (lr*dw1)
w2 = w2 - (lr*dw2)
b = b - (lr*db)


# In[18]:


z = w1*x1 + w2*x2 + b

#activation of values 
a = sigmoid(z)
print('activation value before update: ', my_dic['activation'])
print('activation value after update: ', a)
print('')

# compute the loss of the function(since we have training exmple equal to 1, so cost==loss )
cost = -1 * (y * np.log(a) + (1 - y) * (np.log(1 - a)))  # compute cost
print('loss of function before update: ', my_dic['cost'])
print('loss of function after update: ', cost)


# From the above result we can easily recognize the change in activation and loss in one step **Backward Pass**. Before performing update in weights, we are getting higher loss while in one step update we yield in smaller loss than before. Let's do update multiple time to see how the weight and bias update effect the loss. 

# ### Update Paramters w and b multiple times:
# In this section we will update pramaters weights and bias multiple times by looping over the process we have followed above and see the results, following are the steps:
# 1. Define the function that will compute activation and loss
# 2. Define the function to update the paramaters
# 3. Define the function plot the results
# 4. iterate over the above function multiple times and plot the results

# In[19]:


def get_activation_loss(x1, x2, w1, w2, b):
    # this function compute activations, cost and z
        # x : input features
        # w : weight
        # b : bias
    z = w1*x1 + w2*x2 + b

    #activation of values 
    a = sigmoid(z)

    # compute the loss of the function(since we have training exmple equal to 1, so cost==loss )
    cost = -1 * (y * np.log(a) + (1 - y) * (np.log(1 - a)))  # compute cost
    
    return(a,cost, z)


# In[20]:


def update_paramters(x1, x2, w1, w2, b, a, y, lr):
    # This function computes gradient of parmaters and then update them
    # returns upadated parameters weights and bias
        # x: input features
        # w: weights
        # b: bias
        # a: activation
        # y: actual label
        # lr: learning rate
      
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw1 = x1*(a-y)
    dw2 = x2*(a-y)
    db = a - y 

    # update parameter w1, w2 and b by the equation

    w1 = w1 - (lr*dw1)
    w2 = w2 - (lr*dw2)
    b = b - (lr*db)
    
    return(w1, w2, b)


# In[21]:


def plt_res(lst, ylab, lr):
    #This will plot the list of values at y axis while x axis will contain number of iteration
    #lst: lst of action/cost
    #ylab: y-axis label
    #lr: learning rate
    plt.plot(lst)
    plt.ylabel(ylab)
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(lr))
    plt.show()


# In[22]:


# define paramters i.e. weights and bias to random to new random values
w1, w2, b = 0.04, 0.75, 0.0015
lst_cost = []
lst_activation = []
#In code below, we will update paramets about 3500 times.
num_iter = 3500
lr = 0.007

# gnerate data with two feature set and get label of xor function
x1, x2 = 1, 0
y = int(np.logical_xor(0,1))

print ('x1: ', x1, '; x2: ',x2)
print('xor value(y): ', y)

for i in range(num_iter):
    a,cost,z = get_activation_loss(x1, x2, w1, w2, b)
#     print('cost at iteration', i,': ', cost)
#     print('activation at iteration', i,': ', a)
    w1, w2, b = update_paramters(x1, x2, w1, w2, b, a, y, lr)
    lst_cost.append(cost)
    lst_activation.append(a)

plt_res(lst_cost, 'loss', lr)
plt_res(lst_activation,'activation', lr)
    


# In the above graphs, we have plotted loss and values to y axis with iteration on the x-axis.  It can be seen easily that in initial iteration steps we are having activation values that are far away from the actual value y thus resulting in higher loss value. As we go through updating the parameters weights and bias we are achiving activation values closer to y i.e. 1 and also error near to 0.
# 

# ---

# 
