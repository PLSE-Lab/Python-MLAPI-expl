#!/usr/bin/env python
# coding: utf-8

# # A Guide to Mathematics of Neural Networks
# ## Agenda
# After completing this tutorial, you will have an idea about:
# 
# 1. activation functions
# 2. general working principles of a neural network
# 3. feedforward algorithm
# 4. backpropagation algorithm
# 5. gradient descent (non-stochastic and non-adaptive)
# 
# In the last section, all of these concepts are merged together and created a complete working neural network.
# 
# ## Acknowledgements
# I would like to thank **Assoc. Prof. Murat Karakaya** for his great efforts in the preparation of this tutorial. I also want to thank other researchers who help and contribute to preparation of this work. You can find the Kaggle profiles each of them at the end of the tutorial.
# 
# This tutorial is heavily based on **Tariq Rashid**'s excellent book [Make Your Own Neural Network 1st Edition](https://www.amazon.com/Make-Your-Own-Neural-Network/dp/1530826608/ref=sr_1_1?ie=UTF8&qid=1548348196&sr=8-1&keywords=tariq+rashid). I definitely recommend the book since it provides the mathematics of neural networks in the simplest way I've ever seen.
# 
# ## Simplest Computation Machine
# A compuatation machine or module is a simple black box that inputs a thing (more specifically a scalar, matrix, etc.), makes some calculations using the input(s) and outputs the result. The computation step is almost always a predefined algorithm. Actually, this is the limiting part of every computation machine. Let's make an example for such a simple machine.

# In[ ]:


# This is our computation machine.
# It takes an input and constructs and output by multiplying the given input with 4.
def multiply(number):
    return 4 * number

# Let's use it.
output = multiply(3)

# Print the result to screen.
print(output)


# The above example might be the simplest ever computation machine. However, it proves the point that the algorithm used to generate the output is fixed. It will never change whatever the input is. The following image represents this type of machines.
# 
# ![A black-box model](https://i.ibb.co/vB4SqQ5/black-box.png)
# 
# Our next example will a little bit more complicated than the previous one. It is a machine that converts kilometers to miles (1). The relationship between kilometers and miles is **linear**. Linear means the change on value for either of them directly affects the value of other one. For example, if you double the number in miles, the same distance in kilometers is also doubled.
# 
# This relationship is very similar to a line's equation in a 2-dimensional space. Equation of a line in a 2-dimensional space is defined as the following equation.
# $$y = m*x + b$$
# Now, let's write the equation for the conversion.
# $$miles = kilometers * c$$
# If you take the $b = 0$ for this example (the line will always pass through the origin), both equations are basically the same. In conversion equation, miles depends on kilometers times a constant $c$. But we don't know the constant at this time. Although we don't know the constant, we know some true km-miles pairs. Let's say we know the following ones.
# 
# | Kilometers | Miles |
# | -------------- | ------- |
# | 0 | 0 |
# | 100 | 62.137 |
# 
# Table 1: Known samples for kilometers to miles.
# 
# This means that we have a baseline to compare our machine's performance. Then, what are we waiting for? Let's give a random value to $c$ and compare the output with the known ones. Here, I will also choose the same value as of the book. Let's evaluate the machine with $c = 0.5$ and see what happens.

# In[ ]:


# Define the c as a separate global variable to change it easily without redefinig the whole function.
c = 0.5

# First define our machine with the decided values.
def kilometers_to_miles(kilometers):
    miles = kilometers * c
    return miles

# Now evaluate the machine and print the result.
output = kilometers_to_miles(100)
print(output)


# This gives us 50.0. Actually this is not a bad result. But we need to quantify how bad or good this result is. How far is our estimation from the real one? This is called the **error**.
# $$error = truth - calculated$$
# In our example,
# $$error = 62.137 - 50 = 12.137$$
# This result tells us we are wrong by 12.137. Let's convert this formula to a function for later use.

# In[ ]:


# The error function.
def error(truth, calculated):
    return truth - calculated

# Calculate the error of the previous calculation.
output = kilometers_to_miles(100)
err = error(62.137, output)
print("The error is", err)


# At this point, we know that increasing $c$ will also increase the output and we can say that we need to increase the output by looking the error (Hint: the error is positive). Let's take $c = 0.6$ and recalculate the output and error.

# In[ ]:


# Redefine the new c
c = 0.6

output = kilometers_to_miles(100)
err = error(62.137, output)
print("The error is", err)


# The error is much better than the previous one. We are now much more close to the solution. Maybe you can live with this error and say that the constant c is 0.6. But we are perfectionist people. Let's increase $c$ a little bit more to reduce the error. See what happens when we take $c = 0.7$.

# In[ ]:


c = 0.7

output = kilometers_to_miles(100)
err = error(62.137, output)
print("The error is", err)


# We may overshoot the correct answer, right? It may be good to increase $c$ not by 0.1 but 0.01.

# In[ ]:


c = 0.61

output = kilometers_to_miles(100)
err = error(62.137, output)
print("The error is", err)


# The error is very low. But let's try out luck with one more step, shall we?

# In[ ]:


c = 0.62

output = kilometers_to_miles(100)
err = error(62.137, output)
print("The error is", err)


# This is a much better error. It is almost 0. This process of changing the variable repeatedly bit by bit is called an **iterative approach**.
# 
# Let's create another dataset and **train** a model to make classification.
# 
# | ID | Width | Length | Bug |
# | --- | -------- | --------- | ----- |
# | 1 | 3.0 | 1.0 | ladybird |
# | 2 | 1.0 | 3.0 | caterpillar |
# Table 2: Bug dataset.
# 
# Now, let's draw the data to a scatter plot. To plotting we will use the library `matplotlib`.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

dataset = {
    "widths": [3.0, 1.0],
    "lengths": [1.0, 3.0],
    "bugs": ["ladybird", "caterpillar"]
}

plt.scatter(dataset["widths"][0], dataset["lengths"][0], c="green", s=150, label="ladybird")
plt.scatter(dataset["widths"][1], dataset["lengths"][1], c="red", s=150, label="caterpillar")
plt.xlabel("Width")
plt.ylabel("Length")
plt.legend()
plt.show()


# We can define a line that splits these two bugs from each other. The process is almost exactly same with the previous example. The equation of our line is as follows.
# $$y = Ax$$
# Let's take $A = 0.25$ and see what happens.

# In[ ]:


# To make these calculations easier and some practice purposes let's import numpy here.
import numpy as np

A = 0.25

x = np.linspace(0, 3)
y = A * x

plt.scatter(dataset["widths"][0], dataset["lengths"][0], c="green", s=150, label="ladybird")
plt.scatter(dataset["widths"][1], dataset["lengths"][1], c="red", s=150, label="caterpillar")
plt.plot(x, y, label="y = 0.25 * x")
plt.xlabel("Width")
plt.ylabel("Length")
plt.legend()
plt.show()


# You can clearly see that this is not a good classifier. The line cannot separate those two cute bugs. It is clear that when we rotate the line a little bit up, the classifier will work just fine. However, we want to find an algorithm to do that for us. But before move on let's make some error calculations.

# In[ ]:


y = A * dataset["widths"][0]

err = error(dataset["lengths"][0], y)
print("The error is", err)


# You can see that our error is different that 0. We can use this error to inform our $A$. Let's make a just enough classifier by targeting $y = 1.1$ instead of $y = 1$. Let's recalculate our error according to this.

# In[ ]:


y = A * dataset["widths"][0]

err = error(1.1, y)
print("The error is", err)


# We can generalize the idea with the following equation.
# $$t = (A + \Delta A)x$$
# $t$ is our target value, $\Delta A$ is a small change in $A$. Let's plug these equations to our previous error equation and call error as $E$.
# $$E = target - actual = Ax + (\Delta A)x - Ax$$
# hence,
# $$E = (\Delta A)x$$
# This part is very important since we make a connection between error and a small change in $A$. You can extract the $\Delta A$ from the equation in terms of $E$.
# $$\Delta A = \frac{E}{x}$$
# Some people here may end up with a conclusion as the change in $A$ is proportional to the change in error as $x$ changes and they are indeed right. Let me give a hint for the future. This definition is very similar to the definition of the derivative. But, of course, there are slight but fundamental differences.
# 
# You can think of $\Delta A$ is an adjuster for the slope of the line. OK, the error was $0.35$ and $x$ was $3.0$. Hence, $\Delta A = \frac{0.35}{3.0} = 0.1167$. This means that the new value of $A$ must be $0.25 + 0.1167$ since the previous value of $A$ was $0.25$ as you can see from the previous plot. Let's draw the previous plot with this new slope and see the difference.

# In[ ]:


deltaA = err / dataset["widths"][0]
A = 0.25 + deltaA

y = A * x

plt.scatter(dataset["widths"][0], dataset["lengths"][0], c="green", s=150, label="ladybird")
plt.scatter(dataset["widths"][1], dataset["lengths"][1], c="red", s=150, label="caterpillar")
plt.plot(x, y, label="y = {0:.2} * x".format(A))
plt.xlabel("Width")
plt.ylabel("Length")
plt.legend()
plt.show()


# You can see that the separator line is barely above of the green dot. It's almost a successful classifier but not quite for now. Luckily, we know another training example which is $x = 1.0$ and $y = 3.0$ also known as the *caterpillar*. Let's calculate our error for that second sample.

# In[ ]:


x = dataset["widths"][1]
y = A * x

# We expect to see that 3.0
print(y)


# This not very close, right? So, let's calculate our error to update our classifier. Notice, that we want the line does not pass through the example but pass through slightly below. So, we choose $2.9$ intentionally.

# In[ ]:


err = error(2.9, y)
print("Error is", err)


# OK, this error is little bit bigger than the previous one. Let's update our $A$ with the previous adjustment equation.

# In[ ]:


# Previous A
A = 0.37
x = np.linspace(0, 3)
deltaA = err / dataset["widths"][1]
A = A + deltaA

y = A * x

plt.scatter(dataset["widths"][0], dataset["lengths"][0], c="green", s=150, label="ladybird")
plt.scatter(dataset["widths"][1], dataset["lengths"][1], c="red", s=150, label="caterpillar")
plt.plot(x, 0.37 * x, label="y = 0.37 * x")
plt.plot(x, y, label="y = {0:.2} * x".format(A))
plt.plot(x, 0.25 * x, label="y = 0.25 * x")
plt.xlabel("Width")
plt.ylabel("Length")
plt.legend()
plt.show()


# The orange line is our refined classifier. You can see that it is no better than the previous (blue) classifier.  The green line shows our initial classifier. This graph shows us a fundamental problem with this concept. If we keep training for each training sample, we are doomed to forget the previous one. So, how do we remember each of the training samples? Actually, the solution is easy. We give weights to each of the samples. The weight of effect of each sample is called **learning rate** in statistical learning (machine learning). We will refer *learning rate* as $L$ in our future examples. Let's update our previous update formula with the addition of *learning rate*.
# $$\Delta A = L\left(\frac{E}{x}\right)$$
# 
# Let's recalculate all of our previous examples with a *learning rate* $L = 0.5$ and $A = 0.25$ and in a semi-automated manner.

# In[ ]:


A = 0.25
L = 0.5
t = np.linspace(0, 3)

x = dataset["widths"][0]
y = A * x
E = error(1.1, y)
print("Error:", E)
deltaA = L * (E / x)
A = A + deltaA

plt.plot(t, A * t, label="First Iteration")
plt.scatter(dataset["widths"][0], dataset["lengths"][0], c="green", s=150, label="ladybird")
plt.scatter(dataset["widths"][1], dataset["lengths"][1], c="red", s=150, label="caterpillar")
plt.xlabel("Width")
plt.ylabel("Length")
plt.legend()
plt.show()


# In[ ]:


x = dataset["widths"][1]
y = A * x
E = error(0.9, y)
print("Error:", E)
deltaA = L * (E / x)
A = A + deltaA

plt.plot(t, A * t, label="Second Iteration")
plt.scatter(dataset["widths"][0], dataset["lengths"][0], c="green", s=150, label="ladybird")
plt.scatter(dataset["widths"][1], dataset["lengths"][1], c="red", s=150, label="caterpillar")
plt.xlabel("Width")
plt.ylabel("Length")
plt.legend()
plt.show()

print("The final value of A is", A)


# ## Neurons
# Neurons don't permit the signals pass through directly, but they wait the signals to grow or pass a **threshold** value.
# 
# >A function that takes the input signal and generates an output signal, but takes into account some kind of threshold is called an **activation function**.
# >
# > -- *Tariq Rashid*
# 
# There are many activation functions. Maybe the simplest one is the **step function** (or Heaviside function). The following Python code shows a step function in action.

# In[ ]:


def step_function(X):
    return (X > 1.0).astype(int)

x = np.linspace(-10, 10, 1000)
y = step_function(x)

plt.plot(x, y)
plt.title("Step Function")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# You can see that a neuron fires only if the input signal can pass the threshold value (of 1). By definition the step function is differentiable. The derivative of step function is the Dirac Delta function. However, if you take the derivative of step function numerically by using the definition of derivation as shown in the following equation (by approaching $h$ to $0$), you cannot get the Dirac Delta function as long as you don't choose a fairly large $h$. The following plots shows the situation in action.
# $$\frac{df(x)}{dx} = \lim_{h\to0} \frac{f(x + h) - f(x)}{h}$$

# In[ ]:


def derivative(function, evaluate, h):
    return (function(evaluate + h) - function(evaluate)) / h


# In[ ]:


plt.figure(figsize=(15,5))

y = derivative(step_function, x, 0.1)
plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.grid(True)

dy = derivative(step_function, x, 0.000001)
plt.subplot(1, 2, 2)
plt.plot(x, dy)
plt.grid(True)
plt.show()


# The solution to this problem, using a similar, non-linear and differentiable function. Luckily, we have one and it is called **sigmoid**.  The equation of sigmoid is shown below.
# $$
# y=\frac{1}{1+e^{-x}}
# $$
# You can see the both shape of the sigmoid itself and its derivative.

# In[ ]:


def sigmoid(X):
    return 1 / (1 + np.exp(-X))

plt.figure(figsize=(15,5))

y = sigmoid(x)
plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title("Sigmoid")
plt.grid(True)

dy = derivative(sigmoid, x, 0.000001)
plt.subplot(1, 2, 2)
plt.plot(x, dy)
plt.title("Derivative of Sigmoid")
plt.grid(True)
plt.show()


# From now on, I will use the `derivative()` function from the `scipy.misc` module instead of the above one. I just implement it to show you how easy to implement such a function instead of dealing with algebraic equations to get the derivative of a function. It may be less efficient however, it is much clear than the algebraic ones.
# 
# ### Feedforward Algorithm
# 
# Naturally, there are one or more inputs to a neuron and one or more outputs from a neuron. Then, **how does a neuron deal with so many inputs?** The answer is the neuron sums up all the inputs. Then, it gives the sum to the activation function and the result is called as the output. From this perspective, the operations are so simple. Let me throw some little complexity in that. **How can we connect neurons?** We connect neurons with **weights**. They are simple lines that have some numbers or coefficients in them. Most of the time weights are indicated by letter $w_{2,3}$. This means that the weights between node 2 in a layer and node 3 in the next layer. When an output of a neuron passes through a weight, it is multiplied with the **weight** and becomes an input for the next neuron. This is the crucial part of a neural network. Because learning in neural networks happen in weights. The little nudges in weights are actually called the learning process. But before moving onto the learning, let's do some examples.
# 
# Consider the following neuron.
# ![Simple Neuron](https://i.ibb.co/4TLFbxB/Untitled-Diagram.png)
# By using the previous information, calculate the output $y$.

# In[ ]:


w1, w2, w3 = 0.9, 0.3, 0.5
x1, x2, x3 = 1, 2, 0.5

y = sigmoid(x1 * w1 + x2 * w2 + x3 * w3)
print(y)


# **What if you have thousands of weights and hundreds of inputs?** Are you still calculating by hand? Of course not. We need to automate the input calculation step. Here the *linear algebra* comes to the rescue. Let's rewrite the previous example. But, this time with vectors and matrices. I will use capital letters to denote matrices.

# In[ ]:


W = np.array([0.9, 0.3, 0.5])
X = np.array([1, 2, 0.5], ndmin=2).T
y = sigmoid(W.dot(X))
print(y)


# The result is same as the above. This is the power of matrix product. If we generalize the idea here:
# 
# $$
# \boldsymbol{X_h}=\boldsymbol{W} \cdot \boldsymbol{I}
# $$
# 
# Assume that we have a 2 layer and each layer has 3 neurons. So, we have a total of 9 weights. We can write the matrices as follows:
# 
# $$
# \boldsymbol{W}=
# \begin{bmatrix}
# 0.9 & 0.3 & 0.4 \\
# 0.2 & 0.8 & 0.2 \\
# 0.1 & 0.5 & 0.6
# \end{bmatrix}
# $$
# $$
# \boldsymbol{I}=
# \begin{bmatrix}
# 0.9 \\
# 0.1 \\
# 0.8
# \end{bmatrix}
# $$
# 
# Now, let's calculate the output.

# In[ ]:


W = np.array([[0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]])
I = np.array([0.9, 0.1, 0.8], ndmin=2).T
X_h = W.dot(I)
print(X_h)


# These numbers are the inputs to the neurons of the next layer. Notice that I didn't put them into the sigmoid function before multiplying with the weights. The reason is that these numbers come from the input layer. So, I can assume that they have been already activated with an activation function. Now, we can calculate the output of the middle layer by just put $X_h$ into the activation function.

# In[ ]:


O_h = sigmoid(X_h)
print(O_h)


# That's it. We made it. This is actually all the steps necessary to **feedforward** a neural network. The next step is a little bit harder than that. So, sit tight.
# 
# ### Learning Weights - Backpropagation
# Remind our first example with simple linear classifier. We calculated the error simply by looking the difference between our value and what the value should be and adjust the slope of the line according to the error. OK, **how can we calculate an error while there is more than one contribuer to the same error?** The solution is repeating exactly the same steps with our previous *feedforward* algorithm. That means a node affects an error proportional with its weight. This idea actually opens the doors of *linear algebra* again. Because it is the same operation that we have performed as in the *feedforward* algorithm. There is a little detail that is worth to mention though. There are more than one weight that affects a single error. At this times, we get the proportion of the weight. For example, if there is two weights that affect a single error - let's say $w_{11}$ and $w_{21}$ - the proportion should be:
# 
# $$
# \frac{w_{11}}{w_{11} + w_{21}}
# $$
# 
# This gives us only one layer back. But it highly possible that there are more than one layers. So, **how can we go even more further?** We just repeat the exactly same steps with the previous one. We will distribute the newly found errors with proportional to the weights that affect them. So, let's permit *linear algebra* to talk for us here.
# 
# $$
# \boldsymbol{E_h}=\boldsymbol{W_h}^T \cdot \boldsymbol{E_o}
# $$
# 
# $$
# \boldsymbol{E_h}=
# \begin{bmatrix}
# w_{11} & w_{12} \\
# w_{21} & w_{22}
# \end{bmatrix}
# \cdot
# \begin{bmatrix}
# e_1 \\
# e_2
# \end{bmatrix}
# $$
# 
# This final touch gives us the ability to calculate the errors from end to the start. But, we don't yet learn the ability to calculate the weights.
# 
# ### Updating the Weights
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import scipy.misc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Activation function
def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def ReLU(X):
    return X * (X > 0)


# In[ ]:


# NN Class
class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, activation=sigmoid):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        self.lr = learningrate
        
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5
        
        self.activation = activation
    
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation(final_inputs)
        
        return final_outputs
    
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        
        self.who += self.lr * np.dot((output_errors * scipy.misc.derivative(self.activation, final_inputs, 0.00001)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * scipy.misc.derivative(self.activation, hidden_inputs, 0.00001)), np.transpose(inputs))
        
        return np.sqrt(np.sum(np.power(output_errors, 2)))


# In[ ]:


raw_data = pd.read_csv("../input/trainSimple.csv")
raw_data.head(20)


# In[ ]:


X = raw_data.drop(["A", "B"], axis=1)
y = raw_data[["A", "B"]]

print("First 5 X values:")
print(X.head())

print("First 5 Y values:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[ ]:


X_scalar = StandardScaler()
y_scalar = StandardScaler()

X_scalar.fit(X)
y_scalar.fit(y)

X_norm = X_scalar.transform(X_train)
y_norm = y_scalar.transform(y_train)


# In[ ]:


print("First 5 X values:")
print(X_norm[:5])

print("First 5 Y values:")
print(y_norm[:5])


# In[ ]:


print(X_norm.shape)
print(y_norm.shape)
#print(X_test.shape)
#print(y_test.shape)


# In[ ]:


input_nodes = 6
hidden_nodes = 100
output_nodes = 2
learning_rate = 0.9

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


# In[ ]:


for count in range(100):
    total = 0
    print("Count:", count)
    for i in range(len(X_norm)):
        total += nn.train(X_norm[i], y_norm[i])
    print("Train error:", total)


# In[ ]:


error = 0
X_test = X_scalar.transform(X_test)
for i in range(len(X_test)):
    y_hat = nn.query(X_test[i]).T
    y_hat = y_scalar.inverse_transform(y_hat)
    y_real = y_test.iloc[i].values
    error += np.sum(np.power(y_real - y_hat, 2))

print("Error:", error / len(X_test))

