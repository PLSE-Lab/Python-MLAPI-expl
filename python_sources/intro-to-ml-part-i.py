#!/usr/bin/env python
# coding: utf-8

# # Part I - Coding the ML ourselves 

# ## Collect data
# 
# Define some (x, y) data.
# 
# x | y
# - | -:
# 1 | 4
# 2 | 6
# 3 | 7
# 4 | 9
# 5 | 10

# In[ ]:


import numpy

x = numpy.array([1, 2, 3, 4, 5])
y = numpy.array([4, 6, 7, 9, 10])


# ## Visualize the data
# Plot the raw data.

# In[ ]:


import matplotlib.pyplot as pyplot
get_ipython().run_line_magic('matplotlib', 'inline')

fig1 = pyplot.figure()
axes1 = pyplot.axes(title='Vizualization of the data')
scatter1 = axes1.scatter(x, y)


# ## Choose a model
# It looks like the data is fairly linear, so let's use a linear model as the "brain" that will be trained to predict this data.
# 
# $y = w x + b$

# In[ ]:


def Predict(x, w, b):
    return w * x + b 


# We don't yet know the values for the parameters (w, b), so initially let them be zero.

# In[ ]:


w = 0
b = 0


# Plot the untrained model.

# In[ ]:


yp = Predict(x, w, b)
axes1.plot(x, yp, color='red')
fig1


# ## Train the model
# We need to train the model to better predict the data.  The goal of the training is to find the value of (w, b) that would minimize the prediction error.
# 
# ### Define a loss function
# The optimum solution can be found directly by defining a "**squared error**" loss function and then minimizing that loss function.
# 
# $L = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2 = \frac{1}{m} \sum_{i=1}^{m} (w x_i + b - y_i)^2$
# 
# 

# In[ ]:


def Loss(x, y, w, b):
    yp = Predict(x, w, b)
    J = (yp - y)**2      
    loss = numpy.average(J)
    return loss    


# There are several available techniques to find the minimum of a function.  We will explore three ways:
# * Find the approximate minimum visually using a graph
# * Find the exact minimum using calculus
# * Find the approximate minimum using an iterative numerical technique

# ### Find the minimum loss visually
# Compute the loss function over a range of w and b values.

# In[ ]:


ws, bs = numpy.meshgrid(numpy.linspace(0, 5, 20), numpy.linspace(0, 5, 20))
ws, bs = ws.ravel(), bs.ravel()
yp = numpy.outer(ws, x) + bs.reshape(-1, 1)
losses = numpy.average((yp - y)**2, axis=1).ravel()


# Plot the loss function.

# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

trace0 = go.Mesh3d(x=ws, y=bs, z=losses, opacity=0.5)
layout = dict(scene=dict(xaxis=dict(title='w'), yaxis=dict(title='b'), zaxis=dict(title='loss')))
fig2 = go.Figure(data=[trace0], layout=layout)
iplot(fig2)


# Search the graph points to find the point of minimum loss (brute force search).

# In[ ]:


idx = numpy.where(losses == numpy.amin(losses))
w, b, loss = ws[idx][0], bs[idx][0], losses[idx][0]
print('w:', w, 'b:', b, 'loss:', loss)


# Add a marker at that point on the plot of the loss function.

# In[ ]:


trace1 = go.Scatter3d(x=(w,), y=(b,), z=(loss,), marker=dict(size=5, color='cyan'))
fig3 = go.Figure(data=[trace0, trace1], layout=layout)
iplot(fig3)


# Plot the trained model.

# In[ ]:


yp = Predict(x, w, b)
axes1.plot(x, yp, color='cyan')
fig1


# ### Minimize the loss function using calculus
# To minimize the loss function we set the partial derivitives of the loss function with respect to w and b equal to zero...
# $L = \frac{1}{m} \sum_{i=1}^{m} (w x_i + b - y_i)^2$
# 
# $\frac{\delta L}{\delta w} = \frac{2}{m} \sum_{i=1}^{m} (w x_i + b - y_i) x_i = 0$
# <br><br>
# $\frac{\delta L}{\delta b} = \frac{2}{m} \sum_{i=1}^{m} (w x_i + b - y_i) = 0$
# <br><br>
# and then solve the resulting system of two equations in two unknowns to get w and b:
# <br><br>
# $w = \frac{(\frac{1}{m} \sum_{i=1}^{m} x_i y_i) - (\frac{1}{m} \sum_{i=1}^{m} x_i) (\frac{1}{m} \sum_{i=1}^{m} y_i)}
#       {(\frac{1}{m} \sum_{i=1}^{m} x_i^2) - (\frac{1}{m} \sum_{i=1}^{m} x_i)^2}$
# <br><br>
# $b = (\frac{1}{m} \sum_{i=1}^{m} y_i) - w (\frac{1}{m} \sum_{i=1}^{m} x_i)$ 
# <br><br>
# These equations for w and b can be simplified by recognizing that each summation is an average of some quantity.
# <br><br>
# $w = \frac{\text{avg}(x y) \, - \, \text{avg}(x) \, \text{avg}(y)}{\text{avg}(x^2) \, - \, \text{avg}(x)^2}
# = \frac{(24.6) - (3)(7.2)}{(11) - (3)^2} = 1.5$
# <br><br>
# $b = \text{avg}(y) - w \, \text{avg}(x) = (7.2) - (1.5)(3) = 2.7$
# 
# $x$ | $y$ | $x y$ | $x^2$
# --- | --- | ----- | ----:
# 1   | 4   | 4     | 1
# 2   | 6   | 12    | 4
# 3   | 7   | 21    | 9
# 4   | 9   | 36    | 16
# 5   | 10  | 50    | 25

# Define a function to compute the exact minimum point by using the calculus solution.

# In[ ]:


def Fit(x, y):
    xavg = numpy.average(x)
    yavg = numpy.average(y)

    xyavg = numpy.average(x * y)
    x2avg = numpy.average(x**2) 

    w = (xyavg - xavg * yavg) / (x2avg - xavg**2)
    b = yavg - w * xavg

    loss = Loss(x, y, w, b)
    
    return w, b, loss


# Call the function to get the fit.

# In[ ]:


w, b, loss = Fit(x, y)
print('w:', w, 'b:', b, 'loss:', loss)


# Add a marker at that point on the plot of the loss function.

# In[ ]:


trace2 = go.Scatter3d(x=(w,), y=(b,), z=(loss,), marker=dict(size=5, color='green'))
fig4 = go.Figure(data=[trace0, trace1, trace2], layout=layout)
iplot(fig4)


# Plot the trained model.

# In[ ]:


yp = Predict(x, w, b)
axes1.plot(x, yp, color='green')
fig1


# ### Minimize the loss function using an iterative numerical technique
# 
# The most common optimization technique used in ML is called "**Gradient Descent**".  The idea is to start anywhere on the loss surface and walk down-hill until you find the minimum point.  The **gradient** of a function is denoted using the $\nabla$ symbol, and is defined as a vector describing the slope of the function along each axis at a given point.
# 
# $\nabla L = (\frac{\delta L}{\delta w}, \frac{\delta L}{\delta b})$
# 
# $\nabla L = (\frac{2}{m} \sum_{i=1}^{m} (w x_i + b - y_i) x_i, \frac{2}{m} \sum_{i=1}^{m} (w x_i + b - y_i))$
# 
# $\nabla L = (\frac{2}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) x_i, \frac{2}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i))$
# 
# Define a function to compute the gradient.

# In[ ]:


def Gradient(x, y, w, b):
    yp = Predict(x, w, b)
    dLdw = 2 * numpy.average((yp - y) * x)
    dLdb = 2 * numpy.average(yp - y)    
    return dLdw, dLdb


# Define a function that implements the gradient descent algorithm.

# In[ ]:


def GradientDescent(x, y, gradient, alpha, max_steps, goal):
    w = 0
    b = 0
    loss = Loss(x, y, w, b)
    ws=[w]; bs=[b]; losses=[loss]
    for i in range(max_steps):
        dLdw, dLdb = gradient(x, y, w, b)
        w = w - alpha * dLdw
        b = b - alpha * dLdb
        loss = Loss(x, y, w, b)
        ws.append(w); bs.append(b); losses.append(loss)
        if loss < goal:
            break
    return ws, bs, losses


# Call the function to get the fit.

# In[ ]:


ws, bs, losses = GradientDescent(x, y, Gradient, alpha=0.01, max_steps=10000, goal=0.06)
w, b, loss = ws[-1], bs[-1], losses[-1]
print('w:', w, 'b:', b, 'loss:', loss, 'steps:', len(losses)-1)


# Plot the history of the gradient descent.

# In[ ]:


trace3 = go.Scatter3d(x=ws, y=bs, z=losses, marker=dict(size=2, color='blue'))
fig5 = go.Figure(data=[trace0, trace1, trace2, trace3], layout=layout)
iplot(fig5)


# Plot the trained model.

# In[ ]:


yp = Predict(x, w, b)
axes1.plot(x, yp, color='blue')
fig1


# The gradient can also be estimated numerically using $\frac{\delta L}{\delta w} \approx \frac{L(w + \epsilon, b) - L(w, b)}{\epsilon}$ and $\frac{\delta L}{\delta b} \approx \frac{L(w, b + \epsilon) - L(w, b)}{\epsilon}$  in $\lim \epsilon \to 0$ .
# <br>
# Define a function to estimate the gradient numerically.

# In[ ]:


def NumGradient(x, y, w, b):
    eps = 1E-12         
    loss = Loss(x, y, w, b)
    dLdw = (Loss(x, y, w + eps, b) - loss) / eps
    dLdb = (Loss(x, y, w, b + eps) - loss) / eps
    return dLdw, dLdb


# Call the GradientDescent function again (but using NumGradient) to get the fit.

# In[ ]:


ws, bs, losses = GradientDescent(x, y, NumGradient, alpha=0.01, max_steps=10000, goal=0.06)
w, b, loss = ws[-1], bs[-1], losses[-1]
print('w:', w, 'b:', b, 'loss:', loss, 'steps:', len(losses)-1)


# Plot the history of the gradient descent.

# In[ ]:


trace4 = go.Scatter3d(x=ws, y=bs, z=losses, marker=dict(size=2, color='purple'))
fig6 = go.Figure(data=[trace0, trace1, trace2, trace3, trace4], layout=layout)
iplot(fig6)


# Plot the trained model.

# In[ ]:


yp = Predict(x, w, b)
axes1.plot(x, yp, color='purple')
fig1


# ## Summary
# 
# The model has been trained by using the data to find the best choice for the "network parameters" (w, b) such that the model predicts the data with minimum error.
# 
# The minimum steps were the following:
# * Collect data
# * Visualize the data (optional)
# * Choose a model and write the corresponding Predict() function
# * Choose a loss function and write the corresponding Loss() function
# * Use the GradientDescent() function to train the model
# 
# Note that if you use Keras or TensorFlow, then most of the coding is done for you automatically, as shown in [Intro to ML - Part II](https://www.kaggle.com/tagoodr/intro-to-ml-part-ii).
# 
