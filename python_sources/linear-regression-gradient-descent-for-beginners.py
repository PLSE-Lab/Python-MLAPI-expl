#!/usr/bin/env python
# coding: utf-8

# **Linear Regression** is an approach to model the relationship between two continous variables(in this case x(distance travelled) and y(amount of calories burnt)).
# Its a supervised learning approach, where a single line is selected out of numerous lines with differenet slopes and y-intercept values, which have the least error or distance between points. Just like shown in the figure
# ![](http://www.sthda.com/english/sthda-upload/images/machine-learning-essentials/linear-regression.png)
# 
# Here, we used gradient descent algorithm to find the best values for both slope and y-intercept to acheive least error
# 
# For better explanation [YouTube Link](https://www.youtube.com/watch?v=XdM6ER7zTLk) check out this link.
# 
# **This kernel instead of using in-built libraries, implements the whole algorithm in python**
# 
# **Don't forget to upvote**

# These are python libraries that are useful of several mathematical computations.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Code starts from the bottom of the kernel. I will clear this issue in few days**

# This function calculates the mean of sum of squares of errors in our dataset. Here, we used mean sum squared error/loss function, which somewhat looks like this
# ![MSE formula](https://cdn-images-1.medium.com/max/800/1*20m_U-H6EIcxlN2k07Z7oQ.png)

# So, if we look closely to this line of our code.
# ![](https://i.imgur.com/i941VYA.jpg)
# 

# We will find that it looks familiar with our previous mean square error formula as **mx+b** part indicates the predicted value of y for the specific value of x, lets's say this value as y' (y-dash) and  the **y** part indicates the value of y for x. 

# So the error will calculate this part of space
# ![](http://wiki.fast.ai/images/5/55/Linear_line_w_cost_function.png)

# In[ ]:


def errorForGivenPoints(b,m,points):
    totalErrors = 0
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        totalErrors += (y- (m * x) + b) ** 2
    return(totalErrors/float(len(points)))


# In this function, we found the partial derivaitve *w.r.t*  b and m of MSE formula and use them to calculate new values for b and m. In order to calculate the least error value, we have to derive the MSE formula and as it is a total hit and try method (remember we are not using backpropogation here). We have to search the whole result all by ourselves. As you can see in the image, the local minima is the point where the least amount of error is occuring. So, we are searching that point in this algorithm.
# ![Gradient display](https://spin.atomicobject.com/wp-content/uploads/gradient_descent_error_surface.png)
# 
# This is the most easy example to show the least error is occuring in the minima
# ![](https://cdn-images-1.medium.com/max/1600/0*rBQI7uBhBKE8KT-X.png)
# 
# 
# This image proves that whenever the ball moves to a minima, the MSE error is minimum
# ![](https://cdn-images-1.medium.com/max/800/1*E-5K5rHxCRTPrSWF60XLWw.gif)

# In[ ]:


def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += (-2/N) * (y - ((m_current * x) + b_current))
        m_gradient += (-2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return([new_b,new_m])


# Both b and m values start from 0 and will iterate for already specified number (num_iterations). This function is calling another function that is going to do some serious calculations (basically, implementing the whole gradient descent algorithm) and then returning both the b and m values back to run() function.

# In[ ]:


def gradient_descent(points,initial_b,initial_m,learning_rate,num_iterations):
    b = initial_b
    m = initial_m
    for i in range(num_iterations):
        b,m = step_gradient(b, m, np.array(points), learning_rate)
#         print(errorForGivenPoints(b,m,points))
#         print("value of {0} and {1}".format(b,m))
    return [b,m]


# Here, **genfromtxt()** function takes two params, namely, the path of csv file and delimiter. Delimiter helps in splitting the data. By default, **genfromtxt()** uses *dtype = float*, so you should be careful while accessing the table.
# 
# **learning_rate** is a hyper-parameter that completely matches with the tuning knob of a radio but in this case we are considering model. **learning_rate** defines how fast our model learns. So, if the learning rate is too low, our model will never come to a conclusion and if its too high, our model will lose the most optimum solution. *I randomly selected (0.001) this value, you can select your own and check the model efficiency and let me know in the comment section*
# ![Difference b/w small and large learning rates](https://cdn-images-1.medium.com/max/1600/0*QwE8M4MupSdqA3M4.png)
# 
# Linear regression depends on the basic formula of "y = mx + b", where m is the slope of the line and b is the y-intercept. We set both the b and m values as zero because we want to let our model to learn optimum the slope and the y-intercept over time.
# 
# **num_iterations** is the variable to store total number of iterations that we want our model to run in order to find an optimum solution. Here, we set 1000 because our dataset contains only 100 rows(small size).
# 
# 
# 

# In[ ]:


def run():
    points = np.genfromtxt('../input/data.csv',delimiter=',')
    learning_rate = 0.0001 # hyper-parameter
#      y = mx + b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b,initial_m,errorForGivenPoints(initial_b,initial_m,points)))
    print("Working...")
    [b,m] = gradient_descent(points,initial_b,initial_m,learning_rate,num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error= {3}".format(num_iterations,b,m,errorForGivenPoints(b,m,points)))


# ***CODE WILL START FROM THIS POINT***
# 
# Since, there is no main() function in python, execution of the code starts from the indentation level 0 (zero) from top to bottom. To avoid this, we can use **__name__**, a **special buit-in variable** in python, which stores the name of current module.

# In[ ]:


if __name__ == '__main__':
    run()

