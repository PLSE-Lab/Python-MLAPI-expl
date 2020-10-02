#!/usr/bin/env python
# coding: utf-8

# # Understanding Local Minimum

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# 
# 
# ## $f(x) = x^4 - 4x^2 +5$

# In[ ]:


def f(x):
    return x**4-(4*x**2)+5


# In[ ]:


x = np.linspace(start=-3, stop=3, num=500) #returns evenly spaced numbers over specified interval
#print(x)


# In[ ]:


plt.xlim([-3,3])
plt.ylim([0,8])
plt.xlabel('x', fontsize=18,color='red')
plt.ylabel('f(x)',fontsize=18,color='red')
plt.plot(x,f(x));


# # Finding the lowest steep

# ### Idea : The point where the slop stops to change when we descend from the right side of the parabola

# The derivate of the function $f(x) = x^4 - 4x^2 +5$ is
#   $f(x)=4x^3- 8x$ , so creating a python function for that

# In[ ]:


def df(x):
    return 4*x**3-8*x


# In[ ]:


plt.figure(figsize=[15, 5])

#Graph for the function
plt.subplot(1,2,1)
plt.xlim([-2,2])
plt.ylim([0.5,5.5])
plt.title('Function')
plt.xlabel('x', fontsize=18,color='red')
plt.ylabel('f(x)',fontsize=18,color='red')
plt.style.use('ggplot')
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}                          
plt.rc('font', **font);
plt.plot(x,f(x));

#Graph for the slope
plt.subplot(1,2,2)
plt.title('Derivative of the function')
plt.xlabel('x', fontsize=18,color='red')
plt.ylabel('df(x)',fontsize=18,color='red')
plt.xlim([-2,2])
plt.ylim([-6,8])
plt.style.use('ggplot')
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}                          
plt.rc('font', **font);
plt.plot(x,df(x),color='red');


# #### We can find here we have two minimum points, and there is a possibility that a line can intersect three points in the derivate slope (graph 2)

# In[ ]:


def gradient_descent(derivate_func,o_start,learningrate, precision,rangevalue):

    start = o_start
    runcount =0


    x_list = [start]
    y_list = [derivate_func(start)]

    #stop the loop once we reach limit
    for i in range(rangevalue):
        previous  = start 
        #calculate the error (if the slope is very high which means it is very far from zero)
        gradient = derivate_func(previous)
        #print('slope at the point ' +str(previous)+' is '+ str(gradient))
        start = previous - learningrate*gradient

        step_size = abs(start-previous)
        x_list.append(start)
        y_list.append(derivate_func(start))
        print(step_size)
        runcount = runcount+1
        if(step_size<precision):break
            
    return start, x_list, y_list
    

 
    


# In[ ]:


local_min,list_x,list_y = gradient_descent(df,-2,0.02,0.001,1000)
print('local minimum at ', local_min)
print('number of steps ',len(list_x))


# #### Superimpose the previous outputs on the graph

# In[ ]:


plt.figure(figsize=[15, 5])

#Graph for the function
plt.subplot(1,2,1)
plt.xlim([-3,3])
plt.ylim([0,8])
plt.title('Function')
plt.xlabel('x', fontsize=18,color='red')
plt.ylabel('f(x)',fontsize=18,color='red')
plt.style.use('ggplot')
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}                          
plt.rc('font', **font);
plt.plot(x,f(x),alpha=0.8);

#Just the below two lines added to superimpose the scatter plot on the existing image
values = np.array(list_x)
plt.scatter(list_x, f(values), color='green', s=100, alpha=0.6) #scatter plot the previous results

#Graph for the slope
plt.subplot(1,2,2)
plt.title('Derivative of the function')
plt.xlabel('x', fontsize=18,color='red')
plt.ylabel('df(x)',fontsize=18,color='red')
plt.xlim([-2,3])
plt.ylim([-7,6])
plt.style.use('ggplot')
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}                          
plt.rc('font', **font);
plt.plot(x,df(x),color='red');
plt.scatter(list_x, list_y,color='blue',s=100,alpha=0.6);


# 
# #### Observation:
# we can see that, when we provide initial guess to -2, the gradient descent approaces
#  to the first slope =0 but when the intial guess is changed to zero, the gradient descent approaches
#  to the next slope = 0, so the initial guess influences which minima it might be landing.
#  
#  ****

# 

# 
