#!/usr/bin/env python
# coding: utf-8

# ## Importing Lib :
# 

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
x = np.linspace(0, 5, 11)
y = x ** 2


# In[ ]:


x


# In[ ]:


y


# ## So we have two methods for plotting :
# 
# ### - Function method
# ### - Object-oriented method

# ### 1- Function method :
# ### Similar as matlab :

# In[ ]:


plt.plot(x, y)


# ### - Subplot Functional method:
# 

# In[ ]:


plt.subplot(1, 2, 1) # 1 row , 2 cols , plot number you referring to (1)
plt.plot(x, y, 'r') # plot in 1 x, y and color is red

plt.subplot(1, 2, 2) # 1 row , 2 cols , plot number you referring to (2)
plt.plot(y, x, 'b') # plot in 2 x, y and color is red


# ## 2- Object-oriented method :
# 

# In[ ]:


fig = plt.figure() # create figure

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # draw the plot 10% from the left , 10% from the bottom 80% from the width, 80% from the hieght

axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # draw the plot 20% from the left , 50% from the bottom 40% from the width, 30% from the hieght
axes1.plot(x, y)
axes2.plot(y, x)


# ### - Subplot for OO method:
# 

# In[ ]:


fig, axes = plt.subplots(nrows= 1, ncols=2) # notice here we deal with the figure like a matrix


# ### If want to draw in this subplot "OO Method"

# In[ ]:


fig, axes = plt.subplots(figsize=(12,3)) #here we control the size of fig , 12 px in x axis, 3 px in y axis.

axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title');


# ## - We can control the color and marker by matplot lib:
# 

# In[ ]:


fig = plt.figure() # create figure
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(x, y, color='purple', lw=1, ls='-', marker='o', markersize=20) #lw  isline width, ls is line style, marker of the points and its size.


# # AND THANK U 

# In[ ]:




