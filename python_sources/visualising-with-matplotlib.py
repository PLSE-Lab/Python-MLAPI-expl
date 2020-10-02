#!/usr/bin/env python
# coding: utf-8

# # Visualising with Matplotlib

# A basic tutorial for noobs.

# ## ---------------------------------------------------------------------------------------------------------------------------------

# In[1]:


import matplotlib.pyplot as plt
import numpy as np


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


x = np.linspace(1,10,10)
x


# In[4]:


y = x**2
y


# In[5]:


plt.plot(x,y) 
plt.xlabel('Squares')
plt.ylabel('Numbers')
plt.title('Exponential')
# or 
#plt.plot(x,y, 'r')


# In[6]:


plt.subplot(1,2,1)
plt.plot(x,y)
plt.subplot(1,2,2)
plt.plot(y,x,'g')


# ** Object oriented Approach **

# In[7]:


fig = plt.figure()
axes = fig.add_axes([0,0,0.9,0.9]) #([from left, from bottom, width, height])
axes.plot(x,y)


# In[8]:


fig = plt.figure()
axes = fig.add_axes([0,0,0.9,0.9]) #([from left, from bottom, width, height])
axes2 = fig.add_axes([0.1,0.3,0.5,0.5])
axes.plot(x,y)
axes2.plot(x,y)
axes.set_title('Larger Plot')
axes2.set_title('Smaller Plot')


# In[9]:


fig, axes = plt.subplots(nrows = 3,ncols = 3)
plt.tight_layout()


# 
# 
# 'plt.tight_layout()' makes sure there is not overlapping over the subplots.

# In[10]:


fig ,axes = plt.subplots(nrows = 1, ncols = 2)
print(axes)

for ax in axes:
    ax.plot(x,y)


# Notice that 'axes' is iterable. 

# In[11]:


fig ,axes = plt.subplots(nrows = 1, ncols = 2)
axes[0].plot(x,y)
axes[1].plot(y,x)
axes[0].set_title('Plot 1')
axes[1].set_title('Plot 2')


# ** Figure Size and DPI**

# In[14]:


plt.figure(figsize = (8,2))
plt.plot(x,y)


# In[43]:


fig, axes = plt.subplots(nrows = 2, ncols = 1,figsize = (8,3))
axes[0].plot(x,y)
axes[1].plot(y,x)
plt.tight_layout()
axes


# In[92]:


fig.savefig('squeeze.png')


# In[134]:


fig = plt.figure()
axes = fig.add_axes([0,0,0.9,0.9])
axes.plot(x,x**3,'-r', label = 'X Squared')
axes.plot(x,x**2,'-b', label = 'X Cubed')

axes.legend(loc = 0)


# Notice that, legend() function looks for a label parameter in plot function.  
# 'legend()' takes in loc argument for the location of the legend to be placed.

# In[17]:


fig = plt.figure()
axes = fig.add_axes([0,0,0.9,0.9])
axes.plot(x,x**3,'-r', label = 'X Squared')
axes.plot(x,x**2,'-b', label = 'X Cubed')

axes.legend(loc = (0.1,0.2))


# legend() also takes in a tuple in loc parameter for manual location.

# Multiple plots in one canvas can also be plotted as  
# **plt.plot(x, y, w, z)**
# ### --------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### Plot Appearance

# In[69]:


fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
axes.plot(x,y, color = 'purple',alpha = 0.5, linestyle = '-.', linewidth = 3) # linewidth or lw


# Here **alpha** denotes the transparency  
# **linewidth** is for broadening of the plot line  
# **color** parameter can take RGB values too.

# **linestyle** chooses the style of line such as 'steps', ' - ', ' -- ' or ' : '  

# In[65]:


fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (6,6))
axes[0][0].plot(x,y, color = 'green', linestyle = '-.')
axes[0][1].plot(x,y, color = 'purple', linestyle = ':')
axes[1][0].plot(x,y, color = '#3399FF', linestyle = '-')
axes[1][1].plot(x,y, color = '#FF007F', linestyle = 'steps')
axes[2][0].plot(x,y, color = '#CCCC00', linestyle = 'dotted')
axes[2][1].plot(x,y, color = '#808080', linestyle = '-')


# In[12]:


fig = plt.figure()
axes = fig.add_axes([0,0,1,1])

axes.plot(x,y, color = 'black', ls = '--',lw = 4, marker = 'o', markersize = 20, markerfacecolor = 'pink'
         ,markeredgewidth = 4, markeredgecolor = 'red')

axes.plot(x,x*1.1, color = 'blue', ls = '--',lw = 3, marker = '+', markersize = 20, markeredgecolor = 'green')


# ** Now setting the limit of graph **

# In[130]:


x = np.linspace(0,50,50)
y= x**2
fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
axes.plot(x,y, color = 'purple') # linewidth or lw

axes.set_xlim(0,20)
axes.set_ylim(0,1500)


# In[ ]:




