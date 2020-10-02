#!/usr/bin/env python
# coding: utf-8

# # Larn Yersel Python
# ## Part 4 External Libraries

# Python is most often used as a *scripting language*. A language that is designed to pull together commands from external libraries of functions. Advanced users can create their own libraries, but there are many pre-written libraries available. Some of them are pre-installed with a Python installation. The ones you get depend on which version you install. Kaggle has pre-installed many of the most popular mathematical, scientific and statistical libraries. This naotebook will introduce you to how to use libraries and to a few of the functions available. It is not a comprehensive set of documentation for any of these libraries. Any search engine will give you lots to consult.

# Our first example is from the `math` library. This comes with even the most basic installation and contains lots of useful standard functions like trig functions, logs and exponentials etc.
# 
# Before we can use any library we have to import it. After that we can use the functions by preceding them with the library name and a dot.

# In[ ]:


import math

print(math.pi)
print(round(math.pi,3))
for i in range(10):
    x = round(i*math.pi/10,3)
    y = round(math.sin(x),3)
    print(x,y)


# We can now use functions from the math library anywhere in this notebook without putting an import statement every time.

# In[ ]:


x = math.cos(math.radians(60))
print(round(x,1))


# If you're doing numerical work there will be a lot of rounding of decimal values needed to produce nice output. Generally, your output needs some words of explanation - i.e. a string. Python allows you to insert values into a string for printing. You can call the `.format()` function on your string. It will insert any arguments in order into the slots indicated by `{}`. Here is a simple example.

# In[ ]:


x = 3.25789
y = 2.841598
print('The coordinates are({},{})'.format(x,y))


# You can tidy this up by indicating how each slot is to be formatted. There are different formatting options for different types of variables. Floats can be formatted with a specified number of places after the decimal and a specific total width. Here are a few examples.

# In[ ]:


print('Here is x to 2dp--{:.2f}'.format(x))
print('Here is x to 3dp--{:.3f}'.format(x))
print('Here is x with total width 6 and 2dp--{:6.2f}'.format(x))


# If we plan to use a library in lots of places it can be annoying to have to type the full name every time. Python allows us to specify a shortcut. This next example re-imprts the math library but this time using a shortcut name.

# In[ ]:


import math as m
print('  x  | cos(x) ')
print('--------------')  
for i in range(11):
    x = i*m.pi/10
    y = m.cos(x)
    print('{:.2f} | {:5.2f}'.format(x,y))


# Kaggle includes the powerful graphing library `matplotlib`. This is huge (over 70000 lines of code) and very complicated. We will use a sublibrary called `pyplot` and give it a handy shortcut:

# In[ ]:


import matplotlib.pyplot as plt


# Here is a simple example. It takes a Python list of x coordinates and a Python list of y coordinates and joins them with straight lines.

# In[ ]:


plt.plot([1,2,3,4],[5,7,4,8])
plt.show()


# We could use this to make a really low res graph of a function. The `label` attribute, together with the `legend()` function defines and creates a key.

# In[ ]:


x = [i*m.pi/10 for i in range(21)]
y = [m.cos(i*m.pi/10) for i in range(21)]
plt.plot(x,y,label='y=cos(x)')
plt.legend()
plt.show()


# Now lets add a second graph. The extra commands here should be obvious.

# In[ ]:


x = [i*m.pi/10 for i in range(21)]
y1 = [m.cos(i*m.pi/10) for i in range(21)]
y2 = [m.sin(i*m.pi/10) for i in range(21)]
plt.plot(x,y1,label='y=cos(x)')
plt.plot(x,y2,label='y=sin(x)')
plt.xlabel('x in radians')
plt.ylabel('y')
plt.title('Graphs of sin, cos and tan')
plt.legend()
plt.show()


# The default behaviour is to put the graph inside an empty box with the axes at the left and bottom. This can be changed with some hard work. The `subplots` function returns a `Figure`, which is the actual graph object and a set of `axes`

# In[ ]:


x = [i*m.pi/10 for i in range(21)]
y1 = [m.cos(i*m.pi/10) for i in range(21)]
y2 = [m.sin(i*m.pi/10) for i in range(21)]
# Here is the key command
fig, ax = plt.subplots()
# This sets the axis positions to be centred where the data is equal to 0.0
ax.spines['left'].set_position(('data', 0.0))
ax.spines['bottom'].set_position(('data', 0.0))
# This hides the rest of the outside box
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# We plot to the axes (this allows us to superimpose graphs with different axes)
ax.plot(x,y1,label='y=cos(x)')
ax.plot(x,y2,label='y=sin(x)')
# Specify the box limits
plt.xlim(-1,7)
plt.ylim(-1.5,1.5)
# Add Labels and position at the ends of the axes
plt.xlabel('x in radians',horizontalalignment='right', x=1.0)
plt.ylabel('y', rotation=0, horizontalalignment='right', y=1.0)
plt.title('Graphs of sin and cos')
# This locates the legend 3/20 of the way in and 1/10 of the way up
plt.legend(loc=(0.15,0.1))
plt.show()


# In[ ]:





# Some of the code above will be needed for every graph. We can type less by creating a handy function.
# 
# We need to use the "set" functions to define limits etc. directly for a set of axes. This is because the `axes` come directly from matplotlib, not the nicer pyplot library.
# 

# In[ ]:


def makeaxes(axes,xmin,xmax,ymin,ymax):
    axes.spines['left'].set_position(('data', 0.0))
    axes.spines['bottom'].set_position(('data', 0.0))
    axes.spines['right'].set_color('none')
    axes.spines['top'].set_color('none')
    axes.set_xlim(xmin,xmax)
    axes.set_ylim(ymin,ymax)


# The graph is much nicer than before, but still a little jagged. We could tidy it up by adding lots more points but the `numpy` library contains a nice function to automatically generate lists with values that may be decimals. The `numpy` functions look like the `math` functions, but can be applied to lists directly.

# In[ ]:


import numpy as np
# This creates a list of points to plot, spaced 0.1 apart
x = np.arange(0, 3 * np.pi, 0.1) 
# Now apply numpy functions to our list
y1 = np.cos(x) 
y2 = np.sin(x)
# Now the formatting stuff
fig, ax = plt.subplots()
makeaxes(ax,-0.5,3*np.pi,-1.5,1.5)
# Make the ticks nice
plt.xticks(np.arange(0, 3 * np.pi+0.5, np.pi/2),['0','$\pi$/2','$\pi$','$3\pi$/2','$2\pi$','$5\pi$/2','$3\pi$'] )
# Add Labels and position at the ends of the axes
plt.xlabel('x in radians',horizontalalignment='right', x=1.0)
plt.ylabel('y', rotation=0, horizontalalignment='right', y=1.0)
plt.title('Graphs of sin and cos')
# Now plot
ax.plot(x,y1,label='y=cos(x)',color='red')
ax.plot(x,y2,label='y=sin(x)',color='purple')
# This locates the legend 1/10 of the way in and at the bottom
plt.legend(loc=(0.1,0))
plt.show()


# Here is an example with a `for` loop to plot lots of graphs on the same axes. There's no legend because it would be huge.

# In[ ]:


x = np.arange(-2, 2, 0.1) 
# Now the formatting stuff
fig, ax = plt.subplots()
makeaxes(ax,-2,2,-3,3)
plt.xlabel('x',horizontalalignment='right', x=1.0)
plt.ylabel('y', rotation=0, horizontalalignment='right', y=0.95)
plt.title('Graphs of $y=ln(x^2+c)$ for $0\leq c<\leq 3$')
# Calculate y values and plot
for c in np.arange(0,3,0.1):
    y = np.log(x**2+c)
    ax.plot(x,y, color=(c/3,0,1-c/3))

plt.show()


# To finish off this brief demo, here is a plot of several graphs side by side. 
# 
# This example draws vertical cross sections of the graph $z=x^3-3xy^2$ for various values of $y$. Note the use of dollar signs within the strings to force matplotlib to use Tex-style formatting.

# In[ ]:


# This line changes the maths font to Computer Modern
plt.rc('mathtext', fontset="cm")
# This sets up an array of graphs. The variable "axes" is an array of the axes for each subgraph
fig, axes = plt.subplots(nrows=4, ncols=5,figsize=(15,15))
# Add a main title
fig.suptitle('Cross Sections of $z=x^3-3xy^2$',fontsize=18, y=1.03)
# We use the same x values for every graph
x = np.arange(-2,2,0.1)
# Now use nested loops to set up the graphs
for row in range(4):
    for col in range(5):     
        y = row-2+col/5
        # Numpy lists can be used in arithmetic
        z = x**3-3*x*y**2
        # extract the correct set of axes and use it to plot
        subplt=axes[row,col]
        makeaxes(subplt,-2,2,-8,8)
        subplt.set_xlabel('$x$',x=1.0)
        subplt.set_ylabel('$z$',y=1.0, rotation=0)
        subplt.set_title('$y={:.1f}$'.format(y), loc='right')
        subplt.plot(x,z)
        
fig.tight_layout()
plt.show()

    

