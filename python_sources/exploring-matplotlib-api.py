#!/usr/bin/env python
# coding: utf-8

# # Exploring Matplotlib API
# 
# I have used Matplotlib for a while and found the library quite confusing.  
# When I need to find out how to do something, I usually go to [Stack Overflow](https://stackoverflow.com/questions/tagged/matplotlib) (that site where you find people who read the docs for you).  
# I find examples with different methodologies and end up with different styles in my notebooks. I don't like it, so I decided to dive into Matplotlib and learn once and for all how it works.  
# 
# The reason for this diversty of approaches to set a plot is that there are two API in Matplotlib. The `pyplot` API and an `object-oriented` API. It is advise to learn the `object-oriented` API and I'll try to stick to it as much as I can in this kernel.
# 
# Why should I bother learning how to use Matplotlib when libraries like *seaborn* help us creating complex plots easily and pandas has basic plotting functions?  
# Well, both are built on Matplotlib and at the end of the day, when you want to customize your plots, you need to know a bit how it works.
# 
# I hope that this kernel will help beginners like me get the most out of this awesome plotting library.  
# 
# Finaly, if you find any mistake or wish to add something, feel free to leave me your feedback and I'll update the kernel.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('../input/train.csv')


# ## Figure
# The `figure` is the top level container that will contain everything we draw.  
# When creating a figure, we can specify its size using the `figsize` parameter. `figsize` is a *width*, *height* tuple.
# 
# ```python
#     # creating a figure with width = 8 and height = 5
#     fig = plt.figure(figsize=(8, 5))
# ```
# 
# We can also use the `figaspect` function to set the aspect ratio of our figure:
# 
# ```python
#     # create a figure that is twice as tall as it is wide
#     fig = plt.figure(figsize=figaspect(2.0))
# ```

# ## Axes
# `Axes` are the area in which the data is plotted. A `Figure` can have multiple `Axes` but `Axes` belong to one `Figure` only.  
# We can add `Axes` to a `Figure` using the `add_axes()` or `add_subplot()` methods. `Subplots` and `Axes` are the same thing.  
# `add_axes()` takes the `rect` parameter. `rect` is a sequence of floats that specifies [left, bottom, width, heights]. So the `Axes` are positioned in absolute coordinates.  
# 
# ```python
#     ax = fig.add_axes([0,0,1,1])
# ```
# 
# `add_subplots()` takes 3 integers as parameter. Those 3 numbers set the number of rows and columns and the position on the subplot in the grid: `add_subplots(ijk)` add an `Axes` in the kth position of a grid that has i rows and j columns.  
# `add_subplot()` is the easiest way to setup your layout while `add_axes()` will give you more control over the position of your `Axes`.
# 
# ```python
#     # create a new axes at the first position of a 2 rows by 3 columns grid
#     ax = fig.add_subplot(231)
# ```

# ### single Axes

# In[2]:


# create a figure
fig = plt.figure(figsize=(6, 4))

# add an axes
ax = fig.add_subplot(111)

plt.show()


# ### Multiple subplots
# We can create a figure with multiple subplots by calling `add_subplot()` for each subplot we want to create.  
# Here's an example for a 2 by 2 layout:

# In[3]:


fig = plt.figure(figsize=(6, 4))

ax1 = fig.add_subplot(221)
ax1.set_title('first subplot')

ax2 = fig.add_subplot(222)
ax2.set_title('second subplot')

ax3 = fig.add_subplot(223)
ax3.set_title('third subplot')

ax4 = fig.add_subplot(224)
ax4.set_title('fourth subplot')

fig.tight_layout()
plt.show()


# This method is not the most efficient, especially if we want to draw a lot of subplots.  
# An alternative way is to use the `plt.subplots()` function.  
# The function returns a figure and an array of axes.

# In[4]:


fig, axes = plt.subplots(nrows=2, ncols=2)

# we can now access any Axes the same way we would access
# an element of a 2D array
axes[0,0].set_title('first subplot')
axes[0,1].set_title('second subplot')
axes[1,0].set_title('third subplot')
axes[1,1].set_title('fourth subplot')

fig.tight_layout()
plt.show()


# Another way to create grid layouts is to use the `gridspec` module.  It lets us specify the location of subplots in the figure. It also makes it easy to have plots that span over multiple columns

# In[5]:


import matplotlib.gridspec as gridspec

fig = plt.figure()

# I use gridspec de set the grid
# I need a 2x2 grid
G = gridspec.GridSpec(2, 2)

# the first subplots is on the first row and span over all columns
ax1 = plt.subplot(G[0, :])

# the second subplot is on the first column of the second row
ax2 = plt.subplot(G[1, 0])

# the third subplot is on the second column of the second row
ax3 = plt.subplot(G[1, 1])

fig.tight_layout()
plt.show()


# Or we can have a subplot that spans over multiple rows

# In[6]:


fig = plt.figure()

# the first subplot is two rows high
ax1 = plt.subplot(G[:, :1])

# the second subplot is on the second column of the first row
ax2 = plt.subplot(G[0, 1])

# the third subplot is on the second column of the second row
ax3 = plt.subplot(G[1, 1])

fig.tight_layout()
plt.show()


# Using `gridspec`, we can also have different sizes for each subplot by specifying ratios for *width* and *heights*.

# In[7]:


fig = plt.figure()

G = gridspec.GridSpec(2, 2,
                       width_ratios=[1, 2], # the second column is two times larger than the first one
                       height_ratios=[4, 1] # the first row is four times higher than the second one
                       )

# in this example, I use a different way to refer to a grid element
# note that it is not clear in which part of the grid the subplot is
ax1 = plt.subplot(G[0]) # same as plt.subplot(G[0, 0])
ax2 = plt.subplot(G[1]) # same as plt.subplot(G[0, 1])
ax3 = plt.subplot(G[2]) # same as plt.subplot(G[1, 0])
ax4 = plt.subplot(G[3]) # same as plt.subplot(G[1, 1])

fig.tight_layout()
plt.show()


# ## Artists
# `Artists` are everything we can see on a `Figure`. Most of them are tied to an `Axes`.

# ## Plotting
# I won't spend to much time on the ploting functions. Everybody has already used them extensively and there are a lot of tutorials out there.  
# I'll just add that the plotting functions we usually call using pyplot (e.g. `plt.scatter()`) can be called on Axes: `ax.scatter()`. This make it easy to manage multiple subplots

# In[8]:


fig, ax = plt.subplots(figsize=(6, 4), nrows=1, ncols=2)

# plot different charts on each axes
ax[0].bar(np.arange(0, 3), df["Embarked"].value_counts())
ax[1].bar(np.arange(0, 3), df["Pclass"].value_counts())

# customize a bit
ax[0].set_title('Embarked')
ax[1].set_title('Pclass')

fig.tight_layout()
plt.show()


# ## Customization
# Now that we have set the layout and plot some data, we can have a look at how to customize the elements or `Artists` of our plot.    

# ### Axis and Ticks
# `Axis` are the X and Y axis of our plot, not to be confused with the `Axes`.  
# Each `Axis` has `Ticks` that can also be customized. Let's look at a sample plot.

# In[9]:


x = np.arange(0, 10, 0.1)
y = x**2
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)
plt.show()


# The `Ticks` on the X-axis are the marks with the numbers 0 to 10 and the ticks on the Y-axis are the marks with numbers 0 to 100.  
# **Customizing the axis**  
# Some useful customizations on the axis are:  
# 
# - `ax.set_xlabel()`, `ax.set_ylabel()`: add a label to the axis
# - `ax.set_xlim()`, `ax.set_ylim()`: to set the data limits on the axis. We can use `get_xlim()` or `get_ylim()` to see what those are those limits.  
# 

# **Customizing the ticks**
# We can customize the `Ticks` using the `tick_params()` method.  
# The most useful options are:
# - bottom, top, left, right: set to True/False or 'on'/'off' to show or hide the ticks
# - labelbottom, labeltop, labelleft, labelright: set to True/False or 'on'/'off' to show or hide the ticks labels
# - labelrotation: rotate the tick labels
# - labelsize: resize the ticks labels
# - labelolor: change the color of ticks labels
# - color: change the ticks colors

# In[10]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)

# add labels
# we can also use LaTex notations in titles and labels
ax.set_xlabel('$x$')
ax.set_ylabel('$y = x^2$')

# reduce the axis limites to let the line touch the borders
ax.set_xlim(0, 10)
ax.set_ylim(0, 100)

# customize the ticks
ax.tick_params(labelleft=False,
               labelcolor='orange',
               labelsize=12, 
               bottom=False,
               color='green',
               width=4, 
               length=8,
               direction='inout'
              )

plt.show()


# ### Spines
# The spines are the lines that surround your plot. Two of them define te x and y axis. The two others close the frame.  
# We can set bounds to limit the spine length and show or hide a spine.

# In[11]:


fig = plt.figure()
ax = fig.add_subplot(111)

# shorten the spines used for the x and y axis
ax.spines['bottom'].set_bounds(0.2, 0.8)
ax.spines['left'].set_bounds(0.2, 0.8)

# other customizations
ax.spines['bottom'].set_color('r')
ax.spines['bottom'].set_linewidth(2)

# remove the two other spinces
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()


# ### Bars, lines, markers, ...
# Plotting functions return objects. For example `ax.bar(x, y)` returns a container with all the bars. We can catch the bars and call methods on them.  
# We can find the methods in the matplotlib [documentation](https://matplotlib.org/api/_as_gen/matplotlib.patches.Rectangle.html#matplotlib.patches.Rectangle). The bars are instances of the `Rectangle` class from the `patches` module.  
# We can get the bars attributes:
# - get_height()
# - get_width()
# - get_x(), get_y(), get_xy()
# 
# or set a new value for those attributes:
# - set_height()
# - set_width()
# - set_x(), set_y(), set_xy()
# - set_color()
# - ...

# In[12]:


# lets create some bars
x = np.arange(0, 5)
y = [2, 6, 7, 3, 4]
bars = plt.bar(x, y, color='b')

# we can get the height of the bars
for i, bar in enumerate(bars):
    print('bars[{}]\'s height = {}'.format(i, bar.get_height()))

# or we can set a different color for the third bar
bars[2].set_color('r')

# or set a different width for the first bar
bars[0].set_width(0.4)

plt.show()


# The `hist()` method returns an array of the bins values, an array of bins and the patches used to draw the histogram

# In[13]:


fig = plt.figure()
ax = fig.add_subplot(111)

x = df['Age'].dropna()
bins = np.arange(0, 95, 5)
values, bins, bars = ax.hist(x, bins=bins)

# get the value of each bin
for bin, value in zip(bins, values):
    print('bin {}: {} passengers'.format(bin, int(value)))

# change the highest bin color    
max_idx = values.argmax()
bars[max_idx].set_color('r')

plt.show()


# ### Annotations
# We can add text to our plot with `ax.annotation()` and `ax.text()`.  
# `ax.annotation('text', (x, y)`  
# `ax.text(x, y, 'text', **kwargs)`
# 
# If needed, we can add arrows to point our annocation to specific point. Just use `ax.arrow(x, y, dx, dy)` where `x` and `y` are the coordinates of the origin of the arrow and `dx`, `dy` are the length of the arrao along the coordinates.  
# We can also add more arguments to customize the arrow:
# - width: width of the arraow tail
# - head_width: width of the arrow head
# - head_length: length of the arrow head
# - shape: shape of the arrow head ('full', 'left' or 'right')
# - edgecolor, facecolor
# - color (will override edgecolor and facecolor)
# - ...

# In[14]:


x = np.arange(0, 10, 0.1)
y = x**2
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)

ax.annotate('bar', (5, 20))

ax.text(1, 
        50, 
        'foo',
        fontsize=14,
        color='red')

# add an arrow that starts at 'foo' and point at the line
ax.arrow(1.5, 48,
         0.5, -44,
         length_includes_head=True,
         width=0.3,
         head_length=4,
         facecolor='y',
         edgecolor='r',
         shape='left')

# we can also set an arrow in the annotate method
ax.annotate('quz', 
            xytext=(6, 60), # text coordinates 
            xy=(7.8, 60),   # arrow head coordinates
            arrowprops=dict(arrowstyle="->"))

plt.show()


# ## Practice
# Now, let's use everything we've learn to create some nice visualizations.
# 
# ### Exercise 1
# In this first exercise, I am going to improve a simple bar chart showing the number of passengers per port of embarkation.  
# Here's the default chart rendered by pyplot.

# In[15]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(np.arange(0, 3), df["Embarked"].value_counts())
plt.show()


# I am going to remove unecessary elements and change the colors.  
# I'll add some meaningful information like labels and values on bars and a title.

# In[16]:


fig = plt.figure()
ax = fig.add_subplot(111)

x = np.arange(0, 3)
y = df['Embarked'].value_counts()
bars = ax.bar(x, y, color='lightslategrey')

# remove the frame
ax.set_frame_on(False)

# we need only 3 ticks (one per bar)
ax.set_xticks(np.arange(0, 3))

# we don't want the ticks, only the labels
ax.tick_params(bottom='off')
ax.set_xticklabels(['Southampton', 'Cherbourg', 'Queenstown'],
                   {'fontsize': 12,
                    'verticalalignment': 'center',
                    'horizontalalignment': 'center',
                    })

# remove ticks on the y axis and show values in the bars
ax.tick_params(left='off',
               labelleft='off')

# add the values on each bar
for bar, value in zip(bars, y):
    ax.text(bar.get_x() + bar.get_width() / 2, # x coordinate
            bar.get_height() - 5,              # y coordinate
            value,                             # text
            ha='center',                       # horizontal alignment
            va='top',                          # vertical alignment
            color='w',                         # text color
            fontsize=14)

# use a different color for the first bar
bars[0].set_color("firebrick")

# add a title
ax.set_title('Most of passengers embarked at Southampton',
             {'fontsize': 18,
              'fontweight' : 'bold',
              'verticalalignment': 'baseline',
              'horizontalalignment': 'center'})

plt.show()


# ### Exercise 2
# In this second exercise, I am going to plot the distribution of passengers per age and color the bars depending on survival rate.

# In[17]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

x = df['Age'].dropna()
age_bins = np.arange(0, 95, 5)
values, bins, bars = ax.hist(x, bins=age_bins)

ax.set_xticks(bins)

ax.set_ylim(values.min(), values.max() + 10)

ax.spines['bottom'].set_bounds(bins.min(), bins.max())
ax.spines['right'].set_bounds(values.min(), values.max())

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

cm = plt.cm.get_cmap('viridis')

for i in range(0, len(bins)):
    if i < len(bins) - 1:
        survival = df[(df.Age >= bins[i]) & (df.Age < bins[i + 1])]['Survived'].mean()
    else:
        survival = df[(df.Age >= bins[i])]['Survived'].mean()
    try:
        bars[i].set_color(cm(survival))
    except:
        pass

# add colorbar
# the survival rate is already normalized so we don't need norm=plt.Normalize(vmin=0, vmax=1)
# I left it as an example
sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=1)) 
sm._A = []
plt.colorbar(sm)
    
# add a title
ax.set_title('Survival rate per passenger age',
             {'fontsize': 18,
              'fontweight' : 'bold',
              'verticalalignment': 'baseline',
              'horizontalalignment': 'center'})

plt.show()


# In[ ]:




