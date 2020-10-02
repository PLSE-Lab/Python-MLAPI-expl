#!/usr/bin/env python
# coding: utf-8

# <img src="https://matplotlib.org/_static/logo2.png">

# # Overview
# ## Introducing Matplotlib and pyplot
# * Anatomy of a figure
# * Plots, title, labels, lines, markers, watermarks
# 
# ## Basic, intermediate and advance plots
# * Shapes and curves
# * Text and annotations
# * Different units on the same axis
# * Scaled axis
# 
# ## Visualizing data
# * Boxplots and violin plots
# * Histrograms and pie charts
# * Stem charts
# * Autocorrelations
# * Stackplots
# 
# <img src="https://i.ibb.co/WPdDgKp/Screenshot-2018-12-23-at-11-14-53-PM.png">

# * Matplotlib is a Python 2D plotting library which produces **publication quality** figures in a variety of **hardcopy formats** and interactive environments **across platforms**.
# 
# <img src="https://i.ibb.co/5nVJ85R/Screenshot-2018-12-24-at-9-59-07-AM.png">
# <img src="https://i.ibb.co/XjsCn8v/Screenshot-2018-12-24-at-10-02-29-AM.png">

# * **Object level APIs** Includes granular low-level APIs to control each object in a plot.
# * **matplotlib pyplot** is a higher level API that controls the "state-machine".
# * **Pylab** is a convenience module imports portions of Matplotlib and NumPy to give users a Matlab -like access to functions.

# ## Matplotlib
# * Python ploting library
# * Easy to create plots
# * Embeddable GUI for application development
# * can be used across platfroms

# <img src="https://i.ibb.co/k8jQJxZ/Screenshot-2018-12-24-at-10-21-07-AM.png">

# * **Figure** overall window or page, within which all operations are performed
# * **Axes** Area within a figure where actual graphs are plotted
# * Axes has an X-axis and a Y-axis
# * Contains tick, tick locations, labels
# 
# <img src="https://i.ibb.co/NCHQ1R6/Screenshot-2018-12-24-at-10-27-59-AM.png">
# 
# * A figure can have more than one axes.
# 
# ## Anatomy of a figure
# <img src="https://i.ibb.co/dB54X8N/Screenshot-2018-12-24-at-10-31-27-AM.png">

# # Introducing Matplotlib and pyplot

# ## Basic Plots

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


plt.plot([2, 4, 6, 8],
        [4, 8, 12, 16])


# In[ ]:


plt.plot([2, 4, 6, 8],
        [4, 8, 12, 16], color='red')


# In[ ]:


plt.plot([2, 4, 6, 8],
        [4, 8, 12, 16])
plt.xlabel('x', fontsize=15, color='green')   # naming x-axis
plt.ylabel('2*x', fontsize=15, color='green') # naming y-axis


# In[ ]:


plt.plot([4, 8, 12, 16]) # means this is y-axis and x-axis its assume as index


# In[ ]:


x = np.linspace(start = 0, stop = 10, num = 50) # Give value in ascending order


# In[ ]:


plt.plot(x, np.sin(x))   # (x, y)
plt.xlabel('x', fontsize=15, color='green')
plt.ylabel('Sin(x)', fontsize=15, color='green')


# In[ ]:


plt.tick_params(axis='y',
               color='red',
               labelcolor='blue',
               labelsize='xx-large')


# In[ ]:


plt.tick_params(axis='x',
               bottom=False,
               labelbottom=False)


# In[ ]:


plt.plot(x, np.sin(x), label='sin curve')
plt.xlabel('x', fontsize=15, color='green')
plt.ylabel('sin(x)', fontsize=15, color='green')
plt.legend()   # for label
plt.title('Playing with Plots')  # for title


# In[ ]:


plt.plot(x, np.sin(x), label='sin curve')
plt.xlabel('x', fontsize=15, color='green')
plt.ylabel('sin(x)', fontsize=15, color='green')
plt.legend()   # for label
plt.title('Playing with Plots')  # for title
plt.xlim(1, 5)  # limit x-axis our paramter


# In[ ]:


plt.plot(x, np.sin(x), label='sin curve')
plt.xlabel('x', fontsize=15, color='green')
plt.ylabel('sin(x)', fontsize=15, color='green')
plt.legend()   # for label
plt.title('Playing with Plots')  # for title
plt.xlim(1, 5)  # limit x-axis our paramter
plt.ylim(-1, 0.5)  # limit y-axis our paramter


# ## Lines and Markers

# In[ ]:


x = np.linspace(start = 0, stop = 10, num = 50)


# In[ ]:


plt.plot(x, np.sin(x))


# In[ ]:


plt.plot(x, np.sin(x), label='sine curve')
plt.plot(x, np.cos(x), label='cosine curve')
plt.legend()
plt.title('Playing with Plots')


# In[ ]:


plt.plot(x, np.sin(x), label='sine curve', color='green')
plt.plot(x, np.cos(x), label='cosine curve', color='m')
plt.legend()
plt.title('Playing with Plots')


# In[ ]:


random_array = np.random.randn(20)


# In[ ]:


plt.plot(random_array,
        color='green')
plt.show()


# In[ ]:


plt.plot(random_array,
        color='green',
        linestyle=':')  # dot style graph
plt.show()


# In[ ]:


plt.plot(random_array,
        color='green',
        linestyle='--') # line style
plt.show()


# In[ ]:


plt.plot(random_array,
        color='green',
        linestyle=':')
plt.show()


# In[ ]:


plt.plot(random_array,
        color='green',
        linestyle='--',
        linewidth=3)  # line size
plt.show()


# In[ ]:


plt.plot(random_array,
        color='green',
        marker ='d') # diamond shape default size is 6
plt.show()


# In[ ]:


plt.plot(random_array,
        color='green',
        marker ='d', # diamond shape 
        markersize=10) # changing size of diamond shape
plt.show()


# In[ ]:


plt.plot(random_array,
        color='green',
        marker ='d', # diamond shape 
        markersize=10, # changing size of diamond shape
        linestyle='None') # Remove the line show only diamond shape
plt.show()


# In[ ]:


plt.scatter(range(0, 20),  # Scatter plot
           random_array,
           color='green',
           marker='d')
plt.show()


# ## Figures and Axes

# In[ ]:


fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
plt.show()


# * **At 1st position** 0 represent extreme left of the figure and 1 represent extreme right of the figure.
# * **At 2nd position** 0 represents bottom of the figure and 1 represents top of the figure.
# * **At 3rd position** width of the figure.
# * **At 4th position** height of the figure.

# In[ ]:


type(ax)


# In[ ]:


# This for comparison between two graphs
fig = plt.figure()
ax1 = fig.add_axes([0, 0.6, 1, 0.4])
ax2 = fig.add_axes([0, 0, 0.8, 0.4])
plt.show()


# In[ ]:


x = np.linspace(start = 0, stop = 10, num = 50)


# In[ ]:


fig = plt.figure()

ax1 = fig.add_axes([0, 0.6, 1, 0.4])
ax2 = fig.add_axes([0, 0, 0.8, 0.4])

ax1.plot(x, np.sin(x))
ax2.plot(x, np.cos(x))

plt.show()


# In[ ]:


fig = plt.figure()

ax1 = fig.add_axes([0, 0.6, 1, 0.4])
ax2 = fig.add_axes([0, 0, 0.8, 0.4])

ax1.plot(x, np.sin(x))
ax1.set_xlabel('x', fontsize=15, color='r')
ax1.set_ylabel('sin(x)', fontsize=15, color='r')

ax2.plot(x, np.cos(x))
ax2.set_xlabel('x', fontsize=15, color='r')
ax2.set_ylabel('cos(x)', fontsize=15, color='r')

plt.show()


# In[ ]:


# Figure inside another figure
fig = plt.figure()
ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes([0.5, 0.5, 0.4, 0.4])
plt.show()


# In[ ]:


# Figure inside another figure
fig = plt.figure(figsize=(8,8))  # change size of the fig
ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes([0.5, 0.5, 0.4, 0.4])
plt.show()


# In[ ]:


fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(221) # create 2*2 figure and 1 represent no of the figure
plt.show()


# In[ ]:


type(ax1)


# In[ ]:


isinstance(ax1, matplotlib.axes._axes.Axes)


# In[ ]:


fig = plt.figure(figsize=(8,8))

ax1 = fig.add_subplot(221) # create 2*2 figure and 1 represent no of the figure
ax1.plot([1, 2, 3, 4],
        [2, 4, 6, 8])

ax2 = fig.add_subplot(222) # create 2*2 figure and 2 represent no of the figure
ax2.plot(x, np.sin(x))


# In[ ]:


fig = plt.figure(figsize=(8,8))

ax1 = fig.add_subplot(221) # create 2*2 figure and 1 represent no of the figure
ax1.plot([1, 2, 3, 4],
        [2, 4, 6, 8])

ax2 = fig.add_subplot(222) # create 2*2 figure and 2 represent no of the figure
ax2.plot(x, np.sin(x))

ax3 = fig.add_subplot(223) # create 2*2 figure and 3 represent no of the figure
ax3.plot(x, np.cos(x))


# In[ ]:


fig = plt.figure(figsize=(8,8))

ax1 = fig.add_subplot(221) # create 2*2 figure and 1 represent no of the figure
ax1.plot([1, 2, 3, 4],
        [2, 4, 6, 8])

ax2 = fig.add_subplot(222) # create 2*2 figure and 2 represent no of the figure
ax2.plot(x, np.sin(x))

ax3 = fig.add_subplot(224) # create 2*2 figure and 4 represent no of the figure
ax3.plot(x, np.cos(x))


# In[ ]:


ax1 = plt.subplot2grid((2, 3), (0, 0))
ax1.plot(x, np.sin(x))
ax1.set_label('sine curve')

ax2 = plt.subplot2grid((2, 3), (0, 1))
ax2.plot(x, np.cos(x))
ax2.set_label('cosine curve')

ax3 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
ax3.plot([1, 2, 3, 4],
        [2, 4, 5, 8])
ax3.set_label('straight line')
ax3.yaxis.tick_right()

ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
ax4.plot(x, np.exp2(x))
ax4.set_label('exponential curve')


# * subplot2grid(shape, loc, rowspan=1, colspan=1)
# * **loc** : sequence of 2 ints.Location to place axis within grid. First entry is row number, second entry is column number.
# * **rowspan** : int. Number of rows for the axis to span to the right.
# * **colspan** : int. Number of columns for the axis to span downwards.

# In[ ]:


# A simple way to get a figure with one set of axes
fig, ax = plt.subplots()


# In[ ]:


type(fig)


# In[ ]:


type(ax)


# ## Watermarks

# In[ ]:


fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4],
       [2, 4, 6, 8])


# In[ ]:


fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4],
       [2, 4, 6, 8])
ax.text(1, 4, 'Do not distribute',
       fontsize=30,
       color='red',
       ha='left',   # horizontal alignment
       va='bottom', # vertical alignment
       alpha=0.5)


# In[ ]:


fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4],
       [2, 4, 6, 8])
ax.text(1, 4, 'Do not distribute',
       fontsize=30,
       color='red',
       ha='right',   # horizontal alignment
       va='top', # vertical alignment
       alpha=0.5)


# In[ ]:


fig = plt.figure(figsize=(8, 8))

ax1 = fig.add_subplot(221)
ax1.plot([1, 2, 3, 4],
       [2, 4, 6, 8])
ax1.set_label('straight line')
ax1.text(1, 4, 'Do not distribute',
       fontsize=20,
       color='red',
       ha='left',   # horizontal alignment
       va='bottom', # vertical alignment
       alpha=0.5)


# In[ ]:


fig = plt.figure(figsize=(8, 8))

ax1 = fig.add_subplot(221)
ax1.plot([1, 2, 3, 4],
       [2, 4, 6, 8])
ax1.set_label('straight line')
ax1.text(1, 4, 'Do not distribute',
       fontsize=20,
       color='red',
       ha='left',   # horizontal alignment
       va='bottom', # vertical alignment
       alpha=0.5)

ax2 = fig.add_subplot(222)
ax2.plot(x, np.sin(x))

ax3 = fig.add_subplot(223)
ax3.plot(x, np.cos(x))


# ## Visualizing Stock Data

# In[ ]:



from subprocess import check_output
import pandas as pd
print(check_output(["ls", "../input"]).decode("utf8"))
stock_data = pd.read_csv('../input/matplotlib/stocks.csv')
stock_data.head()


# In[ ]:


stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.head()


# In[ ]:


fig = plt.figure(figsize=(10, 6))

ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes([0.05, 0.65, 0.5, 0.3])


# In[ ]:


fig = plt.figure(figsize=(10, 6))

ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes([0.05, 0.65, 0.5, 0.3])

ax1.plot(stock_data['Date'],
        stock_data['AAPL'],
        color='green')
ax1.set_title('AAPL vs IBM (inset)')


# In[ ]:


fig = plt.figure(figsize=(10, 6))

ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes([0.05, 0.65, 0.5, 0.3])

ax1.plot(stock_data['Date'],
        stock_data['AAPL'],
        color='green')
ax1.set_title('AAPL vs IBM (inset)')

ax2.plot(stock_data['Date'],
        stock_data['IBM'],
        color='blue')


# In[ ]:


fig = plt.figure(figsize=(10, 6))
fig.suptitle('Stock price comparison 2007-2017', fontsize=20)

ax1 = fig.add_subplot(221)
ax1.set_title('MSFT')
ax1.plot(stock_data['Date'],
        stock_data['MSFT'],
        color='green')

ax2 = fig.add_subplot(222)
ax2.set_title('GOOG')
ax2.plot(stock_data['Date'],
        stock_data['GOOG'],
        color='purple')

ax3 = fig.add_subplot(223)
ax3.set_title('SBUX')
ax3.plot(stock_data['Date'],
        stock_data['SBUX'],
        color='magenta')

ax3 = fig.add_subplot(224)
ax3.set_title('CVX')
ax3.plot(stock_data['Date'],
        stock_data['CVX'],
        color='orange')


# # Building Basic, Intermediate and Advanced Plots
# ## Plotting Shapes

# In[ ]:


import matplotlib.patches as patches
fig, ax = plt.subplots()
print(fig)
print(ax)
ax.add_patch(     # A patch in matplotlib represents 2D objects
    patches.Rectangle(
        (0.1, 0.1),  # (left edge, bottom edge) of rectangle
        0.5,         # width of the rectangle
        0.5,         # height of the rectagle
        fill=False   # not fill with the color
    )
)
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

ax.add_patch(     # A patch in matplotlib represents 2D objects
    patches.Rectangle(
        (0.1, 0.1),  # (left edge, bottom edge) of rectangle
        0.5,         # width of the rectangle
        0.5,         # height of the rectagle
        fill=False   # not fill with the color
    )
)
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

ax.add_patch(     # A patch in matplotlib represents 2D objects
    patches.Rectangle(
        (0.1, 0.1),  # (left edge, bottom edge) of rectangle
        0.5,         # width of the rectangle
        0.5,         # height of the rectagle
        facecolor='yellow',
        edgecolor='green'
    )
)
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

for p in [
    patches.Rectangle((0.1, 0.1), 0.3, 0.6,
    hatch='.'
    ),
    patches.Rectangle((0.5, 0.1), 0.3, 0.6,
    hatch='\\',
    fill=False
    )
]:
    ax.add_patch(p)
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

for p in [
    patches.Rectangle((0.1, 0.1), 0.2, 0.6, alpha=None,
    ),
    patches.Rectangle((0.4, 0.1), 0.2, 0.6, alpha=1.0,
    ),
    patches.Rectangle((0.7, 0.1), 0.2, 0.6, alpha=0.6,
    ),
    patches.Rectangle((1.0, 0.1), 0.2, 0.6, alpha=0.1,
    )
]:
    ax.add_patch(p)
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

for p in [
    patches.Rectangle((0.1, 0.1), 0.2, 0.6, alpha=None,
    ),
    patches.Rectangle((0.4, 0.1), 0.2, 0.6, alpha=1.0,
    ),
    patches.Rectangle((0.7, 0.1), 0.2, 0.6, alpha=0.6,
    ),
    patches.Rectangle((1.0, 0.1), 0.2, 0.6, alpha=0.1,
    )
]:
    ax.add_patch(p)

ax.set_xlim(0, 1.5)
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

for p in [
    patches.Rectangle((0.1, 0.1), 0.2, 0.6, facecolor=None # Default color blue
    ),
    patches.Rectangle((0.4, 0.1), 0.2, 0.6, facecolor='none' # Not fill color
    ),
    patches.Rectangle((0.7, 0.1), 0.2, 0.6, facecolor='red'  # fill red color
    ),
    patches.Rectangle((1.0, 0.1), 0.2, 0.6, facecolor='#00ffff' # fill blue
    )
]:
    ax.add_patch(p)

ax.set_xlim(0, 1.5)
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

for p in [
    patches.Rectangle((0.1, 0.1), 0.2, 0.6, fill=False, edgecolor=None
    ),
    patches.Rectangle((0.4, 0.1), 0.2, 0.6, fill=False, edgecolor='none'
    ),
    patches.Rectangle((0.7, 0.1), 0.2, 0.6, fill=False, edgecolor='red'
    ),
    patches.Rectangle((1.0, 0.1), 0.2, 0.6, fill=False, edgecolor='#00ffff'
    )
]:
    ax.add_patch(p)

ax.set_xlim(0, 1.5)
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

for p in [
    patches.Rectangle((0.1, 0.1), 0.2, 0.6, fill=False, linestyle='solid'  # Default
    ),
    patches.Rectangle((0.4, 0.1), 0.2, 0.6, fill=False, linestyle='dashed'
    ),
    patches.Rectangle((0.7, 0.1), 0.2, 0.6, fill=False, linestyle='dashdot'
    ),
    patches.Rectangle((1.0, 0.1), 0.2, 0.6, fill=False, linestyle='dotted'
    )
]:
    ax.add_patch(p)

ax.set_xlim(0, 1.5)
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

for p in [
    patches.Circle((0.1, 0.4), 0.1,
    hatch='/'
    ),
    patches.Circle((0.5, 0.4), 0.1,
    hatch='*',
    facecolor='red'
    ),
    patches.Circle((0.9, 0.4), 0.1,
    hatch='\\',
    facecolor='green'
    ),
    patches.Circle((0.5, 0.7), 0.1,
    hatch='//',
    fill=False
    )
]:
    ax.add_patch(p)
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

polygon = patches.Polygon([[0.1, 0.1],
                           [0.2, 0.8],
                           [0.5, 0.7],
                           [0.8, 0.1],
                           [0.4, 0.3]],
                          fill=False)

ax.add_patch(polygon)
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

polygon = patches.Polygon([[0.1, 0.1],
                           [0.2, 0.8],
                           [0.5, 0.7],
                           [0.8, 0.1],
                           [0.4, 0.3]],
                          closed=False,
                          fill=False)

ax.add_patch(polygon)
plt.show()


# In[ ]:


# Arrow is polygon with seven sides
fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')
polygon = patches.Arrow(0.1, 0.2, # centre of the bace of the arrow
                       0.7, 0.7)
ax.add_patch(polygon)
plt.show()


# ## Bezier Curve
# <img src="https://i.ibb.co/Mh073tb/Screenshot-2018-12-25-at-2-41-07-PM.png">

# In[ ]:


from matplotlib.path import Path
fig, ax = plt.subplots()
p = patches.PathPatch(Path([(0.1, 0.1), (0.8, 0.8), (0.8, 0.1), (0.4, 0.2)],
                              [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]),
                     fill=None)
ax.add_patch(p)
plt.show()


# In[ ]:


from matplotlib.path import Path
fig, ax = plt.subplots()
p = patches.PathPatch(Path([(0.1, 0.1), (0.8, 0.8), (0.8, 0.1), (0.4, 0.2)],
                              [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]),
                     fill=None)
ax.add_patch(p)
plt.show()


# In[ ]:


from matplotlib.path import Path
fig, ax = plt.subplots()
p = patches.PathPatch(Path([(0.1, 0.1), (0.8, 0.8), (0.8, 0.1), (0.4, 0.2)],
                              [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3]),
                     fill=None)
ax.add_patch(p)
plt.show()


# In[ ]:


from matplotlib.path import Path
fig, ax = plt.subplots()
p = patches.PathPatch(Path([(0.1, 0.1), (0.8, 0.8), (0.8, 0.1), (0.4, 0.2)],
                              [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.MOVETO]),
                     fill=None)
ax.add_patch(p)
plt.show()


# ## Annotations

# In[ ]:


fig, ax = plt.subplots()
ax.plot([1, 2, 3],
       [2, 4, 6])

ax.annotate('min value',
           xy=(1, 2),                   # points to datapoint
           xytext=(1.5, 2.0),           # where to write the text
           arrowprops=dict(color='g'))  # Arrow color
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.plot([1, 2, 3],
       [2, 4, 6])

ax.annotate('min value',
           xy=(1, 2),                   # points to datapoint
           xytext=(1, 3),               # where to write the text
           arrowprops=dict(color='g'))  # Arrow color
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.plot([1, 2, 3],
       [2, 4, 6])

ax.annotate('min value',
           xy=(1, 2),                   # points to datapoint
           xytext=(1, 3),               # where to write the text
           arrowprops=dict(facecolor='y', edgecolor='green', alpha=0.3))
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.plot([1, 2, 3],
       [2, 4, 6])

ax.annotate('Significant point',
           xy=(2, 4),                   
           xytext=(2.0, 2.5),               
           arrowprops=dict(color='green')
           )
ax.plot([2], [4], 'ro')
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.plot([1, 2, 3],
       [2, 4, 6])

ax.annotate('Significant point',
           xy=(2, 4),                   
           xytext=(2.0, 2.5),               
           arrowprops=dict(color='green', shrink=0.1)
           )
ax.plot([2], [4], 'ro')
plt.show()


# In[ ]:


import numpy as np

x1 = -1 + np.random.randn(100)
y1 = -1 + np.random.randn(100)

x2 = 1 + np.random.randn(100)
y2 = 1 + np.random.randn(100)


# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x1, y1, color='r')
ax.scatter(x1, y2, color='g')
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x1, y1, color='r')
ax.scatter(x1, y2, color='g')

bbox_props = dict(boxstyle='square', facecolor='w', alpha=0.5)
ax.text(-2, -2, 'Sample A', ha='center', va='center', size=20, bbox=bbox_props)
ax.text(0, 2, 'Sample B', ha='center', va='center', size=20, bbox=bbox_props)


# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x1, y1, color='r')
ax.scatter(x1, y2, color='g')

bbox_props = dict(boxstyle='square', facecolor='w', alpha=0.5)
ax.text(-2, -2, 'Sample A', ha='center', va='center', size=20, bbox=bbox_props)
ax.text(0, 2, 'Sample B', ha='center', va='center', size=20, bbox=bbox_props)

arrow_bbox_props= dict(boxstyle='rarrow',
                      facecolor='#EBF5FB',
                      edgecolor='b',
                      linewidth=2,
                      alpha=0.7)
ax.text(0, 0,
       'Direction',
       ha='center',
       va='center',
       rotation=45,
       size=15,
       bbox=arrow_bbox_props)


# ## Scales

# In[ ]:


y = np.random.uniform(low=0.0, high=1000, size=(1000,))
y.sort()
x = np.arange(len(y))


# In[ ]:


plt.plot(x, y)
plt.grid(True)
plt.show()


# In[ ]:


plt.plot(x, y)
plt.grid(True)
plt.yscale('log')
plt.show()


# In[ ]:


plt.plot(x, y)
plt.grid(True)
plt.yscale('log', basey=2) # base log2
plt.show()


# In[ ]:


plt.plot(x, y)
plt.grid(True)
plt.yscale('log', basey=2) # base log2 on y-axis
plt.xscale('log', basex=2) # base log2 on x-axis
plt.show()


# ## Twin Axis

# In[ ]:


austin_weather = pd.read_csv('../input/matplotlib/austin_weather1.csv')
austin_weather.head()


# In[ ]:


austin_weather = austin_weather[['Date', 'TempAvgF', 'WindAvgMPH']].head(30)
austin_weather


# In[ ]:


fig, ax_tempF = plt.subplots()

fig.set_figwidth(12)
fig.set_figheight(6)

ax_tempF.set_xlabel('Date')

ax_tempF.tick_params(axis='x', bottom=False, labelbottom=False) # disable scale in x-axis


# In[ ]:


fig, ax_tempF = plt.subplots()

fig.set_figwidth(12)
fig.set_figheight(6)

ax_tempF.set_xlabel('Date')

ax_tempF.tick_params(axis='x', bottom=False, labelbottom=False)
ax_tempF.set_ylabel('Temp (F)', color='red', size='x-large')

ax_tempF.tick_params(axis='y', labelcolor='red', labelsize='large')


# In[ ]:


fig, ax_tempF = plt.subplots()

fig.set_figwidth(12)
fig.set_figheight(6)

ax_tempF.set_xlabel('Date')

ax_tempF.tick_params(axis='x', bottom=False, labelbottom=False)
ax_tempF.set_ylabel('Temp (F)', color='red', size='x-large')

ax_tempF.tick_params(axis='y', labelcolor='red', labelsize='large')

ax_tempF.plot(austin_weather['Date'],
                            austin_weather['TempAvgF'],
             color='red')


# In[ ]:


fig, ax_tempF = plt.subplots()

fig.set_figwidth(12)
fig.set_figheight(6)

ax_tempF.set_xlabel('Date')

ax_tempF.tick_params(axis='x', bottom=False, labelbottom=False)
ax_tempF.set_ylabel('Temp (F)', color='red', size='x-large')

ax_tempF.tick_params(axis='y', labelcolor='red', labelsize='large')

ax_tempF.plot(austin_weather['Date'],
                            austin_weather['TempAvgF'],
             color='red')
ax_wind = ax_tempF.twinx() # The function creates another Y axis using the same X axis


# In[ ]:


fig, ax_tempF = plt.subplots()

fig.set_figwidth(12)
fig.set_figheight(6)

ax_tempF.set_xlabel('Date')

ax_tempF.tick_params(axis='x', bottom=False, labelbottom=False)
ax_tempF.set_ylabel('Temp (F)', color='red', size='x-large')

ax_tempF.tick_params(axis='y', labelcolor='red', labelsize='large')

ax_tempF.plot(austin_weather['Date'],
                            austin_weather['TempAvgF'],
             color='red')
ax_wind = ax_tempF.twinx() # The function creates another Y axis using the same X axis

ax_wind.set_ylabel('Avg wind Speed (MPH)',
                  color='blue',
                  size='x-large')
ax_wind.tick_params(axis='y',
                  labelcolor='blue',
                  labelsize='large')


# In[ ]:


fig, ax_tempF = plt.subplots()

fig.set_figwidth(12)
fig.set_figheight(6)

ax_tempF.set_xlabel('Date')

ax_tempF.tick_params(axis='x', bottom=False, labelbottom=False)
ax_tempF.set_ylabel('Temp (F)', color='red', size='x-large')

ax_tempF.tick_params(axis='y', labelcolor='red', labelsize='large')

ax_tempF.plot(austin_weather['Date'],
                            austin_weather['TempAvgF'],
             color='red')
ax_wind = ax_tempF.twinx() # The function creates another Y axis using the same X axis

ax_wind.set_ylabel('Avg wind Speed (MPH)',
                  color='blue',
                  size='x-large')
ax_wind.tick_params(axis='y',
                  labelcolor='blue',
                  labelsize='large')

ax_wind.plot(austin_weather['Date'],
            austin_weather['WindAvgMPH'],
            color='blue')


# In[ ]:


def fahrenheit2celsius(f):
    return (f - 32) * 5 / 9


# In[ ]:


fig, ax_tempF = plt.subplots()

fig.set_figwidth(12)
fig.set_figheight(6)

ax_tempF.set_xlabel('Date')

ax_tempF.tick_params(axis='x', bottom=False, labelbottom=False)
ax_tempF.set_ylabel('Temp (F)', color='red', size='x-large')

ax_tempF.tick_params(axis='y', labelcolor='red', labelsize='large')

ax_tempF.plot(austin_weather['Date'],
                            austin_weather['TempAvgF'],
             color='red')
ax_tempC = ax_tempF.twinx() # The function creates another Y axis using the same X axis

ymin, ymax = ax_tempF.get_ylim()

ax_tempC.set_ylim(fahrenheit2celsius(ymin),
                  fahrenheit2celsius(ymax))

ax_tempC.tick_params(axis='y',
                    labelcolor='blue',
                    labelsize='large')


# # Visualizing Data
# ## Box plot

# In[ ]:


x = np.random.randint(low=0, high=20, size=20)
x.sort()
x


# In[ ]:


plt.boxplot(x)
plt.show()


# * The box represents values between the 25th and 75th percentiles
# * The central line is the median value of the data.
# * The caps represent the range of values(excluding outliers) in the data
# * The vertical line represent the whiskers are the bars which connect the box with the caps.

# In[ ]:


x = np.append(x, 22)
plt.boxplot(x)
plt.show()


# In[ ]:


x = np.append(x, 37)
x = np.append(x, 40)
plt.boxplot(x)
plt.show()


# * Outliers are represented by circles outside the box called fliers

# In[ ]:


plt.boxplot(x, vert=False) # print in vertical
plt.show()


# In[ ]:


plt.boxplot(x, vert=False, notch=True)
plt.show()


# In[ ]:


plt.boxplot(x, vert=False, notch=True, showfliers=False)
plt.show()


# In[ ]:


# The default boxplot is a line2D object which only allows formatting of the edges.
# By setting patch_artist-True, we turn it into a 2D patch
bp = plt.boxplot(x, patch_artist=True)
bp


# In[ ]:


bp = plt.boxplot(x, patch_artist=True)
bp['boxes'][0].set(facecolor='lightyellow', edgecolor='maroon', hatch='.')


# In[ ]:


bp = plt.boxplot(x, patch_artist=True)
bp['boxes'][0].set(facecolor='lightyellow', 
                   edgecolor='maroon', hatch='.')
bp['whiskers'][0].set(color='red',
                     linewidth=2)
bp['whiskers'][1].set(color='blue')


# In[ ]:


bp = plt.boxplot(x, patch_artist=True)
bp['boxes'][0].set(facecolor='lightyellow', 
                   edgecolor='maroon', hatch='/')
bp['fliers'][0].set(marker='D', 
                   markerfacecolor='blue')


# In[ ]:


bp = plt.boxplot(x, patch_artist=True)
bp['boxes'][0].set(facecolor='lightyellow', 
                   edgecolor='maroon')
bp['medians'][0].set(linestyle='--',
                    linewidth=3)


# In[ ]:


print(check_output(["ls", "../input/score-of-exams/"]).decode("utf8"))
exam_data = pd.read_csv('../input/score-of-exams/exams.csv')


# In[ ]:


exam_data.head()


# In[ ]:


exam_scores = exam_data[['math score', 'reading score', 'writing score']]
exam_scores.head()


# In[ ]:


exam_scores.describe()


# In[ ]:


exam_scores = np.array(exam_scores) # convet the dataframe to array


# In[ ]:


bp = plt.boxplot(exam_scores)
plt.show()


# In[ ]:


bp = plt.boxplot(exam_scores, patch_artist=True)
plt.show()


# In[ ]:


colors = ['blue', 'grey', 'lawngreen']


# In[ ]:


bp = plt.boxplot(exam_scores, patch_artist=True)

for i in range(len(bp['boxes'])):
    bp['boxes'][i].set(facecolor=colors[i])
    bp['caps'][2*i+1].set(color=colors[i])

plt.show()


# In[ ]:


bp = plt.boxplot(exam_scores, patch_artist=True)

for i in range(len(bp['boxes'])):
    bp['boxes'][i].set(facecolor=colors[i])
    bp['caps'][2*i+1].set(color=colors[i])

plt.xticks([1, 2, 3], ['Math', 'Reading', 'Writing']) 
plt.show()


# ## Violin plots

# In[ ]:


vp = plt.violinplot(exam_scores)
plt.show()


# * The graph displays the density of the data set over this range of values

# In[ ]:


vp = plt.violinplot(exam_scores, showmedians=True)
plt.xticks([1, 2, 3], ['Math', 'Reading', 'Writing'])
plt.show()


# In[ ]:


vp = plt.violinplot(exam_scores, showmedians=True, vert=False)
plt.yticks([1, 2, 3], ['Math', 'Reading', 'Writing'])
plt.show()


# In[ ]:


vp


# In[ ]:


vp = plt.violinplot(exam_scores, showmedians=True, vert=False)
plt.yticks([1, 2, 3], ['Math', 'Reading', 'Writing'])

for i in range(len(vp['bodies'])):
    vp['bodies'][i].set(facecolor=colors[i])

plt.show()


# In[ ]:


vp = plt.violinplot(exam_scores, showmedians=True, vert=False)
plt.yticks([1, 2, 3], ['Math', 'Reading', 'Writing'])

for i in range(len(vp['bodies'])):
    vp['bodies'][i].set(facecolor=colors[i])

vp['cmaxes'].set(color='maroon')
vp['cmins'].set(color='black')
vp['cbars'].set(linestyle=':')
vp['cmedians'].set(linewidth=6)
    
plt.show()


# In[ ]:


vp = plt.violinplot(exam_scores, showmedians=True, vert=False)
plt.yticks([1, 2, 3], ['Math', 'Reading', 'Writing'])

for i in range(len(vp['bodies'])):
    vp['bodies'][i].set(facecolor=colors[i])

plt.legend(handles = [vp['bodies'][0], vp['bodies'][1]],
           labels = ['Math', 'Reading'],
           loc = 'upper left')


# ## Histograms

# In[ ]:


np_data = pd.read_csv('../input/matplotlib/national_parks.csv')
np_data.head()


# In[ ]:


np_data.describe()


# In[ ]:


plt.hist(np_data['GrandCanyon'],
        facecolor='cyan',
        edgecolor='blue',
        bins=10)
plt.show()


# In[ ]:


n, bins, patches = plt.hist(np_data['GrandCanyon'],
                            facecolor='cyan',
                            edgecolor='blue',
                            bins=10)
print('n: ', n)  # frequency of the data point
print('bins: ', bins)  # the middel value of the bin
print('patches: ', patches)


# In[ ]:


n, bins, patches = plt.hist(np_data['GrandCanyon'],
                            facecolor='cyan',
                            edgecolor='blue',
                            bins=10,
                            density=True)
print('n: ', n)  # frequency of the data point
print('bins: ', bins)  # the middel value of the bin
print('patches: ', patches)


# In[ ]:


n, bins, patches = plt.hist(np_data['GrandCanyon'],
                            facecolor='cyan',
                            edgecolor='blue',
                            bins=10,
                           cumulative=True)
plt.show()


# In[ ]:


data = pd.read_csv('../input/matplotlib/sector_weighting.csv')
data


# In[ ]:


plt.pie(data['Percentage'],
       labels=data['Sector'])
plt.show()


# In[ ]:


plt.pie(data['Percentage'],
       labels=data['Sector'])
plt.axis('equal')  # perfect circle
plt.show()


# In[ ]:


colors = ['deeppink', 'aqua', 'magenta', 'silver', 'lime']


# In[ ]:


plt.pie(data['Percentage'],
       labels=data['Sector'],
       colors=colors,  # color for each sector
       autopct='%.2f') # represents the format for the displyaed values
plt.axis('equal')
plt.show()


# In[ ]:


plt.pie(data['Percentage'],
       labels=data['Sector'],
       colors=colors,  # color for each sector
       autopct='%.2f', # represents the format for the displyaed values
       startangle=90,  # start angle
       counterclock=False)
plt.axis('equal')
plt.show()


# In[ ]:


explode = (0, 0.1, 0, 0.3, 0)


# In[ ]:


plt.pie(data['Percentage'],
       labels=data['Sector'],
       colors=colors,  # color for each sector
       autopct='%.2f', # represents the format for the displyaed values
       explode=explode)
plt.axis('equal')
plt.show()


# In[ ]:


wedges, texts, autotexts = plt.pie(data['Percentage'],
       labels=data['Sector'],
       colors=colors,  # color for each sector
       autopct='%.2f') # represents the format for the displyaed values
plt.axis('equal')

print('Wedges: ', wedges)
print('Texts: ', texts)
print('Autotexts: ', autotexts)


# In[ ]:


wedges, texts, autotexts = plt.pie(data['Percentage'],
                                   labels=data['Sector'],
                                   colors=colors,  # color for each sector
                                   autopct='%.2f', # represents the format for the displyaed values
                                   explode=explode)

plt.axis('equal')

wedges[1].set(edgecolor='blue', linewidth=2)
texts[1].set(family='cursive', size=20)
autotexts[1].set(weight='bold', size=15)


# ## Autocorrelation
# * **Correlation:-** Measure of the relationship between two items or variables.
# * **Autocorrelation:-** Measures the relationship between a variable's current value and past value. 
# <img src="https://i.ibb.co/ZhtwjYD/Screenshot-2018-12-26-at-3-49-14-PM.png">
# <img src="https://i.ibb.co/wzDWv0Q/Screenshot-2018-12-26-at-3-49-04-PM.png">

# In[ ]:


grand_canyon_data = pd.read_csv('../input/grand-visits/grand_canyon_visits.csv')
grand_canyon_data.head()


# In[ ]:


grand_canyon_data['NumVisits'].describe()


# In[ ]:


grand_canyon_data['NumVisits'] = grand_canyon_data['NumVisits'] / 1000 # oveflow the correlation
grand_canyon_data['NumVisits'].describe()


# In[ ]:


plt.figure(figsize=(16, 8))
plt.acorr(grand_canyon_data['NumVisits'],
         maxlags=20) # range of x-axis (-20, 20)
plt.show()


# In[ ]:


plt.figure(figsize=(16, 8))
lags, c, vlines, hline = plt.acorr(grand_canyon_data['NumVisits'],
         maxlags=20) # range of x-axis (-20, 20)
plt.show()
print('lags: ', lags, '\n')
print('c: ', c, '\n')             # correlation values
print('vlines: ', vlines, '\n')
print('hline: ', hline, '\n')


# ## Stacked Plots

# In[ ]:


np_data = pd.read_csv('../input/matplotlib/national_parks.csv')
np_data.head()


# In[ ]:


x = np_data['Year']


# In[ ]:


y = np.vstack([np_data['Badlands'],
             np_data['GrandCanyon'],
             np_data['BryceCanyon']])
y


# In[ ]:


labels = ['Badlands',
         'GrandCanyon',
         'BryceCanyon']


# In[ ]:


plt.stackplot(x, y,
              labels=labels)
plt.legend(loc='upper left')
plt.show()


# In[ ]:


colors = ['sandybrown',
         'tomato',
         'skyblue']


# In[ ]:


plt.stackplot(x, y,
              labels=labels,
             colors=colors,
             edgecolor='grey')
plt.legend(loc='upper left')
plt.show()


# In[ ]:


np_data[['Badlands',
        'GrandCanyon',
        'BryceCanyon']] = np_data[['Badlands',
                                   'GrandCanyon',
                                   'BryceCanyon']].diff()
np_data.head()


# In[ ]:


# Stem plots are a good way to analyze fluctuating data
plt.figure(figsize=(10, 6))

plt.stem(np_data['Year'],
        np_data['Badlands'])
plt.title('Change in Number of Visitors')
plt.show()

