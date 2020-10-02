#!/usr/bin/env python
# coding: utf-8

# # Matplotlib

# # Imports and settings

# In[ ]:


# Standard library
import random

# Specific imports from standard library
from cycler import cycler

# Basic imports
import numpy as np
import pandas as pd

# Graphs
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


# # Default graphs settings
# 
# The first thing we should do is to set default graph settings. Some reasons to do this are:
# - Graphs are very small by default
# - Small text and values are hard to read
# - Colours are not readable by colourblind people
# 
# Just for comparison, we can compare the default graph with the one plotted after setting default parameters.

# In[ ]:


x = np.random.rand(100)
y = np.random.rand(100)


# In[ ]:


plt.figure()
plt.plot(x, y, "o", label="Points")
plt.xlabel("Label X")
plt.ylabel("Label Y")
plt.legend()
plt.show()


# In[ ]:


# Default graph settings

# Seaborn advanced                                                                                                                                                           
sns.set(style='ticks',          # 'ticks', 'darkgrid'                                                                                                                        
        palette='colorblind',   # 'colorblind', 'pastel', 'muted', 'bright'                                                                                                  
        #palette=sns.color_palette('Accent'),   # 'Set1', 'Set2', 'Dark2', 'Accent'                                                                                          
        rc = {                                                                                                                                                               
           'figure.autolayout': False,   # Automaticall set the figure size to fit in canvas                                                                       
           'figure.figsize': (16, 10),   # Figure size - width, height (in inches)    
           'figure.max_open_warning': False,
           'figure.titlesize': 32,      # Whole figure title size (plt.suptitle)
           'legend.frameon': True,      # Frame around the legend                                                                                                              
           'patch.linewidth': 2.0,      # Width of frame around the legend                                                                                                        
           'lines.markersize': 6,       # Size of marker points                                                                                                                      
           'lines.linewidth': 2.0,      # Width of lines                                                                                                                      
           'font.size': 14,             # Size of font on axes values                                                                                                           
           'legend.fontsize': 18,       # Font size in the legend                                                                                                           
           'axes.labelsize': 22,        # Font size of axes names                                                                                                                  
           'axes.titlesize': 26,        # Font size of subplot titles (plt.title)                                                                                                                 
           'axes.grid': True,           # Set grid on/off                                                                                                                             
           'grid.color': '0.9',         # Color of grid lines - 1 = white, 0 = black                                                                                          
           'grid.linestyle': '-',       # Style of grid lines                                                                                                              
           'grid.linewidth': 1.0,       # Width of grid lines                                                                                                                
           'xtick.labelsize': 22,       # Font size of values on X-axis                                                                                                  
           'ytick.labelsize': 22,       # Font size of values on Y-axis                                                                                                       
           'xtick.major.size': 8,       # Size of ticks on X-axis                                                                                                    
           'ytick.major.size': 8,       # Size of ticks on Y-axis                                                                                                 
           'xtick.major.pad': 10.0,     # Distance of axis values from X-axis                                                                                               
           'ytick.major.pad': 10.0,     # Distance of axis values from Y-axis   
           'image.cmap': 'viridis'      # Default colormap
           }                                                                                                                                                                 
       )     


# In[ ]:


plt.figure()
plt.plot(x, y, "o", label="Points")
plt.xlabel("Label X")
plt.ylabel("Label Y")
plt.legend()
plt.show()


#  # How to draw graphs in matplotlib?
# 
# ## Scatterplot, lineplot and some basics
# 
# There are only a few things you need to create a graph. For the very basic chart you will need just:
# - `plt.figure(figsize=(16,10))` - Creates a figure, a canvas to draw on. Probably the most used argument here is specifying the size of a figure, which can be set in default settings and changed only for a figure with multiple subplots. 
# - `plt.plot(X, Y, "-bo", label="Label")` - The basic command to draw on the canvas. `X` and `Y` have to be iterable (lists, np.arrays, pd.Series) of the same size. The third argument is visually very important, it specifies both colour and type of the figure. If you write `"-"`, then it will be a line plot, if you write `"o"`, it will be a scatterplot, if you write both, it will be line plot with highlighted points. You can also write a shortcut for colour inside like `"b"` as blue in our case. A label is a text which will be shown in the legend for this particular data. There are usually more `plt.plot` commands, because you may want to plot different groups of data.
# - `plt.xlabel` - Sets label of x-axis.
# - `plt.ylabel` - Sets label of y-axis.
# - `plt.legend` - Draws the legend with labels specified in `plt.plot` commands.
# - `plt.show` - Shows the graph. Note that in Jupyter the graph is shown even without calling this command, but it also returns the last object you called (in our case the legend).

# In[ ]:


x = np.random.rand(100)
y = np.random.rand(100)
t = np.arange(100)


# In[ ]:


plt.figure()
plt.plot(x, y, "o", label="Points - type 1")
plt.plot(y, x, "o", label="Points - type 2")
plt.xlabel("Label X")
plt.ylabel("Label Y")
plt.legend()
plt.show()


# In[ ]:


plt.figure()
plt.plot(t, y, "-o", label="Points - type 1")
plt.xlabel("Label X")
plt.ylabel("Label Y")
plt.legend()
plt.show()


# We used as markers circles, however it is possible to use squares (`s`), diamonds (`d`), triangles (`v`) or other. You can find the full list of available markers in the [documentation](https://matplotlib.org/3.1.1/api/markers_api.html). (It is also interesting to check out, [how markers can be filled](https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/marker_fillstyle_reference.html).)

# # Scatterplot variations
# 
# ## Colourfull bubble plot
# 
# Apart from drawing scatterplots using `plt.plot`, matplotlib has a special function for it called `plt.scatter`. For a simple scatterplot, there is no difference between these function, however, if we wished to draw a bubble plot (points can differ in size) or draw points with different colours, `plt.plot` wouldn't allow us to do that. Instead, we have to use `plt.scatter`.
# 
# To `plt.scatter` you can pass additional arguments `s` and `c`. `s` for "size" is a scalar or iterable. If it is scalar, then all points will have the same size, if it is iterable, every point will have the size you specify. It has to be passed ordered by which I mean that N-th point in `X` and `Y` will have the size of N-th field in `s`. Note, that if we wish to have markers of size 6 (as we set in default settings), we need to pass square of that value here. For `c` as "colour" it is similar you only pass a single string or an iterable of strings with the same length as `X` and `Y`. 

# In[ ]:


N_points = 100
x = np.random.rand(N_points)
y = np.random.rand(N_points)
z = np.random.rand(N_points)
colors = [random.choice(["b", "r", "g", "y", "m", "c", "k", "w"]) for i in range(N_points)]


# In[ ]:


plt.figure()
plt.scatter(x, y, s=z*300, c=colors, alpha=0.5)
plt.show()


# ## Colours
# 
# In matplotlib there are eight basic colours for which a shortcut exist:
# - b - Blue
# - r - Red
# - g - Green
# - y - Yellow
# - m - Magenta
# - c - Cyan
# - k - Black
# - w - White
# 
# A strange thing is that we used exactly those colours in the previous bubble plot, but there is no red and magenta, however, you can see orange and pink. 
# Let's try to plot it again, but not to use shortcuts this time.

# In[ ]:


N_points = 100
x = np.random.rand(N_points)
y = np.random.rand(N_points)
z = np.random.rand(N_points)
colors = [random.choice(["blue", "red", "green", "yellow", "magenta", "cyan", "black", "white"]) for i in range(N_points)]


# In[ ]:


plt.figure()
plt.scatter(x, y, s=z*300, c=colors, alpha=0.5)
plt.show()


# Now colours match descriptions, but they are a bit more aggressive and I think that we can all agree that the previous graph was visually more plausible. The reason why the previous graph shows different colours are the default settings. We set the seaborn palette to `colorblind` and it changed default colours for colour shortcuts. 

# ## Opacity
# 
# There are multiple way how to set a color:
# - Named color - `mediumvioletred`
# - Hex code - `#C71585`
# - RGB channels - `(0.7812, 0.0859, 0.5234) == (199, 21, 133) / 256`
# 
# It is not possible to pass an array or a list to `alpha` argument, so if we wish to set opacity different for each point we need a different way to do it. If we pass colours as RGB channels, it is possible to add the fourth alpha channel and that is how we pass opacity:
# 
# - RGBA channels - `(0.7812, 0.0859, 0.5234, 1)`

# In[ ]:


N_points = 100
x = np.random.rand(N_points)
y = np.random.rand(N_points)
z = np.random.rand(N_points)
colours = [(0.7812, 0.0859, 0.5234, opacity) for opacity in z]

plt.figure()
plt.scatter(x, y, s=15**2, c=colours)
plt.show()


# ## Heatmap
# 
# Heatmaps are the most easily created using `np.meshgrid` and `plt.pcolormesh`. To mesh grid you need to pass `x` and `y` range and a function to compute `z` values for all `x` and `y` combinations. To `plt.colormesh` you just then simply pass all three, `x`, `y` and `z`. Don't forget to plot colour bar using `plt.colobar()` as well.

# In[ ]:


def f(x, y):
    return x**2 + y**2

x, y = np.meshgrid(np.linspace(-50, 50, 100), np.linspace(-50, 50, 100))
Z = f(x, y)


# In[ ]:


plt.figure()
plt.pcolormesh(x, y, Z)
plt.colorbar()
plt.show()


# ## Barchart

# In[ ]:


y = np.random.rand(5)
x = range(len(y))
xlabels = ["A", "B", "C", "D", "E"]


# In[ ]:


plt.figure()
plt.bar(x, y, edgecolor="k")
plt.xticks(x, xlabels)
plt.show()


# ## Symbols in legend
# 
# Sometimes, you may want to show different markers in a legend than are actually in a graph. We illustrate it here in two cases:
# - You have a lot of data points and want to make a scatterplot with very low opacity. In this case, the marker in the legend will have the same low opacity as in the graph, so no-one will see it at all in the legend. Here, you want to change the marker in the legend to have different opacity.
# - You have a scatterplot where there are multiple symbols for different classes, however, markers have colour as if they would lie on a heatmap.
# 
# Lets first look at the scatterplot where you can not see the marker in the legend due to low opacity.

# In[ ]:


x = np.random.normal(0, 0.1, 100000)
y = np.random.normal(0, 0.1, 100000)


# In[ ]:


plt.figure()
plt.plot(x, y, "o", label="Points - type 1", alpha=0.01)
plt.xlabel("Label X")
plt.ylabel("Label Y")
plt.legend()
plt.show()


# In order to create a custom legend, we need to create a list of rows to show in a legend and pass it to `plt.legend` afterwards. Elements are created using `Line2D` or `Patch` elements. It happens to me very often that I want to show only a symbol in a legend, not a line, however, `Line2D` plots a line and a marker. An easy fix to this is to set the colour of the line to white, so it is not visible in the legend.

# In[ ]:


legend_elements = [Line2D([0], [0],
                          marker='o',              # Marker to show in a legend
                          color='w',               # Set this to white, so that line behind marker is not visible
                          label='Points - type 1', # Label next to marker
                          markerfacecolor='b',     # Color of the marker
                          markersize=15            # Size of the marker
                         )]

plt.figure()
plt.plot(x, y, "o", alpha=0.01)
plt.xlabel("Label X")
plt.ylabel("Label Y")
plt.legend(handles=legend_elements)
plt.show()


# The next example is a little bit more intricate. We plot three-dimensional information in the scatterplot, i.e. we traditionally use `x` and `y` axes as usual and then use colour as a representation of the third dimension. Colour is more frequently used for a distinction between classes, however now, we will use the shape of markers instead. (Bonus tip: If you plot normal scatterplot with points in multiple classes it is better to use both colour and shape to distinguish between classes, not just one of them.)
# 
# You can now proceed and check how markers are created, it is the same as in the last example, the only difference is that we plot markers white with a black border as their colour has not a meaning of a class. For those interested, we will describe, how float numbers are mapped to colours. At first, we need to find the minimum and maximum (together those values are called the "extent") of float numbers plotted as colour. Then we select a colourmap. A colourmap takes as input number between 0 and 1 and maps it to colour. However, if our numbers are lower or higher than 0 and 1, we wouldn't be able to map them to colour, so we create a `Normalize` object to which we pass the extent. The purpose of normalizing is to map numbers within a specified extent to numbers between zero and one. 
# 
# So in the end, if we map numbers to colours we have to:
# - Specify the extent for norm
# - Map float numbers within the extent to numbers within zero and one using `Normalize`
# - Map numbers within zero and one to colour using a colourmap
# 
# The last thing needed to add is to create `ScalarMappable` to which we pass colourmap and `Normalize` and then we pass it to `plt.colorbar` so that matplotlib knows which colourmap to plot in `plt.colorbar` and what is the extent.

# In[ ]:


x = np.arange(1, 10)
y1 = 100 - x**2
y2 = 120 - 1.2 * x**2
z1 = np.sqrt(y1) - 3
z2 = np.sqrt(y2)

min_y = np.minimum(z1, z2).min()
max_y = np.maximum(z1, z2).max()


# In[ ]:


legend_elements = [Line2D([0], [0],                          
                          marker="o",                        
                          color="w",                         
                          label="Circles",                   
                          markerfacecolor="w",               
                          markeredgecolor="k",               
                          markersize=15                      
                         ),
                   Line2D([0], [0],                          
                          marker="d",
                          color="w",
                          label="Diamonds",
                          markerfacecolor="w",
                          markeredgecolor="k",
                          markersize=15)]

cmap = cm.viridis                                            
norm = Normalize(vmin=min_y, vmax=max_y)                     

plt.figure()
plt.scatter(x, y1, c=cmap(norm(z1)), marker="o", s=15**2)
plt.scatter(x, y2, c=cmap(norm(z2)), marker="d", s=15**2)
plt.xlabel("Label X")
plt.ylabel("Label Y")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm)
plt.legend(handles=legend_elements)
plt.show()


# ## The difference between `plt.plot` and `ax.plot` - multiple graphs in a single figure
# 
# I am sure that you have come across a situation where it would be better to plot multiple graphs in a single figure. This is where you should stop using `plt.plot` and instead define a figure and number of subplots using `plt.subplots(N_rows, N_columns)`. Axis here means a single graph in the figure. 
# 
# Matplotlib is an object-oriented library. Two objects you come into contact the most are figures and axes (in the sense of a subplot). There are basically three ways of increasing difficulty of how to create a single figure and axes and to plot something:
# 
# 1. Single graph - Create a figure using `plt.figure` and create an axis and plot at the same time using `plt.plot`
# 2. Multiple graphs of the same size - Create a figure and multiple axes (i.e. subplots) using `plt.subplots`, then call `ax.plot` for every axis, creating graphs one by one.
# 3. Multiple graphs of different sizes - Create an empty figure using `plt.figure`, then use `fig.add_subplot` for creating individual axes. Plot by calling `ax.plot` for every axis.
# 
# If you need only a single graph, there is no need why you should create single subplot and use axes, `plt.plot` is just the easiest and the fastest. 
# 
# Now, let's look at the second option, where all graphs have the same size, but there is a multiple of them in the figure.

# In[ ]:


N_points = 100

x = np.random.rand(N_points)
y = np.random.rand(N_points)
z = np.random.rand(N_points)


# In[ ]:


# fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
# ax1 = axes[0][0]
# ax2 = axes[0][1]
# ax3 = axes[1][0]
# ax4 = axes[1][1]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)   # Identical to preceeding five lines

ax1.plot(x, y, "o", label="Points A")
ax1.set_ylabel("Label Y")
ax1.set_title("Title A")
ax1.legend(loc="upper right")

ax2.plot(y, z, "s", label="Points B")
ax2.set_title("Title B")
ax2.legend(loc="upper right")

ax3.plot(x, z, "d", label="Points C")
ax3.set_ylabel("Label Y")
ax3.set_xlabel("Label X")
ax3.set_title("Title C")
ax3.legend(loc="upper right")

ax4.plot(z, y, "v", label="Points D")
ax4.set_xlabel("Label X")
ax4.set_title("Title D")
ax4.legend(loc="upper right")

plt.suptitle("Four graphs")
plt.subplots_adjust(wspace=0.10, hspace=0.30)
plt.show()


# A few things to note about `plt.subplots` and using individual axes:
# - There are multiple ways how to unpack axes into individual axes. 
# - Use `sharex=True` and `sharey=True` if all data points in your subplots lie in the same range. It prevents you from cluttering the figure with identical numbers. It is enough when only left graphs have specified Y-range and bottom graphs X-range.
# - Don't forget that you have to call everything you want to be plotted for each subplot individually. For example, you need to set title or legend for each subplot. If you used `plt.legend()` before `plt.show()` as usual, only the last subplot would have legend shown.
# - You can specify a common title for all graphs using `plt.suptitle`.
# - You can adjust horizontal and vertical space between subplots using `plt.subplots_adjust`.

# ## Different graph sizes in a single figure and grid settings
# 
# The most complex way of creating subplots with matplotlib is when they have different sizes. In order to start with it, you need to define figure using `plt.figure`. Then, you can use `fig.add_subplot` for creating subplots. However, I find passing arguments to `fig.add_subplot` a bit confusing when not used with grid specifications. Grid specifications are the most easily imagined as Excel cells. You need to specify how many columns and rows you will use and you can also set width and height for each column or row. To `fig.add_subplot` you then pass grid specifications and slice it as NumPy array.
# 
# Let's go through an example I have actually really used. I needed to compare two heatmaps, so I created an A4 size figure with four square plots and four thin rectangle plots. Square plots show new heatmap, old heatmap, the difference between heatmaps, and zoom of difference to origin around (0,0). Then I realized that I need to observe values around axes more closely, so I added zoom along X and Y axes for both new and old heatmap. 

# In[ ]:


def f1(x, y):
    return np.exp(-x)+np.exp(-y)

def f2(x, y):
    return 1/(1+x**2) + 1/(1+3*y**2)

x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)

X, Y = np.meshgrid(x, y)
Z1 = f1(X, Y)
Z2 = f2(X, Y)
Z_diff = Z2 - Z1
Z_diff_max_abs = np.maximum(Z_diff.max(), abs(Z_diff.min()))


fig = plt.figure(figsize=(15,27))
gs = fig.add_gridspec(ncols=2, nrows=6,
                      height_ratios=[1, 1, 0.2, 0.2, 0.2, 0.2])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, :])
ax6 = fig.add_subplot(gs[3, :])
ax7 = fig.add_subplot(gs[4, :])
ax8 = fig.add_subplot(gs[5, :])

im = ax1.pcolormesh(x, y, Z1)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
ax1.set_title("Function 1")

im = ax2.pcolormesh(x, y, Z2)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
ax2.set_title("Function 2")

im = ax3.pcolormesh(x, y, Z_diff, cmap="seismic")
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
im.set_clim(-Z_diff_max_abs, Z_diff_max_abs)
ax3.set_title("Difference 2-1")

im = ax4.pcolormesh(x, y, Z_diff, cmap="seismic")
divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
im.set_clim(-Z_diff_max_abs, Z_diff_max_abs)
ax4.set_xlim((0,2.5))
ax4.set_ylim((0,2.5))
ax4.set_title("Zoomed difference 2-1")

im = ax5.pcolormesh(x, y, Z1)
divider = make_axes_locatable(ax5)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
ax5.set_xlim((0,10))
ax5.set_ylim((0,2.5))
ax5.set_title("Zoom to F1 along X-axis")

im = ax6.pcolormesh(x, y, Z2)
divider = make_axes_locatable(ax6)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
ax6.set_xlim((0,10))
ax6.set_ylim((0,2.5))
ax6.set_title("Zoom to F2 along X-axis")

im = ax7.pcolormesh(x, y, Z1)
divider = make_axes_locatable(ax7)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
ax7.set_xlim((0,2.5))
ax7.set_ylim((0,10))
ax7.set_title("Zoom to F1 along Y-axis")

im = ax8.pcolormesh(x, y, Z2)
divider = make_axes_locatable(ax8)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
ax8.set_xlim((0,2.5))
ax8.set_ylim((0,10))
ax8.set_title("Zoom to F2 along Y-axis")

plt.subplots_adjust(wspace=0.4, hspace=0.5)
plt.show()


# ## How to create colorcycler 
# 
# If you have ever tried to plot spaghetti graph or you just plotted more than ten classes you for sure noticed that after the tenth colour the colours start to repeat. Let's look at it in the figure below.

# In[ ]:


x = np.arange(0, 100)


# In[ ]:


plt.figure()
for i in range(1, 16):
    plt.plot(x, x * i * 0.1, "-", label=f"{i}-th color")
plt.legend()
plt.show()


# Colors start to repeat because matplotlib is using colorcyclers where only ten colors are set. Colorcyclers are just iterators over a list of colors and they can be changed easily. You only need to define a list of colors, create a cycler and set it for an individual axes using `ax.set_prop_cycle(cycler('color', list_of_colors))` or set it globally using `plt.rc('axes', prop_cycle=(cycler('color', list_of_colors)))`.
# 
# Bonus tip: Usually you don't need more than ten colours and if you do, the graph is likely a spaghetti graph and you should think about making it interactive or plotting it as small multiples.

# In[ ]:


list_of_colors = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"]   # Paul Tol colorpalette

fig, ax = plt.subplots()
ax.set_prop_cycle(cycler('color', list_of_colors))
for i in range(1, 16):
    ax.plot(x, x * i * 0.1, "-", label=f"{i}-th color")
plt.legend()
plt.show()


# It might also come you handy to shorten current colour cycler for a single axis. Maybe there are two or multiple groups you want to plot, but there is some relation between groups, so maybe the first elements of each group should have same colour and groups are distinguished by a different marker. The easiest solution in such a case is to shorten the colour cycler, assuming that groups have the same number of elements.

# In[ ]:


fig, ax = plt.subplots()
ax.set_prop_cycle(color=plt.rcParams['axes.prop_cycle'].by_key()['color'][:5])

for i in range(1, 6):
    ax.plot(x, x * i * 0.1, "-", label=f"{i}-A")
for i in range(1, 6):
    ax.plot(x, x * (i+5) * 0.1, "--", label=f"{i}-B")
plt.legend()
plt.show()


# ## Different legend locations
# 
# The last thing discussed in this notebook, for now, is placing of a legend. The legend is automatically placed in a part of a graph with the least amount of data so that everything is visible. Nevertheless, it is possible to change the location of a graph using the `loc` argument. To `loc` we need to pass a string specifying the vertical and horizontal position. Vertical positions might be `lower`, `center` or `upper` and horizontal positions may be `left`, `center` or `right`. So valid location is for example `upper left`.
# 
# Graphs can contain a lot of data and sometimes it might be better to place the legend completely outside of the graph for which you can use `bbox_to_anchor` argument and pass it a position.
# 
# If you wish to learn more about placing a legend, we would recommend you to read [this StackOverflow answer](https://stackoverflow.com/a/43439132/3944404).

# In[ ]:


plt.figure()
for i in range(1, 16):
    plt.plot(x, x * i * 0.1, "-", label=f"{i}-th color")
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
plt.show()


# In[ ]:


plt.figure()
for i in range(1, 16):
    plt.plot(x, x * i * 0.1, "-", label=f"{i}-th color")
plt.legend(bbox_to_anchor=(1.04,0.6), borderaxespad=0, ncol=3)
plt.show()


# In[ ]:


plt.figure()
for i in range(1, 16):
    plt.plot(x, x * i * 0.1, "-", label=f"{i}-th color")
plt.legend(bbox_to_anchor=(0.95,-0.1), borderaxespad=0, ncol=5)
plt.show()


# In[ ]:




