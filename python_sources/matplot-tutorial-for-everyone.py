#!/usr/bin/env python
# coding: utf-8

# This kernel aims at depicting various plots and visualization techniques which can be done using matplotlib library.It covers the basic to advance level plotting functions of the libraray.It contains several examples which will give you hands-on experience in generating plots in python.

# # Please upvote if you like this kernel for further encouragement

# # 1. Introduction to Matplot Library
# 
# It is the python library for visualizing data by creating diffrent graphs and charts. Some of the key points related to it are mentioned below :-
# 
# 1. Matplotlib is a 2-D plotting library that helps in visualizing figures.
# 2. It took inspiration from MATLAB programming language and provides a similar MATLAB like interface for graphics.
# 3. It really integrated well pandas which is used for data manipulation
# 4. It is a robust, free and easy library for data visualization.
# 

# # 2.Matplotlib installations and basics 
# 
# The first step is to install the matplotlib. if you are using Anaconda then it is already installed. 
# 
# ### Installation
# 
# If matplotlib is not already installed, you can install it by using the command
# 
# pip install matplotlib
# 
# ### Import Library
# 
# In the following ways you can import matplot
# 
# 1. [import matplotlib.pyplot as plt](http://)
# 2. from matplotlib import pyplot as plt
# 
# 

# # 3.Anatomy of Matplotlib Figure
# 
# The figure contains the overall window where plotting happens, contained within the figure are where actual graphs are plotted. Every Axes has an x-axis and y-axis for plotting. And contained within the axes are titles, ticks, labels associated with each axis. An essential figure of matplotlib is that we can more than axes in a figure which helps in building multiple plots, as shown below. In matplotlib, pyplot is used to create figures and change the characteristics of figures.
# 
# ![image.png](attachment:image.png)
# 
# 
# 
# 

# The figure contains the overall window where plotting happens, contained within the figure are where actual graphs are plotted. Every Axes has an x-axis and y-axis for plotting. And contained within the axes are titles, ticks, labels associated with each axis. An essential figure of matplotlib is that we can more than axes in a figure which helps in building multiple plots, as shown below. In matplotlib, pyplot is used to create figures and change the characteristics of figures.
# 
# You can think of the figure as a big graph consisting of multiple sub-plots. Sub-plot can be one or more than one on a figure. In graphics world, it is called 'canvas'.
# 
# ![image.png](attachment:image.png)
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt


# In[ ]:


## Basic Plot

x = [10, 20, 30, 40, 50]
y = [45, 75, 30, 80, 40]
plt.bar(x,y)
plt.show()


# ## 4.Functions used for different plots 
# 
# The following methods are used to draw diffrent types of graphs in matplot library
# 
# ![](http://)
# ![image.png](attachment:image.png)

# ## 4.1 Bar Graph
# 
# Bar graph is mainly used to compare between two categories or groups.For example you want to show the comparison between employees and their incomes.

# In[ ]:


x = ['Ajay', 'Tom', 'Vicky', 'Anouska','Mikhail']
y = [4500, 7500, 3000, 8000, 40000]
plt.title(" Bar graph example") # Name title of the graph
plt.xlabel('Employees') # Assign the name of the x axis
plt.ylabel("Salary") # Assign the name of the y axis
plt.bar(x, y, color='red') # Change bar color
plt.show()


# **We can style your graph using the following functions**
# 
# 1. **plt.title( )** for specifying title of your plot.
# 2. **plt.xlabel( )** for labeling x-axis.
# 3. **plt.ylabel( )** for labeling y-axis.
# 4. **color** = option in plt.bar( ) for defining color of bars.

# ### 4.1.1 How to show values and label at the top of the graph
# 

# In[ ]:


barplot = plt.bar(x, y)
for bar in barplot:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom') #va: vertical alignment y positional argument
    
plt.title(" Bar graph example")
plt.xlabel('Employees')
plt.ylabel("Salary")


# **Code Explanation**
# 
# 1. **.get_height()** returns height of rectangle of each bar which is basically a value of y-axis
# 2. **plt.text()** is used to place text on the graph
# 3. **get_x() and get_width()**:- to find the value of the x axis.

# ### 4.1.2 How to increase the size of the graph

# In[ ]:


plt.figure(figsize=(7,7))
barplot = plt.bar(x, y)
for bar in barplot:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom') #va: vertical alignment y positional argument
    
plt.title(" Bar graph example")
plt.xlabel('Employees')
plt.ylabel("Salary")
plt.show()


# **We have used plt.figure(figsize=(7,7))**

# ### 4.1.3 How to hide axis
# 
# Sometimes we hide y-axis to give aesthetic touch to our bar graph. To do this in matplotlib, we can leverage plt.yticks( ) function.[ ] means empty list.[](http://)

# In[ ]:


plt.figure(figsize=(7,7))
barplot = plt.bar(x, y)
for bar in barplot:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom') #va: vertical alignment y positional argument
    plt.yticks([])
plt.title(" Bar graph example")
plt.xlabel('Employees')
plt.ylabel("Salary")
plt.show()


# ## 4.2 Horizontal Bar Graph
# 
# Horizontal bar graph can be used to solve the same purpose as of column bar chart . But this time the bars are horizontal to give a diffrent look and feel

# In[ ]:


plt.barh(x,y)
plt.title("Horizontal Bar graph example")
plt.xlabel("Employees")
plt.ylabel('Salary')


# ### 4.2.1 How to sort or order bars
# 
# We can arrange the bars based on the values. We need to combine both the lists and then sort them based on value of list y. zip( ) function is used to combine items of lists x and y. sorted( )sort them based on y. Then we split and store it in x and y lists.

# In[ ]:


y,x = zip(*sorted(zip(y,x)))
plt.barh(x,y)


# ### 4.2.2 Reverse order of bars
# 
# plt.subplot() is used to find out current axes and then invert function assists to reverse the order.

# In[ ]:


plt.barh(x,y)
ax=plt.subplot()
ax.invert_yaxis()


# ### 4.2.3 Professional Themes / Styles for Graphs

# In[ ]:


print(plt.style.available)


# In[ ]:


plt.barh(x,y)
plt.title("Horizontal Bar graph example")
plt.xlabel("Employees")
plt.ylabel('Salary')
plt.style.use('dark_background')


# In[ ]:


plt.barh(x,y)
plt.title("Horizontal Bar graph example")
plt.xlabel("Employees")
plt.ylabel('Salary')
plt.style.use('seaborn-dark-palette')


# ## 4.3 Line Graph
# 
# A line chart or line graph is a type of chart which displays information as a series of data points called 'markers' connected by straight line segments.

# In[ ]:


import pandas as pd

df = pd.DataFrame({"Year" : [2015,2016,2017,2018,2019], 
                  "Salary_Hike" : [2000, 3000, 4000, 3500, 6000]})


# In[ ]:


# plot line chart
plt.plot(df["Year"], df["Salary_Hike"])
plt.title("Simple Line Plot")
plt.xlabel('Year')
plt.ylabel('Salary_Hike')
plt.style.use('seaborn-white')


# ### 4.3.1 Add Marker in Line Plot
# 
# By making use of style= option, you can include marker with customization in color and style.

# In[ ]:


ax = df.plot(x="Year", y="Salary_Hike", kind="line", title ="Simple Line Plot", legend=True, style = 'b--')
ax.set(ylabel='Salary_Hike', xlabel = 'Year', xticks =df["Year"])


# ### 4.3.2 Adding multiple lines in the line graph
# 

# In[ ]:


import pandas as pd

product = pd.DataFrame({"Year" : [2014,2015,2016,2017,2018], 
                  "ProdAManufacture" : [2000, 3000, 4000, 3500, 6000],
                  "ProdBManufacture" : [3000, 4000, 3500, 3500, 5500]})


# Multi line plot
ax = product.plot("Year", "ProdAManufacture", kind="line", label = 'Product A manufacture')
product.plot("Year", "ProdBManufacture", ax= ax , kind="line", label = 'Product B manufacture', title= 'MultiLine Plot') #ax : axes object

# Set axes
ax.set(ylabel='Sales', xlabel = 'Year', xticks =df["Year"])


# ## 4.4 Scatter Plot
# 
# A scatter plot is mainly used to show relationship between two continuous numeric variables. 
# 
# kind = 'scatter' is used for creating scatter diagram.

# In[ ]:


ax = product.plot("Year", "ProdAManufacture", kind='scatter', color = 'red', title = 'Year by ProductA Manufacture')
ax.set(ylabel='ProdAManufacture', xlabel = 'Year', xticks =df["Year"])
plt.show()


# ## 4.5 Pie Chart
# 
# A pie chart is a circular graph which splits data into slices to show numerical proportion of each category. If you are showing percentages, all of them should add to 100%.

# In[ ]:


Goals = [20, 12, 11, 4, 3]
players = ['Ronaldo', 'Messi', 'Suarez', 'Neymar', 'Salah', ]
comp = pd.DataFrame({"Goals" : Goals, "players" : players})
ax = comp.plot(y="Goals", kind="pie", labels = comp["players"], autopct = '%1.0f%%', legend=False, title='No of Goals scored')

# Hide y-axis label
ax.set(ylabel='')


# ### 4.5.1 Customize Pie Chart
# 
# The default startangle is 0. By making startangle = 90 , everything will be rotated counter-clockwise by 90 degrees. By using explode = option, you can explode specific categories

# In[ ]:


ax = comp.plot(y="Goals", kind="pie", labels = comp["players"], startangle = 90, shadow = True, 
        explode = (0.1, 0.1, 0.1, 0, 0), autopct = '%1.0f%%', legend=False, title='No of Goals scored')
ax.set(ylabel='')
plt.show()


# ## 4.6 Histogram
# 
# 
# Histogram is used to show the frequency distribution of a continuous variable.

# In[ ]:


# Creating random data
import numpy as np
np.random.seed(1)
mydf = pd.DataFrame({"Age" : np.random.randint(low=20, high=100, size=50)})

# Histogram
ax = mydf.plot(bins= 5, kind="hist", rwidth = 0.7, title = 'Distribution - Marks', legend=False)
ax.set(xlabel="Bins")
plt.show()


# ### 4.6.1 How to add multiple sub-plots
# 
# Matplotlib provides two interfaces to do this task - plt.subplots( ) and plt.figure(). 

# In[ ]:


labels = ['Amsterdam', 'Berlin', 'Brussels', 'Paris']
x1 = [45, 30, 15, 10]
x2 = [25, 20, 25, 50]

finaldf = pd.DataFrame({"2017_Score":x1, "2018_Score" : x2, "cities" : labels})


# In[ ]:


# Method 1

fig = plt.figure()

ax1 = fig.add_subplot(121)
ax = finaldf.plot(x="cities",  y="2017_Score", ax=ax1, kind="barh", legend = False, title = "2017 Score")
ax.invert_yaxis()

ax2 = fig.add_subplot(122)
ax = finaldf.plot(x="cities",  y="2018_Score", ax=ax2, kind="barh", legend = False, title = "2018 Score")
ax.invert_yaxis()
ax.set(ylabel='')


# In[ ]:


#Method 2

fig, (ax0, ax01) = plt.subplots(1, 2)

ax = finaldf.plot(x="cities",  y="2017_Score", ax=ax0, kind="barh", legend = False, title = "2017 Score")
ax.invert_yaxis()

ax = finaldf.plot(x="cities",  y="2018_Score", ax=ax01, kind="barh", legend = False, title = "2018 Score")
ax.invert_yaxis()
ax.set(ylabel='')


# ### 4.6.2 How to show sub-plots vertically

# In[ ]:


fig = plt.figure()
ax1 = fig.add_subplot(211)
ax = finaldf.plot(x="cities",  y="2017_Score", ax=ax1, kind="barh", legend = False, title = "2017 vs 2018 Score")
ax.invert_yaxis()
plt.xticks(range(0,60,10))
ax.set(ylabel='')

ax2 = fig.add_subplot(212)
ax = finaldf.plot(x="cities",  y="2018_Score", ax=ax2, kind="barh", legend = False)
ax.invert_yaxis()
ax.set(ylabel='')


# ## 4.7 Contour Plots 
# 
# Contour plots (sometimes called Level Plots) are a way to show a three-dimensional surface on a two-dimensional plane. It graphs two predictor variables X Y on the y-axis and a response variable Z as contours. These contours are sometimes called z-slices or iso-response values

# In[ ]:


def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 -y ** 2)
n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap='jet')
C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)


# ## 4.8 Imshow Plots
# 
# Imshow( RGB ) displays the truecolor image RGB in a figure. imshow( BW ) displays the binary image BW in a figure. For binary images, imshow displays pixels with the value 0 (zero) as black and 1 as white. imshow( X , map ) displays the indexed image X with the colormap map .

# In[ ]:


def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
n = 10
x = np.linspace(-3, 3, 4 * n)
y = np.linspace(-3, 3, 3 * n)
X, Y = np.meshgrid(x, y)
plt.imshow(f(X, Y))
plt.show()


# ## 4.9 Quiver Plots 
# 
# A quiver plot is a type of 2D plot that shows vector lines as arrows. Quiver plots are useful in Electrical Engineering to visualize electrical potential and useful in Mechanical Engineering to show stress gradients.
# 
# 

# ### 4.9.1 Quiver plot with one arrow

# In[ ]:


fig, ax = plt.subplots()


x_pos = 0
y_pos = 0
x_direct = 1
y_direct = 1


ax.quiver(x_pos, y_pos, x_direct, y_direct)
ax.set_title('Quiver plot with one arrow')


plt.show()


# ### 4.9.2 Quiver plot with two arrows

# In[ ]:


fig, ax = plt.subplots()

x_pos = [0, 0]
y_pos = [0, 0]
x_direct = [1, 0]
y_direct = [1, -1]


ax.quiver(x_pos,y_pos,x_direct,y_direct,
         scale=5)
ax.axis([-1.5, 1.5, -1.5, 1.5])


plt.show()


# ## 4.9.3 Quiver plot using a meshgrid
# 
# 
# A quiver plot with two arrows is a good start, but it is tedious and repetitive to add quiver plot arrows one by one. To create a complete 2D surface of arrows, we'll utilize NumPy's meshgrid() function.
# 
# First, we need to build a set of arrays that denote the x and y starting positions of each quiver arrow on the plot. The quiver arrow starting position arrays will be called X and Y.
# 
# We can use the x, y arrow starting positions to define the x and y components of each quiver arrow direction. We will call the quiver arrow direction arrays u and v. For this plot, we will define the quiver arrow direction based upon the quiver arrow starting point using the equations below.
# 
# 
# ![image.png](attachment:image.png)

# In[ ]:


n = 8
X, Y = np.mgrid[0:n, 0:n]
plt.quiver(X, Y)
plt.show()


# ## 4.10 3 D Plots 

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')


# ## 4.11 Polar Axis

# In[ ]:


#Polar Axis 
r = np.arange(0, 2, 0.01)
theta = 2 * np.pi * r

ax = plt.subplot(111, projection='polar')
ax.plot(theta, r)
ax.set_rmax(2)
ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
ax.grid(True)

ax.set_title("A line plot on a polar axis", va='bottom')
plt.show()


# # 5.Useful Tips

# ## 5.1 How to save plot

# # save plot
# x = ['A','B','C', 'D']
# 
# y = [1,2,3,4]
# 
# fig = plt.figure()
# 
# plt.bar(x, y)
# 
# fig.savefig('C:/My Files/Blog/myfirstpic.png')
# 
# plt.close(fig)

# ## 5.2 Setting different limits in axis

# In[ ]:


x = ['A','B','C', 'D']
y = [100,119,800,900]
plt.bar(x, y)
ax = plt.subplot()
ax.set_ylim(0,1000)
plt.show()


# ## 5.3 Mutiple Entries in the Legend

# In[ ]:


plt.legend(["First_Legend","Second_Legend"])


# ## 5.4  Changing font size, weight and color in graph

# In[ ]:


plt.bar(x, y)
plt.title("Cost of Living", fontsize=18, fontweight='bold', color='blue')
plt.xlabel("Cities", fontsize=16)
plt.ylabel("Score", fontsize=16)


# # 6. References 
# 
# 
# 1. https://problemsolvingwithpython.com/06-Plotting-with-Matplotlib/06.15-Quiver-and-Stream-Plots/
# 2. https://www.listendata.com/2019/06/matplotlib-tutorial-learn-plot-python.html
# 3. https://github.com/Saurav6789/Data-visualization/blob/master/Other_Plots.ipynb
# 4. https://gist.github.com/gizmaa/7214002
# 5. https://towardsdatascience.com/data-visualization-using-matplotlib-16f1aae5ce70

# # Please upvote if you like this kernel for further encouragement
