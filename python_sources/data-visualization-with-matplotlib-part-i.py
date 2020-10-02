#!/usr/bin/env python
# coding: utf-8

# # Matplotlib Tutorial

# In[ ]:


# import the required modules
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


# Matplotlib is a 2-D plotting library that helps in visualizing figures. Matplotlib emulates Matlab like graphs and visualizations. Matlab is not free, is difficult to scale and as a programming language is tedious. So, matplotlib in Python is used as it is a robust, free and easy library for data visualization.

# ![image.png](attachment:image.png)

# The figure contains the overall window where plotting happens, contained within the figure are where actual graphs are plotted. Every Axes has an x-axis and y-axis for plotting. And contained within the axes are titles, ticks, labels associated with each axis. An essential figure of matplotlib is that we can more than axes in a figure which helps in building multiple plots, as shown below. In matplotlib, pyplot is used to create figures and change the characteristics of figures.

# The most simple way of creating a figure with an axes is using pyplot.subplots. We can then use Axes.plot to draw some data on the axes:

# # Histograms
# Histograms are generally used when we need to count the number of occurness.

# In[ ]:


N_points = 100000
n_bins = 20

# Generate a normal distribution, center at x=0 and y=5
x = np.random.randn(N_points)
y = .4 * x + np.random.randn(100000) + 5

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(x, facecolor='yellow',edgecolor='green', bins=n_bins)
axs[1].hist(y, facecolor='yellow',edgecolor='green', bins=n_bins)


# We can also create a cumulative version of this histograms.

# In[ ]:


plt.hist(x, facecolor='peru',edgecolor='blue', bins=n_bins, cumulative=True)
plt.show()


# We can specify the range of histogram as well using range.

# In[ ]:


plt.hist(x, facecolor='peru',edgecolor='blue', bins=n_bins, range=(-2,2))
plt.show()


# # Multiple Histogram

# Multiple histograms are useful in understanding the distribution between 2 entity variables. We can see from there which variable is performing better. e.g product A is sold more than product B.

# In[ ]:


plt.hist(x, facecolor='peru',edgecolor='blue', bins=n_bins)
plt.hist(y, facecolor='yellow',edgecolor='green', bins=n_bins)
plt.show()


# # Pie Charts
# A pie chart (or a circle chart) is a circular statistical graphic, which is divided into slices to illustrate numerical proportion. In a pie chart, the arc length of each slice (and consequently its central angle and area), is proportional to the quantity it represents.

# In[ ]:


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# From the above pie chart , it is clearly seen that the what is the percentage of each entry.

# # BarPlot

# In[ ]:


from matplotlib.ticker import FuncFormatter

data = {'Barton LLC': 109438.50,
        'Frami, Hills and Schmidt': 103569.59,
        'Fritsch, Russel and Anderson': 112214.71,
        'Jerde-Hilpert': 112591.43,
        'Keeling LLC': 100934.30,
        'Koepp Ltd': 103660.54,
        'Kulas Inc': 137351.96,
        'Trantow-Barrows': 123381.38,
        'White-Trantow': 135841.99,
        'Will LLC': 104437.60}
group_data = list(data.values())
group_names = list(data.keys())
group_mean = np.mean(group_data)


# In[ ]:


fig, ax = plt.subplots()
ax.barh(group_names, group_data)


# This barplot will help in better understanding of a company profits.

# The plt.bar creates the bar chart for us. If you do not explicitly choose a color, then, despite doing multiple plots, all bars will look the same. This gives us a change to cover a new Matplotlib customization option, however. You can use color to color just about any kind of plot, using colors like g for green, b for blue, r for red, and so on. You can also use hex color codes, like #191970

# In[ ]:


plt.bar([1,3,5,7,9],[5,2,7,8,2], label="Example one")

plt.bar([2,4,6,8,10],[8,6,2,5,6], label="Example two", color='g')
plt.legend()
plt.xlabel('bar number')
plt.ylabel('bar height')

plt.title('Epic Graph\nAnother Line! Whoa')

plt.show()


# # Stack charts
# You can stack bar charts on top of each other. That is particulary useful when you multiple values combine into something greater.

# In[ ]:


data1 = [23,85, 72, 43, 52]
data2 = [42, 35, 21, 16, 9]
plt.bar(range(len(data1)), data1)
plt.bar(range(len(data2)), data2, bottom=data1)
plt.show()


# # Vilion Plot
# 

# A Violin Plot is used to visualise the distribution of the data and its probability density.

# ![image.png](attachment:image.png)

# In[ ]:


np.random.seed(10)
collectn_1 = np.random.normal(100, 10, 200)
collectn_2 = np.random.normal(80, 30, 200)
collectn_3 = np.random.normal(90, 20, 200)
collectn_4 = np.random.normal(70, 25, 200)

## combine these different collections into a list
data_to_plot = [collectn_1, collectn_2, collectn_3, collectn_4]

# Create a figure instance
fig = plt.figure()

# Create an axes instance
ax = fig.add_axes([0,0,1,1])

# Create the boxplot
bp = ax.violinplot(data_to_plot)
plt.show()


# more to come ...
