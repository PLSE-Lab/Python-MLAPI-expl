#!/usr/bin/env python
# coding: utf-8

# # Data Visualization Basics
# This is my personal guide on matplotlib and other visualization basics.

# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


# plotting
plt.plot([1,2,3],[4,5,1])

# Showing the plot
plt.show()


# In[ ]:


# Adding Title and Labels to our graph.
x = [5,8,10]
y = [12,16,6]

# Plotting
plt.plot(x,y)

plt.title('Just plotting a Graph with Title & Labels')
plt.ylabel('Y-Axis')
plt.xlabel('X-Axis')

# Showing the graph
plt.show()


# In[ ]:


# styling the graph
from matplotlib import style
style.use('ggplot')


# In[ ]:


# Plotting multiple lines in a single graph.
x = [5,8,10]
y = [12,16,6]

x2 = [6,9,11]
y2 = [6,15,7]

plt.plot(x,y,'g',label = 'Line One', linewidth = 5)
plt.plot(x2,y2,'c',label = 'Line Two', linewidth = 5)

plt.title('Plotting multiple lines in a Single Graph')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

plt.legend()

plt.grid(True, color = 'k')

plt.show()


# # Bar Graph
# This bar graph are mainly used to compare things between different groups.
# Used with Categorical Variables.

# In[ ]:


plt.bar([1,3,5,7,9], [5,2,7,8,2], color = 'g', label = 'Group 1')
plt.bar([2,4,6,8,10], [8,6,2,5,6], color = 'b', label = 'Group 2')

plt.legend(loc = "best")
'''
Other options are
	best
	upper right
	upper left
	lower left
	lower right
	right
	center left
	center right
	lower center
	upper center
	center
'''

plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

plt.title("My Bar Graph")

plt.show()


# # Histogram
# Used with Quantative Data.

# In[ ]:


population_age = [1,4,7,8,9,44,12,25,48,90,63,41,44,34,46,26,37]

plt.hist(population_age, bins = 10, histtype = 'bar', rwidth = 0.5)
# Check the graph by changing the bins to 5;20 etc
#plt.hist(population_age, bins = 5, histtype = 'bar', rwidth = 0.5)
#plt.hist(population_age, bins = 20, histtype = 'bar', rwidth = 0.5)

plt.xlabel('X-Axis-Population Age')
plt.ylabel('Y-Axis')

plt.title("My Histogram Graph")

plt.show()


# # Scatter Plot
# This is to compare 2 variables when plotting in 2D or 3D, to check the correlation between the variables.

# In[ ]:


x = [1,2,3,4,5,6,7,8]
y = [5,2,4,2,1,4,5,2]

plt.scatter(x,y,label = 'Numbers', color = 'r')

plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

plt.title("My Scatter Plot Graph")
plt.legend(loc = 'best')
plt.show() 


# # SubPlot
# plotting multiple graph in same area of a graph.. or can say using subplot.
# 
# Just copy any 3 graphs from above, and add code plt.subplot()

# In[ ]:


# First Plot
plt.subplot(211)

# Plotting multiple lines in a single graph.

x = [5,8,10]
y = [12,16,6]

x2 = [6,9,11]
y2 = [6,15,7]

plt.plot(x,y,'g',label = 'Line One', linewidth = 5)
plt.plot(x2,y2,'c',label = 'Line Two', linewidth = 5)

plt.title('Plotting multiple lines in a Single Graph')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

plt.legend()

plt.grid(True, color = 'k')

# Second Plot
plt.subplot(212)

#Scatter Plot
x3 = [1,2,3,4,5,6,7,8]
y3 = [5,2,4,2,1,4,5,2]

plt.scatter(x3,y3,label = 'Numbers', color = 'r')

plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

plt.title("My Scatter Plot Graph")
plt.legend(loc = 'best')

plt.show()


# Now changing the subplot values from 211 & 212 to 221 & 222.
# subplot take values as 
# subplot(Number_of_rows, Number_of_columns, plot_number)
# for subplot(211) or subplot(2,1,1) means total 2 plots in a row; 1 column, and 1st graph.

# In[ ]:


# First Plot
plt.subplot(221)

# Plotting multiple lines in a single graph.

x = [5,8,10]
y = [12,16,6]

x2 = [6,9,11]
y2 = [6,15,7]

plt.plot(x,y,'g',label = 'Line One', linewidth = 5)
plt.plot(x2,y2,'c',label = 'Line Two', linewidth = 5)

plt.title('Plotting multiple lines in a Single Graph')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

plt.legend()

plt.grid(True, color = 'k')

# Second Plot
plt.subplot(222)

#Scatter Plot
x3 = [1,2,3,4,5,6,7,8]
y3 = [5,2,4,2,1,4,5,2]

plt.scatter(x3,y3,label = 'Numbers', color = 'r')

plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

plt.title("My Scatter Plot Graph")
plt.legend(loc = 'best')

plt.show()


# # Plotting categorical variables
# Many times we want to create a plot that uses categorical variables in Matplotlib. Matplotlib allows you to pass categorical variables directly to many plotting functions.
# Reference from matplotlib.org 

# In[ ]:


data = {'apples': 10, 'oranges': 15, 'lemons': 5, 'limes': 20}
names = list(data.keys())
values = list(data.values())

fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
axs[0].bar(names, values)
axs[1].scatter(names, values)
axs[2].plot(names, values)
fig.suptitle('Categorical Plotting')

alternate way of above using subplot.
# In[ ]:


plt.figure(figsize=(20,5))

plt.subplot(131)
plt.bar(names, values, label = 'Bar Graph')
plt.legend(loc = "best")
plt.title('Bar Plot')

plt.subplot(132)
plt.scatter(names, values, label = 'Scatter Graph')
plt.legend(loc = "best")
plt.title('Scatter Plot')

plt.subplot(133)
plt.plot(names, values, label = 'Line / Plot Graph')
plt.legend(loc = "best")
plt.title('Line Plot')

plt.suptitle('Categorical Plotting')
plt.show()


# In[ ]:




