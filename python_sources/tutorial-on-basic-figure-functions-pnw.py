#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#BE SURE TO RUN THIS BEFORE STARTING ANYTHING 

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # Fundamental package for comput
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Fundamental package for plotting


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# # <h1><center>Tutorial on Basic Figure Functions</center></h1> 

# #### By Madolyn Kelm and Augustin Kalytiak-Davis
# 
# ## Objective: Learn basics of working with figures in Python
# 
# ### In last week's tutorial, we learned the basics of creating figures in Python. This week, we will review the material we learned last week and delve more deeply into methods of creating informative and aesthetically pleasing figures. 
# 
# 

# ## Activities:
# 
# Read the Introduction below, and do the following exercises.

# ## Review and New Functions
# 
# In this section of the tutorial you will learn a number of basic commands to use when creating figures. These commands are short, but they can change the look of your figure in drastic ways. 
# 
# ## plt.plot()
# Plt.plot is the most basic, essential command for plotting figures. The parameters of your plot are entered within the parantheses as a comma-separated list. These parameters include: your x and y variables, the color and shape, the labels, and even more. We will focus on a few basic specifications today, but feel free to use and share any others you know. 
# 
# When using the plot command sure you also write it out in the order of: plt.plot(x-variable,y-variable,color/shape,label, etc). We will go over the basics for each of these inputs.
# 
# ### X and Y inputs
# There are three main types of inputs for your x values and y values: lists and arrays, ranges, and functions. Each have their own purpose and can be helpful when creating a complex figure.
# 
# ### Lists
# Lists are used to plot specific data points, like experimental data, and take the form of a comma-separated list between two brackets. Data describing bacteria in a petri dish, voltage outputs, distances traveled, or test scores would be best expressed in a list. Here are a few examples:

# In[ ]:


test_scores = [66, 72, 88, 93, 99]
number_bacteria = [1, 4, 9, 16]


# You can also convert a list into a numpy array, as demonstrated below, to access different functions. Here is a link to find out more about numpy array if you are curious: https://cs231n.github.io/python-numpy-tutorial/#:~:text=A%20numpy%20array%20is%20a,the%20array%20along%20each%20dimension.

# In[ ]:


scores_list = [93, 72, 66, 99, 88]
scores_array = np.array(scores_list)
print(scores_list)
print(scores_array)


# ### Ranges
# 
# Ranges describe a series of numbers between two bounds and separated by regular intervals. Let's say, for our graph's x-axis, we want the number of days the student studied for a test between 0 and 7. The appropriate range would be:

# In[ ]:


days_studying = np.arange(0, 8, 1)
#The lower bound is inclusive, while the upper bound is not
print(days_studying)


# ### Functions
# 
# Functions are mathematical functions, like (x+3); often we want to compare and fit our data to a mathematical curve. Functions can be defined and then put into a plt.plot() command (like a list or range) or inserted straight into the command. So:
# 
# y = (x+2) * 3
# 
# plt.plot(x, y)
# 
# will be the same as
# 
# plt.plot(x, (x+2) * 3)
# 

# Let's make a figure using some of the inputs we just learned:

# In[ ]:


#First you need the plt.figure() command, if you are using the notebook function of matplotlib like we are

plt.figure()

#Next you want to define the inputs that will be your x and y. Below, we have set x to be a range; write a function of x to serve as our y.

x = np.arange(-3,4,1) #This is the range of x from -3 to 4 on intervals of 1
y = x+8

#Fill in the blanks here with your x range and y function. Refer back to the notes above if you need help. 
plt.plot( x, y, 'fuchsia', linestyle = '--', label = 'x+8') #The 'b' here is the command for a color. We will go into depth on that next.
a = x**2
plt.plot( x, a, 'c', linestyle = '--', label = 'x^2')
plt.legend()
plt.xlabel('stuff')
plt.ylabel('more stuff')


# Congratulations, you have created a figure!

# ## Customizations
# 
# ### Color
# Next, let's move on to the visuals. In your graph above your data should be represented by a solid blue line. Changing the color of your line is easy and often helpful. It can be done in your plt.plot command, by adding the "name" assigned to a color after your x and y variables (as shown above, the "name" of that shade of blue is 'b').
# 
# Some colors have short "names", while others are more complex. Try changing the color of the line in your graph above to your favorite color. (Refrence the image below for a wide range of colors and their names)
# 
# ![image.png](attachment:image.png)
# 
# ### Style of Line
# 
# Next let's switch the look of our line. We can change this to dashes, single data points or even double dashes using the following commands. These go directly into the plt.plot command, in quotes with the color. For instance, 'ro' will produce red data points while 'b-' will produce a blue dashed line.
# 
# When using colors with more complex names, add another comma in plt.plot after color, and enter the command linestyle = '(type of line you want'. For example:
# 
# plt.plot(x,function,'mediumvioletred', linestyle='-.') will produce a medium violet red line with dashes and dots.
# 
# Try changing the look of your line in your figure above. It should look something like the example below:

# In[ ]:


#EXAMPLE 1

plt.figure()
x = np.arange(1,4,1)
y = x+2

plt.plot(x,y,'mediumvioletred', linestyle='-.')
plt.grid(True)


# ### Mutiple Data Sets on One Graph
# 
# Let's try adding another function to your above graph. 
# 
# Create a new function of x, and name it something other than y. 
# 
# Next, add a second plt.plot command line with the same x range and your new function's name instead of y.
# 
# Lastly, to reduce the possibility of confusion, make your new line a different color and style. Refer to the examples above if you have any trouble.
# 
# If you kept adding more functions, you could get something that looks like this:
# ![image.png](attachment:image.png)
# 
# This figure is pretty busy, and it might be hard to remember which function is which. Let's try adding a legend to you graph.
# 
# In the plt.plot command, simply add another comma after linestyle (or color) and use the command label=''
# 
# This could look like:
# 
# plt.plot(x, [the name of your new defined function], 'b', label='x+2')
# 
# OR
# 
# plt.plot(x, x+2, 'b', label='x+2')
# 
# Try adding labels to your graph, but be sure to run the command: plt.legend(). Otherwise it will not populate your label onto your figure. 
# 
# If you get stuck here is one example:

# In[ ]:


#EXAMPLE 2

plt.figure()
x = np.arange(1,4,1)
y = x+2
y2 = -x+5

plt.plot(x,y,'mediumvioletred', linestyle='-.', label='x+2')
plt.plot(x,y2,'c--', label='-x+5')

plt.legend()
plt.grid(True)


# ### Finish it up
# There a few more tweaks to make your figure perfect and ready to share. You should add a title, and label the x and y axis, using the following commands:
# 
# Label the y-axis: plt.ylabel('y-label')
# 
# Label the x-axis: plt.xlabel('x-label ')
# 
# Add a Title: plt.title('title_name')
# 
# Try adding labels to your graph or the graph above, and refer to the example below if you get stuck:

# In[ ]:


#EXAMPLE 3

plt.figure()
x = np.arange(1,4,1)
y = x+2
y2 = -x+5

plt.plot(x,y,'mediumvioletred', linestyle='-.', label='Timmy the Turtle')
plt.plot(x,y2,'c--', label='Helene the Hare')

plt.ylabel('Distance (m)')
plt.xlabel('Time (s)')
plt.title('The Race')


plt.legend()
plt.grid(True)


# ### Reminders and Extras
# 
# plt.legend() - this commmand creates the legend in your figure
# 
# plt.xlim(left-limit, right-limit) - expands or limits your graph to these parameters
# 
# plt.ylim(lower-limit,  upper-limit) - expands or limits your graph to these parameters
# 
# plt.grid(True) - Puts a grid in the background
# 
# You can try these commands on your graph above, or explore them below.
# 
# Consider doing your own research to find other graphing options and tricks. Figures in Python are basically completely customizable!

# ## In-class activities  
# Complete the activities below to test you graphing skills

# The following package and command allows you to call on any inputted data. Try running (by hitting Shift+Enter) it to see what data set we have entered. This set will come into play in the final activity

# In[ ]:


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <h2><center>Try it yourself </center></h2> 
# 
# Try graphing the following example.
# 
# A biology student wants to graph the multiplication of two fruit flies, and compare it to an exponential function (x^2) on the same graph. Here is the data she collected over the course of two weeks.
# 
# Number of flies counted: 2, 3, 10, 15, 26, 33, 63, 80, 101, 121, 140, 170, 204, 229
# 
# Try graphing her data with solid data points and the exponential function (x^2) as a dashed line. 
# 
# After you are happy with your graph,click the far right option from under your figure to download it!
# 
# 
# ![image.png](attachment:image.png)
# <center><h5> Choosing the far right option on from these will allow you to download your figure. </h5></center>

# In[ ]:


#Some hints are provided below. If you are feeling brave, feel free to delete all and start from scratch!

plt.figure()
x = np.arange(0, 14, 1)
flies = [2, 3, 10, 15, 26, 33, 63, 80, 101, 121, 140, 170, 204, 229]
y = x**2
plt.plot(x, flies, 'co', label = "Number Flies")
plt.plot(x, y, 'k--', label = "x^2")
plt.legend()


# Examine and run the following code if you want to see one way to create the graph they wanted.

# In[ ]:


#Solution (ONLY LOOK IF YOU'RE STUCK)

plt.figure()
flies =[2, 3, 10, 15, 26, 33, 63, 80, 101, 121, 140, 170, 204, 229]
x = np.arange(0, 14, 1)


plt.plot(x, flies, 'go', label='Flies')
plt.plot(x, x**2, 'b', label='Exponential Function')

plt.title('Fruit Fly Experimental Data')
plt.xlabel('Day')
plt.ylabel('Number of Flies')
plt.grid(True)

plt.legend()
plt.show()


# <h2><center> Taking it a Step Further </center></h2>
# 
# Using these basic graphing tools and a little bit more complicated list we can use python to generate graphs for some really interesting data. Take a minute and watch this short video on the dropping of a water droplet onto a superhydrophic source and a hydrophobic source: https://www.youtube.com/watch?v=lYBoRozgYog.
# 
# 
# The students in the research lab want to compare the contact angles a drop of water makes with there possibly hydrophobic surface. One way they will know if the surface is hydrophobic is is the contact angles of the water droplet are at least greater than 90 degrees (most hydrophobic surfaces will have an average angle of 120 degrees, refrence the image below for a visual).
# 
# ![image.png](attachment:image.png)
# <center><h5> Mattone, Manuela & Rescic, Silvia & Fratini, Fabio & Mangan, Rachel. (2017). Experimentation of Earth-Gypsum Plasters for the Conservation of Earthen Constructions. International Journal of Architectural Heritage. 1-10. 10.1080/15583058.2017.1290850. </h5></center>
# 
# 
# 
# 

# After shooting a video of a water droplet being placed onto their surface, they were able to use python to break it up into indivual frames and measure the contact angle on the right and left sides of the droplet. The only down side is the data is a bit long and hard to understand visually. They need your help in creating a plot for their data. 
# 
# Try printing the following two list: CALA (the left contact angles) and CARA (the right contact angles)

# In[ ]:


#If you would like to more about how we created this section to read their data and populate the lists,
#let us know and we can walk through it!
g = 25 # number of frames: MODIFY
All=[]
with open('/kaggle/input/ContactAngles_Single.txt') as f:
    lines = f.readlines()
    All = [line.split() for line in lines]

#These populates lists for just the data from baseline, left contact angle, and right contact angle.
    
frame = []
for n in range(g):
    frame.append(float(All[n*8][1][5:7]))

#baselineA = []
#for n in range(g):
    #baselineA.append(float(All[n*8+4][1]))

CALA = []
for n in range(g):
    CALA.append(float(All[n*8+5][3]))

CARA = []
for n in range(g):
    CARA.append(float(All[n*8+6][3]))
    
#Run your print commands here:
print(CALA)
print(CARA)


# These two sets of data may seem a bit dawnting, but lets see what happens when we plot these two list as a function of frame numbers.

# In[ ]:


#Try creating your graph here. Plotting both the right and left contact angles.
#Try plotting on the same graph with a legend to distingush between the two.


# In[ ]:


#Solution

#This plots our baseline, left contact angle, and right contact angle.   
plt.figure()
plt.plot(frame,CALA,'ro', label='Left Contact Angle')
plt.plot(frame,CARA,'mo', label='Right Contact Angle')

#plt.plot(frame,baselineA, 'bo', label='Baseline')
plt.title('Contact Angle Data: 11.14.19_Multiple_Clicks')
plt.xlabel('Frame Number')
plt.ylabel('Contact Angle (degress)')

plt.legend()
plt.grid()
plt.show()


# No doubt you were able to create a wonderful looking graph ready to share this experimental data, but what does it show. Looking at the x-axis from frames 38 to 61 all contact angles are above 90 degrees! This is great news for the research students as they were able to create a hydrophobic surface!
# 
# Congrats! You are now on your way to creating visually pleasing graphs and now have the ability to share your future research in an easily understandable way. 
