#!/usr/bin/env python
# coding: utf-8

# If you are using R, there is a very good chance that you are using the almighty [ggplot2](https://ggplot2.tidyverse.org/index.html) package for your data visualization. As a big fan of ggplot, it was a very nice surprise for me to find out the [plotnine](https://plotnine.readthedocs.io/en/stable/) library for Python 3.
# 
# Plotnine is a great library that has a very intuitive syntax, and can help you to create easy and elegant visualizations in python. The main problem is that most of the information online is written in R, and some of the Python programmers might struggle with the syntax. 
# 
# In this notebook, I want to present to you how simple and easy can plotnine be, and add some tricks that will be helpful for you while creating knit and elegant graphs.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option("display.max_rows",8)
from plotnine import *

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#read data
PATH = "/kaggle/input/fifa-world-cup/"

players = pd.read_csv(PATH+"WorldCupPlayers.csv")
cups = pd.read_csv(PATH+"WorldCups.csv")
matches = pd.read_csv(PATH+"WorldCupMatches.csv").dropna()


# # World Cup Data
# 
# As a first step, We want to have a quick peek on the WC data. This is a data about all the FIFA World Cups in history, since the first one in 1930 up to 2014.
# 

# ### World Cups Dataframe
# 
# We can use pandas head() function to see the top five rows of our data:

# In[ ]:


cups.head()


# So, here we have quite simple dataframe, with 10 features. As a starter, let's try to plot the number of goals scored on each world cup.

# In[ ]:


#ggplot(dataframe) is creating the graph objectm and geom_line() is a line plot layer.
# note: when we use columns from the dataframe, we must put them inside the aesthetics (aes=()...)
p = ggplot(cups) + geom_line(aes(x="Year", y="GoalsScored")) 
#show the object. When using Pycharm, print it.
p


# Great! our very first plotnine figure! Now, we might want to make it a bit more attractive.
# Let's add title,fix the axis labels, and add more xticks. Note that different objects from the library can be added to the graph object.

# In[ ]:


# Nice trick: if you wrap your ggplot object with round brackets, you can drop down to a new row.
# This way, you can make your code more readable and add comments

#add labels
p = (p + labs(title = "Fifa World Cup Goals Scored per Year",x="Tournament Year", y='Goals Scored') + 
    #add more xticks
    scale_x_continuous(breaks = range(1930,2014,8)) + 
    #add theme.
    theme_bw())

# you can use draw() function and add ; if you want avoid the <ggplot...> object tag.
p.draw();


# You can read more about ggplot themes here:<br>
# https://plotnine.readthedocs.io/en/stable/generated/plotnine.themes.theme.html
# 
# Now, let say we want to create a bar plot, that present the number of cups each national team won (among the teams that won at least one time). 
# You have guessed it right! We can use the "geom_bar" function.

# In[ ]:


#create our barplot and plot it
(ggplot(cups) + geom_bar(aes(x="Winner"))).draw();


# Again, we will want to spruce up our figure a little bit, and make it more readable.
# 1. Order the bars
# 2. Give the xticks an angle for better readability
# 3. Add colors
# 4. Add title, labels, etc.

# In[ ]:


#create a sorted list of national teams, by number of wins
sorted_champions = cups['Winner'].value_counts().index.tolist()

print("Sorted list by winnings:\n",sorted_champions)

#create the plot object
(ggplot(cups)    
     #add bars colored by values
    + geom_bar(aes(x='Winner', fill='Winner')) 
     #sort by sorted list of values
    + scale_x_discrete(limits=sorted_champions) 
     #add xticks angle
    + theme(axis_text_x = element_text(angle = 45)) 
     #add title and axis labels
    + labs(title="World Cup Wins", x='National Team',y='Titles')).draw();


# **Note:** When using "color" attribute inside the aesthetics (aes() function), you must use a feature name in your data. For example, the following code will raise an error:

# In[ ]:


(ggplot(cups)    
     #add bars colored by values
    + geom_bar(aes(x='Winner', fill='blue')));


# So, the right way to color your geometric layer with a solid color, is doing it outside of the aes() function.

# In[ ]:


(ggplot(cups)    
     # add bars colored by plain color
    + geom_bar(aes(x='Winner'), fill='orangered')
     # add angle to xticks
    + theme(axis_text_x = element_text(angle = 45))
     # add titles
    + labs(title="World Cup Wins", x='National Team',y='Titles')).draw();


# Now, what if we want to add the winners graph the runner up national teams?
# 
# The first method (and for my opinion the less elegant way) is just "moving" the bars a bit and make some space for new bars.

# In[ ]:


(ggplot(cups)    
     # add winners bars colored in red, width smaller width
    + geom_bar(aes(x='Winner'), fill='orangered', width=.3)
     # add runner up thin bars colored in blue, nuged to the right
    + geom_bar(aes(x="Runners-Up"), fill='lightblue', position = position_nudge(x = 0.3), width=.3)
     # rotate xticks
     + theme(axis_text_x = element_text(angle = 45))
     # add title
    + labs(title="World Cup Wins", x='National Team',y='Titles')).draw();


# The second method is a very nice trick that can help you create very easy plots with legend in few lines of code.
# We will use the `melt` function from `pandas` library, to meltdown the data we want to visualize, and then use aes colors to seperate it.
# 
# After using `melt`, our data should look like this:

# In[ ]:


cups[['Winner','Runners-Up']].melt()


# Now, we can use the same methods we have already seen to draw this plot. Notice that we can easily split our columns by using the `variable` column to fill our bars with the right color.

# In[ ]:


#create ggplot object over melted data
(ggplot(cups[['Winner','Runners-Up']].melt()) + 
     # add bars, color splitted by variable (winner or runner up)
     # position = 'dodge' is for side-by-side bars. for stacked bars use position="stack"
     geom_bar(aes(x='value', fill='variable'),position = "dodge", width=0.9)
     # rotate xticks and  set figure size
     + theme(axis_text_x = element_text(angle = 45), figure_size=[10,4])
     # add title
    + labs(title="World Cup Wins", x='National Team',y='Titles')
     # change legend title and set new legend labels
    + scale_fill_discrete(name = "Final Place", labels=["2nd Place","1st Place"])).draw();


# ### Matches Dataframe
# 
# Let's take a look on another dataframe we have, the matches.

# In[ ]:


matches.head()


# Now, we want to draw a boxplot of the match attendance per World Cup tournament.
# To do so we can use the `geom_boxplot`.

# In[ ]:


#create ggplot object for matches data
(ggplot(matches) 
     # add boxplots. We need to use the Year.astype() since year is a float
     + geom_boxplot(aes(x='Year.astype("str")', y='Attendance'))
     # rotate xticks and  set figure size
     + theme(axis_text_x = element_text(angle = 45), figure_size=[8,4])
     # add title
    + labs(title="World Cups Attendance by Year", x='Year')
     # change legend title and set new legend labels
    ).draw();


# At the beginning of this notebook, we counted the goals scored on each tournament.
# Now, we can create a more sofisticated plot of this feature, using boxplot over the matches of each year.

# In[ ]:


matches['Total Goals'] = matches['Home Team Goals'] + matches['Away Team Goals']

p3 = (ggplot(matches) + geom_boxplot(aes(y="Total Goals", x='Year.astype(str)', fill='Year'))
      + theme(axis_text_x = element_text(angle = 90), figure_size=[8,4])
     # add title
    + labs(title="World Cups Goals by Year", x='Year'))
     # change legend title and set new legend labels

p3.draw();


# We can also split this figure using the `facet_wrap` function.
# 
# You can read more about facets here:
# https://plotnine.readthedocs.io/en/stable/generated/plotnine.facets.facet_wrap.html

# In[ ]:


matches.Stage[matches['Stage'].str.contains("Group")] = "Group Stage"
matches.Stage[~matches['Stage'].str.contains("Group")] = "Knockout"

p3 + facet_wrap("~Stage")

