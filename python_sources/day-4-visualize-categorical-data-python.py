#!/usr/bin/env python
# coding: utf-8

# This time I'm gon plot a bar chart of dog colors for the peeps in Zurich. We'll read in our data first, though.
# 
# Please note that the column names are in German, as noted below...

# **German**
# 
# Since German is the official language of Zurich most of the columns are in German but the translations to English aren't too tricky
# 
# * ALTER -> Age
# * GESCHLECHT -> Gender
# * STADTKREIS -> City Quarter or District
# * RASSE1 -> Dog's Primary Breed
# * RASSE2 -> Dog's Secondary Breed
# * GEBURTSJAHR_HUND -> Dog's Year of Birth
# * GESCHLECHT_HUND -> Dog's Gender
# * *HUNDEFARBE* -> Dog's Color
# 
# Note that I'm goingh to pick the dog's color column to plot my bars on

# In[ ]:


# call required Python packages
import numpy as np # linear-algebra
import pandas as pd # data processsin, .csv file I/O( e.g. pd.read_csv)

import matplotlib.pyplot as plt # for plotting
##*  # Input data files are available in the "../input/" directory.
##* # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
##* # Any results you write to the current directory are saved as output.

#read in your data to use
dataFrame= pd.read_csv("../input/20170308hundehalter.csv")# cud have benn 20160307hundehalter.csv

#head to the first few rows
dataFrame.head()


# Plot a barchart off variable HUNDEFARBE i.e. the dog' color. NB: These are still the *chosen* select colors for my subset:
# * schwarz/braun
# * tricolor
# * brindle
# * braun
# * schwarz
# * schwarz/weis
# * weis

# In[ ]:


# select the ff colors tricolor, brindle, braun, schwarz, schwarz/weis & weis 
###*      print(dataFrame.loc[dataFrame['HUNDEFARBE'].isin(['tricolor','brindle'  ,'braun', 'schwarz', 'schwarz/weis' , 'weis'])])
catPlotDF= (dataFrame.loc[dataFrame['HUNDEFARBE'].isin(['tricolor','brindle'  ,'braun', 'schwarz', 'schwarz/weis' , 'weis'])])


###*                               ## the data manipulation for plotting
# count the color frequencies using the "value_counts()" function?* & store this in a dataset
colorFreqTable= catPlotDF['HUNDEFARBE'].value_counts()
# this will get us a list of the names  ##*
list(colorFreqTable.index)
# this will get us a list of the counts ##*
colorFreqTable.values
# get all the names from our frequency plot & save them for later
labelNames = list(colorFreqTable.index)
# generate a list of numbers as long as our number of labels
positionsForBars = list(range(len(labelNames)))


###*                               ## the actual plotting

# pass the names and counts to the bar function
### make sure of (import matplotlib.pyplot as plt # for plotting) above!!!*
plt.bar(positionsForBars,colorFreqTable.values) # plot our bars
plt.xticks(positionsForBars,labelNames) # add lables
plt.title("Zurich: Dog colors", fontweight="bold",fontsize=21)


# Esle we can sommar use the 'seaborn' package

# In[ ]:


# We may also use the 'seaborn' package to do this plot

# import seaborn and alias it as sns
import seaborn as sns

###*                               ## fonts
# Bigger than normal fonts
sns.set(font_scale=1.5)

###*                               ## plottin
# make a barplot of your, chosen categorical, column from the dataframe
sns.countplot(catPlotDF['HUNDEFARBE']).set_title("Zurich: Dog colors")

