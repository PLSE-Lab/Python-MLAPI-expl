#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:





# In[ ]:


import pandas as pd
impeachment_polls = pd.read_csv("../input/trump-impeachment-polls/impeachment-polls.csv")
impeachment_topline = pd.read_csv("../input/trump-impeachment-polls/impeachment_topline.csv")


# In[ ]:


impeachment_polls.head()


# In[ ]:


impeachment_polls[["Rep Yes", "Rep No", "Dem Yes", "Dem No", "Ind Yes", "Ind No"]].corr()


# In[ ]:


from scipy import stats
print(impeachment_polls["Dem Yes"])

# Find length of valid x and y data
print(len(impeachment_polls["Dem Yes"].dropna()))
print(len(impeachment_polls["Ind Yes"].dropna()))

# Make variables to store new data
x = impeachment_polls["Dem Yes"].dropna()[:447] # Make x the same length as the shortest length
y = impeachment_polls["Ind Yes"].dropna()
pearson_coef, p_value = stats.pearsonr(x, y)
print(pearson_coef)
print(p_value)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.regplot(x=impeachment_polls["Dem Yes"], y=impeachment_polls["Ind Yes"],marker="*")
plt.title("Democrats and Indepdents who voted yes for impeaching Donald Trump", loc="center")
plt.show()
 


# Something I found interesting was that for the number of democrats that voted yes, about half the amount of independents also voted yes. For example, when 80 democrats voted yes, about 40 independents voted yes. I think this has to do with the amount of democrats and indepdents being sampled, as there are usually more democrats than independents in general. 

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
 
x = impeachment_polls["Dem Yes"]
y = impeachment_polls["Ind Yes"]

plt.scatter(x, y, c="indigo", alpha=0.5)
plt.xlabel("Democrats who said yes")
plt.ylabel("Indpendents who said yes")
plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 
plt.title("Democrats and Indepdents who voted yes for impeaching Donald Trump", loc="center")

plt.show()


# I noticed darker circles towards the middle of the graph. For example, when democrats were around 85, there seemed to be quite a few dark indigo circles when there were about 45 independents, and when democrats were at 75 and indepdnents were at 40. I think this signifies more overlap, meaning the specific amounts of independents correlated to the specific amounts democrats is more popular. I chose to use indigo because it combines blue and purple, the two parties' colors. 

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
 
plt.figure(figsize=(34,20))
plt.rc('xtick', labelsize=32) 
plt.rc('ytick', labelsize=32) 
plt.xlabel('xlabel', fontsize=35)
plt.ylabel('ylabel', fontsize=35)
plt.title("Democrats and Indepdents who voted yes for impeaching Donald Trump", loc="center", fontsize=35)
boxplot = impeachment_polls.loc[impeachment_polls['Dem Yes'] >= 80][impeachment_polls['Ind Yes'] >= 40]
sns.boxplot(x="Dem Yes", y="Ind Yes", data=boxplot, palette="Blues")

plt.show()


# In general, it seems as though that the relationship of indepdents who said yes with an increasing amount of democrats saying yes is direct. However, there are some outliars. For example, when Dem Yes is at 90, The mean of independents who said yes was significantly lower than than when 89 democrats and 91 democrats said yes. I decided to choose a blue color scheme because it represents the democratic party, and as more democrats are saying yes, the blue is becoming darker. 
