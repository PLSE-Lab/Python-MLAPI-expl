#!/usr/bin/env python
# coding: utf-8

# <h1> COMET - Project Sprint </h1>
# <p> This notebook is based on the data set of Video Game Sales. This will hopefully answer some questions and produce explanatory outputs.</p>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import collections as co
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Read whole data set to a variable 
vgsales = pd.read_csv("../input/Video_Game_Sales_as_of_Jan_2017.csv")
vgsales = vgsales.dropna()


# <h4> <b> Question 1: </b> Does user score affect the sales of a game? </h4>

# In[ ]:


x = vgsales['User_Score']
y = vgsales['Global_Sales']

plt.scatter(x, y)
plt.show()


# In[ ]:


samp = {"User Score": vgsales['User_Score'], "Global Sales": vgsales['Global_Sales']}
dsamp = pd.DataFrame(samp)
dsamp.corr('pearson')


# <h4> <b> Question 2: </b> In each year, how many games were released? </h4>

# In[ ]:


years = vgsales.groupby(['Year_of_Release']).count()
years


# In[ ]:


<h4> <b> Question 3: </b> Which game genre is the most popular? </h4>


# In[ ]:


vgsales.groupby(['Genre']).count()['Name']


# According to the given data set, the most popular genre of video game is Action.
