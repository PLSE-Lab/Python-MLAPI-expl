#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 


# Kobe has had great success in his career and in his last game, he scored an amazing 60 points and so in my project, I would like to analyse the shots he has made, where they were made and his success rate for each type of shot along with many other useful data.

# In[ ]:


data = pd.read_csv("../input/data.csv")


# Before we start, we should take a look at the different data that has been given to us, and from there, we should choose the data that we need and the data that we don't need.'

# In[ ]:


data.head()


# # Which shots went in and which ones missed?

# First, before I plot the shots graph, I have to remove all the null values.

# In[ ]:


nonull =  data[pd.notnull(data['shot_made_flag'])]


# First, let us look at where his shots have been taken. We have loc_x and loc_y and from the name, I am thinking it is location of x and y at which the shot was made. Lets see if thats right. NOTE: Alpha used here is lower than HIT or MISS plot because the points are more dense, making it uncleaer is a higher alpha was used.

# In[ ]:


alpha = 0.02
plt.figure(figsize=(7,6))

# loc_x and loc_y
plt.subplot(121)
plt.scatter(nonull.loc_x, nonull.loc_y, color='blue', alpha=alpha)
plt.title('Plot of loc_x against loc_y')


# # Shots that went in

# Now out of all the shots, lets see at which points the shots that went in were.

# In[ ]:


alpha = 0.05
plt.figure(figsize=(5,5))
points = nonull.loc[nonull.shot_made_flag == 1]
plt.scatter(points.loc_x, points.loc_y,color = "#008000", alpha=alpha)
plt.title("Went in, Goodjob Kobe.")


# # Shots that missed

# Now lets look at all the shots that missed

# In[ ]:


alpha = 0.05
plt.figure(figsize=(5,5))
points = nonull.loc[nonull.shot_made_flag == 0]
plt.scatter(points.loc_x, points.loc_y,color = "#FF0000", alpha=alpha)
plt.title("Missed, Nice try kobe.")


# # Side-by-side comparison

# In[ ]:


alpha = 0.05
plt.subplot(121)
points = nonull.loc[nonull.shot_made_flag == 1]
plt.scatter(points.loc_x, points.loc_y,color = "#008000", alpha=alpha)
plt.title("Went in")

alpha = 0.05
plt.subplot(122)
points = nonull.loc[nonull.shot_made_flag == 0]
plt.scatter(points.loc_x, points.loc_y,color = "#FF0000", alpha=alpha)
plt.title("Missed")


# Now, from the graph above, we can see that Kobe misses more when he is closer to the hoop, well my assumption is this is because he was trying to layup but then gets blocked, however he does not miss from far away because maybe his teammates pass the ball and he quickly goes for the shot.

# # Deeper analysis of Kobe's shots

# Now I want to go into deeper analysis and see how many shots Kobe has missed and made in each season.

# In[ ]:


data_shots = data[data['shot_made_flag']>=0]


# In[ ]:


data_missed = data_shots.shot_made_flag == 0
data_success = data_shots.shot_made_flag == 1

shot_missed = data_shots[data_missed].season.value_counts()
shot_success = data_shots[data_success].season.value_counts()
shots = pd.concat([shot_success,shot_missed],axis=1)
shots.columns=['Success','Missed']


fig = plt.figure(figsize=(16,5))
shots.plot(ax=fig.add_subplot(111), kind='bar',stacked=False,rot=1,color=['#008000','#FF0000'])
plt.xlabel('Season')
plt.ylabel('Number of shots')
plt.legend(fontsize=15)


# # Conclusion

# In conclusion, we can see that Kobe has had a very successful career. The peak of his career was in 2005-2006 season where he has had the most success in shooting. However, as you can see in the 2013-2014 season, total number of sucess and missed shots do not exceed 200. This could possibly be because of an injury which caused him to be benched. However, towards his retirement you can see he missed significantly more shots than he got in, hinting towards him reaching a plateu in his skills and being close to retiring.

# In[ ]:




