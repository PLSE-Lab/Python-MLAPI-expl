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


# # 1. Foreword
# 
# This Notebook is created for learning purpose for beginners specially for those who have very little knowledge of Python but have nice experience with other programming languages for example c#, java, c++, SQL. I will be using lot od SQL in there for data wrangling instead of Pandas or any other library.
# 
# In addition to that I have created a small utility to load data from/to CSV/SQL while I will upload once it gets stabalized.

# # 2. Data Load and Libraries Import

# In[ ]:


import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv')
data.head()


# # 3. Data Cleaning

# In[ ]:


#DELETE FROM Games
#WHERE [Average User Rating] = ''

#DELETE FROM Games
#WHERE [User Rating Count] = ''
data = pd.read_csv('../input/appstore17k/appstore_games2.csv')
data.head()


# ### 3.1 Adding two more columns 

# In[ ]:


#SELECT DATEDIFF(day, [Original Release Date], '2019-08-04') [Number_Of_Days], [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') [Ratings_Per_Day]
#,*
#FROM Games

# Number of days = Number of days between Original Release Date and Data Extraction Date
# Ratings Per Day = Total Number of Ratings / Number of days  

data_3 = pd.read_csv('../input/appstore17k/appstore_games3.csv')
data_3.head()


# ### 3.2 Removing Ratings Per Day = 0

# In[ ]:



#SELECT DATEDIFF(day, [Original Release Date], '2019-08-04') [Number_Of_Days], [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') [Ratings_Per_Day]
#,*
#FROM Games
#WHERE [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') >0
#ORDER BY [User Rating Count] DESC
data_4 = pd.read_csv('../input/appstore17k/appstore_games4.csv')
data_4.shape


# # 4. BCG Matrix
# 
# Dividing all the rows into four categories
# 
# ##### High Growth and High Market Share:
#          Those apps that gets reviewed more than or equal to 40 times a day (on average)and have more than 10000 reviews, no matter what their rating is.
# ##### High Growth and Low Market Share:
#          Apps that gets reviewed more than or equal to 40 times a day (on average) but are reviewed less than 10000 times.
# ##### Low Growth and High Market Share:
#          Apps that have been reviewd more than 10000 times but does not gets reviewd frequently.
# ##### Low Growth and Low Market Share:
#          Apps that have both reviews and reviews per day are on low side.

# In[ ]:


plt.figure(figsize=(20,16))
nyc_img=plt.imread('../input/appstore17k/Boston-Matrix2.jpg')
#scaling the image based on the latitude and longitude max and mins for proper output specially when drawing scattter plot
plt.imshow(nyc_img,zorder=0,extent=[1, 20000, 1,12000], alpha=0.5)
p = plt.axis('off')
title = plt.title('BCG Matrix', fontsize=20)
title.set_position([0.5, 1.05])


# ### 4.1 Adjusting Reviews Per Day and Total Reviews Count
# 
# Since the data is segregated in non-semtric way while we want to show it symetrically that is why we have to intriduce a couple of new columns
# 
# 1) Adjusted Rating Count    
# 2) Adjusted Rating Per Day
# 
# 
# __*Formula to calculate*__
# 
# __Adj_Rate_Count__ = If *User Rating Count* is greater than or equal to 10000 then 10000/3032734*[User Rating Count]+10000 Else *User Rating Count*
# where:
#     10000: is the offset and maximmum a *User Rating Count* can be
#     3032734: is the largest User Rating Count in data
#     
# __Adj_Rating_Per_Day__: If Rating Per day is greater than or equal to 40 then 6000/1414*[Rating Per Day]+6000
#                     Else 6000/1414*[Rating Per Day]
# where:
#     6000: is the offset and maximmum a *Rating Per Day* can be
#     1414: is the largest *Rating Per Day*
#     We have to adjust both segments that is greater than 

# In[ ]:


#SELECT DATEDIFF(day, [Original Release Date], '2019-08-04') [Number_Of_Days], [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') [Ratings_Per_Day],[User Rating Count],
#CASE WHEN [User Rating Count] >= 10000 THEN 10000/3032734.00*[User Rating Count]+10000 ELSE [User Rating Count] END Adj_Rate_Count,
#CASE WHEN [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') >= 40 THEN 6000/1414.00*[User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04')+6000 ELSE 6000/39.99*[User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') END Adj_Rating_Per_Day
#FROM Games
#WHERE [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') >0
#ORDER BY [User Rating Count] DESC
data_5 = pd.read_csv('../input/appstore17k/appstore_games5.csv')
data_5.head()


# In[ ]:


plt.figure(figsize=(20,16))
nyc_img=plt.imread('../input/appstore17k/Boston-Matrix2.jpg')
#scaling the image based on the latitude and longitude max and mins for proper output specially when drawing scattter plot
plt.imshow(nyc_img,zorder=0,extent=[1, 20000, 1,12000], alpha=0.1)
title = plt.title('BCG Matrix', fontsize=20)
title.set_position([0.5, 1.05])
ax=plt.gca()
sns.scatterplot(data_5.Adj_Rate_Count, data_5.Adj_Rating_Per_Day, ax=ax)
c = ax.set_xticklabels(['1', '', '', '', '', '10000', '', '', '', '3032734'], rotation=0, horizontalalignment='center')
c = ax.set_yticklabels(['', '', '', '40', '', '', '1414'], rotation=0, horizontalalignment='right')
ax.set_xlabel('Ratings Per Day')
ax.set_ylabel('Number of Ratings')


# ### 4.2 Zooming in
# 
# Since data is very much scattered and we have a few records which is not letting us view proper scatter in the data, so removing a few records from data 

# In[ ]:


#SELECT [Average User Rating], DATEDIFF(day, [Original Release Date], '2019-08-04') [Number_Of_Days], [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') [Ratings_Per_Day],[User Rating Count],
#CASE WHEN [User Rating Count] >= 10000 THEN 10000/300000.00*[User Rating Count]+10000 ELSE [User Rating Count] END Adj_Rate_Count,
#CASE WHEN [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') >= 40 THEN 6000/259.00*[User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04')+6000 ELSE 6000/39.99*[User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') END Adj_Rating_Per_Day
#FROM Games
#WHERE [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') >0
#AND [User Rating Count] < 300000 AND [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') < 300
#ORDER BY [Ratings_Per_Day] DESC
data_6 = pd.read_csv('../input/appstore17k/appstore_games6.csv')
data_6.head()


# In[ ]:


plt.figure(figsize=(20,16))
nyc_img=plt.imread('../input/appstore17k/Boston-Matrix2.jpg')
#scaling the image based on the latitude and longitude max and mins for proper output specially when drawing scattter plot
plt.imshow(nyc_img,zorder=0,extent=[1, 20000, 1,12000], alpha=0.1)
title = plt.title('BCG Matrix Zoomed In', fontsize=20)
title.set_position([0.5, 1.05])
ax=plt.gca()
sns.scatterplot(data_6.Adj_Rate_Count, data_6.Adj_Rating_Per_Day, ax=ax, hue=data_6["Average User Rating"],palette="Set3", legend="full", size=data_6["Average User Rating"])
c = ax.set_xticklabels(['1', '', '', '', '', '10000', '', '', '', '300000'], rotation=0, horizontalalignment='center')
c = ax.set_yticklabels(['', '', '', '40', '', '', '259'], rotation=0, horizontalalignment='right')
ax.set_xlabel('Ratings Per Day')
ax.set_ylabel('Number of Ratings')

