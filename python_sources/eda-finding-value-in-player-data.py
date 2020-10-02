#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/fifa19/data.csv')


# In[ ]:


df.head(5)


# In[ ]:


# Check the dimensions of the dataframe
df.shape


# Lets get a sense of the columns that we are dealing with.

# In[ ]:


# Get an idea of all the skill values
df.info()


# Now lets get an estimate of the dat for all the numerical columns

# In[ ]:


df.describe()


# As we can see, the Mean age for the players dataset is **25.122206**
# 
# Mean potential for the players is **71.307299**, while the mean overall is **66.238699**

# Lets create a box-plot to look at the distribution for Overall vs Potential

# In[ ]:


sns.set(style="whitegrid")
plt.figure(figsize=(12,7))
df_compare = df[['Overall','Potential']]
ax = sns.violinplot(data=df_compare)


# Now lets compare players above and below the mean ago. How does the overall rating and Potential match up by age?

# In[ ]:


fig, axs = plt.subplots(ncols=2,figsize=(12,7))

ax1 = sns.scatterplot(x="Overall", y="Age", data=df, ax=axs[0])
ax2 = sns.scatterplot(x="Potential", y="Age", data=df,ax=axs[1])


# We see that the scatter points from ages 15-25 look different in both the plots.(The area for ages between 15-25 for Potential < 60 is heavily plotted for Overall whereas its is much less dense for Potential). This is because the players younger than the mean age have a lower rating but are valued to have a much higher potential. Its quite intuitive becasue players get better as they reach their peak. 

# How many players above the mean age are yet to fulfill their potential? Players whose overall < Potential and age > 25(mean age). The question we must ask is are these the players who are about to enter their prime? What is there markey value?

# In[ ]:


df_player_pot = df[(df['Overall'] < df['Potential'])]
df_player_pot = df_player_pot[df_player_pot['Age'] > 25]
df_player_pot.shape


# There are 1957 total players above the mean age who are yet to fulfil their potential. Lets look at top 10 by their current rating, potential, Age and position.

# In[ ]:


df_player_pot = df_player_pot.sort_values('Potential', ascending = False)
df_player_pot.head()


# In[ ]:


from matplotlib.image import imread
import matplotlib.patches as patches
import matplotlib.cbook as cbook
from urllib.request import urlopen
from skimage import io

plt.figure(figsize=(15,8))
url = df_player_pot.iloc[0]['Photo']
age = df_player_pot.head(10)['Age'].astype(int)
age = (age.tolist())
low_limit = df_player_pot.iloc[9]['Potential']
high_limit = df_player_pot.iloc[0]['Potential']

# Potential
plt.bar(df_player_pot.head(10)['Name'], df_player_pot.head(10)['Potential'],alpha=0.6,edgecolor='black',color='gray')

# Overall
plt.bar(df_player_pot.head(10)['Name'], df_player_pot.head(10)['Overall'],alpha=0.9,edgecolor='black', color='seagreen')




# Tweak spacing to prevent clipping of tick-labels
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)


for index, row in df_player_pot.head(10).iterrows():
    plt.text(row['Name'],row['Potential']+ 1 ,row["Age"],color='seagreen',bbox=dict(facecolor='white', edgecolor='seagreen', boxstyle='round,pad=1.2'))
    plt.text(row['Name'],row['Potential']+ 2.2 ,row["Position"],color='white',bbox=dict(facecolor='seagreen', edgecolor='white', boxstyle='round,pad=1.2'))
plt.ylim(low_limit -5, high_limit + 3)  

plt.show()

# To add image url for each player
# try:
#     f = urlopen(df_player_pot.iloc[0]['Photo'])
#     image = io.imread(url)
#     plt.imshow(image)
#     plt.show()
# except Exception as e:
#     print(e)


# To be continued...

# In[ ]:




