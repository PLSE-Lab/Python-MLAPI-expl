#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
menu = pd.read_csv("../input/nutrition-facts/menu.csv")
menu.head(10)


# In[ ]:


menu.describe(include='all')


# Missing Data : -
# Count returns number of not null values.  **

# In[ ]:


len(menu)


# ### So its clear that the number of rows in the dataset menu is 260. Also the count of all the items are 260. From this we can conclude that there is no missing data in this dataset.******

# ### Finding missing values in each column

# In[ ]:


menu.isnull().sum()


# In[ ]:


menu.isnull()


# In[ ]:


sns.heatmap(menu.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

### categorical scatter plots
plot = sns.swarmplot(x='Category',y='Sodium', data=menu)
plot.set_xticklabels(plot.get_xticklabels(),rotation = 90)
plt.title('Sodium')
plt.show()


# In[ ]:


menu['Sodium'].describe()


# In[ ]:


menu['Sodium'].idxmax()


# In[ ]:


menu.at[82,'Item']


# In[ ]:


menu.columns


# In[ ]:


menu['Dietary Fiber'].idxmax()


# In[ ]:


menu.at[32,'Item']


# In[ ]:


subset = menu.loc[ [32] ,['Item','Serving Size', 'Protein','Total Fat','Dietary Fiber','Dietary Fiber (% Daily Value)'] ]
subset


# **Food that contains highest fiber content**
# 

# In[ ]:


subset['%Fiber'] = subset['Dietary Fiber']/434 * 100
subset


# In[ ]:


menu['Protein'].idxmax()


# In[ ]:


subset1 = menu.loc[ [82] ,['Item','Serving Size', 'Protein','Calories','Calories from Fat'] ]
subset1


# 

# In[ ]:


subset1['Calories from Protein'] = subset['Protein']/4 
subset


# In[ ]:



df = menu[['Item','Serving Size','Calories','Protein']]

df.head()


# 

# In[ ]:


df['Protein'].idxmax()


# In[ ]:


subset1


# In[ ]:


menu['Calories from Protein'] = menu['Protein']*4 
menu['Calories from Carbs'] = menu['Carbohydrates']*4
menu['%protein calorie'] = menu['Calories from Protein']/menu['Calories']*100
menu['%carbs calorie'] = menu['Calories from Carbs']/menu['Calories']*100
menu['%fat calorie'] = menu['Calories from Fat']/menu['Calories']*100
subset1 = menu.loc[ [82] ,['Item','Category','Serving Size','Calories','Protein','Carbohydrates',
                           'Calories from Protein','Calories from Fat','Calories from Carbs',
                          '%protein calorie','%carbs calorie','%fat calorie'] ]
subset1


# 
# Eventhough above food contains highest protein content. it is a high fat food. it will become nutitious, only if we add a protein shake and salad.
# 
# Protein shake will balance the high fat content and the fiber in the salad will reduce the carbs effect.
# 
# Some nutritionists recommend a ratio of 40 percent carbohydrates, 30 percent protein, and 30 percent fat as a good target for healthy weight loss.

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

### categorical scatter plots
plot = sns.swarmplot(x='Category',y='%fat calorie', data=menu)
plot.set_xticklabels(plot.get_xticklabels(),rotation = 90)
plt.title('High Fat')
plt.show()


# In[ ]:


menu['Category'].unique()

