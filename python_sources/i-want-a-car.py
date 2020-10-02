#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


cars=pd.read_csv('../input/Automobile_data.csv')


# In[ ]:


cars.head()


# ### Checking Data Quality

# In[ ]:


cars.isnull().sum()


# Looks like there are no null values in the dataset. But there are some characters like ? in the dataset that doesn't make any sense. Thus they have to be removed.

# The dataset doesn't contain any null values. But the problem with the dataset is that it contains a lot of '?' characters in it. Also the datatypes of the columns are majorly of type object. For proper analysis these object types have to be changed to either float or int datatype.
#       Now the value to be replaced depends on the other values of the column. Somewhere we replace it with the mean value whereas somewhere we need to use other means for determining the value. 

# ## Cleaning And Transformation

# ### Cleaning the normalized losses column

# In[ ]:


a=cars[cars['normalized-losses']!='?']
b=(a['normalized-losses'].astype(int)).mean()
cars['normalized-losses']=cars['normalized-losses'].replace('?',b).astype(int)
# in this case we replace the '?' characters with the mean value of the normalizzed losses so that there is no deviation from the mean value


# ### Cleaning the num-of-doors column
# 

# In[ ]:


cars[cars['num-of-doors']=='?']


# Both the cars are of sedan style. Lets see the number of doors in sedan cars

# In[ ]:


a=cars[cars['body-style']=='sedan']
a['num-of-doors'].value_counts()


# We can see that the sedan cars have 4 doors in maximum cases. So we will replace '?' with 4

# In[ ]:


a=cars['num-of-doors'].map({'two':2,'four':4,'?':4})
cars['num-of-doors']=a


# ### Cleaning the price column

# In[ ]:


a=cars[cars['price']!='?']
b=(a['price'].astype(int)).mean()
cars['price']=cars['price'].replace('?',b).astype(int)


# ### Cleaning the Horsepower column

# In[ ]:


a=cars[cars['horsepower']!='?']
b=(a['horsepower'].astype(int)).mean()
cars['horsepower']=cars['horsepower'].replace('?',b).astype(int)


# ### Cleaning the bore column

# In[ ]:


a=cars[cars['bore']!='?']
b=(a['bore'].astype(float)).mean()
cars['bore']=cars['bore'].replace('?',b).astype(float)


# ### Cleaning the stroke column

# In[ ]:


a=cars[cars['stroke']!='?']
b=(a['stroke'].astype(float)).mean()
cars['stroke']=cars['stroke'].replace('?',b).astype(float)


# ### Cleaning the peak rpm column

# In[ ]:


a=cars[cars['peak-rpm']!='?']
b=(a['peak-rpm'].astype(float)).mean()
cars['peak-rpm']=cars['peak-rpm'].replace('?',b).astype(float)


# Finally the dataset is free from inconsistency. So now we can analyse the data

# ### Some Basic Analysis

# In[ ]:


cars.describe()


# In[ ]:


print('The different car maufacturers are:',cars['make'].unique())


# **Lets see their total models**

# In[ ]:


sns.countplot(cars['make'])
plt.xticks(rotation='vertical')
plt.show()


# Wooh!! Toyota leads the list with the highest number of cars manufactured.

# In[ ]:


print('the diffferent type of car models are:',cars['body-style'].unique())


# **Lets see the count of each car style**
# 

# In[ ]:


cars_type=cars.groupby(['body-style']).count()['make']
ax=cars_type.sort_values(ascending=False).plot.bar()
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1),fontsize=11)


# **Lets see which car types does manufacturers produce**

# In[ ]:


a=cars.groupby(['make','body-style']).count().reset_index()
a=a[['make','body-style','symboling']]
a.columns=['make','style','count']
a=a.pivot('make','style','count')
a.dropna(thresh=3).plot.barh(width=0.85)
plt.show()


# The above graph shows the type of cars the company makes.I have only included the companies that make more than 3 car styles.
#   
# It clearly shows that Toyota makes hatchback cars on large scale. Similarly Nisssan and volkswagon make sedan cars a lot

# ### Mileage

# In[ ]:


mileage=cars.groupby(['make']).mean()
mileage['avg-mpg']=((mileage['city-mpg']+mileage['highway-mpg'])/2)
mileage['avg-mpg'].sort_values(ascending=False).plot.bar()
plt.show()


# I have calculated the average mileage according to the make by taking the average of city-mpg and highway-mpg. The graph clearly shows that that Chevrolet has the highest avg mileage and Jaguar has the lowest.

# ### Let us see the distribution of some key attributes of the car

# In[ ]:


cars[['engine-size','peak-rpm','curb-weight','horsepower']].hist(color='Y')
plt.show()


# ## How Does Price Vary?
# 
# The important parameters for the Price variance depends mainly upon the engine parameters like horsepower,engine-size,mpg and the type of car i.e length and width.
# 
# So now we will see the correlation  between these price determining factors.

# In[ ]:


plt.figure(figsize=(11,4))
sns.heatmap(cars[['length','width','height','curb-weight','engine-size','horsepower','city-mpg','highway-mpg','price']].corr(),annot=True)
plt.show()


# The above heatmap shows following correlations:
# 
# Strong Correaltions: (width,curb-weight,engine-size,horsepower)---->Price
# 
# Weak Correlations: (city-mpg,highway-mpg)---->Price

# **Stay Tuned**

# In[ ]:




