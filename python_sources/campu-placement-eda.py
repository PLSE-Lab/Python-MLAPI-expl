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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# Let us see if the data has any NaN values

# In[ ]:


data.info()


# From the above output we can see that salary column has missing values. 
# Let us check the data in the column and see if we can replace with any other value

# In[ ]:


data.head()


# It quite okay that students who are not placed will have their salaries zero so let us replace NaN with 0 in the salary column

# In[ ]:


data['salary']=data['salary'].fillna(0)


# In[ ]:


data.info()


# I guess the data cleaning process is done . Let us see if all the columns contain the actual type of values

# In[ ]:


data.head()


# In[ ]:


print(data['gender'].unique())
print(data['status'].unique())
print(data['workex'].unique())
print(data['hsc_b'].unique())
print(data['ssc_b'].unique())


# I'm always used to change the column type if the column contains two different values in whole dataset . Sometimes it will be usefull for the visualizations . If there is anything else useful just let me know in the comments .

# In[ ]:


data['gender']=data['gender'].astype('category')
data['status']=data['status'].astype('category')
data['workex']=data['workex'].astype('category')
data['hsc_b']=data['hsc_b'].astype('category')
data['ssc_b']=data['ssc_b'].astype('category')


# I'm setting the sl_no as index

# In[ ]:


data.set_index('sl_no',inplace=True)


# In[ ]:


data.describe()


# Not all the columns are present in the above result . it become those simple statistics can only be done on number not on strings ;-)

# # Let us get into the Visualizing the data

# In[ ]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
langs = ['Male','Femlae']
students = [139,76]
ax.pie(students, labels = langs,autopct='%1.2f%%',colors = ["#1f77b4", "#ff7f0e"])
plt.title('Pie chart ')
plt.show()


# In[ ]:


col_list = ["shit","pistachio"]
col_list_palette = sns.xkcd_palette(col_list)
sns.set_palette(col_list_palette)
sns.set(rc={"figure.figsize": (10,6)},
 palette = sns.set_palette(col_list_palette),
 context="talk",
 style="ticks")
# How many smokers are in this study?
sns.catplot(x = 'status', data=data, kind='count',height=5,aspect=1.5, edgecolor="black")
plt.title('No of students placed', fontsize = 20)
print(data['status'].value_counts())


# In[ ]:


#average percentage 
values = [(data['ssc_p'].mean()),(data['hsc_p'].mean()),(data['mba_p'].mean()),(data['degree_p'].mean())]
print('scc_p mean = ' +str(data['ssc_p'].mean()))
print('hsc_p mean = ' +str(data['hsc_p'].mean()))
print('mba_p mean = ' +str(data['mba_p'].mean()))
print('degree_p mean = ' +str(data['degree_p'].mean()))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
names = ['ssc_p','hsc_p','mba_p','degree_p']
ax.set_ylabel('Average percentages')
ax.set_title('Average Percentage')
ax.bar(names,values,width = 0.5,color=["#2ca02c"])
plt.show()


# Let me replace the workex column with yes=1 and no=0 . so that i can calculate the corr between salary

# In[ ]:


data['workex'].replace(to_replace ="Yes", 
                 value =1,inplace=True) 
data['workex'].replace(to_replace ="No", 
                 value =0,inplace=True)


# In[ ]:


print('ssc_p to salary ', round(data['salary'].corr(data['ssc_p'])*100,1),'%')
print('hsc_p to salary ', round(data['salary'].corr(data['hsc_p'])*100,1),'%')
print('mba_p to salary ', round(data['salary'].corr(data['mba_p'])*100,1),'%')
print('degree_p to salary ', round(data['salary'].corr(data['degree_p'])*100,1),'%')
print('etest_p to salary ', round(data['salary'].corr(data['etest_p'])*100,1),'%')

print('workexpto salary ', round(data['salary'].corr(data['workex'])*100,1),'%')


# Based on the corr values we can make the feature selection and build the model for prediction
