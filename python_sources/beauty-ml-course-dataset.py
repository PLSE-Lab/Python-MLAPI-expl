#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")


# ## Import dataset

# In[ ]:


my_filepath = ('../input/mlcourse/beauty.csv')
df=pd.read_csv(my_filepath)
df


# ## Check shape of dataset

# In[ ]:


df.shape


# > There are 1260 instances and 10 attributes in the data set

# ## Preview dataset
# > Summary statistics of dataset

# In[ ]:


df.describe()


# ## Check for missing values

# In[ ]:


df.isnull().sum()


# In[ ]:


for col in df.columns:
    print(col, len(df[col].unique()))


# ## Data Visualization

# In[ ]:


sns.heatmap(data=df, annot=True)


# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(data=df.corr(), annot=True)     #correlation matrix


# In[ ]:


sns.pairplot(data=df[['wage', 'female', 'educ','exper']]);


# In[ ]:


df['looks'].value_counts()


#  > We have 5 unique values for our looks(beauty). 5 being the most beautiful and then decreasing to 1.
# 

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x=df['looks'])            #df['looks'].value_counts().plot(kind='bar')
plt.xlabel('Looks')
plt.ylabel('Number of People')
plt.title('Distribution of Looks',fontsize = 18)


# > The graph shows that looks are more likely to be 3 which is an average value 

# In[ ]:


df['wage'].sort_values()


# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(df['wage'], kde=False)
plt.title('Histogram of wage')


# In[ ]:


plt.figure(figsize=(10,8))
sns.lineplot(x=df['educ'],y=df['wage'])
plt.title('Average Variation in Wage with Education Level')
plt.xlabel('Education Level')
plt.ylabel('Wage Level')


# > more education more wage you earn

# In[ ]:


plt.figure(figsize=(10,8))
sns.barplot(x=df.educ,y=df.wage , hue=df.female);


# In[ ]:


sns.scatterplot(x = 'wage' , y = 'exper' , hue='female', data=df )
plt.xlabel('Wage'), 
plt.ylabel('Years of Expertise') 
plt.title('Level of Expertise compared to Wage with the distinction gender')
plt.legend(['Female','Male'])
plt.show()


# > looks like men tend to earn more the more expertise they gain, while women earn the same

# In[ ]:


women_wage = df[df['female']==1]['wage'].groupby(df['exper']).mean()


# In[ ]:


plt.figure(figsize = (10 , 6))
sns.lineplot(data= women_wage)
plt.xlabel('Years of Expertise'), 
plt.ylabel('Mean Wage') 
plt.title('Years of Expertise compared to Wage with the distinction gender for females')


# > there is one (out of range) with a high wage and low exper

# In[ ]:


plt.figure(figsize=(10,6))
sns.regplot(x="educ", y="exper", data=df)
plt.title('Relationship between Education level and Expertise year')


# > The more education you have, the less expertise you'll have.

# In[ ]:


plt.figure(figsize=(10,8))
plt.plot(df.groupby('goodhlth')['wage'].mean())
plt.title('Relationship between Wage and Good health')
plt.xlabel('Good Health')
plt.ylabel('Wage Level')


# In[ ]:


df.married.value_counts()


# In[ ]:


sns.countplot(df.married ,hue=df.female);


# > male who are married are more than female

# In[ ]:


plt.figure(figsize=(10,6))
sns.jointplot(x="looks",y="wage", data = df, kind="kde");


# > The beauty doesn't play a role.

# In[ ]:


plt.figure(figsize=(10,6))
sns.regplot(x="looks",y="wage", data = df)

