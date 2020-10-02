#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


Ramen_df= pd.read_csv('/kaggle/input/ramen-ratings-latest-update-jan-25-2020/Ramen_ratings_2020.csv')


# In[ ]:


Ramen_df.head()


# In[ ]:


figure(figsize=(20,4))
plt.xticks(rotation=90)
sns.countplot(x='Country', data=Ramen_df, palette='Paired', order = Ramen_df['Country'].value_counts().index).set_title('Count of Votes by Country')


# In[ ]:


Ramen_df['Starsnum']=pd.to_numeric(Ramen_df['Stars'], errors='coerce')


# In[ ]:


figure(figsize=(15,4))
plt.xticks(rotation=90)
sns.countplot(x='Style', data=Ramen_df, palette='Paired', order = Ramen_df['Style'].value_counts().index).set_title('Count of Votes by Style')


# In[ ]:


Ramen_df.info()


# In[ ]:


sns.barplot(data=Ramen_df, x="Style", y="Starsnum").set_title('Average Star Vote by Style')


# In[ ]:


figure(figsize=(20,4))
plt.xticks(rotation=90)
sns.barplot(data=Ramen_df, x="Country", y="Starsnum").set_title('Average Star Vote by Country')

