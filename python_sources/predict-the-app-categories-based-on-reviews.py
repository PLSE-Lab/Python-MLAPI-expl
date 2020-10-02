#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_apps = pd.read_csv('../input/googleplaystore.csv')
data_reviews = pd.read_csv('../input/googleplaystore_user_reviews.csv')


# In[ ]:


data_reviews.head()


# In[ ]:


data_apps.head()


# In[ ]:


for i in range(22):
    print(data_apps.Installs.values[i])


# group the data by different categories. There are 34 different categories in this dataset ('1.9' means?), like ART_AND_DESIGN, BEAUTY, AND BUSINESS.

# In[ ]:


category_list = list(data_apps.groupby(['Category']).groups.keys())


# In[ ]:


category_list 


# In[ ]:


len(category_list)


# To see which apps have the highest rating score, the data was sorted by 'Rating'. The first one with Rating 19.0 was clearly an outlier. Hence the rating score of 5.0 was the highest.  Although they are highly rated, they are not quite popular, most of which were installed by only 100+ times.

# In[ ]:


data_apps.sort_values(by=['Rating'],ascending=False)


# In[ ]:


Install_list = list(data_apps.groupby(['Installs']).groups.keys())
n_count=[]
for str in Install_list:
    n_count.append(data_apps[(data_apps['Rating']==5.0) & (data_apps['Installs']==str)].shape[0])
plt.bar(np.arange(len(Install_list)),np.array(n_count))
plt.xticks(np.arange(len(Install_list)), Install_list,rotation='vertical')
plt.show()


# It's noticeable that the feature 'Installs' is categorical, not numerical. There are 22 different levels of the number of installs including 'Free'.

# In[ ]:


Install_list = list(data_apps.groupby(['Installs']).groups.keys())


# In[ ]:


len(Install_list)


# In[ ]:


print(Install_list)


# Here, I am interested in the distribution of the number of installment times in each category. For example, in the 'ARTS_AND_DESIGN' category, the peak is at 100,000, which means most of the apps in this category have been installed by more than 100,000 times. On the other hand, the majority of the apps in the category 'WEATHER' were installed 1000,000. This can be seen the 'WEATHER' category is more popular among people. The other intresting category is 'VEDIO_PLAYERS'. The postion of the peak is also at 1000,000. But one thing is noticeable that there are around three apps in 'VIDEO_PLAYERS' was installled more than one billion times, which is not shown in the other two categories.

# In[ ]:


n_count=[]
for str in Install_list:
    n_count.append(data_apps[(data_apps['Category']=='ART_AND_DESIGN') & (data_apps['Installs']==str)].shape[0])
plt.bar(np.arange(len(Install_list)),np.array(n_count))
plt.xticks(np.arange(len(Install_list)), Install_list,rotation='vertical')
plt.show()


# In[ ]:


n_count=[]
for str in Install_list:
    n_count.append(data_apps[(data_apps['Category']=='WEATHER') & (data_apps['Installs']==str)].shape[0])
plt.bar(np.arange(len(Install_list)),np.array(n_count))
plt.xticks(np.arange(len(Install_list)), Install_list,rotation='vertical')
plt.show()


# In[ ]:


n_count=[]
for str in Install_list:
    n_count.append(data_apps[(data_apps['Category']=='VIDEO_PLAYERS') & (data_apps['Installs']==str)].shape[0])
plt.bar(np.arange(len(Install_list)),np.array(n_count))
plt.xticks(np.arange(len(Install_list)), Install_list,rotation='vertical')
plt.show()


# I am more interested in the apps installed by more than 1 billion times, hence the data of these apps was analyze. There are 53 apps with installment times larger than 1 billion. The apps with the highest rating score of 4.5 are Google Photes, Instagram, Subway Surfers. Although Facebook has a lower rating score of 4.1, the number of reviews on it is the largest, 78 158 306, followed by the WhatsAppMesseger.

# In[ ]:


top_apps = data_apps[data_apps.Installs == '1,000,000,000+']


# In[ ]:


top_apps.sort_values(['Rating'],ascending=False)


# In[ ]:


top_apps['Reviews'] = top_apps['Reviews'].astype(int)


# In[ ]:


top_apps.sort_values(['Reviews'],ascending=False)


# In[ ]:


plt.hist(top_apps.Rating)


# In[ ]:


top_apps.shape


# In[ ]:





# In[ ]:





# In[ ]:




