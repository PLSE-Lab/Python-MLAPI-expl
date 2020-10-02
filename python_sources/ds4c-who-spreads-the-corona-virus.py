#!/usr/bin/env python
# coding: utf-8

# ## Following the [last EDA](https://www.kaggle.com/incastle/ds4c-eda-with-floating-population-data),
# 
# - Koreans are campaigning for self-isolation and social distance as a measure of infection.
# - I would like to analyze through the data of the floating population whether corona is well-prepared according to age group.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import datetime

patient = pd.read_csv('../input/coronavirusdataset/patient.csv')
trend = pd.read_csv('../input/coronavirusdataset/trend.csv')
time = pd.read_csv('../input/coronavirusdataset/time.csv')
route = pd.read_csv('../input/coronavirusdataset/route.csv')


# ### I have floating population data through January 31st.
# - It is often not diagnosed until two weeks after corona exposure.
# - If person A is exposed to corona on January 31, it will be confirmed on February 14.
# 

# In[ ]:


patient = patient.query('confirmed_date < "2020-02-14"')# and infection_reason.str.con not like "%visit"')
for i in ['visit', 'residence']:
    patient = patient[~patient['infection_reason'].str.contains(i)]


# In[ ]:


patient['age'] = 2020 - patient['birth_year'] + 1
patient['age_group'] = patient['age'] // 10
patient['age_group'] = [str(a).replace('.','') for a in patient['age_group']]

print('Female 20, 40, 50 age rank top with significant difference.')
print('What happens to them? It might be a next topic of my analysis')
plt.figure(figsize = (15,8))
ax = sns.countplot(patient['age_group'], order = patient['age_group'].value_counts().sort_index().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()


# ### The most infected age group is 40 ~ 50s.
# - Let's look at some types of demographic features depending on the age group.

# In[ ]:


## floating population data(only January)
fp_01 = pd.read_csv("../input/seoul-floating-population-2020/fp_2020_01_english.csv")


# In[ ]:


# trim data
fp_01['date'] = fp_01['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y.%m.%d").date()).astype('str')
fp_01['date'] = fp_01['date'].apply(lambda x: x[8:]) ## only use day

fp_01 = fp_01.sort_values(['date', 'hour', 'birth_year', 'sex'])  ## this data is not sorted.
fp_01.reset_index(drop= True, inplace = True)


# In[ ]:


fp_01.head()


# ## The first corona confirmed in Korea is January 20.
# - I will compare ***January 17 (Friday)*** before the first outbreak and ***January 31 (Friday)***, the most recent date of my data.
# 
# ### plot 1 : Changes in the floating population by age group
#     - Divide the two dates and draw each.
# ### plot 2 : Differences in the Change of the Current Population by Age Group for Two Dates
#     - Before outbreak-After the outbreak, values were drawn.
#     - In other words, if the value is *positive*, many people go around despite the corona, and if the value is *negative*, the floating population contracted.

# In[ ]:


def make_brith_hour_plot(date1, date2, city):
    
    if city == 'all district':        
        gan17 = fp_01[(fp_01['date'] == date1)]
        gan31 = fp_01[(fp_01['date'] == date2)]
    else:
        gan17 = fp_01[(fp_01['date'] == date1) & (fp_01['city'] == city)]
        gan31 = fp_01[(fp_01['date'] == date2) & (fp_01['city'] == city)]

    gan17 = pd.DataFrame(gan17.groupby(['hour', 'birth_year'])['fp_num'].sum())
    gan17.reset_index(inplace = True)

    gan31 = pd.DataFrame(gan31.groupby(['hour', 'birth_year'])['fp_num'].sum())
    gan31.reset_index(inplace = True)

    fig, ax = plt.subplots(1,2, figsize = (18,8),  gridspec_kw={'wspace': 0.2})
    
    fig.suptitle('{} : Changes by age group and hour'.format(city), fontsize = 18)
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, wspace=0.05)
    
    t = ax[0].scatter(x=gan17['hour'], y=gan17['birth_year'], c=gan17['fp_num'], s= 100, cmap=plt.cm.RdYlBu_r)
    ax[0].set_title("{}day".format(date1), fontsize=15)
    ax[0].set_xlabel('hour', fontsize=13)
    ax[0].set_ylabel('birth_year', fontsize=13)    
    plt.colorbar(t, ax = ax[0])

    t2 = ax[1].scatter(x=gan31['hour'], y=gan31['birth_year'], c=gan31['fp_num'], s= 100, cmap=plt.cm.RdYlBu_r)
    ax[1].set_title("{}day".format(date2), fontsize=15)
    ax[1].set_xlabel('hour', fontsize=13)
#     ax[1].set_ylabel('birth_year', fontsize=13)    
    cbar1  = plt.colorbar(t2, ax = ax[1])
    cbar1.set_label('fp_num', fontsize=13)        
    plt.show()


# In[ ]:


def make_brith_hour_diff_plot(date1, date2, city):
    
    if city == 'all district':        
        gan17 = fp_01[(fp_01['date'] == date1)]
        gan31 = fp_01[(fp_01['date'] == date2)]
    else:
        gan17 = fp_01[(fp_01['date'] == date1) & (fp_01['city'] == city)]
        gan31 = fp_01[(fp_01['date'] == date2) & (fp_01['city'] == city)]
    
    gan17_no_groupby = pd.DataFrame(gan17.groupby(['hour'])['fp_num'].sum())
    gan17_no_groupby.reset_index(inplace = True)

    gan31_no_groupby = pd.DataFrame(gan31.groupby(['hour'])['fp_num'].sum())
    gan31_no_groupby.reset_index(inplace = True)
    
    gan17_groupby = pd.DataFrame(gan17.groupby(['hour', 'birth_year'])['fp_num'].sum())
    gan17_groupby.reset_index(inplace = True)

    gan31_groupby = pd.DataFrame(gan31.groupby(['hour', 'birth_year'])['fp_num'].sum())
    gan31_groupby.reset_index(inplace = True)
    
    fp_num_diff = gan17_groupby.iloc[:,-1:] -  gan31_groupby.iloc[:,-1:]
    axix_=gan17_groupby.iloc[:,:-1]
    df = pd.concat([axix_, fp_num_diff ], axis = 1)

        
    fig, ax = plt.subplots(1,2, figsize = (18,8),  gridspec_kw={'wspace': 0.2, 'hspace': 0.4})
    fig.suptitle('{} : Diff between (before corona) and (after_corona)'.format(city), fontsize = 18)
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, wspace=0.05)
    t = ax[0].scatter(x=df['hour'], y=df['birth_year'], c=df['fp_num'], s= 200, cmap=plt.cm.RdYlBu_r, vmax =7000) #, vmax)
    ax[0].set_title("{}day-{}day diff, groupby hour and age_group".format(date1, date2), fontsize=15)
    ax[0].set_xlabel('hour', fontsize=13)
    ax[0].set_ylabel('birth_year', fontsize=13)    
    cbar1 = plt.colorbar(t, ax = ax[0])
    cbar1.set_label('fp_num', fontsize=13)

    
    
    
    sns.lineplot(data=gan17_no_groupby, x='hour', y='fp_num', color='green', ax=ax[1]).set_title('fp_num', fontsize=16)
    sns.lineplot(data=gan31_no_groupby, x='hour', y='fp_num', color='purple', ax=ax[1]).set_title('fp_num', fontsize=16)
    ax[1].set_title("{}day-{}day diff, total".format(date1, date2), fontsize=15)
    ax[1].set_xlabel('hour', fontsize=13)
    ax[1].set_ylabel('total fp_num', fontsize=13)    
    ax[1].legend([date1, date2])
    plt.show()
    
    
    plt.show()
    


# ## All district

# In[ ]:


make_brith_hour_plot('17', '31', 'all district')


# In[ ]:


make_brith_hour_diff_plot('17', '31', 'all district')


# - If the point on the diff plot is red, it means that the ***flow population has decreased***.
# 
# - And the blue point is that despite the corona, the ***floating population has increased.***
# 
# - The floating population in their 40s has a high value despite the corona, and we should not forget the plot we drew earlier.

# In[ ]:


plt.figure(figsize = (15,8))
ax = sns.countplot(patient['age_group'], order = patient['age_group'].value_counts().sort_index().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()


# ## Now let's draw a picture divided by region.

# In[ ]:


route[(route['date'] < '2020-02-01') & ( route['province'] == 'Seoul')].groupby('city')['patient_id'].nunique()


# #### Like the [last EDA](https://www.kaggle.com/incastle/ds4c-eda-with-floating-population-data), I'm only going to look at the top four districts in Seoul
# - Gangnam-gu
# - Jongno-gu
# - Jung-gu  
# - Jungnang-gu  

# ### Gangnam-gu

# In[ ]:


make_brith_hour_plot('17', '31', 'Gangnam-gu')


# In[ ]:


make_brith_hour_diff_plot('17', '31', 'Gangnam-gu')


# ### In all age groups, the floating population decreased, but only the 20s decreased or maintained.

# # Jongno-gu

# In[ ]:


make_brith_hour_plot('17', '31', 'Jongno-gu')


# In[ ]:


make_brith_hour_diff_plot('17', '31', 'Jongno-gu')


# # Jung-gu

# In[ ]:


make_brith_hour_plot('17', '31', 'Jung-gu')


# In[ ]:


make_brith_hour_diff_plot('17', '31', 'Jung-gu')


# # Jungnang-gu

# In[ ]:


make_brith_hour_plot('17', '31', 'Jungnang-gu')


# In[ ]:


make_brith_hour_diff_plot('17', '31', 'Jungnang-gu')


# 
# - If you look at the graph on the right, the graph itself is different from other regions.
# 
# - The area is a residential area, not an industrial district. Therefore, the population seems to have increased as a result of self-containment.

# In[ ]:





# ## Conclusion
# 
# - some meaningful interpretation seems to be possible.
# 
# - When the floating population data is uploaded in February, I will come to a more accurate analysis.

# In[ ]:




