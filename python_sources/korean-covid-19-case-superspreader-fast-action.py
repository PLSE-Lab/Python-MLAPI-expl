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


# # **Korean Corona Case**
# ![](https://i.imgur.com/fVZuwSj.png?1) 

# # CONTENT 
# * What happen in Korean COVID-19 case?
# * How Korean face the COVID-19?
# * What we can learn from COVID-19 in Korea?
# > 

# # What's happen in Korean COVID-19 case?

# We gonna start to look up the case dataset of Korean COVID-19 case.

# In[ ]:


df_kor_corona_case = pd.read_csv('/kaggle/input/coronavirusdataset/Case.csv')
df_kor_corona_case.head()


# Get deeper with our dataset. This dataset contain 81 rows and 8 columns or features.

# In[ ]:


df_kor_corona_case.shape


# In[ ]:


df_kor_corona_case.columns


# Find percentage of missing value in each column. It is important to get to know how 'rich' our dataset.

# In[ ]:


df_kor_corona_case.isna().sum()/df_kor_corona_case.shape[0]*100


# Great! our dataset does not have missing value at all! let's get work!

# In[ ]:


df_province_confirmed = df_kor_corona_case.groupby(['province'])['confirmed'].sum().sort_values(ascending=False).reset_index()
df_province_confirmed.head()


# In[ ]:


import seaborn as sns
sns.catplot(x='confirmed',y= 'province', data = df_province_confirmed ,kind='bar',height=5, aspect = 2)


# In[ ]:


df_city_confirmed = df_kor_corona_case.groupby(['city'])['confirmed'].sum().sort_values(ascending=False).reset_index()


# In[ ]:


sns.catplot(x='confirmed',y= 'city', data = df_city_confirmed ,kind='bar',height=5, aspect = 2)


# What happen with these province and city. How come the amount of confirmed case raised to top?

# In[ ]:


df_kor_corona_case.loc[df_kor_corona_case['province']=='Daegu'].head(1)


# Wow! only in one place, it has taken 53% of confirmed case. I could say **Shincheonji Church** be a place of **superspreader** happen.
# Let see another infection case.

# In[ ]:


df_infection_confirmed = df_kor_corona_case.groupby(['infection_case'])[['confirmed']].sum().sort_values(by=['confirmed'],ascending=False).reset_index()


# In[ ]:


sns.catplot(y='infection_case',x= 'confirmed', data = df_infection_confirmed ,kind='bar',height=5, aspect = 2)


# Is it only stop Shincheonji Church? or they are move around?

# In[ ]:


df_kor_corona_case.loc[(df_kor_corona_case['infection_case']=='Shincheonji Church') 
                       & (df_kor_corona_case['province'] != 'Daegu')][['province','city','infection_case','confirmed']].sort_values(by=['confirmed'],ascending=False).reset_index()


# In[ ]:


temp_df_kor_corona_case= df_kor_corona_case.loc[(df_kor_corona_case['infection_case']=='Shincheonji Church') 
                       & (df_kor_corona_case['province'] != 'Daegu')][['province','city','infection_case','confirmed']].sort_values(by=['confirmed'],ascending=False).reset_index().merge(df_province_confirmed, on='province', how='left')


# We can see from above dataframe that some people from Shincheonji Church not only stay in Daegu province but has spreaded to 14 provinces. so, let's count how many percentage people from shincheoji church have a part in their province.

# In[ ]:


temp_df_kor_corona_case.rename(columns={"confirmed_x": "confirmed_from_shincheonji", "confirmed_y":"confirmed_total"}, inplace=True)


# In[ ]:


temp_df_kor_corona_case.columns


# In[ ]:


temp_df_kor_corona_case['percentage_from_shincheonji'] = temp_df_kor_corona_case['confirmed_from_shincheonji']/temp_df_kor_corona_case['confirmed_total']*100
temp_df_kor_corona_case


# Below is the bar chart confirmed case from shincheonji churc that spread to some province.

# In[ ]:


sns.catplot(y='confirmed_from_shincheonji',x= 'province', data = temp_df_kor_corona_case ,kind='bar',height=5, aspect = 2).set_xticklabels(rotation=90)


# The spreader from shincheonji church in some province take high number of confirmed case in its province.

# In[ ]:


sns.catplot(y='percentage_from_shincheonji',x= 'province', data = temp_df_kor_corona_case ,kind='bar',height=5, aspect = 2).set_xticklabels(rotation=90)


# # Then, how Korean face the COVID-19?

# Let's jump into time dataframe. i am gonaa use this dataframe to get deeper insight of what Korean goverment did to face COVID-19.

# In[ ]:


df_kor_corona_time = pd.read_csv('/kaggle/input/coronavirusdataset/Time.csv')
df_kor_corona_time.head()


# Just another day of work with data, let's make sure there is less missing value in this dataframe.

# In[ ]:


df_kor_corona_time.isna().sum()/df_kor_corona_time.shape[0]*100


# In[ ]:


total_test = df_kor_corona_time['test'].sum()
test_duration = df_kor_corona_time.shape[0]
first_day = df_kor_corona_time['date'].head(1).values[0]
last_day = df_kor_corona_time['date'].tail(1).values[0]
print("This dataset show us there are {0} test during {1} days since {2} until {3} period".format(total_test,test_duration,first_day,last_day))


# So, how's the result? check it out.

# In[ ]:


negative = df_kor_corona_time['negative'].sum()
confirmed = df_kor_corona_time['confirmed'].sum()
total_test = df_kor_corona_time['test'].sum()
unknown = (df_kor_corona_time['test'].sum()-df_kor_corona_time['confirmed'].sum()-df_kor_corona_time['negative'].sum())
unknown_percentage = round((total_test - negative - confirmed)/total_test*100)
print("From {0} test has been taken by Korean government, they got {1} negative case and {2} confirmed case. Unfortunately there is unknown result from the test. There are {3} case that the results is unknwown. It takes {4} % from total test.".format(total_test,negative,confirmed,unknown,unknown_percentage))


# In[ ]:


import matplotlib.pyplot as plt

negative_percentage = round(df_kor_corona_time['negative'].sum()/df_kor_corona_time['test'].sum()*100)
confirmed_percentage = round(df_kor_corona_time['confirmed'].sum()/df_kor_corona_time['test'].sum()*100)

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Confirmed COVID-19', 'Negative COVID-19', 'Unknown Result'
sizes = [confirmed_percentage, negative_percentage,unknown_percentage ]
explode = (0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
my_colors = ['Coral','turquoise','plum']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90, colors=my_colors)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[ ]:


import matplotlib.pyplot as plt

released_percentage = df_kor_corona_time['released'].sum()/df_kor_corona_time['confirmed'].sum()
deceased_percentage = df_kor_corona_time['deceased'].sum()/df_kor_corona_time['confirmed'].sum()
isolated_percentage = (df_kor_corona_time['confirmed'].sum()-df_kor_corona_time['released'].sum()-df_kor_corona_time['deceased'].sum())/df_kor_corona_time['confirmed'].sum()

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Released', 'Deceased', 'Isolated'
sizes = [released_percentage, deceased_percentage,isolated_percentage ]
explode = (0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
my_colors = ['Coral','turquoise','plum']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90, colors=my_colors)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# Korean govement can push the deceased rate until under 1% of confirmed case. It is great number compare to other country that struggling to push the deceased rate.
# What did they do?

# If we check 'test' column, we can see that Koren goverment take action to prevent the raising spreading number of COVID-19 by make a massive test to their people.

# In[ ]:


percentage_tes_population = round(df_kor_corona_time['test'].sum()/(51.47*1000000)*100)
print('Until 22nd March 2020, Korean goverment success did COVID-19 test to {}% of Korean population.'.format(percentage_tes_population))


# Below graph show us that the amount of test that Korean goverment did increase significantly day by day.

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.style.use('seaborn')

#size
plt.figure(figsize=(17,7))

#data
x_labels = list(df_kor_corona_time['date'].values)
y_case_test = df_kor_corona_time['test'].values
y_case_confirmed = df_kor_corona_time['confirmed'].values

#plot colors
plt.plot(y_case_test, color='blue', label='Tested')
plt.plot(y_case_confirmed, color='red', label='Confirmed')

#title
plt.title('The amount of tested people increase significantly')

#labels
plt.xlabel('Date')
plt.ylabel('sum')
plt.xticks(np.arange(len(x_labels)),x_labels, rotation=90)

# #legend
# blue_patch = mpatches.Patch(color='blue', label='Tested')
# red_patch = mpatches.Patch(color='red', label='Confirmed')

#grid
plt.grid(True)

plt.legend()
plt.show()


# Below are the graph that shows us the distribution of confirmed case of COVID-19 in Korea.

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.style.use('seaborn')

#size
plt.figure(figsize=(17,7))


#data
x_labels = list(df_kor_corona_time['date'].values)
y_case_deceased = df_kor_corona_time['deceased'].values
y_case_confirmed = df_kor_corona_time['confirmed'].values
y_case_released = df_kor_corona_time['released'].values

#plot colors
plt.plot(y_case_deceased, color='red',label='Deceased')
plt.plot(y_case_confirmed, color='blue', label='Confirmed/Postive')
plt.plot(y_case_released, color='green', label='Released')

#title
plt.title('Confirm COVID-19 case distribution')

#labels
plt.xlabel('Date')
plt.ylabel('sum')
plt.xticks(np.arange(len(x_labels)),x_labels, rotation=90)

#legend
# red_patch = mpatches.Patch(color='red', label='Deceased')
# blue_patch = mpatches.Patch(color='blue', label='Confirmed/Postive')
# green_patch = mpatches.Patch(color='green', label='Released')

#grid
plt.grid(True)

plt.legend()
plt.show()


# So, what we can learn from Korean goverment is "take fast action!". Less than a month they already tested 10% of population and start to isolate the confrim case.
# 
# We can see from below chart that, the confirmed case going slightly flat in last 2 weeks. The number of healthy patient from COVID-19 also start getting higher and the deceased graph flat tend to go down.
