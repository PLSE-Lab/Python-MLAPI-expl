#!/usr/bin/env python
# coding: utf-8

# # Andy Estevez & Jordan Roman: UFC Data from 1993 to 2019 Project

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
data = pd.read_csv("../input/ufcdata/data.csv")
raw_fighter_data = pd.read_csv("../input/ufcdata/raw_fighter_details.csv")


# In[ ]:


data.head(5)


# In[ ]:


# Size of the original data
data.shape


# In[ ]:


# Feature Names
print(", ".join(data.columns))


# In[ ]:


# Data types of each feature
data.dtypes


# ### Change the winner column to the name of fighter

# In[ ]:


data['Winner'].value_counts()


# In[ ]:


data.head()


# In[ ]:


# Replace Winner column with the winner of the fight
winners = []
for i in data.index:
    
    decision = data.loc[i, 'Winner']
    
    if decision == 'Draw':
        winners.append("No Contest")
    else:
        if decision == 'Red':
            fighter_name = data.loc[i, 'R_fighter']

        elif decision == 'Blue':
            fighter_name = data.loc[i, 'B_fighter']
        winners.append(fighter_name)
print(winners)


# In[ ]:


data['Winner'] = winners


# In[ ]:


# remove columns we won't be using
cols = ['R_fighter', 'B_fighter', 'date', 'location', 'Winner', 'title_bout', 'weight_class','no_of_rounds', 
        'B_total_time_fought(seconds)',
        'B_current_lose_streak', 'B_current_win_streak', 'B_longest_win_streak', 'B_losses', 'B_Height_cms', 'B_age',
        'R_current_lose_streak', 'R_current_win_streak', 'R_longest_win_streak', 'R_losses', 'R_Height_cms', 'R_age',
        'B_avg_BODY_att', 'B_avg_BODY_landed', 'B_avg_GROUND_att', 'B_avg_GROUND_landed'
       ]
new_data = data[cols]
new_data.head(5)


# In[ ]:


# missing values for the features
values = pd.isnull(new_data).sum()

# remove rows that are less than 5%
indices = np.where((values[:] / 5144 < 0.05) & (values[:] / 5144 >0.00))
rows = [indices]
new_data.dropna(how = 'any', inplace = True)
pd.isnull(new_data).sum()

new_data.shape


# In[ ]:


# How many fights per year
new_data.date = pd.to_datetime(new_data.date)
new_data['year'] = new_data['date'].dt.year
new_data.groupby(['year']).size()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
plt.plot(new_data.groupby(['year']).size())
plt.show()


# In[ ]:


women_indices = (new_data['weight_class'] == "Women's Strawweight") | (new_data['weight_class'] == "Women's Bantamweight") | (new_data['weight_class'] == "Women's Flyweight") | (new_data['weight_class'] == "Women's Featherweight")
women_data = new_data[women_indices]
plt.plot(women_data.groupby(['year']).size())
plt.title("Women Fights Per Year")
plt.show()


# In[ ]:


winners[2107]


# In[ ]:


# Subset of data before USADA implementation 
before2015 = new_data['year'] < 2015
pre2015data = new_data[before2015]
pre2015data.tail(5)

# Subset of data after USADA implementation
after2015 = new_data['year'] >= 2015
post2015data = new_data[after2015]
post2015data.tail(6)


# In[ ]:


# Find the average age pre 2015
avg_age_pre = []
for x in pre2015data.index:
    age_of_fight = pre2015data['R_age'][x] + pre2015data['B_age'][x]
    avg_age_pre.append(age_of_fight / 2)

# Find the average age post 2015
avg_age_post = []
for x in post2015data.index:
    age_of_fight = post2015data['R_age'][x] + post2015data['B_age'][x]
    avg_age_post.append(age_of_fight / 2)


# In[ ]:


plt.figure(figsize = (10,10))

sns.distplot(avg_age_pre, bins = 10, label = 'pre 2015')
sns.distplot(avg_age_post, bins = 10, label = 'post 2015')
plt.legend()
plt.title("Average Age")


# In[ ]:


raw_fighter_data = pd.read_csv("../input/ufcdata/raw_fighter_details.csv")

raw_fighter_data['Weight'] = raw_fighter_data['Weight'].str.strip(' lbs.')
raw_fighter_data['Weight'] = pd.to_numeric(raw_fighter_data['Weight'], errors='coerce')

raw_fighter_data = raw_fighter_data.dropna(subset=['Weight'])
raw_fighter_data['Weight'] = raw_fighter_data['Weight'].astype(int)
raw_fighter_data.dtypes

print(raw_fighter_data['Weight'].mean())

plt.xlim(90,350)
sns.distplot(raw_fighter_data['Weight'], bins=30)
plt.title("Distribution of Weight")


# In[ ]:


# Graph for Weight Class pre USADA 2015
plot = (pre2015data['weight_class'].value_counts().plot.pie(figsize = (10, 10), autopct='%1.1f', fontsize = 20, explode = (0.15, 0.15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
plt.title("Distribution of amount of fights for weight classes before 2015")
plt.show(plot)

# This shows that Welterweight & Lightweight make up the majority of fights 


# In[ ]:


# Graph for Weight Class post USADA 2015
plot = (post2015data['weight_class'].value_counts().plot.pie(figsize = (10,10), autopct='%1.1f', fontsize = 20, explode = (0.15, 0.15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
plt.title("Distribution of amount of fights for weight classes after 2015")
plt.show(plot)


# Both before and after USADA implementations shows that Lightweight & Welterweight STILL make up majority of fights
# However, all Women's weight classes have increased in the amount of fights since 2015


# In[ ]:


# Graph for pre 2015 USADA implementation Title Fights by weight class distributions
champspre2015 = (pre2015data['title_bout'] == False)
pre2015title = pre2015data[~champspre2015]
plot = pre2015title['weight_class'].value_counts().plot.pie(figsize = (10,10), autopct = '%1.1f', fontsize = 20, explode = (0.15, 0.15, 0, 0, 0, 0, 0, 0, 0, 0))
plt.title("Distribution of title fights for weight classes before 2015")


# In[ ]:


# Graph for post 2015 Title Fights by weight class distributions
champspost2015 = (post2015data['title_bout'] == False)
post2015title = post2015data[~champspost2015]
plot = post2015title['weight_class'].value_counts().plot.pie(figsize = (10,10), autopct = '%1.1f', fontsize = 20, explode = (0.15, 0.15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
plt.title("Distribution of title fights for weight classes after 2015")


# In[ ]:


# # using information from the all fights seeing how have Lightweight & Welterweight have changed post 2015 in longest win streaks, lose streaks, age
LightWeight_indices = post2015data['weight_class'] == 'Lightweight'
WelterWeight_indices = post2015data['weight_class'] == 'Welterweight'

LW_data = post2015data[LightWeight_indices]
WW_data = post2015data[WelterWeight_indices]


LightWeight_indices = pre2015data['weight_class'] == 'Lightweight'
WelterWeight_indices = pre2015data['weight_class'] == 'Welterweight'

LW_pre2015data = pre2015data[LightWeight_indices]
WW_pre2015data = pre2015data[WelterWeight_indices]


# In[ ]:


# Comparing Win Streaks for Red & Blue on the Lightweight class
plt.figure(figsize=(10,5))

sns.distplot(LW_data['R_longest_win_streak'].value_counts(), label = 'Lightweight post 2015 for Red Fighter');
sns.distplot(LW_pre2015data['R_longest_win_streak'].value_counts(), label = 'Lightweight pre 2015 for Red Fighter');

plt.title("Difference between pre and post 2015 win streaks for Red fighter ")


# In[ ]:


plt.figure(figsize=(10,5))

sns.distplot(LW_data['B_longest_win_streak'].value_counts(), label = 'Lightweight post 2015 for Blue Fighter');
sns.distplot(LW_pre2015data['B_longest_win_streak'].value_counts(), label = 'Lightweight pre 2015 for Blue Fighter');

plt.title("Difference between pre and post 2015 win streaks for Blue fighter ")


# In[ ]:


# Comparing Win Streaks for Red & Blue on the Welterweight class
plt.figure(figsize=(10,5))

sns.distplot( WW_data['R_longest_win_streak'].value_counts(), label = 'Welterweight post 2015 for Red Fighter');
sns.distplot( WW_pre2015data['R_longest_win_streak'].value_counts(), label = 'Welterweight pre 2015 for Red Fighter');

plt.legend()


# In[ ]:


plt.figure(figsize=(10,5))

sns.distplot( WW_data['B_longest_win_streak'].value_counts(), label = 'Welterweight post 2015 for Blue Fighter');
sns.distplot(WW_pre2015data['B_longest_win_streak'].value_counts(), label = 'Welterweight pre 2015 for Blue Fighter');

plt.legend()


# In[ ]:


import re
raw_fighter_data['Height'] = raw_fighter_data['Height'].astype(str)
r = re.compile(r"([0-9]+)' ([0-9]*\.?[0-9]+)\"")
def get_inches(el):
    m = r.match(el)
    if m == None:
        return float('NaN')
    else:
        return int(m.group(1))*12 + float(m.group(2))
 
raw_fighter_data['HeightInCm'] = raw_fighter_data['Height'].apply(lambda x:get_inches(x))
 
print(raw_fighter_data['HeightInCm'].mean())
plt.xlim(55,100)
sns.distplot(raw_fighter_data['HeightInCm'], bins=30)
raw_fighter_data.head()


# In[ ]:


pre2015data = pre2015data.dropna(subset=['B_avg_BODY_att'])
post2015data = post2015data.dropna(subset=['B_avg_BODY_att'])

plt.xlim(-5,40)

sns.distplot(pre2015data['B_avg_BODY_att'], label='Pre 2015')
sns.distplot(post2015data['B_avg_BODY_att'], label='Post 2015')

plt.legend()
plt.title('Number of body shots attempted, pre and post 2015')


# In[ ]:


pre2015data = pre2015data.dropna(subset=['B_avg_BODY_landed'])
post2015data = post2015data.dropna(subset=['B_avg_BODY_landed'])

plt.xlim(-5,25)

sns.distplot(pre2015data['B_avg_BODY_landed'], label='Pre 2015')
sns.distplot(post2015data['B_avg_BODY_landed'], label='Post 2015')

plt.legend()
plt.title('Number of body shots landed, pre and post 2015')


# In[ ]:


pre2015data = pre2015data.dropna(subset=['B_avg_GROUND_att'])
post2015data = post2015data.dropna(subset=['B_avg_GROUND_att'])

plt.xlim(-5,35)

sns.distplot(pre2015data['B_avg_GROUND_att'], label='Pre 2015')
sns.distplot(post2015data['B_avg_GROUND_att'], label='Post 2015')

plt.legend()
plt.title('Number of ground shots attempted, pre and post 2015')


# In[ ]:


pre2015data = pre2015data.dropna(subset=['B_avg_GROUND_landed'])
post2015data = post2015data.dropna(subset=['B_avg_GROUND_landed'])

plt.xlim(-5,25)

sns.distplot(pre2015data['B_avg_GROUND_landed'], label='Pre 2015')
sns.distplot(post2015data['B_avg_GROUND_landed'], label='Post 2015')

plt.legend()
plt.title('Number of ground shots landed, pre and post 2015')

