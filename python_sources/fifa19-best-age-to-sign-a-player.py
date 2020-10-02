#!/usr/bin/env python
# coding: utf-8

# # Load the data and have a look

# In[ ]:


import pandas as pd


# In[ ]:


get_ipython().system(' chmod 777 data.csv')


# In[ ]:


get_ipython().system(' cat data.csv | head -10')


# In[ ]:


pd.set_option('display.max_columns', None)


# In[ ]:


df = pd.read_csv('../input/data.csv', index_col=0)
df.head()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


list(df.columns)


# ## Players by country

# In[ ]:


df2 =df[['Name', 'Nationality']]
df2_gb = df2.groupby('Nationality')


# In[ ]:


df2_gb = df2_gb.count().reset_index()


# In[ ]:


by_nation = df2_gb.sort_values(by='Name', ascending=False)
by_nation.head()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.style.use('ggplot')


# In[ ]:


labels = by_nation.Nationality
sizes = by_nation.Name
explode = 1

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, shadow=True, startangle=90);


# In[ ]:


by_nation = by_nation[by_nation['Name'] > 400]


# In[ ]:


len(labels)


# In[ ]:


labels = by_nation.Nationality
sizes = by_nation.Name
explode = []

for i in range(len(labels)):
    explode.append(0.2)    

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%');


# # Which country has the best players by overall mean?

# In[ ]:


df3 =df[['Name', 'Nationality', 'Overall']]
df3_gb = df3.groupby('Nationality')


# In[ ]:


df3 = df3_gb.mean().reset_index()
df3.head(10)


# In[ ]:


top_nation = df3.sort_values(by='Overall', ascending=False).head(10)
top_nation


# In[ ]:


labels = top_nation.Nationality
sizes = top_nation.Overall

plt.figure(figsize=(7,7))
plt.bar(labels, sizes, color='g', alpha=0.7)
plt.title('TOP AVERAGE OVERALL COUNTRIES')
plt.xticks(fontsize=15,rotation=90)


# # Let's code in order to calculate the best age to sign a player by position

# In[ ]:


import numpy as np


# In[ ]:


df.columns


# In[ ]:


df4 = df[['Name', 'Age', 'Nationality', 'Overall', 'Position']]
df4.head()


# In[ ]:


df4.count()


# In[ ]:


df4 = df4.dropna()


# In[ ]:


df4.count()


# In[ ]:


from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[ ]:


fix, ax = plt.subplots(figsize=(10,10))

ax.scatter(df4.Age, df4.Overall, alpha=0.2)
ax.set_title('AGE VS OVERALL', loc='left')
divider = make_axes_locatable(ax)
axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

axHistx.xaxis.set_tick_params(labelbottom=False)
axHisty.yaxis.set_tick_params(labelleft=False)

bins=30
axHistx.hist(df4.Age, bins=bins, )
axHisty.hist(df4.Overall, bins=bins, orientation='horizontal', density=True)
("")


# In[ ]:


positions = list(set(df4.Position))
positions.sort()
positions


# In[ ]:


len(positions)


# In[ ]:


fig, axs = plt.subplots(4,7)
plt.figure(figsize=(10,10))
fig.set_figheight(15)
fig.set_figwidth(15)

for i, position in enumerate(positions):
    if i < 7:
        axs[0,i].scatter(df4[df4.Position == position].Age, df4[df4.Position == position].Overall, alpha=0.4)
        axs[0,i].set_title(position)
    if i < 14 and i >= 7:
        axs[1,i-7].scatter(df4[df4.Position == position].Age, df4[df4.Position == position].Overall, alpha=0.4)
        axs[1,i-7].set_title(position)
    if i < 21 and i >= 14:
        axs[2,i-14].scatter(df4[df4.Position == position].Age, df4[df4.Position == position].Overall, alpha=0.4)
        axs[2,i-14].set_title(position)
    if i < 28 and i >= 21:
        axs[3,i-21].scatter(df4[df4.Position == position].Age, df4[df4.Position == position].Overall, alpha=0.4)
        axs[3,i-21].set_title(position)
("")


# ### There are many position, let's group them in fewer

# In[ ]:


df4[df4.Position == 'LAM'].count()


# In[ ]:


df4['Real_Position'] = ''


# In[ ]:


df4.loc[df4['Position'] == 'GK', 'Real_Position'] = 'GOAL KEEPER'


# In[ ]:


df4.loc[df4['Position'] == 'CB', 'Real_Position'] = 'CENTRE BACK'
df4.loc[df4['Position'] == 'RCB', 'Real_Position'] = 'CENTRE BACK'
df4.loc[df4['Position'] == 'LCB', 'Real_Position'] = 'CENTRE BACK'


# In[ ]:


df4.loc[df4['Position'] == 'RB', 'Real_Position'] = 'LATERAL BACK'
df4.loc[df4['Position'] == 'LB', 'Real_Position'] = 'LATERAL BACK'
df4.loc[df4['Position'] == 'LWB', 'Real_Position'] = 'LATERAL BACK'
df4.loc[df4['Position'] == 'RWB', 'Real_Position'] = 'LATERAL BACK'


# In[ ]:


df4.loc[df4['Position'] == 'CDM', 'Real_Position'] = 'MIDFIELDER'
df4.loc[df4['Position'] == 'CM', 'Real_Position'] = 'MIDFIELDER'
df4.loc[df4['Position'] == 'CAM', 'Real_Position'] = 'MIDFIELDER'
df4.loc[df4['Position'] == 'CAM', 'Real_Position'] = 'MIDFIELDER'
df4.loc[df4['Position'] == 'RM', 'Real_Position'] = 'MIDFIELDER'
df4.loc[df4['Position'] == 'LM', 'Real_Position'] = 'MIDFIELDER'
df4.loc[df4['Position'] == 'LDM', 'Real_Position'] = 'MIDFIELDER'
df4.loc[df4['Position'] == 'RDM', 'Real_Position'] = 'MIDFIELDER'


# In[ ]:


df4.loc[df4['Position'] == 'RW', 'Real_Position'] = 'LATERAL MIDFIELDER'
df4.loc[df4['Position'] == 'LW', 'Real_Position'] = 'LATERAL MIDFIELDER'
df4.loc[df4['Position'] == 'RCM', 'Real_Position'] = 'LATERAL MIDFIELDER'
df4.loc[df4['Position'] == 'LCM', 'Real_Position'] = 'LATERAL MIDFIELDER'
df4.loc[df4['Position'] == 'LAM', 'Real_Position'] = 'LATERAL MIDFIELDER'
df4.loc[df4['Position'] == 'RAM', 'Real_Position'] = 'LATERAL MIDFIELDER'


# In[ ]:


df4.loc[df4['Position'] == 'CF', 'Real_Position'] = 'STRIKER'
df4.loc[df4['Position'] == 'RF', 'Real_Position'] = 'STRIKER'
df4.loc[df4['Position'] == 'LF', 'Real_Position'] = 'STRIKER'
df4.loc[df4['Position'] == 'ST', 'Real_Position'] = 'STRIKER'
df4.loc[df4['Position'] == 'LS', 'Real_Position'] = 'STRIKER'
df4.loc[df4['Position'] == 'RS', 'Real_Position'] = 'STRIKER'


# In[ ]:


df4.drop('Position', axis=1, inplace=True)


# In[ ]:


positions = list(set(df4['Real_Position']))
positions


# In[ ]:


fig, axs = plt.subplots(3,2)
plt.figure(figsize=(10,10))
fig.set_figheight(15)
fig.set_figwidth(15)

for i, position in enumerate(positions):
    if i < 2:
        axs[0,i].scatter(df4[df4.Real_Position == position].Age, df4[df4.Real_Position == position].Overall, alpha=0.4)
        axs[0,i].set_title(position)
    if i < 4 and i >= 2:
        axs[1,i-2].scatter(df4[df4.Real_Position == position].Age, df4[df4.Real_Position == position].Overall, alpha=0.4)
        axs[1,i-2].set_title(position)
    if i < 6 and i >= 4:
        axs[2,i-5].scatter(df4[df4.Real_Position == position].Age, df4[df4.Real_Position == position].Overall, alpha=0.4)
        axs[2,i-5].set_title(position)
("")


# In[ ]:


df4_g = df4.groupby(['Real_Position', 'Age'])


# In[ ]:


df4_mean = df4_g.mean().reset_index()
df4_mean


# In[ ]:


fig, axs = plt.subplots(3,2)
plt.figure(figsize=(10,10))
fig.set_figheight(15)
fig.set_figwidth(15)

for i, position in enumerate(positions):
    if i < 2:
        axs[0,i].scatter(df4[df4.Real_Position == position].Age, df4[df4.Real_Position == position].Overall, alpha=0.4)
        axs[0,i].set_title(position)
        axs[0,i].plot(df4_mean[df4_mean.Real_Position == position].Age, df4_mean[df4_mean.Real_Position == position].Overall, color='b')
    if i < 4 and i >= 2:
        axs[1,i-2].scatter(df4[df4.Real_Position == position].Age, df4[df4.Real_Position == position].Overall, alpha=0.4)
        axs[1,i-2].set_title(position)
        axs[1,i-2].plot(df4_mean[df4_mean.Real_Position == position].Age, df4_mean[df4_mean.Real_Position == position].Overall, color='b')
    if i < 6 and i >= 4:
        axs[2,i-5].scatter(df4[df4.Real_Position == position].Age, df4[df4.Real_Position == position].Overall, alpha=0.4)
        axs[2,i-5].set_title(position)
        axs[2,i-5].plot(df4_mean[df4_mean.Real_Position == position].Age, df4_mean[df4_mean.Real_Position == position].Overall, color='b')
("")


# ### Let's remove outliers in order to get better results

# In[ ]:


df5 = df4[['Age', 'Overall', 'Real_Position']]
df5.head()


# In[ ]:


df5_g = df5.groupby(['Real_Position', 'Age'])
df5_g


# In[ ]:


df5_g2 = df5_g.agg(['mean', 'std', 'count']).reset_index()
df5_g2.sample(5)


# In[ ]:


df5_g2.columns


# In[ ]:


df5_g2['Overall', 'count'].sum()


# In[ ]:


df5_merged = df5.merge(df5_g2, how='left', left_on=['Real_Position', 'Age'], right_on=['Real_Position', 'Age'])
df5_merged.head(10)


# In[ ]:


list(df5_merged.columns)


# In[ ]:


df5_merged.columns = ['Age',
 'Overall',
 'Real_Position',
 'O_mean',
 'O_std',
 'O_count']


# In[ ]:


df5_merged.head()


# In[ ]:


df5_merged['Distance_to_mean'] = abs(df5_merged.Overall - df5_merged.O_mean)
df5_merged.head(5)


# In[ ]:


df5_merged['Outliers'] = df5_merged.Distance_to_mean >= (df5_merged.O_std * 3)
df5_merged.head(5)


# In[ ]:


df5_clean = df5_merged[df5_merged.Outliers == False]
df5_clean.sample(5)


# In[ ]:


fig, axs = plt.subplots(3,2)
plt.figure(figsize=(10,10))
fig.set_figheight(15)
fig.set_figwidth(15)

for i, position in enumerate(positions):
    if i < 2:
        axs[0,i].scatter(df5_clean[df5_clean.Real_Position == position].Age, df5_clean[df5_clean.Real_Position == position].Overall, alpha=0.4)
        axs[0,i].set_title(position)
    if i < 4 and i >= 2:
        axs[1,i-2].scatter(df5_clean[df5_clean.Real_Position == position].Age, df5_clean[df5_clean.Real_Position == position].Overall, alpha=0.4)
        axs[1,i-2].set_title(position)
    if i < 6 and i >= 4:
        axs[2,i-5].scatter(df5_clean[df5_clean.Real_Position == position].Age, df5_clean[df5_clean.Real_Position == position].Overall, alpha=0.4)
        axs[2,i-5].set_title(position)
("")


# In[ ]:


df5_clean_g = df5_clean.groupby(['Real_Position', 'Age'])


# In[ ]:


df5_clean_g_mean = df5_clean_g['Overall'].mean().reset_index()
df5_clean_g_mean.sample(5)


# In[ ]:


fig, axs = plt.subplots(3,2)
plt.figure(figsize=(10,10))
fig.set_figheight(15)
fig.set_figwidth(15)

for i, position in enumerate(positions):
    if i < 2:
        axs[0,i].scatter(df5_clean[df5_clean.Real_Position == position].Age, df5_clean[df5_clean.Real_Position == position].Overall, alpha=0.4)
        axs[0,i].set_title(position)
        axs[0,i].plot(df5_clean_g_mean[df5_clean_g_mean.Real_Position == position].Age, df5_clean_g_mean[df5_clean_g_mean.Real_Position == position].Overall, color='b')
    if i < 4 and i >= 2:
        axs[1,i-2].scatter(df5_clean[df5_clean.Real_Position == position].Age, df5_clean[df5_clean.Real_Position == position].Overall, alpha=0.4)
        axs[1,i-2].set_title(position)
        axs[1,i-2].plot(df5_clean_g_mean[df5_clean_g_mean.Real_Position == position].Age, df5_clean_g_mean[df5_clean_g_mean.Real_Position == position].Overall, color='b')
    if i < 6 and i >= 4:
        axs[2,i-5].scatter(df5_clean[df5_clean.Real_Position == position].Age, df5_clean[df5_clean.Real_Position == position].Overall, alpha=0.4)
        axs[2,i-5].set_title(position)
        axs[2,i-5].plot(df5_clean_g_mean[df5_clean_g_mean.Real_Position == position].Age, df5_clean_g_mean[df5_clean_g_mean.Real_Position == position].Overall, color='b')


# ## We can see in the last graphs that over 35 years the data is not normal.
# 
# #### That makes sense because players older than 35 are special cases* 
# *Except for goal keepers
# 
# #### Therefore, let's remove players older than 35

# In[ ]:


df5_clean = df5_clean[df5_clean.Age < 35]
df5_clean.head()


# In[ ]:


df5_clean_g = df5_clean.groupby(['Real_Position', 'Age'])
df5_clean_g_mean = df5_clean_g['Overall'].mean().reset_index()
df5_clean_g_mean.sample(5)


# In[ ]:


fig, axs = plt.subplots(3,2)
plt.figure(figsize=(10,10))
fig.set_figheight(15)
fig.set_figwidth(15)

for i, position in enumerate(positions):
    if i < 2:
        axs[0,i].scatter(df5_clean[df5_clean.Real_Position == position].Age, df5_clean[df5_clean.Real_Position == position].Overall, alpha=0.4)
        axs[0,i].set_title(position)
        axs[0,i].plot(df5_clean_g_mean[df5_clean_g_mean.Real_Position == position].Age, df5_clean_g_mean[df5_clean_g_mean.Real_Position == position].Overall, color='b')
    if i < 4 and i >= 2:
        axs[1,i-2].scatter(df5_clean[df5_clean.Real_Position == position].Age, df5_clean[df5_clean.Real_Position == position].Overall, alpha=0.4)
        axs[1,i-2].set_title(position)
        axs[1,i-2].plot(df5_clean_g_mean[df5_clean_g_mean.Real_Position == position].Age, df5_clean_g_mean[df5_clean_g_mean.Real_Position == position].Overall, color='b')
    if i < 6 and i >= 4:
        axs[2,i-5].scatter(df5_clean[df5_clean.Real_Position == position].Age, df5_clean[df5_clean.Real_Position == position].Overall, alpha=0.4)
        axs[2,i-5].set_title(position)
        axs[2,i-5].plot(df5_clean_g_mean[df5_clean_g_mean.Real_Position == position].Age, df5_clean_g_mean[df5_clean_g_mean.Real_Position == position].Overall, color='b')
("")


# # Conclusions:
# 
# 1) GOAL KEEPERS are better the older they are, **the curve stabilizes at age 30**. This may be because goalkeers do not need to be as fit as other players, so the experience is KPI.
# 
# 2) **The curve** of CENTER BACK, STRIKERS and LATERAL BACK players **stabilizes at age 27**. This may be because such players need experience as well as to be fit, therefore experience and fit are KPIs.
# 
# 3) **The curve**  of MILDFIELDERS **stabilizes at age 24**. This may be bacause MILFFIELDERS much depends on their technique, which is a skill. Therefore, thechnique is a KPI.
# 
# 5) **The curve**  of LATERAL MILDFIELDERS **stabilizes at age 23**. This may be because LATERAL MILFFIELDERS much depends on their technique and velocity, which are skills. Therefore, thechnique and velocity are a KPIs.
# 
# *Needless to say, this is football and there a lot of exceptions. I will sign Mbappe even though he is not 27 years old yet. :)*
# 
