#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="whitegrid", palette="Paired")
plt.rcParams['figure.dpi'] = 120
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('../input/summer-olympics-medals/Summer-Olympic-medals-1976-to-2008.csv', encoding = "ISO-8859-1")
data


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.dropna(inplace=True)
data.Year = data.Year.astype(int)


# In[ ]:


data[['City', 'Year']].drop_duplicates().reset_index(drop=True)


# In[ ]:


gold_country = data[data.Medal == 'Gold'].groupby(['Country']).Medal.size()
gold_top_10 = gold_country.sort_values(ascending = False)[:10]
plt.figure(figsize=(9,7))
top_10_gold_pie = plt.pie(gold_top_10, labels=gold_top_10.index, 
                             autopct= lambda x: f'{x*sum(gold_top_10.values)/100 :.0f} golden', pctdistance=0.85)
pr_pie_circle = plt.Circle((0,0), 0.8, color='black', fc='white', linewidth=0)
p=plt.gcf()
p.gca().add_artist(pr_pie_circle)
plt.xlabel('Top 10 countries by gold medals')
plt.show()


# In[ ]:


silver_country = data[data.Medal == 'Silver'].groupby(['Country']).Medal.size()
silver_top_10 = silver_country.sort_values(ascending = False)[:10]
plt.figure(figsize=(9,7))
top_10_silver_pie = plt.pie(silver_top_10, labels=silver_top_10.index, 
                             autopct= lambda x: f'{x*sum(silver_top_10.values)/100 :.0f} silver', pctdistance=0.85)
pr_pie_circle = plt.Circle((0,0), 0.8, color='black', fc='white', linewidth=0)
p=plt.gcf()
p.gca().add_artist(pr_pie_circle)
plt.xlabel('Top 10 countries by silver medals')
plt.show()


# In[ ]:


bronze_country = data[data.Medal == 'Bronze'].groupby(['Country']).Medal.size()
bronze_top_10 = bronze_country.sort_values(ascending = False)[:10]
plt.figure(figsize=(9,7))
top_10_bronze_pie = plt.pie(bronze_top_10, labels=silver_top_10.index, 
                             autopct= lambda x: f'{x*sum(bronze_top_10.values)/100 :.0f} bronze', pctdistance=0.85)
pr_pie_circle = plt.Circle((0,0), 0.8, color='black', fc='white', linewidth=0)
p=plt.gcf()
p.gca().add_artist(pr_pie_circle)
plt.xlabel('Top 10 countries by bronze medals')
plt.show()


# In[ ]:


top_athelte = data[data.Medal == 'Gold'].groupby(['Athlete', 'Country']).Medal.size()
top_athelte = top_athelte[top_athelte>5].sort_values(ascending = False)
top_athelte = pd.DataFrame(top_athelte).reset_index()
top_athelte.columns = ['Athlete', 'Country', 'Gold medals']
print('Athlete who earnds more than 5 gold medals')
print('-'*42)
print(top_athelte)


# In[ ]:


data.Sport.unique()


# In[ ]:


aqua = data[data.Sport == 'Aquatics'].groupby(['Country']).Medal.size()
top_10_aqua = aqua.sort_values(ascending = False)[:10]
top_10_aqua_bar = top_10_aqua.plot.bar()
for p in top_10_aqua_bar.patches:
    top_10_aqua_bar.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2.0, 
                         p.get_height()), 
                         ha = 'center', 
                         va = 'center', 
                         xytext = (0, 4),
                         textcoords = 'offset points')
plt.xticks(rotation=45)
plt.xlabel('Medals earned in Aquatics (top 10 countries)')


# In[ ]:


athl = data[data.Sport == 'Athletics'].groupby(['Country']).Medal.size()
top_10_athl = athl.sort_values(ascending = False)[:10]
top_10_athl_bar = top_10_athl.plot.bar()
for p in top_10_athl_bar.patches:
    top_10_athl_bar.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2.0, 
                         p.get_height()), 
                         ha = 'center', 
                         va = 'center', 
                         xytext = (0, 4),
                         textcoords = 'offset points')
plt.xticks(rotation=45)
plt.xlabel('Medals earned in Athletics (top 10 countries)')


# In[ ]:


foot = data[data.Sport == 'Football'].groupby(['Country']).Medal.size()
top_10_foot = foot.sort_values(ascending = False)[:10]
top_10_foot_bar = top_10_foot.plot.bar()
for p in top_10_foot_bar.patches:
    top_10_foot_bar.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2.0, 
                         p.get_height()), 
                         ha = 'center', 
                         va = 'center', 
                         xytext = (0, 4),
                         textcoords = 'offset points')
plt.xticks(rotation=45)
plt.xlabel('Medals earned in Football (top 10 countries)')


# In[ ]:


gender_group = data.groupby(['Year', 'Gender']).size().unstack()
gender_group


# In[ ]:


gender_group.apply(lambda x:x/x.sum(), axis=1).plot(kind='barh', stacked=True, legend=False)
plt.legend(['Men', 'Women'], bbox_to_anchor=(1.0, 0.7))
plt.xlabel('Men / Women ratio')


# In[ ]:


top_women = data[data.Gender == 'Women'].groupby(['Country']).Medal.size()
top_women.sort_values(ascending = False)[:10]


# In[ ]:


top_men = data[data.Gender == 'Men'].groupby(['Country']).Medal.size()
top_men.sort_values(ascending = False)[:10]


# In[ ]:


data_ua = data[data.Country == 'Ukraine']
data_ua


# In[ ]:


ua_medals = data_ua.Medal.value_counts().plot.bar(color=['sienna', 'silver', 'gold'])
for p in ua_medals.patches:
    ua_medals.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2.0, 
                         p.get_height()), 
                         ha = 'center', 
                         va = 'center', 
                         xytext = (0, 4),
                         textcoords = 'offset points')
plt.xticks(rotation=0)
plt.xlabel('Ukraine medals')


# In[ ]:


ua_medals_gender = sns.countplot(x="Medal", hue="Gender", data=data_ua)
for p in ua_medals_gender.patches:
    ua_medals_gender.annotate(p.get_height(), 
                        (p.get_x() + p.get_width() / 2.0, 
                         p.get_height()), 
                         ha = 'center', 
                         va = 'center', 
                         xytext = (0, 3),
                         textcoords = 'offset points')


# In[ ]:


data_ua.Sport.value_counts()


# In[ ]:


data_ua[data.Medal == 'Gold'].Sport.value_counts()


# In[ ]:


data_ua[data.Medal == 'Gold'].Athlete.value_counts()


# In[ ]:


data_ua[data_ua.Athlete == 'KLOCHKOVA, Yana']

