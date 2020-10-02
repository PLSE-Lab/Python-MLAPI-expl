#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import warnings
warnings.filterwarnings('ignore')
try:
    t_file = pd.read_csv('../input/globalterrorismdb_0616dist.csv', encoding='ISO-8859-1')
    print('File load: Success')
except:
    print('File load: Failed')

Global_terror = t_file.copy()
Global_terror = Global_terror.rename(columns={'iyear':'year','attacktype1_txt':'attacktype','country_txt':'country_name','region_txt':'region_name'})
Global_terror = Global_terror[['year','country_name','region_name','attacktype','nkill','nwound']]
Global_terror['count'] = 1     #creating a new column with value 1 for calculations

# by year
by_year = (Global_terror.groupby('year').agg({'count':'sum'}))

by_year.plot(kind='barh', color='lightblue',figsize=[10,10])
plt.ylabel('Year', size=16)
plt.xlabel('Total count', size=16)
plt.text('16826','42.5','16840',color='red', size=15)
plt.text('14806','43.5','14806',color='red', size=15)
plt.text('11990','41.5','11990',color='red', size=15)


# 2014 has been the most troublesome year, with a count of 16840 terrorist attacks, followed by 2015 with 14806 attacks and 2013 with 11990 attacks.

# In[ ]:


# Attack type
import seaborn as sns
attack_type = Global_terror.groupby('attacktype')['count'].count().reset_index()
total = attack_type['count'].sum()
attack_type['Percentage'] = attack_type.apply(lambda x : (x['count']/total) * 100, axis=1)

plt.figure(figsize=[16,8])
sns.pointplot(x='attacktype', y='Percentage', data=attack_type, color='red', rotation=30)
plt.xlabel('Attack Type', size=16)
plt.ylabel('Percentage', size=16)


# Bombing/Explosion leads the list with around 48-50% of the total terrorist related cases over the years. Followed by Armed Assault with 25%.

# In[ ]:


# Let's look at the attack types for the top 2 years, i.e. 1992 and 2011

attack_2013 = Global_terror[Global_terror.year == 2013].groupby('attacktype')['count'].count().reset_index()
attack_2014 = Global_terror[Global_terror.year == 2014].groupby('attacktype')['count'].count().reset_index()
attack_2015 = Global_terror[Global_terror.year == 2015].groupby('attacktype')['count'].count().reset_index()

def attack_year(df,col_name):
    new_df = df.rename(columns={'count':'total_attacks'})
    total = new_df['total_attacks'].sum()
    new_df['Percentage'] = new_df.apply(lambda x: (x['total_attacks']/total)*100, axis=1)
    return new_df

new_attack2013 = attack_year(attack_2013,'attacktype')
new_attack2014 = attack_year(attack_2014,'attacktype')
new_attack2015 = attack_year(attack_2015,'attacktype')

plt.figure(figsize=[16,6])
sns.pointplot(x='attacktype', y='Percentage', data=new_attack2013, color='green')
sns.pointplot(x='attacktype', y='Percentage', data=new_attack2014, color='blue')
sns.pointplot(x='attacktype', y='Percentage', data=new_attack2015, color='red')
plt.xlabel('Attack Type', size=16)
plt.ylabel('Percentage', size=16)
plt.title('Attack Types: 2013 vs 2014 vs 2015', size=18)

import matplotlib.patches as mpat
gre = mpat.Patch(color='green', label='2013')
blue = mpat.Patch(color='blue', label='2014')
red = mpat.Patch(color='red', label='2015')
plt.legend(handles=[gre,blue, red])


# All these 3 years follows the same trend. There is very little difference in the amount of attack types.
# Bombing/Explosions --> 52-57%
# Armed Assaults --> 22-26%

# In[ ]:


#dividing years into groups
def year_group(year):
    yr_grp=''
    if year < 1980:
        yr_grp = 'Group 1'
    elif year < 1990:
        yr_grp = 'Group 2'
    elif year < 2000:
        yr_grp = 'Group 3'
    else: 
        yr_grp = 'Group 4'
    return yr_grp

new_globalterror = Global_terror.copy()
new_globalterror['Group'] = new_globalterror.apply(lambda row: year_group(row['year']),axis=1)

group_1 = new_globalterror[new_globalterror.Group == 'Group 1'].groupby('attacktype')['count'].count().reset_index()
group_2 = new_globalterror[new_globalterror.Group == 'Group 2'].groupby('attacktype')['count'].count().reset_index()
group_3 = new_globalterror[new_globalterror.Group == 'Group 3'].groupby('attacktype')['count'].count().reset_index()
group_4 = new_globalterror[new_globalterror.Group == 'Group 4'].groupby('attacktype')['count'].count().reset_index()


# In[ ]:


#using the previous function to calculate the percentage
new_grp1 = attack_year(group_1,'attacktype')
new_grp2 = attack_year(group_2,'attacktype')
new_grp3 = attack_year(group_3,'attacktype')
new_grp4 = attack_year(group_4,'attacktype')

plt.figure(figsize=[16,8])
sns.pointplot(x='attacktype',y='Percentage', data=new_grp1, color='red')
sns.pointplot(x='attacktype',y='Percentage', data=new_grp2, color='green')
sns.pointplot(x='attacktype',y='Percentage', data=new_grp3, color='blue')
sns.pointplot(x='attacktype',y='Percentage', data=new_grp4, color='violet')

plt.xlabel('Attack Type', size=16)
plt.ylabel('Percentage', size=16)
plt.title('Attack Types: Different year groups', size=18)

red_l = mpat.Patch(color='red', label='1970-1980')
gre_l = mpat.Patch(color='green', label='1981-1990')
blue_l = mpat.Patch(color='blue', label='1991-2000')
vio_l = mpat.Patch(color='violet', label='2001-2015')
plt.legend(handles=[red_l,gre_l,blue_l,vio_l])


# In[ ]:


# Which attack type has been most effective - w.r.t number of kills and wounds
num_kills = Global_terror.groupby('attacktype')['nkill'].sum().reset_index()
num_wound = Global_terror.groupby('attacktype')['nwound'].sum().reset_index()

def attack_kills(df, col_name):
    new_df = df.rename(columns={'nkill':'kill_count'})
    total = new_df['kill_count'].sum()
    new_df['Percentage'] = new_df.apply(lambda x: (x['kill_count']/total)*100, axis=1)
    return new_df

def attack_wound(df,col_name):
    new_df = df.rename(columns={'nwound':'wound_count'})
    total = new_df['wound_count'].sum()
    new_df['Percentage'] = new_df.apply(lambda x: (x['wound_count']/total)*100, axis=1)
    return new_df
    
new_kills = attack_kills(num_kills,'attacktype')
new_wound = attack_wound(num_wound,'attacktype')

plt.figure(figsize=[16,7])
sns.pointplot(x='attacktype', y='Percentage', data=new_kills, color='blue')
sns.pointplot(x='attacktype', y='Percentage', data=new_wound, color='green')
plt.xlabel('Attack Type', size=16)
plt.ylabel('Percentage', size=16)
plt.title('Total Kills vs Total Wounds', size=18)

gre_lab = mpat.Patch(color='green', label = 'Wounded')
blue_lab = mpat.Patch(color='blue', label = 'Killed')
plt.legend(handles=[blue_lab,gre_lab])


# As we saw in the previous topic. bombing/explosions was the highest form of attack, followed by armed assault. But this is not the case when it comes to number of people killed.
# Armed assault clearly takes the lead killing 42% of the total people, whereas bombing/explosions killed 38%.
# 
# In case of people getting wounded, bombing/explosions left 75% of total people wounded, whereas armed assault with 15%.
# 
# Conclusion: Most effective way that killed maximum number of people --> Armed Assault [42%]
#             Most effective way that wounded maximum number of people --> Bombing/Explosions [75%]

# In[ ]:


# Killed & Wounded --> w.r.t Year Groups
x_kill = new_globalterror.groupby('Group')['nkill'].sum()
y_wound = new_globalterror.groupby('Group')['nwound'].sum()

plt.figure(figsize=[16,7])
plt.subplot(121)
x_kill.plot.pie(subplots=True, autopct = '%.2f', figsize=[6,6], fontsize=20)
plt.ylabel("")
plt.title("Percentage of people killed in each Year Group", size=18)

plt.subplot(122)
y_wound.plot.pie(subplots=True, autopct = '%.2f', figsize=[6,6], fontsize=20)
plt.ylabel("")
plt.title("Percentage of people wounded in each Year Group", size=18)


# The pie chart clearly depicts how Group 4 [2001-2015] dominates in both killed and wounded.

# In[ ]:


# Region
region_cases = Global_terror.groupby('region_name').agg({'count':'sum'})
region_cases.plot(kind='barh',figsize=[16,5])
plt.xlabel('Count of terrorist attacks', size=14)
plt.ylabel('Region', size=14)
plt.title('Terrorist Attacks in different regions of the world', size=16)


# Most troubled region: 1. Middle East & North Africa
#                                       2. South Asia

# In[ ]:


#kills and wounds in each region
x = Global_terror.groupby('region_name')['nkill'].sum()
y = Global_terror.groupby('region_name')['nwound'].sum()

plt.figure(figsize=[5.5,12])
plt.subplot(211)
x.plot.pie(subplots=True, autopct = '%.2f', figsize=[6,6], fontsize=11)
plt.ylabel("")
plt.title("Percentage of people killed in each Region", size=18)

plt.subplot(212)
y.plot.pie(subplots=True, autopct = '%.2f', figsize=[6,6], fontsize=11)
plt.ylabel("")
plt.title("Percentage of people wounded in each Region", size=18)


# Middle East & N.Africa --> 30.55% people killed and 41.26% wounded from the total people killed/wounded by terrorist attacks.
# South Asia --> 24.52% killed and 27.78 wounded.
# 
# Lets go deeper into these 2 regions:

# In[ ]:


mideast = Global_terror[Global_terror.region_name == 'Middle East & North Africa'].groupby('country_name')['count'].sum()
Sasia = Global_terror[Global_terror.region_name == 'South Asia'].groupby('country_name')['count'].sum()

# Middle East & North Africa
mideast.plot(kind='barh', figsize=[10,6])
plt.title('Middle East & North Africa', size=18)
plt.xlabel('Total Attacks', size=16)
plt.ylabel('Country', size=16)


# In[ ]:


# South Asia
Sasia.plot(kind='barh', figsize=[10,6])
plt.title('South Asia', size=18)
plt.xlabel('Total Attacks', size=16)
plt.ylabel('Country', size=16)


# Iraq from Middle East and Pakistan from South Asia have the highest number of terrorist attacks.

# In[ ]:


# Iraq vs Pakistan
iraq_df = Global_terror[Global_terror.country_name == 'Iraq'].groupby('attacktype')['count'].sum().reset_index()
pak_df = Global_terror[Global_terror.country_name == 'Pakistan'].groupby('attacktype')['count'].sum().reset_index()

new_iraq = attack_year(iraq_df,'attacktype')
new_pak = attack_year(pak_df,'attacktype')

plt.figure(figsize=[16,8])
sns.pointplot(x='attacktype', y='Percentage', data=new_iraq, color='blue')
sns.pointplot(x='attacktype', y='Percentage', data=new_pak, color='red')
plt.title('Iraq vs Pakistan', size=18)
plt.xlabel('Attack Types', size=16)
plt.ylabel('Percentage', size=16)

blue_l = mpat.Patch(color='blue', label='Iraq')
red_l = mpat.Patch(color='red', label='Pakistan')
plt.legend(handles=[blue_l,red_l])


# Bombing/Explosions are in extreme numbers in Iraq, totalling upto 72%, whereas all the other types are below 20%.
# But in Pakistan, the rate of bombing/explosions and armed assaults are about 53% and 28% respectively.

# In[ ]:


# Iraq
iraq_kills = Global_terror[Global_terror.country_name == 'Iraq'].groupby('attacktype')['nkill'].sum().reset_index()
iraq_wound = Global_terror[Global_terror.country_name == 'Iraq'].groupby('attacktype')['nwound'].sum().reset_index()

new_iraqkill = attack_kills(iraq_kills,'attacktype')
new_iraqwound = attack_wound(iraq_wound,'attacktype')

plt.figure(figsize=[16,6])
sns.pointplot(x='attacktype', y='Percentage', data=new_iraqkill, color='blue')
sns.pointplot(x='attacktype', y='Percentage', data=new_iraqwound, color='green')
plt.xlabel('Attack Type', size=16)
plt.ylabel('Percentage', size=16)
plt.title('Iraq: Kills vs Wounds', size=18)

blu_lab = mpat.Patch(color='blue', label='Killed')
gr_lab = mpat.Patch(color='green', label='Wounded')
plt.legend(handles=[blu_lab,gr_lab])


# This is some extreme finding, around 75% of people killed and more than 95% wounded in Iraq due to terrorism, are a result of bombing/explosions.
# Armed assault only leads to 17-18% kills and 4-5% wounded people.

# In[ ]:


# Pakistan
pak_kills = Global_terror[Global_terror.country_name == 'Pakistan'].groupby('attacktype')['nkill'].sum().reset_index()
pak_wound = Global_terror[Global_terror.country_name == 'Pakistan'].groupby('attacktype')['nwound'].sum().reset_index()

new_pakkill = attack_kills(pak_kills,'attacktype')
new_pakwound = attack_wound(pak_wound,'attacktype')

plt.figure(figsize=[16,6])
sns.pointplot(x='attacktype', y='Percentage', data=new_pakkill, color='blue')
sns.pointplot(x='attacktype', y='Percentage', data=new_pakwound, color='green')
plt.xlabel('Attack Type', size=16)
plt.ylabel('Percentage', size=16)
plt.title('Pakistan: Kills vs Wounds', size=18)

blu_lab = mpat.Patch(color='blue', label='Killed')
gr_lab = mpat.Patch(color='green', label='Wounded')
plt.legend(handles=[blu_lab,gr_lab])


# In[ ]:


# top 15 dangerous place - w.r.t terrorism [1970 to 2015]

top15_dan = Global_terror.groupby('country_name')['count'].count().sort_values(ascending=False)
plt.figure(figsize=[18,6])
top15_dan.head(15).plot(kind='bar', rot=0)
plt.xlabel('Country', size=16)
plt.ylabel('Total Attacks', size=16)
plt.title('Top 15 Dangerous Country - Terrorism [1970-2015]', size=18)

