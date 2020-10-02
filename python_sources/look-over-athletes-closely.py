#!/usr/bin/env python
# coding: utf-8

# In this kernel, i'll try to find out relation between atletes' height and weight. And i'll inspect connection between BMI and medals.
# 
# **Content**
# 
# 1. [Importing and analyzing the data](#1)
# 1. [Top gold medalled sports](#2)
# 1. [Height - Weight plots](#3)
# 1. [BMI Analysis](#4)
# 1. [Other Analysis](#5)
# 1. [Ending](#6)

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 
import warnings

warnings.filterwarnings('ignore')


# <a id="1"></a> <br>
# ## 1. Importing and analyzing the data

# In[ ]:


#import data from csv file.
df_olympic = pd.read_csv("../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv")
df_noc = pd.read_csv("../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv")
olympic = df_olympic.copy()


# In[ ]:


olympic.info()


# The olympic data has 15 columns. I'll deal with Sports, Sex, Medal, Height, Weight columns mostly.

# In[ ]:


olympic.head()


# There are null (NaN) values both Height and Weight columns. On the top five row, we can see it. We'll have to handle them.

# In[ ]:


olympic.describe()


# In[ ]:


#I'm omitting ID (column no 0) with iloc not to affect our correlation.
#I'm trying to find out correlation between our columns.

f,ax = plt.subplots(figsize=(8,6))
sns.heatmap(olympic.iloc[:, 1:].corr(),annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()


# <a id="2"></a> <br>
# ## 2. Top gold medalled sports

# It seems there are direct correlation between our Height and Weight columns. So i will examine connection between Height and Weight of gold medalists.

# In[ ]:


#filtering gold medal winners as gold_medal

gold_medal = olympic[olympic['Medal']=='Gold']


# I want to learn the top gold medalled sports branches. Because of that, i'm creating a series group by sport and adding elements from series to dictionary.

# In[ ]:


sports = gold_medal.groupby(['Sport']).size()
top_gold_medal_sports = pd.DataFrame({'Sports':sports.index, 'Count':sports.values})
#Ranking the count of medalled sport branches in descending order.
top_gold_medal_sports.sort_values(['Count', 'Sports'], ascending=[False, True], inplace=True)

print(top_gold_medal_sports.head())


# There are an easier way though. 

# In[ ]:


print(gold_medal['Sport'].value_counts(dropna=False).head()) 


# I'll use this top records to select brances for my weight-height plot later.  Now, i'm able to create bar plot of top 25 gold medalled sports branches. 

# In[ ]:


###I will use of this graphic to determine that i decide to examine which branches

plt.figure(figsize=(18,9))
sns.barplot(x=top_gold_medal_sports['Sports'][:20], y=top_gold_medal_sports['Count'][:20])
plt.xticks(rotation= 60)
#plt.xlabel('Sports')
#plt.ylabel('Gold Medal Count')
plt.title('Top 20 Gold Medalled Sports')

plt.show()


# <a id="3"></a> <br>
# ## 3. Height - Weight plots

# I'm trying plot scatter diagram of all gold medalled athlethes at once. 

# In[ ]:


gold_medal.plot(kind='scatter', x='Weight', y='Height', alpha=0.5, color='darkblue', figsize = (15,9))
plt.xlabel='Weight'
plt.ylabel='Height'
plt.title('Gold Medal Winner at Athletics')

plt.show()


# There are a significant correlation between Height and Weight. it's a bit complicated because we plotted all branches at same time. So I want to examine sport branches separately.

# In[ ]:


def medalled_athletes_scatter(medal_type,sport):
    """
    takes a medal type and sport type, 
    gives a scatter plot for weight-height relation
    """
    athletes=olympic[(olympic['Medal']==medal_type) & (olympic['Sport']==sport)]
    athletes.plot(kind='scatter', x='Weight', y='Height', alpha=0.5, color='darkgreen', label="All Genders", figsize = (12,7))
    patch = mpatches.Patch(color='darkgreen', label='All')
    plt.legend(handles=[patch], loc='lower right')
    plt.xlabel='Weight'
    plt.ylabel='Height'
    plt.title(medal_type+' Medal Winners at '+sport)
    
    plt.show()


# Let's try to plot gold medalled gymnastics athletes with our function.

# In[ ]:


medalled_athletes_scatter('Gold', 'Gymnastics')


# As you can see, height and weight are directly proportional explicitly. I think it will be good if we add gender division.

# In[ ]:


def medalled_athletes_gender_scatter(medal_type,sport):
    """
    takes a medal type and sport type, gives a scatter plot 
    for weight-height relation with gender discrimination
    """
    athletes=olympic[(olympic["Medal"]==medal_type) & (olympic["Sport"]==sport)].loc[:,["Sex","Height","Weight","Sport","Medal"]]
    athletes["Color"] = ["blue" if each =="M" else "red" for each in athletes.Sex]     
    athletes.plot(kind='scatter', x='Weight', y='Height', alpha=0.5, color=athletes["Color"], figsize = (12,7))
    red_patch = mpatches.Patch(color='red', label='Female')
    blue_patch = mpatches.Patch(color='blue', label='Male')
    plt.legend(handles=[red_patch, blue_patch], loc='lower right')
    plt.title(medal_type+" Medal Winners at "+sport)
    
    plt.show()


# In[ ]:


medalled_athletes_gender_scatter('Gold', 'Gymnastics')


# We can inspect any branches now. For example I wonder silver medalist at basketball.

# In[ ]:


medalled_athletes_gender_scatter('Silver', 'Basketball')


# It's a tidier plot. It's almost a line chart.

# <a id="4"></a> <br>
# ## 4. BMI Analysis

# Body mass index (BMI) is a measure of body fat.The formula for BMI is : weight (kg) / (height (m))2
# As you remember, we have NaN values at Weight and Height. We can assign them mean values but i prefer to ignore them. I calculate bmi column below.

# In[ ]:


olympic['BMI'] = olympic['Weight']/(olympic['Height']/100)**2


# When weight or height have a NaN value, bmi is NaN too. You can see this below.

# In[ ]:


olympic.loc[:,['Weight', 'Height', 'BMI']].head()


# We can use histogram plot for bmi analysis. I want to compare two type sport. For example wrestling and basketball.

# In[ ]:


Wrestling=olympic[(olympic["Medal"]=='Gold') & (olympic["Sport"]=='Wrestling')].loc[:,["BMI","Sport","Medal"]]    
Basketball=olympic[(olympic["Medal"]=='Gold') & (olympic["Sport"]=='Basketball')].loc[:,["BMI","Sport","Medal"]]    

f,ax=plt.subplots(1,2,figsize=(15,7))

Wrestling.BMI.plot.hist(ax=ax[0],bins=30,edgecolor='black',color='purple')

ax[0].set_title('BMI Distribution of Gold Wrestlers')
x1=list(range(15,50,5))
ax[0].set_xticks(x1)

Basketball.BMI.plot.hist(ax=ax[1],bins=30,edgecolor='black',color='orange')

ax[1].set_title('BMI Distribution of Gold Basketballers')
x2=list(range(10,40,5))
ax[1].set_xticks(x2)

plt.show()


# We can say based on the graph above that wrestling athletes have more weight proportion than basketball players. Because they have wide range of BMIs. Basketball players those have above 35 bmi are more than wrestlers.

# In[ ]:


#Summer Olympics since 1928 
df_summer = olympic[(olympic.Season == 'Summer') & (olympic.Year >= 1928)]

#Winter Olympics since 1928 
df_winter = olympic[(olympic.Season == 'Winter') & (olympic.Year >= 1928)]


# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12, 8))

sns.boxplot(x="Year", y="BMI", data=df_summer, hue="Sex", 
            palette="muted", ax=ax) 
           
ax.set_ylim([15, 30])
plt.xticks(rotation=90)
plt.title("Athlete BMI over time (Summer Oliympics)")

plt.show()


# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12, 8))
sns.boxplot(x="Year", y="BMI", data=df_winter, hue="Sex", palette="viridis_r", ax=ax)
ax.set_ylim([15, 30])
plt.xticks(rotation=90)
plt.title("Athlete BMI over time (Winter Oliympics)")

plt.show()


# It seems Summer Oliympics have more uniform bmi than Winter Oliympics over time.

# <a id="5"></a> <br>
# ## 5. Other Analysis

# In[ ]:


df_winter_ages = df_winter.groupby(['Year','Sex'], as_index=False)['Age','Year','Sex'].agg('mean')
df_summer_ages = df_summer.groupby(['Year','Sex'], as_index=False)['Age','Year','Sex'].agg('mean')
df_winter_ages.head()


# In[ ]:


plt.figure(figsize=(12,6))
sns.barplot(x="Year", y="Age", data=df_winter_ages, hue="Sex", palette="hls")
plt.title('Age Distribution in Winter Olympic by Sex') 
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()


# In[ ]:


plt.figure(figsize=(12,6))
sns.barplot(x="Year", y="Age", data=df_summer_ages, 
            hue="Sex", palette="rocket") 

plt.title('Age Distribution in Summer Olympic by Sex') 
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()


# The average of females age peaks in 1904.

# In[ ]:


OL = olympic.loc[:,["Year", "ID", "Sex"]].drop_duplicates().groupby(["Year", "Sex"]).size().reset_index()
OL.columns = ["Year","Sex","Count"]


# In[ ]:


Years = olympic["Year"].sort_values().unique().tolist()

Female = []
Male = []

for year in Years:
    Female.append(
        OL[(OL["Year"] == year) & (OL["Sex"] == 'F')]["Count"].sum()/
        OL[(OL["Year"] == year)]["Count"].sum()*100
    )
    
    Male.append(
        OL[(OL["Year"] == year) & (OL["Sex"] == 'M')]["Count"].sum()/
        OL[(OL["Year"] == year)]["Count"].sum()*100
    )


# In[ ]:


f,ax = plt.subplots(figsize=(16,6))
sns.barplot(x=Years, y=Female, label='Female', color='r', alpha = 0.7)
sns.barplot(x=Years, y=Male, label='Male', color='b', alpha = 0.4)

ax.set(xlabel='Hour', ylabel='Percentage', 
       title='Percentage Distribution of Female & Male Athletes by Years'
      ) 
      
ax.legend(loc='upper right',frameon= True)

plt.show()


# In[ ]:


plt.figure(figsize=(16,6))
sns.barplot(x="Year", y="Count", data=OL, 
            hue="Sex", palette="muted"
           ) 
           
plt.title('Number of Female & Male Athletes by Years') 

plt.show()


# In[ ]:


Cities = olympic.loc[:,["ID", "City"]].drop_duplicates().groupby(["City"]).size().reset_index()
Cities.columns = ["City", "Count"]
Cities = Cities.sort_values("Count", ascending=False)


# In[ ]:


plt.figure(figsize=(16,6))
sns.barplot(x="City", y="Count", data=Cities.head(25), palette="rocket")

plt.title('Cities Hosting The Greatest Number Of Olympic Games') 
plt.xticks(rotation=45)

plt.show()


# In[ ]:


Teams = olympic.loc[:,["ID", "NOC"]].drop_duplicates().groupby(["NOC"]).size().reset_index()
Teams.columns = ["NOC", "Count"]
TeamsCountry = pd.merge(Teams,df_noc, on=['NOC','NOC'])


# In[ ]:


data = [ dict(
        type = 'choropleth',
        locations = TeamsCountry['NOC'],
        locationmode = 'ISO-3',
        z = TeamsCountry['Count'],
        text = TeamsCountry['region'],
        
        colorscale=
            [[0.0, "rgb(251, 237, 235)"],
            [0.09, "rgb(245, 211, 206)"],
            [0.12, "rgb(239, 179, 171)"],
            [0.15, "rgb(236, 148, 136)"],
            [0.22, "rgb(239, 117, 100)"],
            [0.35, "rgb(235, 90, 70)"],
            [0.45, "rgb(207, 81, 61)"],
            [0.65, "rgb(176, 70, 50)"],
            [0.85, "rgb(147, 59, 39)"],
            [1.00, "rgb(110, 47, 26)"]],
        
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) 
        ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Count'),
      ) 
       ]

layout = dict(
    title = "Countries With the Most Teams",
    geo = dict(
        showframe = False,
        showcoastlines = True,
        width=500,height=400
    )
)

w_map = dict(data=data, layout=layout)

iplot( w_map, validate=False)


# <a id="6"></a> <br>
# ## 6. Ending

# This was my first kernel. I'll continue working on. If you like the kernel, please upvote.
