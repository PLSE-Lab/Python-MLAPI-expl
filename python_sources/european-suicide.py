#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# ### We will be looking at the European segment of the WHO statistics for suicide. There are 3 main segments that we will take a look at:
# ## 1. The role of sex on suicide 
# ### Here we observe the difference in suicide statisics of each European country based on sex. How this difference occurs through time is also analysed.
# ## 2. The effect of age
# ### The contribution of age towards a country's suicide statistic is displayed with a violin plot, which also further highlights the difference in sex.
# ## 3. The geopandas package
# ### We take a little look at the geopandas package, which enables some nice plots of the world to be quickly drawn.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
sns.set_style("darkgrid")


import os
print(os.listdir("../input"))

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
# Any results you write to the current directory are saved as output.


# In[ ]:


def fill_and_filter(df, nas=-999):
    df.fillna(nas, inplace=True)
    df = df[df['suicides_no']>=0]
    df = df[df['population']>=0]
    return df

def filter_year(df, year=2015):
    return df[df['year']==year]

def filter_country(df, country='Brazil'):
    return df[df['country']==country]

def filter_continent(df, continent='Europe'):
    return df[df['continent']==continent]

def groupings(df):
    """
    Calculates Percentage Statistics based on the country's sex and age demographics
    """
    tmp = pd.DataFrame({'total_population_country' : df.groupby(['country'])['population'].sum()}).reset_index()
    df = df.merge(tmp, on = 'country')
    tmp = pd.DataFrame({'total_suicides_country' : df.groupby(['country'])['suicides_no'].sum()}).reset_index()
    df = df.merge(tmp, on = 'country')
    tmp = pd.DataFrame({'total_suicides_sex_country' : df.groupby(['country', 'sex'])['suicides_no'].sum()}).reset_index()
    df = df.merge(tmp, on = ['country', 'sex'])
    tmp = pd.DataFrame({'total_suicides_age_country' : df.groupby(['country', 'age'])['suicides_no'].sum()}).reset_index()
    df = df.merge(tmp, on = ['country', 'age'])

    df['suicide_percentage_country'] = df['total_suicides_country']*100 / (df['total_population_country'])
    
    df['suicide_percentage_sex_country'] = df['total_suicides_sex_country']*100 / (df['total_suicides_country']+1)
    df['suicide_percentage_age_country'] = df['total_suicides_age_country']*100 / (df['total_suicides_country']+1) 
    return df

def world_statistics(df):
    """
    Calculates more statistics based on the continent of the country
    """
    tmp = pd.DataFrame({'total_population_continent' : df.groupby(['continent'])['population'].sum()}).reset_index()
    df = df.merge(tmp, on = 'continent')
    tmp = pd.DataFrame({'total_suicides_continent' : df.groupby(['continent'])['suicides_no'].sum()}).reset_index()
    df = df.merge(tmp, on = 'continent')
    tmp = pd.DataFrame({'total_suicides_sex_continent' : df.groupby(['continent', 'sex'])['suicides_no'].sum()}).reset_index()
    df = df.merge(tmp, on = ['continent', 'sex'])
    tmp = pd.DataFrame({'total_suicides_age_continent' : df.groupby(['continent', 'age'])['suicides_no'].sum()}).reset_index()
    df = df.merge(tmp, on = ['continent', 'age'])
    
    df['suicide_percentage_country_continent'] = df['total_suicides_country']*100 / (df['total_population_continent'])
    
    df['percentage_suicide_per_continent'] = df['total_suicides_continent'] *100/ df['total_population_continent']
    df['suicide_percentage_sex_continent'] = df['total_suicides_sex_continent']*100 / (df['total_suicides_continent']+1)
    df['suicide_percentage_age_continent'] = df['total_suicides_age_continent']*100 / (df['total_suicides_continent']+1) 
    return df


# ### I am mostly going to be looking at European statistics here, as they are the most well-covered compared with the other continents. 

# In[ ]:


df = pd.read_csv('../input/who_suicide_statistics.csv')
df = fill_and_filter(df, nas=-999)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world[(world.pop_est>0) & (world.name!="Antarctica")]
world = world.merge(df, left_on = 'name', right_on = 'country')


#world = filter_continent(world,continent= 'Europe')


# In[ ]:


def filter_and_plot(df, year=None,country=None, continent=None, column = 'total_suicides_country'):
    """
    Filters for the relevant year and country/continent, using geopandas to plot the column
    """
    if year!=None:
        df = filter_year(df, year=year)
    if country!=None:
        df = filter_country(df, country=country)
    elif continent!=None:
        df = filter_continent(df, continent=continent)
    df = groupings(df)
    df = world_statistics(df)
    fig, ax = plt.subplots(1, figsize=(15,15))
    df.plot(column = column, ax=ax)
    ax.axis('off')
    # Create colorbar as a legend
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=np.max(df[column])))
    # empty array for the data range
    sm._A = []
    # add the colorbar to the figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_ylabel('%s' % (column))
    
def plot_bar(df, year=2014, col = 'suicide_percentage_country', title = ''):
    sns.set_style("white")
    sns.set_context({"figure.figsize": (24, 10)})
    plt.xticks(rotation=90)
    
    df.fillna(0, inplace=True)
    df = filter_year(df, year=year)
    df = groupings(df)
    df = world_statistics(df)

#Plot 2 - overlay - "bottom" series
    bottom_plot=sns.barplot(x = df.country, y = df[col], color = "orange")
    

    #topbar = plt.Rectangle((0,0),1,1,fc="blue", edgecolor = 'none')
    bottombar = plt.Rectangle((0,0),1,1,fc='orange',  edgecolor = 'none')
   

#Optional code - Make plot look nicer
    sns.despine(left=True)
    bottom_plot.set_ylabel("%s" % (col))
    bottom_plot.set_xlabel("Country")
    bottom_plot.set_title("%s" % (title),fontsize=40 )

#Set fonts to consistent 16pt size
    for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
                 bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
        item.set_fontsize(24)   
def plot_sex(df, year=2014, col = 'suicide_percentage_sex_country', title = ''):
    sns.set_style("white")
    sns.set_context({"figure.figsize": (24, 10)})
    plt.xticks(rotation=90)
    
    df.fillna(0, inplace=True)
    df = filter_year(df, year=year)
    df = groupings(df)
    df = world_statistics(df)
    #Set general plot properties
    
    male = df[df['sex']=='male']
    female = df[df['sex']=='female']
    male.sort_values('country')
    female.sort_values('country')
#Plot 1 - background - "total" (top) series
    top_plot = sns.barplot(x = male.country, y = male[col],color = "blue")
    

#Plot 2 - overlay - "bottom" series
    bottom_plot=sns.barplot(x = female.country, y = female[col], color = "pink")
    

    topbar = plt.Rectangle((0,0),1,1,fc="blue", edgecolor = 'none')
    bottombar = plt.Rectangle((0,0),1,1,fc='pink',  edgecolor = 'none')
    l = plt.legend([bottombar, topbar], ['Female', 'Male'], loc=1, ncol = 2, prop={'size':18})
    l.draw_frame(False)

#Optional code - Make plot look nicer
    sns.despine(left=True)
    bottom_plot.set_ylabel("%s" % (col))
    bottom_plot.set_xlabel("Country")
    bottom_plot.set_title("%s" % (title),fontsize=40 )

#Set fonts to consistent 16pt size
    for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
                 bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
        item.set_fontsize(24)

def plot_sex_time(df, country,col = 'suicide_percentage_sex_country', title = ''):
    sns.set_style("white")
    sns.set_context({"figure.figsize": (24, 10)})
    plt.xticks(rotation=90)
    
    males = []
    females = []
    years = []
    df.fillna(0, inplace=True)
    df = filter_country(df, country=country)
    for year in sorted(df['year'].unique()):
        tmp = filter_year(df, year)
        tmp = groupings(tmp)
        tmp = world_statistics(tmp)

        male = tmp[tmp['sex']=='male']
        female = tmp[tmp['sex']=='female']
        males.append(male[col].max())
        females.append(female[col].max())
        years.append(year)
        
        
#Plot 1 - background - "total" (top) series
    top_plot = sns.barplot(x = years, y = males,color = "blue")
    

#Plot 2 - overlay - "bottom" series
    bottom_plot=sns.barplot(x = years, y = females, color = "pink")
    

    topbar = plt.Rectangle((0,0),1,1,fc="blue", edgecolor = 'none')
    bottombar = plt.Rectangle((0,0),1,1,fc='pink',  edgecolor = 'none')
    l = plt.legend([bottombar, topbar], ['Female', 'Male'], loc=1, ncol = 2, prop={'size':18})
    l.draw_frame(False)

#Optional code - Make plot look nicer
    sns.despine(left=True)
    bottom_plot.set_ylabel("%s" % (col))
    bottom_plot.set_xlabel("%s" % (country))
    bottom_plot.set_title("%s" % (title),fontsize=40 )

#Set fonts to consistent 16pt size
    for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
                 bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
        item.set_fontsize(24)
        
def plot_sex_diff_time(df, years = [2004, 2014],sex='female', col = 'total_suicides_sex_country', title = ''):
    sns.set_style("white")
    sns.set_context({"figure.figsize": (24, 10)})
    plt.xticks(rotation=90)
    if sex=='female':
        color_ = 'pink'
    else:
        color_ = 'blue'
    
    tmp0 = filter_year(df, years[0])
    tmp0 = groupings(tmp0)
    tmp0 = world_statistics(tmp0)
    tmp1 = filter_year(df, years[1])
    tmp1 = groupings(tmp1)
    tmp1 = world_statistics(tmp1)
    
    f_tmp1 = pd.DataFrame(tmp1[tmp1['sex']==sex].groupby('country')[col].max().reset_index())
    f_tmp0 = pd.DataFrame(tmp0[tmp0['sex']==sex].groupby('country')[col].max().reset_index())
    f_tmp0.columns = ['country', 'new_column']
    f_tmp1 = f_tmp1.merge(f_tmp0, on='country')
    f_tmp1['diff'] = (f_tmp1[col] - f_tmp1['new_column']) / f_tmp1['new_column']
   
    #m_tmp1 = pd.DataFrame(tmp1[tmp1['sex']=='male'].groupby('country')[col].max().reset_index())
    #m_tmp0 = pd.DataFrame(tmp0[tmp0['sex']=='male'].groupby('country')[col].max().reset_index())
    #m_tmp0.columns = ['country', 'new_column']
    #m_tmp1 = m_tmp1.merge(m_tmp0, on='country')
    #m_tmp1['diff'] = (m_tmp1[col] -m_tmp1['new_column']) / m_tmp1['new_column']
  

    #top_plot = sns.barplot(x = m_tmp1.country, y = m_tmp1['diff'],color = "blue")
#Plot 2 - overlay - "bottom" series
    bottom_plot=sns.barplot(x = f_tmp1.country, y = f_tmp1['diff'], color = color_)
    

    #topbar = plt.Rectangle((0,0),1,1,fc="blue", edgecolor = 'none')
    bottombar = plt.Rectangle((0,0),1,1,fc=color_,  edgecolor = 'none')
   # l = plt.legend([bottombar, topbar], ['Female', 'Male'], loc=1, ncol = 2, prop={'size':18})
   # l.draw_frame(False)

#Optional code - Make plot look nicer
    sns.despine(left=True)
    bottom_plot.set_ylabel("%s" % (col))
    bottom_plot.set_xlabel("Country")
    bottom_plot.set_title("%s" % (title),fontsize=40 )

#Set fonts to consistent 16pt size
    for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
                 bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
        item.set_fontsize(24)


# ### Let us first look at the distribution of suicide percentages per country

# In[ ]:


plot_bar(world[world['continent']=='Europe'], col = 'suicide_percentage_country', title = 'Suicide Percentage by Country')


# ### Lithuania's suicide rate is clearly very worrying

# # **Role of Sex on Suicide in Europe**
# ### The first statistic that I'm interested in is the role of sex in suicide. Let us first pick a year - 2014 - to compare the countries of Europe

# In[ ]:


plot_sex(world[world['continent']=='Europe'])


# ### It appears Belarus, Poland, Romania and Slovakia have the worst suicide rates for men in Europe. The Netherlands and Sweden have the worst suicide rates for women in Europe, which is personally very surprising considering their renowned quality of life.

# In[ ]:


plot_sex(world[world['continent']=='Europe'], col = 'total_suicides_sex_country', title = 'Total Suicides per country')


# ### Germany and Ukraine have the most male suicides in 2014 in Europe
# ### Germany and France have the most female suicides.

# ### Let us now observe how a country's suicide based on sex changes throughout time. I will pick my country - The Netherlands.

# In[ ]:


plot_sex_time(world[world['continent']=='Europe'], country='Netherlands',col = 'suicide_percentage_sex_country', title = 'Suicides in The Netherlands')


# In[ ]:


plot_sex_time(world[world['continent']=='Europe'], country='Netherlands',col = 'total_suicides_sex_country', title = 'Total Suicides in The Netherlands')


# ## This is a worrying trend for the Netherlands - Suicides have risen over the past decade. 
# 
# ### What about Lithuania? As we saw earlier, this country is #1 in Europe - is it getting better or worse?

# In[ ]:


plot_sex_time(world[world['continent']=='Europe'], country='Lithuania',col = 'suicide_percentage_sex_country', title = 'Suicides in Lithuania')


# In[ ]:


plot_sex_time(world[world['continent']=='Europe'], country='Lithuania',col = 'total_suicides_sex_country', title = 'Total Suicides in Lithuania')


# ### This is good news for Lithuania - the suicide rate is dropping!
# 
# ### Let us now turn to percentage change

# In[ ]:


plot_sex_diff_time(world[world['continent']=='Europe'], years = [2004, 2014], sex='female',
                   col = 'total_suicides_sex_country', title = 'Percentage Change in Female European Suicide from 2004-2014')


# ### Over the last 10 years, most European countries are showing a decrease in suicide for women. There are clearly a few noteiceable exceptions, however

# In[ ]:


plot_sex_diff_time(world[world['continent']=='Europe'], years = [2004, 2014], sex='male',
                   col = 'total_suicides_sex_country', title = 'Percentage Change in Male European Suicide from 2004-2014')


# ### Suicide in Greece has also increased for men. This is unfortunately true for my homeland (UK) and current residence - The Netherlands.
# 
# ### It seems that in general, suicides in Eastern Europe are rapidly dropping for both sexes.

# # The Role of Age on Suicide

# In[ ]:


def violin(df, year=2014, column='suicides_no', log=False):
    #column='total_suicides_age_country'
    sns.set(style="whitegrid", palette="pastel", color_codes=True)
    sns.set_context("notebook", font_scale=2)
    tmp = filter_year(df, year=year)
    tmp = groupings(tmp)
    tmp = world_statistics(tmp)
    if log==True:
        tmp[column] = np.log(tmp[column]+1)
    vio = sns.violinplot(x="age", y=column, hue="sex",
               split=True, inner="quart",
               palette={"female": "y", "male": "b"},
               data=tmp)
    vio.set_title("Number of Suicides Across Europe based on Age (Logarithm)",fontsize=40 )
sns.despine(left=True)
violin(world[world['continent']=='Europe'], year=2014, log=True)


# ### We see that most suicides come from the age group 35-54 years, for both sexes. Thankfully, the number of young suicides in Europe is quite low. 
# 
# ### The asymmetry that we saw before between sexes can also be seen again; more males kill themselves than females. 

# ### Unfortunately, the WHO statistics are only primarily reliable for the continent of Europe. As we can see below, the number of reported suicides elsewhere is very lacking

# In[ ]:


for i in sorted(df['year'].unique()):
    tmp = filter_year(world, year=i)
    tmp = groupings(tmp)
    tmp = world_statistics(tmp)

    max_ = np.max(tmp['total_suicides_continent'])
    continent = tmp[tmp['total_suicides_continent']==max_]['continent'].unique()
    print('The continent in year %s with the largest number of suicides is %s with %s' % (i, continent, max_))


# ### Initially it seems European suicides are through the roof compared to other continents. I am assuming, however, this is due to the fact that European suicides are much more likely to be recorded. Let's compare the supposed populations

# ### Yes, the recorded population is much bigger in Europe. A more important / accurate measure, therefore, would be percentages

# # Geopandas!
# 
# ### Geopandas is a lovely piece of kit that enables you to visualise maps easily. It's a shame the dataset doesn't come with data on cities.

# In[ ]:


filter_and_plot(world, year=2014,country=None, continent='South America', column = 'total_suicides_country')


# In[ ]:


filter_and_plot(world, year=2014,country=None, continent='Europe', column = 'total_suicides_country')

