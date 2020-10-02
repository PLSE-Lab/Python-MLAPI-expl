#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import libraries
import pandas as pd
import numpy as np


# ## NB: Can only download one or two blocks of 30min data on my system at a time

# In[ ]:


#Import block & house info data, rename columns
files = ['block_0.csv']
block_0 = pd.concat([pd.read_csv("../input/archived-smartmeter-data/" + f, dtype=str) for f in files])
house_info = pd.read_csv('../input/archived-smartmeter-data/informations_households.csv')
block_0.columns = ['LCLid','DateTime','KWh']


# First block (Block_0): 39786 30min recordings of energy consumption in KWh across 390 homes

# # 1. Data Clean&Merge
# ## 1.1 KWh Nulls
# There are 'Null' values in the KWh column, there are 390 of them in Block_0 and as such appears assigned one null value per household. The time of the null value is in-between each 30min recording and is constant over groupings of houses. Suspect this was the smart meters resetting. Do not know if this invalidates a recording before or after. Quick check below shows that it appears not to have impacted recordings, however we should keep this in mind. Soln to the Null recordings is to delete all records with a Null in the KWh column (390 records in total) 

# In[ ]:


block_0 = block_0[block_0.KWh != 'Null']
block_0['KWh'] = block_0['KWh'].astype(float)


# ## 1.2 DateTime reformat
# Reformatting the DateTime column to make it easier to manage

# In[ ]:


DateTime = block_0.iloc[:, 1].values
Year = np.zeros(len(DateTime), dtype=int)
Month = np.zeros(len(DateTime), dtype=int)
Day = np.zeros(len(DateTime), dtype=int)
Hour = np.zeros(len(DateTime), dtype=float)
for i in range(0,len(DateTime)):
    Year[i] = (DateTime[i][0:4])
    Month[i] = (DateTime[i][5:7])
    Day[i] = (DateTime[i][8:10])
    if DateTime[i][14:16]=='00':
        Hour[i] = DateTime[i][11:13]
    else: 
        Hour[i] = DateTime[i][11:13]
        Hour[i] += 0.5
        
block_0.loc[:,'Day']=Day
block_0.loc[:,'Month']=Month
block_0.loc[:,'Year']=Year
block_0.loc[:,'Hour']=Hour
block_0 = block_0.drop('DateTime',1)


# ## 1.3 Check for Acorn labelling errors 
# There is one error type: 'ACORN-' instead of 'ACORN-*' which occur for two houses, therefore delete these from analysis

# In[ ]:


#LCLids where a miss catagorisation has occured i.e. when Acorn == ACORN- in this insatnce
LCLids_del = house_info.loc[house_info.Acorn == 'ACORN-'].LCLid.values
LCLids_del


# In[ ]:


#Delete these houses from the current block being analysed
for i in range (0,len(LCLids_del)): 
    if any(block_0.LCLid == LCLids_del[i]):
        block_0 = block_0.loc [block_0.LCLid != LCLids_del[i]]
        print(i)


# ## 1.4 Merging data
# Many to one merge, remove 'ACORN-' string in acorn classification 

# In[ ]:


block_0_info = pd.merge(block_0,house_info)
block_0_info = block_0_info.drop('Acorn_grouped',1)
block_0_info = block_0_info.drop('file',1)
block_0_info.Acorn = block_0_info.Acorn.str.replace('ACORN-',"")


# ## 1.4 What data do we have? 
# Apears that only 2012 and 2013 are complete years

# In[ ]:


test_1 = block_0_info.groupby(['Acorn','Year','Month']).mean().drop(['Day','Hour'],1).reset_index()
Years = test_1.Year.unique()
Months = test_1.Month.unique()
for yr in Years: print(yr,':\n',(test_1.loc[test_1.Year == yr]).count())


# In[ ]:


#keep 2012 and 2013, reset index with drop=False
block_0_info_2012_2013 = block_0_info.loc[(block_0_info.Year == 2012) | (block_0_info.Year == 2013)].reset_index(drop=False)


# ## 1.4 Include a weekend flag and acorn description for analysis purposes
# **THINK I COULD HAVE DONE THIS EASIER BY JUST USING ORIGINAL DATETIME COLUMN IN TABLEAU**

# In[ ]:


import datetime


DayOfWeek = []
DayOfYear = []
for i in range(0,len(block_0_info_2012_2013)):
    DayOfWeek.append(datetime.datetime(block_0_info_2012_2013.Year[i], 
                                       block_0_info_2012_2013.Month[i], 
                                       block_0_info_2012_2013.Day[i]).weekday())
    DayOfYear.append(datetime.date(block_0_info_2012_2013.Year[i], 
                                       block_0_info_2012_2013.Month[i], 
                                       block_0_info_2012_2013.Day[i]).timetuple()[7])
#put in new column
block_0_info_2012_2013['DayOfWeek'] = DayOfWeek
block_0_info_2012_2013['DayOfYear'] = DayOfYear

#create weekend flag, days 5 and 6 are Sat and Sun
block_0_info_2012_2013['Weekend'] = ""
tmp = block_0_info_2012_2013[(block_0_info_2012_2013.DayOfWeek == 5) | (block_0_info_2012_2013.DayOfWeek == 6)].index
block_0_info_2012_2013['Weekend'][tmp] = 1
tmp = block_0_info_2012_2013[block_0_info_2012_2013.Weekend != 1] .index
block_0_info_2012_2013['Weekend'][tmp] = 0


# In[ ]:


block_0_info_2012_2013['AcornDescription'] = ""
AcornName = pd.DataFrame({'Acorn':['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','U'],
                           'AcornName':['Lavish Lifestyles','Executive Wealth','Mature Money',
                                        'City Sophisticates','Career Climbers','Countryside Communities',
                                        'Successful Suburbs','Steady Neighbourhoods','Comfortable Seniors',
                                        'Starting Out','Student Life','Modest Means','Striving Families',
                                        'Poorer Pensioners','Young Hardship','Struggling Estates',
                                        ' Difficult Circumstances','Unknown']})
AcornDesc = pd.read_csv('../input/archived-smartmeter-data/Acorn Descriptions v2.csv')
AcornDesc = pd.merge(AcornName,AcornDesc)
block_0_info_2012_2013_wkend_desc = pd.merge(block_0_info_2012_2013,AcornDesc)


# ## 1.5 Extract to txt file for analysis in Tableau

# In[ ]:


#block_0_info_2012_2013_wkend_desc.to_csv('../results/SmMe block_0_2012_2013_weekend_desc.txt')


# # 2. Analysis in Tableau
# Below are two Dashboards created based on the above data in Tableau
# As there are many ways in which the data can be viewed, for example by Acorn class, household,  day, weekend, month etc. a Tableau dashboard is convenient to do some exploration of the data.
# I have provided the HTML and link to the dashboards (please open fullscreen), some conclusions based on what I have seen from the dashboards has been given below.

# ## 2.1 Overview Dashboard
# Displayed in the horizontal plot on the left is the average energy consumed broken down by acorn classification & the week or weekend flag. Displayed in the horizontal plot on the right is the average energy consumed per hour for each acorn classification (the size of the circles are proportional to the energy consumption). All filters on the right apply to all the figures.

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1512990824106' style='position: relative'><noscript><a href='#'><img alt='Overview ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Sm&#47;SmartMeters_1&#47;Overview&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='SmartMeters_1&#47;Overview' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Sm&#47;SmartMeters_1&#47;Overview&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1512990824106');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# link: https://public.tableau.com/views/SmartMeters_1/Overview?:embed=y&:display_count=yes&publish=yes

# ## 2.2 Uncertainty in the mean energy consumption within each Acorn class
# Displayed in this dashboard are the mean and standard deviation of the energy consumption over time for a particular month, year and weekend flag. The left figure shows the breakdown by acorn classification. The figure on the right shows the breakdown by household. The filters in the middle of the pane apply to the left figure and the filters on the right of the pane apply to the right figure.

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1512991679641' style='position: relative'><noscript><a href='#'><img alt='Uncertainty ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Sm&#47;SmartMeters_1_2&#47;Uncertainty&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='SmartMeters_1_2&#47;Uncertainty' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Sm&#47;SmartMeters_1_2&#47;Uncertainty&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1512991679641');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# link:
# https://public.tableau.com/views/SmartMeters_1_2/Uncertainty?:embed=y&:display_count=yes&publish=yes

# ## 2.3 Hourly ave. energy consumption as a function of day of year
# Displayed in this dashboard is a nice view of mean of the energy consumption for a particular time of day and for a particular day. Filters are available for Acorn classifcation and House ID. The image displays nicely changes in patterns due to variations in seasons and time of day.

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1512991980407' style='position: relative'><noscript><a href='#'><img alt='TimeOfDay and DayOfYear ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Sm&#47;SmartMeters_1_3&#47;TimeOfDayandDayOfYear&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='SmartMeters_1_3&#47;TimeOfDayandDayOfYear' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Sm&#47;SmartMeters_1_3&#47;TimeOfDayandDayOfYear&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1512991980407');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# Link:
# https://public.tableau.com/views/SmartMeters_1_3/TimeOfDayandDayOfYear?:embed=y&:display_count=yes&publish=yes

# ## 2.4 Some conclusions:
# Some of the things I saw when building and playing around with this data
# * The mean energy consumption does not align perfectly with the acorn classification order (A -> Q) i.e. it does not align with the premise that A consumes more than B, B consumes more than C etc. The premise is true from A to F, however after F (countryside communities), energy consumption no longer correlates with acorn class. O (Young Hardship) consumes the least and M (Striving Families) consume an amount similar to D (City Sophisticates).
# 
# * During the week almost all acorn households show clear morning and evening peaks in energy consumption, reflecting the expected classical electricity demand curve. For M (Striving Families) the difference between average and peak demand is exaggerated especially in the morning. 
# 
# * Night time use of energy is very low except for some households (MAC000003 for example) in acorn class P (Struggling Estates) during winter months. Although according to the data these households still have standard use tariffs one would assume they must be using electrical storage heaters. If standard they are using standard tariffs and storage heaters this would have been very expensive.
# 
# * The variance of energy use is very large even within a particular acorn class, month and weekend flag. This indicates the large amount of individual behavior occurring within acorn classifications even on comparable days
# 

# ## 2.5 Some interesting questions:
# * For a given month can we cluster the electricity demand patterns of these households into distinct groupings and do those groupings align with acorn classifications?
# * ... is there a difference in energy consumption behaviour between different tariff types within a certain acorn classification and month?
# 

# ## 2.6 Plan
# * Run unsupervised learning algorithms on data... 
#     * For this make use of the preprepared daily_dataset.zip folder created by Jean-Michel D. on Kaggle.
#     * This data contains summary stats of each day for each household. Summary stats inc. mean, std.dev, max, min, sum...

# # 3. Daily Dataset
# ## 3.1 Importing data, cleaning and exporting 

# In[ ]:


#import data, files from 0 to 111 blocks
MAX_FILES = 112
files = [('block_'+str(i)+'.csv') for i in range (0,MAX_FILES)]

block_Daily = pd.concat([pd.read_csv("../input/smart-meters-in-london/daily_dataset/daily_dataset/" + f, dtype={'LCLid': np.str, 'day': np.str, 
                                                                          'energy_median': np.float64,
                                                                          'energy_mean': np.float64, 'energy_max': np.float64,
                                                                          'energy_count': np.float64, 'energy_std': np.float64,
                                                                          'energy_sum': np.float64,'energy_min': np.float64}) \
                         for f in files])

house_info = pd.read_csv('../input/archived-smartmeter-data/informations_households.csv')


# In[ ]:


print('Percentage of days when no. of energy readings was less than 47 is: ',
      round(100*len(block_Daily[block_Daily.energy_count < 47])/len(block_Daily),2)
     ,'%')


# Remove these days to avoid invalid summary statistics where maximums and minimums etc. have been missed by the recording equipment. From now on we can be confident that the summary stats per day apply for recordings where 47 out of 48 potetnial recordings each day have been captured

# In[ ]:


LCLids_del = house_info.loc[house_info.Acorn == 'ACORN-'].LCLid.values
LCLids_del


# In[ ]:


#Delete these houses from the current block being analysed
for i in range (0,len(LCLids_del)): 
    if any(block_Daily.LCLid == LCLids_del[i]):
        block_Daily = block_Daily.loc [block_Daily.LCLid != LCLids_del[i]]
        print(i)


# In[ ]:


block_Daily.drop(block_Daily.index[block_Daily.energy_count < 47], axis=0)

block_Daily_info = pd.merge(block_Daily,house_info)
block_Daily_info = block_Daily_info.drop(['Acorn_grouped','file'],1)
block_Daily_info.Acorn = block_Daily_info.Acorn.str.replace('ACORN-',"")
block_Daily_info.Acorn.unique()


# In[ ]:


block_Daily_info.head()


# Extract this data for cluster analysis in Tableau ... 

# In[ ]:


#block_Daily_info.to_csv('../results/SmMe block_Daily_info.txt')


# ## 3.2 Standard (Std) and Time of Use (ToU) tariff behaviour
# Dashboard below shows the average energy consumption on a daily basis aggregated over acorn classifications and months. As observed in most months and acorn catagories ToU tariff users consume less energy. Acorn catagories which are not consumming less energy appear to be F, O, N and Q

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1512991411868' style='position: relative'><noscript><a href='#'><img alt='Std or ToU ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Sm&#47;SmartMeters_2&#47;StdorToU&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='SmartMeters_2&#47;StdorToU' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Sm&#47;SmartMeters_2&#47;StdorToU&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1512991411868');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# link: 
# https://public.tableau.com/views/SmartMeters_2/StdorToU?:embed=y&:display_count=yes

# ## 3.3 Cluster analysis based on daily summary stats
# k-means++, with Calinski-Harabasz criterion to assess the number of clusters, was performed on the daily data for each house. Data included: Ave.Energy_sum, Ave.Energy_mean, Ave.Energy_max, Ave.Energy_min and Ave.Energy_median. The analysis returned 3 clusters which are not very well correlated with acorn calssification, highlighting the large amount of variability in the data across households. The 'shape' of the energy consumption curve may provide an additional insight for the cluster analysis and show return more clusters.

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1512991498592' style='position: relative'><noscript><a href='#'><img alt='Clustering DayData ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Sm&#47;SmartMeters_2_2&#47;ClusteringDayData&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='SmartMeters_2_2&#47;ClusteringDayData' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Sm&#47;SmartMeters_2_2&#47;ClusteringDayData&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1512991498592');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# link:
# https://public.tableau.com/views/SmartMeters_2_2/ClusteringDayData?:embed=y&:display_count=yes&publish=yes

# *I would like to do more analysis on this dataset. I struggled with regard to running speed and memory when trying to download and analyse the 30min energy consumption data across all households. There may be a way around the memory issue. I think if we could get a sumarised view of the hourly usage behaviour for every house then then some additional insights would be avaiable re. the shape of the energy consumption curve and how that may characterise various behaviours across acron classifications.*

# In[ ]:




