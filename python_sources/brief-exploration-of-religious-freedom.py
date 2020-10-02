#!/usr/bin/env python
# coding: utf-8

# ### A Brief Examination of World Religious Freedom 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Import Additional Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from IPython.core.display import HTML 
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Read in data and view the first 5 rows
hfi_data = pd.read_csv('../input/hfi_cc_2018.csv')
hfi_data = hfi_data[hfi_data['ISO_code'] != 'BRN'] # We drop this country as it does not contain pf_religion data
hfi_data.head()


# In[ ]:


# Define function for plotting yearly median for given metrics with std_dev error bars
def plot_med_year(df, metric, title=None):
    df_year = df.groupby('year')
    df_med = pd.merge(df_year[metric].median().to_frame().reset_index(), 
                  df_year[metric].std().to_frame(name='std').reset_index())
    plt.figure()
    plt.errorbar(x=df_med['year'], y=df_med[metric], 
                 yerr=df_med['std'], linestyle='None', marker='s')
    if title:
        plt.title(title)
    else:
        plt.title('Median {} by Year'.format(metric))


# #### Overall Freedom
# Here, we will view the overall changes in average world freedom for the time period contained in the datset (2008-2016). In particular, we will view the trends in overall human freedom and in personal freedom. While the human freedom score is made up of an average of personal freedom and economic freedom, we will not be analyzing economic freedom as our interest is primairly in religious freedom (a sub-category of personal freedom).

# In[ ]:


# Plot World Average hfi by year
plot_med_year(hfi_data, 'hf_score', 'World Median HF Score by Year')
# Plot World Average personal freedom by year
plot_med_year(hfi_data, 'pf_score', 'World Median PF Score by Year')


# The above graphs show a downward trend in both personal and overall human freedom from 2008 to 2016. Let's break these scores down further by world region. 

# In[ ]:


# Explore yearly regional metrics in human freedom, personal freedom, and religious freedom
# Other religion related metrics are excluded as many have missing data
region_yr = hfi_data.groupby(['region', 'year']).median().reset_index()

# Plot human freedom by region
plt.figure()
sns.relplot(x='year', y='hf_score', hue='region', data=region_yr)
plt.title('Median Yearly hf_score by Region')

#Plot personal freeodm by region
plt.figure()
sns.relplot(x='year', y='pf_score', hue='region', data=region_yr)
plt.title('Median Yearly pf_score by Region')


# Not surprisingly, North America and Western Europe consistently exhibit the highest scores for personal and overall human freedom. What is more surprising from these visualizations is the large spread in human and personal freedom between the five highest reigons and the five lowest regions in the world. These groups are spread by nearly a full point or more in both the personal and human freedom scores. Let's dive deeper into the sub-category of interest here - religious freedom, and see if similar trends exist in religious freedom. 

# #### Religious Freedom

# In[ ]:


# Plot World Median religious freedom score by year
plot_med_year(hfi_data, 'pf_religion', 'World Med Religious Freedom Score by Year')


# Religious freedom exhibits an overall downard trend over the 2008 to 2016 period, similar to those of personal and human freedom, albeit with more fluctuation. Next, we will explore the regional breakdown of religious freedom over this period. 

# In[ ]:


#Plot religious freedom by region
plt.figure()
sns.relplot(x='year', y='pf_religion', hue='region', data=region_yr)
plt.title('Median Yearly pf_religion Score by Region')


# From this regional breakdown, it is clear that there has been more significant variance in religious freedom for each region over 2008-2016 than there has been for overall personal or human freedom. Additionally, while Western Europe has generally ranked very highly in personal and human freedom, it ranks in the middle for religious freedom. Let's look further into the overall and regional medians for government religious restrictions and religious harassment. 

# In[ ]:


#Plot religious freeodm - restrictions
plot_med_year(hfi_data, 'pf_religion_restrictions', 'World Med Religious Freedom Restrictions Score by Year')
#Plot religious freeodm - harassment 
plot_med_year(hfi_data, 'pf_religion_harassment', 'World Med Religious Freedom Harassment Score by Year')
#Plot religious freeodm - restrictions by region
plt.figure()
sns.relplot(x='year', y='pf_religion_restrictions', hue='region', data=region_yr)
plt.title('Median Yearly pf_religion_restrictions Score by Region')
#Plot religious freeodm - harassment by region
plt.figure()
sns.relplot(x='year', y='pf_religion_harassment', hue='region', data=region_yr)
plt.title('Median Yearly pf_religion_harassment Score by Region')


# Western Europe is again low in both sub-categories of religious freedom. There is also an interesting and somewhat surprising downward trend for North America, particularly from 2014-2016. To get a better understanding of how these metrics relate to each other and to overall religious, personal, and human freedom, we will view the correlation matrix.  

# In[ ]:


# First, we will create a data frame containing only the metrics of interest
religion = hfi_data[['countries', 'region', 'year', 'pf_religion_harassment','pf_religion_restrictions', 
                     'pf_religion', 'pf_score', 'hf_score']]


# In[ ]:


# Next, we create a heat map of the correlation matrix
plt.figure()
sns.heatmap(religion.drop(columns='year').corr(), annot=True)


# Not surprisingly given the trends observed above, around the world there is a high degree of correlation between government restrictions on religious freedom and the presence of religious harassment. Additionally, there is a somewhat strong correlation (.49) between pf_religion and overall pf_score. Somewhat surprisingly however, the correlation between pf_religion and overall hf_score is only .39. Thus, it exhbits a somewhat strong but not overwhelmingly strong correlation. Let's view these metrics another way to get a better sense for how they relate.    

# In[ ]:


# Create a scatter matrix to view correlation at a more granular level
plt.figure()
sns.pairplot(religion.drop(columns='year'), hue='region')


# The above scatter matrix allows us to visualize the correlation between each of the key metrics in a more granular way. This allows us to see what the correlaiton matrix already told us, that pf_religion is only mildly correlated with hf_score, but that govrnment restrictions of religion and religious harassment are more strongly correlated. 

# Next, we will examine religious freedom at an even more granular level, and look at the world's 5 best and 5 worst countries in terms of religious freedom for 2015 and 2016.

# In[ ]:


# Create new df for these years only
years = [2015, 2016]
religion_15_16 = religion[religion['year'].isin(years)]


# In[ ]:


# Bottom 5 - 2015
religion_15_16[religion_15_16['year'] == 2015].nsmallest(n=5, columns='pf_religion')


# From here, we see that the five lowest countries for pf_religion also have very low scores for religious restrictions, but moderate scores for religious harassment. Let's see if anything changed from 2015 to 2016. 

# In[ ]:


# Bottom 5 - 2016
religion_15_16[religion_15_16['year'] == 2016].nsmallest(n=5, columns='pf_religion')


# Interestingly, Iran is no longer in the bottom 5 for religious freedom in 2016, but Malaysia now is. The other four countries remain the same, albeit in a different order. Additionally, the scores in government restrictions in the bottom 5 all decreased from 2015-2016. This is not surprising given the overall world downward trend in restrictions scores from 2015 to 2016. Let's see if there is a similar trend amongst the 5 best countries in the world with regards to religious freedom.  

# In[ ]:


# Top 5 - 2015
religion_15_16[religion_15_16['year'] == 2015].nlargest(n=5, columns='pf_religion')


# In[ ]:


# Top 5 - 2016
religion_15_16[religion_15_16['year'] == 2016].nlargest(n=5, columns='pf_religion')


# Unlike the lowest countries for religious freedom, there is a higher degree of variance amongst the countries included in the top 5. Additionally, the countries in the top 5 do not seem to exhibit the same downward trend in government restrictions that the world in general, and the bottom 5 in particular, exhibit. Of course, this is not a perfect comparison given the changes in the countries included in the top 5, but for our brief analysis, it is still helpful. Lastly, it is interesting to note that there is slightly more regional variation amongst the top 5 countries than amongst the bottom 5. 

# #### Conclusion
# 
# Thus, we have briefly answered the basic questions we set out to answer at the beginning of this analysis. While we have not gone too in depth, the analysis has shown us that there is a correlation between high levels of government restrictions on religion and religious harassment. Additionally, one somewhat surprising finding was the downward trend in government religious restrictions scores in Western Europe. We will end this analysis with three geographic visualizations which show how religious freedom scores are distributed around the globe.

# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1548565321797' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;hf&#47;hfi_analysis_v1&#47;RelFreedom&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='hfi_analysis_v1&#47;RelFreedom' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;hf&#47;hfi_analysis_v1&#47;RelFreedom&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1548565321797');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1548565707738' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;hf&#47;hfi_analysis_v1&#47;RelRestrictions&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='hfi_analysis_v1&#47;RelRestrictions' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;hf&#47;hfi_analysis_v1&#47;RelRestrictions&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1548565707738');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1548565668582' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;hf&#47;hfi_analysis_v1&#47;RelHarassment&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='hfi_analysis_v1&#47;RelHarassment' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;hf&#47;hfi_analysis_v1&#47;RelHarassment&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1548565668582');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:




