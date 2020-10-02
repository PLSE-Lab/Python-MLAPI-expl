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


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv('../input/us-accidents/US_Accidents_Dec19.csv')

# Displaying first few records of the data
data.head()


# # Accident Count By State and Severity
# The data is now grouped by 'State' and 'Severity'. This grouped data shows number of accidents that occured in each state and for each Severity.

# In[ ]:


count_bystates = data.groupby(['State','Severity']).size().unstack(fill_value=0)

# Displaying sample of the grouped data
count_bystates.head()


# ## Bar graph Showing Counts of Accidents that Occured State-Wise by Severity

# In[ ]:


fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
count_bystates.plot.bar(ax=ax)

#Adding formatting elements to the graph
ax.set_xlabel('State',fontsize=15,color='red')
ax.set_ylabel('Counts',fontsize=15,color='red')
ax.set_title('State-Wise Counts for Accidents for Each Severity',fontdict={'fontsize': 25, 'fontweight' : 'bold', 'verticalalignment': 'center', 'horizontalalignment': 'center'})
ax.legend(['Small Incident (Caused Short Delay)','Severe','Very Severe','Most Severe (Caused Long Delay)'])
plt.show()

# Figure saved to graphs directory
fig.savefig('State-WiseCounts.png')


# # Accidents based on Weather Conditions
# Another key piece of information from the dataset is the weather conditions that could have been the reasons for the accident.
# 
# Extracting weather data and dropping empty records

# In[ ]:


weatherdata = data['Weather_Condition'].dropna()

# Displaying sample weather information collected
weatherdata.head()


# Some of the text in the Weather_Conditions column are seperated by spaces. While processing the text, this could lead to some confusion. So we are going to remove the spaces between the text.

# In[ ]:


def processText(text):
    try:
        textlist = text.split('/')
        for i in range(len(textlist)):
            textlist[i]=textlist[i].split(' ')
            if '' in textlist[i]:
                textlist[i].remove('')
            textlist[i] = ''.join(textlist[i])
        text = '/'.join(textlist)
        return text
    except Exception as e:
        print(text)
        print(e)
        
weatherdata = weatherdata.apply(processText)

# Showing that the spaces were removed from the Weather Condition Text
# Displaying changed text values
weatherdata.head()


# # Bar Graph showing Top 10 Conditions for Accidents
# Extracting Counts for each Weather Condition and then finding the 10 reasons with highest count.

# In[ ]:


weather = weatherdata.groupby(weatherdata).size().reset_index(name='Count').set_index('Weather_Condition').sort_values(by='Count',ascending=False)
top_10_conditions = weather.head(n=10)

# Displaying Valses
top_10_conditions


# Creating Bar Chart to show the table data

# In[ ]:


fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
top_10_conditions.plot.bar(ax=ax)

ax.set_xlabel('Top Weather Conditions',fontsize=15,color='red')
ax.set_ylabel('Count of Accidents',fontsize=15,color='red')
ax.set_title('Top 10 Weather Conditions in which Most Accidents Occured',fontdict={'fontsize': 25, 'fontweight' : 'bold', 'verticalalignment': 'center', 'horizontalalignment': 'center'})
ax.legend().remove()
plt.tight_layout()
plt.show()

fig.savefig('Top10Reasons.png')


# ## WordCloud to show Most Frequent Condition for accidents
# Importing wordcloud library to constrct and plot wordcloud.

# In[ ]:


from wordcloud import WordCloud


# ### All Text WordCloud
# Gathering all text as a single string to construct the word cloud.

# In[ ]:


text = ' '.join(weatherdata.values.tolist())
wordcloud = WordCloud(background_color="white").generate(text)


# Generating wordcloud with the most frequent appearing in big letters and less frequent in ones small.

# In[ ]:


fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(111)
ax.set_title('Wordcloud using all Text Weather Conditions',fontdict={'fontsize': 25, 'fontweight' : 'bold', 'verticalalignment': 'center', 'horizontalalignment': 'center'})
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

fig.savefig('WordCloudAllText.png')


# ### Text Frequencies WordCloud
# Grouping Weather Conditions and counts to construct the word cloud using frequencies.

# In[ ]:


text_with_frequencies = dict(zip(weather.index.values.tolist(),weather.Count.tolist()))
wordcloud = WordCloud().generate_from_frequencies(text_with_frequencies,max_font_size=68)


# Generating wordcloud text frequecies.

# In[ ]:


fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(111)
ax.set_title('Wordcloud using Weather Condition Frequencies',fontdict={'fontsize': 25, 'fontweight' : 'bold', 'verticalalignment': 'center', 'horizontalalignment': 'center'})
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

fig.savefig('WordClouTextFrequencies.png')

