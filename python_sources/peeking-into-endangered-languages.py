#!/usr/bin/env python
# coding: utf-8

# Data_Exploration

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud
from nltk.corpus import stopwords
from mpl_toolkits.basemap import Basemap
import plotly
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/data.csv") #loading data


# ## Looking at the Dataset ##

# In[ ]:


data.shape


# ## So we see that the data has 2722 rows and 15 columns, to get the idea about these columns lets look at their names ##

# In[ ]:


data.columns


# Seeing the first few rows of dataset

# In[ ]:


data.head()


# ## Seems like the dataset has lots of repeatitive features eg. names in three languages and country codes, I will remove these repeatitive ones##

# In[ ]:


features = ["ID","Name in English","Countries","Degree of endangerment","Alternate names","Name in the language","Number of speakers","Sources","Latitude","Longitude","Description of the location"]


# In[ ]:


data = data[features] #pandas command for selecting features


# checking for the missing values

# In[ ]:


data.isnull().sum()


# **as we see that Alternate names and Name in the language have lots of missing values, so deleting them from our data**

# In[ ]:


data = data.drop(["Name in the language"],axis = 1)
data = data.drop(["Alternate names"],axis = 1)


# In[ ]:


data[data.Countries.isnull() == True] #cheking for that one language which has missing country


# In[ ]:


data.Countries = data.Countries.fillna("nocountry") #assigning the missing country as "nocountry"


# ## Now analysing the data stepwise, making the word cloud of the countries where these endangered languages are spoken to give us the idea about the region wise endangerment of languages ##

# In[ ]:


text = ''
np.array(data.Countries.values)[1].split(",")[0]
for i in np.arange(len(np.array(data.Countries.values))):
    val = np.array(data.Countries.values)[i].split(",")
    for j in np.arange(len(val)):
        text = " ".join([text,val[j]])
        
text = text.strip()


# In[ ]:


wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40).generate(text)
wordcloud.recolor(random_state=312)
plt.imshow(wordcloud)
plt.title("Wordcloud for countries ")
plt.axis("off")
plt.show()


# **So, we can see that United States(USA), Russian Federation, Brazil and India are coutries where you can see most endangered languages being spoken
# It could mean a couple of things**
# 
# 
#  1. Maybe these coutries have been able to preserve these languages upto a long time and we could stil work out with its' preserving
#  2. Or, it could mean that these countries have less compatibility/tolerance for these endangered languages
# 
# 

# **Moving forward, let us see how many of the languages are of different degrees of endangerment**

# In[ ]:


plt.xticks(rotation = -45)
sns.countplot(x="Degree of endangerment", data=data, palette="Greens_d")


# **Now let us try to find the languages which are most spreaded ones versus the least spreaded ones**

# In[ ]:


language_spread = pd.DataFrame(data["Name in English"].values, columns=["Name in English"])
language_spread["num_countries"] = 0
language_spread["ID"] = data.ID
language_spread["Countries"] = data.Countries


# In[ ]:


language_spread.shape


# Below I have used the number of "comas" + 1 as the number of countries a particular language is being spoken in. Hence if the comma count is more then we have more number of countries in which that endangered language is being spoken

# In[ ]:


language_spread.num_countries = data.Countries.apply(lambda x: len(re.findall(r",",x)))
language_spread.num_countries = language_spread.num_countries + 1
language_spread = language_spread.sort_values(by = "num_countries",ascending=False)


# **So, one of the most spread languages are!!**

# In[ ]:


language_spread["Name in English"].head()


# In[ ]:


sns.distplot(language_spread.num_countries)


# **After seeing the distribution of the languages, we could see that most of the endangered languages 
# are from 1-3 countries, so we will take top 40 languages and will see are these most spreaded 
# endangered languages region specific.**

# In[ ]:


text = ''

for i in np.arange(len(np.array(data.Countries.values))):
    val = np.array(language_spread.Countries.values)[i].split(",")
    for j in np.arange(len(val)):
        text = " ".join([text,val[j]])
text = text.strip()


# **Making the word cloud**

# In[ ]:


wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40).generate(text)
wordcloud.recolor(random_state=312)
plt.imshow(wordcloud)
plt.title("Wordcloud for top 40 languages ")
plt.axis("off")
plt.show()


# **We got approximately 10 countries in which the most spreaded languages are there. meaning that if you visit the countries Brazil, America, China, Russia and India, you are more likely to find a similar kind of endangered language :)**

# In[ ]:


data.Latitude = data.Latitude.fillna(0)
data.Longitude = data.Longitude.fillna(0)


# **Now looking at the different on the basis of degree of endangerment of languages**

# ## Looking at Vulnerable ##

# In[ ]:


plt.figure(figsize=(12,6))

m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')
m.drawcoastlines()
m.drawcountries()
en_lon = data[data["Degree of endangerment"] == "Vulnerable"]["Longitude"].astype(float)
en_lat = data[data["Degree of endangerment"] == "Vulnerable"]["Latitude"].astype(float)
x, y = m(list(en_lon), list(en_lat))
m.plot(x, y, 'go', markersize = 6, alpha = 0.7, color = "blue")

plt.title('Vulnerable languages')
plt.show()


# ## Looking at Definitely endangered ##

# In[ ]:


plt.figure(figsize=(12,6))

m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')
m.drawcoastlines()
m.drawcountries()
en_lon = data[data["Degree of endangerment"] == "Definitely endangered"]["Longitude"].astype(float)
en_lat = data[data["Degree of endangerment"] == "Definitely endangered"]["Latitude"].astype(float)
x, y = m(list(en_lon), list(en_lat))
m.plot(x, y, 'go', markersize = 6, alpha = 0.7, color = "blue")

plt.title('Vulnerable languages')
plt.show()


# ## Looking at Severely endangered ##

# In[ ]:


plt.figure(figsize=(12,6))

m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')
m.drawcoastlines()
m.drawcountries()
en_lon = data[data["Degree of endangerment"] == "Severely endangered"]["Longitude"].astype(float)
en_lat = data[data["Degree of endangerment"] == "Severely endangered"]["Latitude"].astype(float)
x, y = m(list(en_lon), list(en_lat))
m.plot(x, y, 'go', markersize = 6, alpha = 0.7, color = "blue")

plt.title('Vulnerable languages')
plt.show()


# ## Looking at Critically endangered ##

# In[ ]:


plt.figure(figsize=(12,6))

m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')
m.drawcoastlines()
m.drawcountries()
en_lon = data[data["Degree of endangerment"] == "Critically endangered"]["Longitude"].astype(float)
en_lat = data[data["Degree of endangerment"] == "Critically endangered"]["Latitude"].astype(float)
x, y = m(list(en_lon), list(en_lat))
m.plot(x, y, 'go', markersize = 6, alpha = 0.7, color = "blue")

plt.title('Vulnerable languages')
plt.show()


# ## Extinct ##

# In[ ]:


plt.figure(figsize=(12,6))

m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')
m.drawcoastlines()
m.drawcountries()
en_lon = data[data["Degree of endangerment"] == "Extinct"]["Longitude"].astype(float)
en_lat = data[data["Degree of endangerment"] == "Extinct"]["Latitude"].astype(float)
x, y = m(list(en_lon), list(en_lat))
m.plot(x, y, 'go', markersize = 6, alpha = 0.7, color = "blue")

plt.title('Vulnerable languages')
plt.show()

