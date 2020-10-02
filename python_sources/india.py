#!/usr/bin/env python
# coding: utf-8

# Purpose of this Kernel is to teach how to do data visualization using packages like matplotlib,seaborn,wordcloud,pandas etc.Here I will be plotting data about India which will give an overview about the country.This is a work in process and if you like my work please do vote. 

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


# ### Representing Preamble of Indian Constitution on a Word Cloud
# Here we will see how to plot a word cloud by analysing the text provided in a csv file.This will give us an idea about the keys words in the preamble of Indian constitution

# In[ ]:


import numpy as np # linear algebra
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

mpl.rcParams['figure.figsize']=(10.0,7.0)    #(6.0,4.0)
mpl.rcParams['font.size']=12                #10 
mpl.rcParams['savefig.dpi']=100             #72 
mpl.rcParams['figure.subplot.bottom']=.1 


stopwords = set(STOPWORDS)
data = pd.read_csv("../input/datacsv1/data_1.csv")
wordcloud = WordCloud(
                          background_color='Greens',
                          stopwords=stopwords,
                          max_words=1000,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(data['Text']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)
plt.ioff()


# In[ ]:


import pandas as pd
import numpy as np
data1 = pd.read_csv("../input/datacsv/data.csv")
from PIL import Image
from scipy import ndimage
mask = np.array(Image.open('../input/pictureind/india-map-img.png'))
txt = " ".join(data1['Text'].dropna())
wc = WordCloud( max_words=2000,  colormap='brg', max_font_size=20, 
                          random_state=42, background_color='White').generate(txt)
plt.figure(figsize=(16,18))
plt.imshow(wc)
plt.axis('off')
plt.title('');


# The constitution of India is like a holy book for all Indians.The functioning of the country is based on the articles in the constitution.As per the constitution, India is **SOVEREGIN SCOCIALIST SECULAR DEMOCRATIC REPUBLIC** with an objective of providing **JUSTICE EQUALITY LIBERTY FRATERNITY** to its citizens.For understanding the preamble of Indian constitution you can go through my blog from the link below http://btplife.blogspot.com/2018/09/india-become-independent-on-15-aug-1947.html

# ### Population growth of India with Line Plot
# Out aim is to plot the Population growth on India and provide label,title,legend,grid,linestyle,color,linewidth and color to the plot.

# In[ ]:


# Importing matplotlib python module
import matplotlib.pyplot as plt
# Year of census
Year=[1871,1881,1891,1901,1911,1921,1931,1941,1951,1961,1971,1981,1991,2001,2011,2018]
#Population data in crores 1 crore=10 million
Population=[23.8,25.3,28.7,29.3,31.5,31.8,35.2,38.8,36.1,43.9,54.8,68.3,84.6,102.8,121,132.2]
plt.plot(Year,Population,color='r',marker='o',linestyle='--',linewidth=2)
plt.xlabel('Year')
plt.ylabel('Population in Crores')
plt.title('Population Growth of India')
plt.grid()
plt.ioff()


# From the graph we can see that the population of India has been rapidly increasing in last 140 years.Only the decade 1940-50 shows a decline in population growth.This was because India got Independence in year 1947 and Pakistan separated out from India.After Independence population growth has been rapid this is partially due to low literacy and awareness.Also post independence health care has improved resulting in lesser child mortality rates and average life span has increased.

# ### Population density of India importing an image 

# In[ ]:


from PIL import Image
img=np.array(Image.open('../input/densityind/Density.jpg'))
fig=plt.figure(figsize=(10,10))
plt.imshow(img,interpolation='bilinear')
plt.axis('off')
plt.ioff()
plt.show()


# From the above picture it can been seen that India has high population density in the Northern Gangetic plans and the costal regions.The States West Bengal,Bihar,Uttar Pradesh,Kerala have high population density compared to other states.The river Ganga flows through plains of Uttar pradesh,Bihar and West Bengal.Fertile soil has lead to the population growth.Indias civilization has its origion in the Indus Valley which is near to todays Pakistan,Punjab and Rajastan.Around 5000 years back Saraswati river river flowed through states of Rajastan and Gujrat.After Saraswati river vanished and large areas of Rajastan and Gujrat now have become deserts.The disappering of Saraswati river caused the migration of people to Gangetic plains which have higher population density today.For more information on population density you can go through my blog http://btplife.blogspot.com/2018/02/population-density-of-kerala-data.html

# ### GDP growth of India with a bar plot

# In[ ]:


Year=[1965,1970,1980,1990,2000,2010,2016]
GDP=[5883.24,6351.72,18959.41,32660.8,47469.16,168000,226000]
plt.bar(Year,GDP,color='g',width=4,edgecolor='k',align='center')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('GDP Of India in Crore Rs')
plt.grid()
plt.ioff() 


# After independence India adopted a socialist model of economic development.Government had big control on who produced what,how much and at what price.So the growth of India economy was very slow in the period of 1947 to 1991.In the year 1991 India liberalized Indian economy under the leadership of Prime Minister N. Rao and Finance Minister Manmohan Singh.Since then Indian economy has picked up.Currently India is the 7th Biggest economy in the world behind countries like USA,China,Japan,Germany,UK,France.Very soon India will be among the top five economies of the world.

# ### Religious beliefs of Indians with a Pie Chart 

# In[ ]:


Religion=['Hindu','Muslim','Christan','Sikh','Buddhist','Jain','Others']
col=['b','g','c','r','y','m','b']
Population=[96.6,17.2,2.78,2.08,0.84,0.44,1.07] #Data is from Year 2011 census
plt.pie(Population,labels=Religion,radius=1.2,colors=col,shadow=True,labeldistance=1.1,rotatelabels=True,startangle=0,frame=False,counterclock=True,explode=(0,0.2,0.3,0.4,0.8,1,1.2),autopct='%1.1f%%',)
plt.title('Religions of India')
plt.axis('equal')  
plt.tight_layout()
plt.ioff()


# Close to 80% people are Hindus.Other major relgions practised by Indias are Islam (14.4%), Christanity (2.3%), Sikhism (1.7%), Bhuddism (0.7%), Jainism (0.4%) and others (0.9%) include Jews,Parsis,Non Beleivers and the people who dont want to disclose their religious identity.India is a highly religious country four major religions like Hinduism,Bhuddism,Jainism and Sikhism has its roots in India.Christianity came to India in AD 52 with the arrival of Apostle St Thomas to Southern state of Kerala.Islam also has an early entry to India through the Arab traders.The second oldest mosque was built in India while Prophet Muhammad was still Alive.People of all religions coexist peacefully in India.
