#!/usr/bin/env python
# coding: utf-8

# We have data about activity on the USA-Canada and USA-Mexico border. 
# 
# In this kernel, I have tried to do an exploratory analysis of the activity at the border. This kernel only contains the trends of activity at the borders over the years. Initially I have focused on the type of activity which involved people. Later on, I will proceed to do exploratory analysis of border activity regarding goods.
# Any suggestions regarding this kernels are welcome :-)

# Lets start with importing the data as below

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


# After that, lets import some libraries.

# In[ ]:


#Import Library
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler


# Data has been taken into a variable and primitive cleaning is done.

# In[ ]:


data=pd.read_csv('/kaggle/input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv')
data['Date']=pd.to_datetime(data['Date'])


# Lets see the levels of activity on the two borders

# In[ ]:



state_cou=data.groupby(['Border']).size().reset_index(name='count')

data.groupby(['Border']).size().reset_index(name='count')

sns.set_context('talk')
sns.barplot(x=state_cou['Border'],y=state_cou['count'],palette='deep')
plt.xticks(rotation=90)
plt.ylabel('Activity at Entry Ports')
plt.title('Border vs Count')
plt.show()


# Wordcloud can be created for so see the kinds of objects that are listed in the data which pass through the borders.

# In[ ]:


abc=','
abc2=data.Measure.unique()
abc3=abc.join(abc2)
abc3=abc3.replace(" ","_")

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                min_font_size = 10).generate(abc3) 

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# Below, two arrays have been created each listing mobiles that pass through border which can be differentiated by if they carry people or goods. 

# In[ ]:


people=['Personal vehicles','Bus Passengers','Personal Vehicle Passengers','Train Passengers','Trains','Buses']
data[data.Measure.isin(people)]
goods=['Truck Containers Empty','Rail Containers Full','Rail Containers Empty','Truck Containers Full','Trucks']
#data[data.Measure.isin(goods)]


# Get the latest date from the data

# In[ ]:


data['Date'].max()


# Get the earliest date from the data

# In[ ]:


data['Date'].min()


# These are just used for checking and getting to know the data better.
# 
# 
# Group the data based on the Measure, Border and the date of activity as below

# In[ ]:


data_border_pass=data[data.Measure.isin(people)].groupby(['Border','Measure','Date'],as_index=False).sum()
#data_border_pass


# Firstly, I have taken data solely for US-Mexico Border and aggregated the activity based on the Measure.
# 
# Port code has been dropped as it the columns was irrelevant for my scope of analysis. Additionally, the date has been mutated and only year from this columns has been extracted as i had to observe the trends of activities at the border over the years. 
# 

# In[ ]:


#data_border_pass[data_border_pass['Border']=='US-Mexico Border']['Border']
dat1=data_border_pass[data_border_pass['Border']=='US-Mexico Border']
dat1=dat1.drop(['Port Code'],axis=1)
dat1=dat1.groupby(['Measure','Date'],as_index=False).sum()
dat1['Year']=pd.DatetimeIndex(dat1['Date']).year
dat1.groupby(['Measure','Year'],as_index=False).sum()


# See the Unique measures which carry people

# In[ ]:


dat1['Measure'].unique()


# A python function can be created to plot bart graphs(which are mostly used in the analysis) to avoid redundant code.

# In[ ]:


def bar_plt(x_bar,y_bar,ylab,titl):
    sns.set_context('talk')
    sns.barplot(x=x_bar,y=y_bar,palette='deep')
    plt.xticks(rotation=90)
    plt.ylabel(ylab)
    plt.title(titl)


# Lets see how the activity of Bus with passengers has been over the years. 

# In[ ]:


dat_temp=dat1[dat1['Measure']=='Bus Passengers']
dat_temp=dat_temp.groupby(['Measure','Year'],as_index=False).sum()
bar_plt(dat_temp['Year'],dat_temp['Value']
        ,'Count of Activity'
        ,'Bus Passengers Activity US-MEX')


# It can be seen from the graph that the Passengers activity at the US-Mex border declined in year 2009 but bounced back up for a brief period of time only to fall again.

# In[ ]:


dat_temp=dat1[dat1['Measure']=='Buses']
dat_temp=dat_temp.groupby(['Measure','Year'],as_index=False).sum()
bar_plt(dat_temp['Year'],dat_temp['Value']
       ,'Count of Activity'
    ,'Buses Activity US-MEX')


# It can be seen from the graog that the frequency of buses on US-Mex border has been declining steadily since 2009. This might be as a result of reduced demand after 2009. This phenomenon of graphs tipping downwards in 2009 is a result of financial crisis of 2009 when economy was slowed down. You will see this phenomenon in the rest of the following graphs as well.
# 

# In the following map we will add two graphs namely, Bus Passengers Activity and Buses activity across the US-MEX border to see the relation between these two trends more clearly.

# In[ ]:


fig, ax1 = plt.subplots()

color = 'tab:red'
#ax1.set_xlabel('time (s)')
ax1.set_ylabel('Buses', color=color)
ax1.bar(dat1[dat1['Measure']=='Buses']['Year'], dat1[dat1['Measure']=='Buses']['Value'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Bus Passengers', color=color)  # we already handled the x-label with ax1
ax2.plot(dat1[dat1['Measure']=='Bus Passengers']['Year'], dat1[dat1['Measure']=='Bus Passengers']['Value'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# Look at the train activity for US-MEX border

# In[ ]:


dat_temp=dat1[dat1['Measure']=='Trains']
dat_temp=dat_temp.groupby(['Measure','Year'],as_index=False).sum()
bar_plt(dat_temp['Year'],dat_temp['Value']
       ,'Count of Activity'
    ,'Trains Activity US-MEX')


# As expected, the trains activity at the border US-Mexico declined sharply in 2009 but increased every year after it. In 2017 the train frequency was greater than 2007, the year in which the economy was at best.
# This might be the reason that frequency of buses was reduced at the border, increased frequencies of trains might have reduced the demand of Buses. It will be interesting to see the passenger activity though trains after this. 

# In[ ]:


dat_temp=dat1[dat1['Measure']=='Train Passengers']
dat_temp=dat_temp.groupby(['Measure','Year'],as_index=False).sum()
bar_plt(dat_temp['Year'],dat_temp['Value']
       ,'Count of Activity'
    ,'Trains Passengers Activity US-MEX')


# What a surprise! Even though the train frequency at the US-MEX border was increasing through 2009-2013, the passenger activity was almost the same throughout that period. It did catch up after 2013 but it wasnt on the same levels as before 2009.   

# We will plot the same kind of graph we plotted for the Bus related activity for trains to see the relation and different between them.

# In[ ]:


fig, ax1 = plt.subplots()

color = 'tab:red'
#ax1.set_xlabel('time (s)')
ax1.set_ylabel('Trains', color=color)
ax1.bar(dat1[dat1['Measure']=='Trains']['Year'], dat1[dat1['Measure']=='Trains']['Value'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Train Passengers', color=color)  # we already handled the x-label with ax1
ax2.plot(dat1[dat1['Measure']=='Train Passengers']['Year'], dat1[dat1['Measure']=='Train Passengers']['Value'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# Lets see the activity at US-MEX border for Personal Vehicles

# In[ ]:


dat_temp=dat1[dat1.Measure.isin(['Personal Vehicle Passengers', 'Personal Vehicles'])]
dat_temp=dat_temp.groupby(['Measure','Year'],as_index=False).sum()
bar_plt(dat_temp['Year'],dat_temp['Value']
       ,'Count of Activity'
    ,'Personal Vehcles Activity US-MEX')


# The activity was at the lowest at 2010 but has been increasing steadily after that.

# Below graphs in which the above mentioned activities were taken and mean frequency for the each one them was taken. After that graph for plotted for the activities over the years with respect to the mean frequency. This was done in hope to create a Facet Grid for the activities. It was important to plot the count of activities with respecto to mean frequency to bring them all to the same scale which is very much needed for the facet grid.  

# In[ ]:


scaler=StandardScaler()
dat_bus=dat1[dat1['Measure']=='Bus Passengers']
dat_bus=dat_bus.groupby(['Measure','Year'],as_index=False).sum()
dat_bus=pd.DataFrame(dat_bus)
scaler.fit(dat_bus.iloc[:,2].values.reshape(-1,1))
dat_bus['Value']=pd.DataFrame(scaler.transform(np.asarray(dat_bus['Value']).reshape(-1,1)))


# In[ ]:


bar_plt(dat_bus['Year'],dat_bus['Value'],'Count of Activity(wrt Mean Activity)','Bus Passenger Activity')


# In[ ]:


scaler1=StandardScaler()
dat_train=dat1[dat1['Measure']=='Train Passengers']
dat_train=dat_train.groupby(['Measure','Year'],as_index=False).sum()
dat_train=pd.DataFrame(dat_train)
scaler1.fit(dat_train.iloc[:,2].values.reshape(-1,1))
dat_train['Value']=pd.DataFrame(scaler1.transform(np.asarray(dat_train['Value']).reshape(-1,1)))


# In[ ]:


bar_plt(dat_train['Year'],dat_train['Value'],'Count of Activity(wrt Mean Activity)','Train Passengers Activity')


# In[ ]:


scaler2=StandardScaler()
dat_pass=dat1[dat1['Measure']=='Personal Vehicle Passengers']
dat_pass=dat_pass.groupby(['Measure','Year'],as_index=False).sum()
dat_pass=pd.DataFrame(dat_pass).reset_index()
scaler2.fit(dat_pass.iloc[:,3].values.reshape(-1,1))
dat_pass['Value']=pd.DataFrame(scaler2.transform(np.asarray(dat_pass['Value']).reshape(-1,1)))


# In[ ]:


bar_plt(dat_pass['Year'],dat_pass['Value'],'Count of Activity(wrt Mean Activity)','Personal Vehicle Passengers Activity')


# In[ ]:


scaler3=StandardScaler()
dat_tra=dat1[dat1['Measure']=='Trains']
dat_tra=dat_tra.groupby(['Measure','Year'],as_index=False).sum()
dat_tra=pd.DataFrame(dat_tra).reset_index()
scaler2.fit(dat_tra.iloc[:,3].values.reshape(-1,1))
dat_tra['Value']=pd.DataFrame(scaler2.transform(np.asarray(dat_tra['Value']).reshape(-1,1)))


# In[ ]:


bar_plt(dat_tra['Year'],dat_tra['Value'],'Count of Activity(wrt Mean Activity)','Trains Activity')


# Following is an attempt to create a facetgrid for the above mentioned activities but somehow the layout is off. So any suggestions to make it better are welcome.

# In[ ]:


dat_append=dat_bus.append(dat_train).append(dat_pass).append(dat_tra)


# In[ ]:


sns.set_context('talk')
g=sns.FacetGrid(dat_append,col="Measure",col_wrap=2)
g.map(sns.barplot,'Year','Value',palette='deep')
g.fig.tight_layout()


# In following graphs the similar trends for US-Canada Border are plotted.

# In[ ]:


dat2=data_border_pass[data_border_pass['Border']=='US-Canada Border']
dat2=dat2.drop(['Port Code'],axis=1)
dat2=dat2.groupby(['Measure','Date'],as_index=False).sum()
dat2['Year']=pd.DatetimeIndex(dat1['Date']).year
dat2.groupby(['Measure','Year'],as_index=False).sum()


# In[ ]:


dat2['Measure'].unique()


# In[ ]:


bar_plt(dat2[dat2['Measure']=='Bus Passengers']['Year'],dat2[dat2['Measure']=='Bus Passengers']['Value'],
        'Count of Activity'
        ,'Bus Passengers Activity US-CAN')


# In[ ]:


bar_plt(dat2[dat2['Measure']=='Buses']['Year'],dat2[dat2['Measure']=='Buses']['Value'],
        'Count of Activity'
        ,'Buses Activity US-CAN')


# In[ ]:


bar_plt(dat2[dat2['Measure']=='Train Passengers']['Year'],dat2[dat2['Measure']=='Train Passengers']['Value'],
        'Count of Activity'
        ,'Train Passengers Activity US-CAN')


# In[ ]:


bar_plt(dat2[dat2['Measure']=='Trains']['Year'],dat2[dat2['Measure']=='Trains']['Value'],
        'Count of Activity'
        ,'Trains Activity US-CAN')


# In[ ]:


bar_plt(dat2[dat2['Measure']=='Personal Vehicle Passengers']['Year'],dat2[dat2['Measure']=='Personal Vehicle Passengers']['Value'],
        'Count of Activity'
        ,'Personal Passengers vehicles Activity US-CAN')

