#!/usr/bin/env python
# coding: utf-8

# IMPORTANT NOTE!
# This kernel is heavily inspired by by Aatish Bhatia and Minute Physics. Please visit <a href="https://aatishb.com/covidtrends/"> their website</a> to learn more!

# # Is Turkey Doing Good vs COVID-19?
# 
# ### Well tell me if I'm being selfish but I really want to know that am I going to be okay? Let's find out whether if Turkey is late to jump the band wagon.
# 
# #### Since COVID-19 pandemic has an exponential rate of transmission, analyzing weekly difference in logarithmic scale creates an easier, safer and much more intentionally pessimistic approach to looking only at day-by-day confirmed cases. If Turkey's graph at the end follows a downward path, it means that we are going in a good direction; if not, it means that we should take action. That simple! This kernel also updates itself so you can use it to know whether if Turkey is fine. You can also change the country by forking this kernel.

# First of I want to have a basic understanding of the data as usual.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


covid_19_data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")


# In[ ]:


covid_19_data.head()


# It looks like I can iterate through every date entry and calculate weekly difference and go to the last entry and get the total confirmed count. However, some countries have multiple entries to the dataset. I will deal with that later. But first, let's just drop the columns that aren't useful to me.

# In[ ]:


covid_19_data = covid_19_data.drop(columns = ["Province/State","Lat","Long"])


# In[ ]:


covid_19_data.head()


# Then, I will store every country and the date series in a dictionary where each value will be a nx2 matrix where n is count of entry per country.

# In[ ]:


#Start every country with a blank list
confirmed_dict = {}
for i in covid_19_data["Country/Region"]:
    confirmed_dict[i] = []
    
#Get dates to a list
dates = list(covid_19_data.columns)
dates.remove('Country/Region')

#Iterate through every row of the dataframe
for count,country in enumerate(covid_19_data["Country/Region"].values.tolist()):
    
    #Checking if a country has duplicate entries
    if confirmed_dict[country] == []:
        
        temp = []
        for date in dates:
            temp.append([date,covid_19_data[date][count]])
        confirmed_dict[country] = temp
    
    else:
        #Adding the duplicate value to the old value
        old = confirmed_dict[country]
        for entry in old:
            for date in dates:
                if date == entry[0]:
                    entry[1] += covid_19_data[date][count]


# That seems right. Now, I want to extract total confirmed count and weekly differences for each day into two seperate lists to plot them later. To achieve this, I will go through every entry for every country and substract 7 days before value. 

# In[ ]:


def createData(list,dict):
    result = {}
    for country in list:
        
        temp_X = []   #This will store total confirmed
        temp_Y = []   #This will store weekly difference by day
        
        for i in range(len(dict[country]) - 7):
            temp_X.append(dict[country][i + 7][1])   #Total Value
            temp_Y.append(dict[country][i + 7][1] - dict[country][i][1])   #Weekly Difference by Day
                
        result[country] = [temp_X,temp_Y]
    return result


# Now, let's plot our graphs. Thanks to the <a href="https://towardsdatascience.com/cyberpunk-style-with-matplotlib-f47404c9d4c5">Towards Data Science</a> article for this beautiful graph design.

# In[ ]:


plt.style.use("seaborn-dark")
for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = '#212946'  # bluish dark grey
for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '0.9'  # very light grey
size = (15,7.5)

fig, ax = plt.subplots(figsize=size)
ax.grid(color='#2A3459')
colors = ['#08F7FE','#FE53BB','#F5D300','#00ff41']
ax.set_title('Scalar Confirmed')

n_lines = 10
diff_linewidth = 1.05
alpha_value = 0.03

countries = ["US","Turkey","Italy","Korea, South"]
plot_data = createData(countries,confirmed_dict)


for idx,country in enumerate(countries):
    ax.plot(plot_data[country][0],plot_data[country][1],label = country,color = colors[idx])

    
#plt.yscale('log')
#plt.xscale('log')
plt.legend()
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=size)
ax.grid(color='#2A3459')
colors = ['#08F7FE','#FE53BB','#F5D300','#00ff41']
ax.set_title('Logarithmic Confirmed')

countries = ["US","Turkey","Italy","Korea, South"]
plot_data = createData(countries,confirmed_dict)

for idx,country in enumerate(countries):
    ax.plot(plot_data[country][0],plot_data[country][1],label = country,color = colors[idx])

plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()


# In[ ]:



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=size)

colors = ['#08F7FE','#FE53BB','#F5D300','#00ff41']
fig.suptitle('Logarithmic Confirmed by Country')



ax1.plot(plot_data[countries[0]][0],plot_data[countries[0]][1],label = countries[0],color = colors[0])
ax2.plot(plot_data[countries[1]][0],plot_data[countries[1]][1],label = countries[1],color = colors[1])
ax3.plot(plot_data[countries[2]][0],plot_data[countries[2]][1],label = countries[2],color = colors[2])
ax4.plot(plot_data[countries[3]][0],plot_data[countries[3]][1],label = countries[3],color = colors[3])
ax1.set_title(countries[0])
ax2.set_title(countries[1])
ax3.set_title(countries[2])
ax4.set_title(countries[3])

ax1.set_xscale("log")
ax2.set_xscale("log")
ax3.set_xscale("log")
ax4.set_xscale("log")

ax1.set_yscale("log")
ax2.set_yscale("log")
ax3.set_yscale("log")
ax4.set_yscale("log")


for ax in fig.get_axes():
    ax.label_outer()


# In[ ]:


plt.style.use("seaborn-dark")
for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = '#212946'  # bluish dark grey
for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '0.9'  # very light grey

fig, ax = plt.subplots(figsize=size)
ax.grid(color='#2A3459')
colors = ['#08F7FE','#FE53BB','#F5D300','#00ff41']
ax.set_title('How is Turkey Doing? (Downwards Better)')

n_lines = 10
diff_linewidth = 1.05
alpha_value = 0.03

countries = ["Turkey"]
plot_data = createData(countries,confirmed_dict)

margin = 0.01

for idx,country in enumerate(countries):
    ax.plot(plot_data[country][0],plot_data[country][1],label = country,color = colors[idx])
    if np.log(plot_data[country][1])[-1] > np.log(plot_data[country][1])[-2] + margin:
        ax.fill_between(x=plot_data[country][0],y1=plot_data[country][1],y2=[0] * len(plot_data[country][0]),color=colors[1],alpha=0.1)
    elif np.log(plot_data[country][1])[-1] < np.log(plot_data[country][1])[-2] - margin:
        ax.fill_between(x=plot_data[country][0],y1=plot_data[country][1],y2=[0] * len(plot_data[country][0]),color=colors[3],alpha=0.1)
    elif np.log(plot_data[country][1])[-1] < np.log(plot_data[country][1])[-2] + margin and plot_data[country][1][-1] > plot_data[country][1][-2] - margin:
        ax.fill_between(x=plot_data[country][0],y1=plot_data[country][1],y2=[0] * len(plot_data[country][0]),color=colors[2],alpha=0.1)

    
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()


# # CONCLUSION:
# 
# ## If everything works, this graph's downward trend should mean that we are doing fine.To indicate a downward trend, you can look at the area below the curve. If it's <span style="color:#FE53BB;background-color:#212946">red</span> this means a bad upward trend whereas <span style="color:#00ff41;background-color:#212946">green</span> means a happy downward trend lastly  <span style="color:#F5D300;background-color:#212946">yellow</span> means it stayed inside the margin that we have decided above. Change threshold is set to 0.01 by default.
