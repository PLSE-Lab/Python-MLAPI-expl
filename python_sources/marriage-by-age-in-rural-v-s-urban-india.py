#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# **In this notebook I will be exploring data on marriages in India. Until recently child marriages(Marriage at a very young age) were very prominent in India. With time the number of child marriages have decreased but still in the rural parts of India these practices still persist. We will visualize Rural India and Urban India for this, since literacy rate in Urban region is more the amount of child marriages will certainly be less as compared to Rural areas.
# **

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Inputting Files 
rural_df = pd.read_csv('/kaggle/input/marriage-by-age-india/Rural_india - Rural_india - Sheet1 (2).csv')
urban_df = pd.read_csv('/kaggle/input/marriage-by-age-india/Urban_india - Sheet1 (1).csv')
india_df = pd.read_csv('/kaggle/input/marriage-by-age-india/Indian Marital status - Sheet1 (2).csv')
#found 1 wrong value, correcting
india_df['Total'].iloc[2] = 4413705
#The datasets were already cleaned beforehand and therefore dont need further cleaning


# **The datasets were already cleaned, they also have the distribution of marriages by some of the more prominent religions in India. We will also see the distribution by religions in this notebook as it might also give us some insight on the dataset.**

# In[ ]:


#Lets look at the datasets
print(india_df.head())
print(urban_df.head())
print(rural_df.head())


# **We can see that the dataset are divided into 8 colums namely, Age, Total/Urban/Rural, Hindu, Muslim, Christian, Sikh, Buddhist, Jain. The first column 'Age' tells us the age-group of married individuals, second column gives us the distribution of everyone who are in that class (India,Urban,Rural) and the rest of the columns give distribution by religion.** 
# 
# **Now lets visualize the total marriages in each region**

# In[ ]:



def total_vis():
    #converting the numbers into percentage
    total = india_df['Total'].loc[0]
    df = pd.DataFrame(data = {'Age': india_df['Age'],'Total':india_df['Total']})
    df[['Total']] = (df[['Total']]/total)*100
    #Drop "all ages" row
    df.drop([0],inplace = True)
    #plotting
    plt.figure(figsize = (13,13))
    pos = np.arange(len(df['Age']))    
    bars = plt.bar(pos,df['Total'],align = 'center',alpha = 0.5,color = 'purple',width = 0.9)    
    plt.xticks(pos,df['Age'],rotation = 30,alpha = 0.8)
    plt.title('Marriage by Age in India',fontdict = {'fontsize' : 20},alpha = 0.5)
    #removing ticks and labels
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=True)
    #removing frame
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    #putting text
    for bar in bars:
        height = bar.get_height()
        if bar.get_height()>0.6:
            plt.gca().text(x = bar.get_x() + bar.get_width()/2,y = (bar.get_height()-0.5),s = str(round(height,1)) + '%',ha='center', color='white', fontsize=11)
        else:
            plt.gca().text(x = bar.get_x() + bar.get_width()/2,y = (bar.get_height()+0.1),s = str(round(height,1)) + '%',ha='center', color='black', fontsize=11)
    
    return pd.DataFrame(data = {'Age': india_df['Age'],'Total':india_df['Total']})
total_vis()


# **Above graph shows Percentage of marriages in different age groups in India. We can see that most marriages are done between the age of 18 to 21, which is still pretty young. We also see that almost 2% of marriages are done before the age of 10, this figure is very worrying. The total number of marriages done are 640,021,008, 2% of that is 12,800,420. This number is still very high. **
# 
# **Next we will see a visualization of Rural v/s Urban.**

# In[ ]:


def rural_urban():
    #Dividing by total marriages in each age group
    total = pd.Series(india_df['Total'])
    df = pd.DataFrame(data = {'Age': india_df['Age'],'Urban':urban_df['Urban'],'Rural':rural_df['Rural']})
    df[['Rural','Urban']] = (df[['Rural','Urban']].divide(total,axis = 0))*100
    #plotting
    plt.figure(figsize = (15,15))
    plt.ylim((0,100))
    pos = np.arange(len(df['Age']))
    bars1 = plt.bar(pos,df['Rural'],alpha = 0.7,bottom = df['Urban'],color='#FF3030')
    bars2 = plt.bar(pos,df['Urban'],alpha = 0.7,color='#00FA9A')
    plt.xticks(pos,df['Age'],rotation = 30,alpha = 0.8)
    plt.title('Rural v/s Urban by Age',fontdict = {'fontsize' : 20},alpha = 0.5)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=True)
    #removing frame
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    for i in range(len(bars2)):
        height = bars2[i].get_height()
        plt.gca().text(x = bars2[i].get_x() + bars2[i].get_width()/2,y = (bars2[i].get_height()-1.5),s = str(round(height,1)) + '%',ha='center', color='grey', fontsize=11)
        plt.gca().text(x = bars2[i].get_x() + bars2[i].get_width()/2,y = (bars2[i].get_height()+0.6),s = str(round(bars1[i].get_height(),1)) + '%',ha='center', color='white', fontsize=11)
    plt.legend(['Rural','Urban'])
    return pd.DataFrame(data = {'Age': india_df['Age'],'Urban':urban_df['Urban'],'Rural':rural_df['Rural']})
rural_urban()


# **Above graph shows the percentage of marriages in each age group in Rural as well as Urban areas. We can immediately see that the total number of marriages in rural areas is more. Also marriage at a young age is also very common in rural. One of the reasons could be the low literacy rate and maybe poverty.**
# 
# **Now lets look at distribution by religious communities. This time we would use line charts to visualise the percentage by religions.**

# In[ ]:


def religions():
    fig,((ax1),(ax2),(ax3)) = plt.subplots(3,1,figsize=(14,14))
    rel = ["Hindu","Muslim","Christian","Buddhist","Jain","Sikh"]
    
    #plotting india
    df1 = india_df
    df1[rel] = (df1.loc[:][rel].divide(df1.loc[0][rel]))*100
    df1 = (df1.drop(['Total'],axis = 1)).drop([0])
    ax1.plot(df1['Age'],df1['Hindu'],'o-')
    ax1.plot(df1['Age'],df1['Muslim'],'o-')
    ax1.plot(df1['Age'],df1['Christian'],'o-')
    ax1.plot(df1['Age'],df1['Buddhist'],'o-')
    ax1.plot(df1['Age'],df1['Jain'],'o-')
    ax1.plot(df1['Age'],df1['Sikh'],'o-')
    ax1.set_ylabel('Percentage')
    ax1.set_title('India')
    ax1.legend(rel)
    
    #plotting rural
    df2 = rural_df
    df2[rel] = (df2.loc[:][rel].divide(df2.loc[0][rel]))*100
    df2 = (df2.drop(['Rural'],axis = 1)).drop([0])
    ax2.plot(df1['Age'],df1['Hindu'],'o-')
    ax2.plot(df1['Age'],df1['Muslim'],'o-')
    ax2.plot(df1['Age'],df1['Christian'],'o-')
    ax2.plot(df1['Age'],df1['Buddhist'],'o-')
    ax2.plot(df1['Age'],df1['Jain'],'o-')
    ax2.plot(df1['Age'],df1['Sikh'],'o-')
    ax2.set_ylabel('Percentage')
    ax2.set_title('Rural')
    ax2.legend(rel)
    
    #plotting urban
    df3 = urban_df
    df3[rel] = (df3.loc[:][rel].divide(df3.loc[0][rel]))*100
    df3 = (df3.drop(['Urban'],axis = 1)).drop([0])
    ax3.plot(df1['Age'],df1['Hindu'],'o-')
    ax3.plot(df1['Age'],df1['Muslim'],'o-')
    ax3.plot(df1['Age'],df1['Christian'],'o-')
    ax3.plot(df1['Age'],df1['Buddhist'],'o-')
    ax3.plot(df1['Age'],df1['Jain'],'o-')
    ax3.plot(df1['Age'],df1['Sikh'],'o-')
    ax3.set_ylabel('Percentage')
    ax3.set_title('Urban')
    ax3.legend(rel)
    return
religions()


# **Here we see that theres not much difference in the percentage of marriages in different agge groups in lower ages. So we cant really say that religion affects the age of marriage.**
# 
# # Conclusion
# **We see that child marriage is still quite prominent in India, more in Rural areas than Urban. We also see that religion might not be a determining factor for the same. So the most obvious reasons would be poverty and illiteracy.**
# 
# This was my first ever visualisation notebook. Please critique it and tell me what you think and what else could be added. Some more ideas I had were including sex of individuals. This visualisation doesnt tell something concrete and was mostly to practice my skills and hence would be pretty sloppy.
# 
# Datasets were obtained from "censusindia.gov.in" and cleaned in google sheets.
# 
# Thanks for reading!!

# In[ ]:




