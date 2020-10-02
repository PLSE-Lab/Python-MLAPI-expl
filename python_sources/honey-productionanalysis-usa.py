#!/usr/bin/env python
# coding: utf-8

# # ANALYSIS OF PRODUCTION OF HONEY IN THE USA FROM YEAR 1998 TO 2012
# 
# ### Keeping the following as the main points to be analyzed (from data)
# 1.  How has honey production yield changed from 1998 to 2012?
# 2.  Over time, which states produce the most honey? Which produce the least? Which have experienced the most change in honey yield?
# 3.  Does the data show any trends in terms of the number of honey producing colonies and yield per colony before 2006, which was when concern over Colony Collapse Disorder spread nationwide?
# 4.Are there any patterns that can be observed between total honey production and value of production every year? How has value of production, which in some sense could be tied to demand, changed every year?
# 
# 
# 
# ### * Import the libraries and configure basic settings

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)


# ### * Load the dataset

# In[2]:


honey = pd.read_csv('../input/honeyproduction.csv')


# ### * Some basic information about the dataset

# In[ ]:


honey.dtypes


# In[ ]:


honey.info()


# In[ ]:


honey.describe().round()


# In[3]:


honey.head(10)


# ### *Let us see the total production of honey in USA(statewise), throughout the given time range 
# 
# #### We can see that the states North Dakota(ND) and California(CA) stand out as major honey production states from 1998-2012

# In[4]:


plt.figure(figsize=(40,15))
sns.barplot(x=honey['state'],y = honey['totalprod'])
plt.title('Statewise Total Honey production in USA',fontsize =30)
plt.xlabel('States',fontsize=30)
plt.ylabel('Total Production of Honey in USA',fontsize=30)


# ### * Visualizing how the total production of honey in USA has fared over the years
# 
# #### Group the dataset by year and use the sum method to get the total honey production in USA every year.Then plot a line graph to visualize the same.

# In[5]:


honey_byYear = honey[['totalprod','year']].groupby('year').sum()
honey_byYear.reset_index(level=0, inplace=True)


# In[6]:


plt.figure(figsize=(30,10))
plt.plot(honey_byYear['year'],honey_byYear['totalprod'])
plt.title('Total Production Of Honey in USA (lbs.) Vs Year',fontsize=30)
plt.xlabel('Year',fontsize = 20)
plt.ylabel('Total Production of Honey',fontsize = 20)


# ### * Analyzing the major and minor honey production states in the year over the period 1998-2012
# 
# #### Like the previous plot here we group the dataset by state and use the sum method to get the total production of honey by each state over the given period. Then a barplot for the same is done
# 
# #### We all know that state of Dakota(s) and California is famous for its honey. This is proven by observing that the states ND(North Dakota),CA(California) and SD(South Dakota) are the 3 major producers as apiculture is one of the major livelihoods in these states, while the states MD,OK and SC contribute least to the honey production in USA. 

# In[7]:


honey_byState = honey[['totalprod','state']].groupby('state').sum()
honey_byState.reset_index(level=0, inplace=True)
honey_byState.sort_values(by='totalprod',ascending=False,inplace=True)
plt.figure(figsize=(20,10))
sns.barplot(x=honey_byState['state'],y=honey_byState['totalprod'])
plt.title('Statewise total honey production',fontsize = 20)
plt.xlabel('State',fontsize = 20)
plt.ylabel('Total Production(lbs)',fontsize = 20)


# #### Below are methods to calculate the percentage change for a quantity in the dataset for the given year range and to plot these percentage changes as a bargraph state wise for the given time range in years.

# In[11]:


def percentage_change(dataframe,column,y1,y2):
    '''Creates 2 dataframes with rows only having the given year values. Then these 2 frames are merged to a single frame
    with respect to the states and returned'''
    honey_change_yearA = dataframe[['state',column,'year']].loc[dataframe['year']==y1].sort_values('state')
    honey_change_yearB = dataframe[['state',column,'year']].loc[dataframe['year']==y2].sort_values('state')
    honey_yearAyearB = pd.merge(honey_change_yearA,honey_change_yearB,on=['state','state']).drop('year_x',axis=1).drop('year_y',axis=1)    
    honey_yearAyearB['percentage_change'] = (honey_yearAyearB.iloc[:,2]-honey_yearAyearB.iloc[:,1])/honey_yearAyearB.iloc[:,1] * 100
    return honey_yearAyearB

def percentage_plot(plot_parameterList,fig_size=(10,15)):
    '''Creates 'n' Bargrph subplots.
    The plot_parameterList is a list of lists that has the following arguments
    0 -> Dataframe
    1 -> Quantity for the bar graphs over which the percentage change has been meeasured using percentage_change method.
    2 -> Start year of the time range
    3 -> End year of the time range
    
    And additional parameter to change the figure size is included.
    Best for maximum of 3x2 plots'''
    sns.set(rc={'figure.figsize':fig_size})
    sns.set(font_scale=0.9)
    import math
    total_plots = len(plot_parameterList)
    plotRows = math.ceil(total_plots/2)
    if total_plots > 1:
        fig, ax = plt.subplots(plotRows,2)
        ax = ax.flatten()
        for x in range(total_plots):
                sns.barplot(x = plot_parameterList[x][0]['percentage_change'], y = plot_parameterList[x][0]['state'], ax = ax[x])
                ax[x].title.set_text('PercentageChange : ' + plot_parameterList[x][1] +' years>'+str(plot_parameterList[x][2])+'-'+str(plot_parameterList[x][3]))
                ax[x].set_xlabel('Percentage Change')
                ax[x].set_ylabel('State')        
    else:
        sns.barplot(x = plot_parameterList[0][0]['percentage_change'], y = plot_parameterList[0][0]['state'])
        plt.title('PercentageChange : ' + plot_parameterList[0][1] +' years>'+str(plot_parameterList[0][2])+'-'+str(plot_parameterList[0][3]))
        plt.xlabel('Percentage Change')
        plt.ylabel('State')


# ### *Observing the percentage change of yield around 2006 (when the flag on colony collapse disease was raised)
# 
# #### It can be seen that the statewise yield of honey has gone down for most of the states from 2004 until 2006(upper 2 plots). In the next two years many states have gradually recovered (bottom 2 plots), mostly because of conservation policies and practises taken to save the American Honey industry and overcome the CCD that was observed in 2006.

# In[12]:


column_name,start_year,end_year = ['totalprod',2004,2005]
column_name,start_year1,end_year1 = ['totalprod',2005,2006]
column_name,start_year2,end_year2 = ['totalprod',2006,2007]
column_name,start_year3,end_year3 = ['totalprod',2007,2008]
honey_yearAyearB_production = percentage_change(honey,column_name,start_year,end_year)
honey_yearAyearB_production1 = percentage_change(honey,column_name,start_year1,end_year1)
honey_yearAyearB_production2 = percentage_change(honey,column_name,start_year2,end_year2)
honey_yearAyearB_production3 = percentage_change(honey,column_name,start_year3,end_year3)
percentage_plot([                 [honey_yearAyearB_production,column_name,start_year,end_year],                 [honey_yearAyearB_production1,column_name,start_year1,end_year1],                 [honey_yearAyearB_production2,column_name,start_year2,end_year2],                 [honey_yearAyearB_production3,column_name,start_year3,end_year3]                ],(10,15))


# ### * Analyzing how the the honey traits fared before and after the CCD was flaged in 2006.
# 
# #### A method to plot a lmplot (Scatter) for the honey production traits is used. This is done to generalize the plotting of the traits for individual states or USA altogether. The 'Yield per colony' and 'Total Production' is plotted as a scatter plot where each dot represents the 'Yield per colony' and 'Total Production' for a particular year

# In[138]:


def scatter_plot(plot_parameterList,fig_size):
    sns.set(rc={'figure.figsize':fig_size})
    sns.set(font_scale=0.9)
    import math
    total_plots = len(plot_parameterList)
    #plotRows = math.ceil(total_plots/2)
    #fig, ax = plt.subplots(plotRows,2)
    #ax = ax.flatten()
    
    for x in range(total_plots):
            sns.lmplot(x = 'yieldpercol', y = 'totalprod',data = plot_parameterList[x][0],fit_reg=False,hue='year')
            plt.title('Year wise -Mean Totalproduction and Num.Colonies scatter plot('+plot_parameterList[x][1]+')')


# #### Here dataframes are derived for the state AL and whole USA (using groupby with respect to year) from the original dataframe. Then passed to the scatter_plot method to plot the respective scatter plots.
# 
# #### It can observed that for USA(mean values) and state AL the yeild per colony, total production has drastically come down after the years 2005. The dots at the top right corner of the plots that belong the years 1998-2004 suggests that during these years the honey produced and the yield per colony was significantly high compared to the years 2006 and onwards, showcased by the dots in the lower right corner of the plots. Though there are few sporadic dots that dont fit the above general observation in case of state AL, producing the plot for different states by changing the value in the honey_colprod_singleState dataframe shows the general trend.

# In[141]:


honey_colprod_singleState = honey[['totalprod','yieldpercol','year','state']].loc[honey['state']=='AL'].drop('state',axis=1)
honey_colprod_USA = honey[['totalprod','yieldpercol','year']].groupby(['year']).mean()
honey_colprod_USA.reset_index(inplace=True)
scatter_plot([[honey_colprod_USA,"USA"],[honey_colprod_singleState,'State-AL']],(10,15))


# In[143]:


# For Additional reference
#g = sns.FacetGrid(honey_colprod_USA, col="state",col_wrap = 5)
#g = (g.map(plt.scatter, 'year','numcol').set_axis_labels('xyx','abx'))


# ### * Analysis of how the value of production has been affected due to the change in total honey production and to see if something related to demand for honey can be summarized from the above analysis.

# In[170]:


honey_priceDemand = honey.groupby('year').mean()
honey_priceDemand.reset_index(level=0,inplace=True)
#honey.loc[honey['state'] == 'ND']


# ### From the below plots we can see that the production value has grown while the total production of honey has declined. This can be attributed to the rise in price of honey (per lbs).
# 
# #### It is seen that the production value during the period 2005-2007 has taken a major hit, maybe due to the demand decreasing in view of the CCD. Also the demand for honey after the above period has increased ,this can be attributed to the slow recovery of the total production of honey against the sharp rise of the price for honey.

# In[173]:


plt.figure(figsize=(20,10))
plt.plot(honey_priceDemand['year'],honey_priceDemand['totalprod'],c='b',marker='o',markersize=12)
plt.plot(honey_priceDemand['year'],honey_priceDemand['prodvalue'],c='g',marker='X',markersize=12)
plt.legend(ncol=2,loc=2,fontsize = 15)
plt.title('Total Production and Production Value of Honey over the years',fontsize = 15)
plt.xlabel('years',fontsize = 15)
plt.ylabel('quantity',fontsize = 15)


# In[174]:


plt.figure(figsize=(20,10))
sns.barplot(x='year',y='priceperlb',data=honey_priceDemand)
plt.title('Price of Honey over the years',fontsize = 15)
plt.xlabel('years',fontsize = 15)
plt.ylabel('Price per Pound',fontsize = 15)


# # MY FIRST DATA SCIENCE PROJECT ON KAGGLE
# 
# ### Still new to python, I'm taking up data sets to analysis and improve my expertise in analysis and visulisation of data in python.
# 
# ### Any suggestions,tips are most welcome. Comment if any corrections are to be done or there are better ways to implement the same, it would help out a lot :D
# 
# # THANK YOU :D
