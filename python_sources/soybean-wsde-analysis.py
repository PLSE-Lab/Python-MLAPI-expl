#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score


# > ## Soybean WSDE Analysis
# 
# #### Note
# - This study focuses on soybean since it is a very important crop and protein source for human beings, which can be made into oil, bean curd and many other yummy food of my favorite.
# - This is just an initial analysis to quickly look at soybean yield, export/import and price time trends and their correlations.  
# - It is not supposed to give any definitive answer to those realistic questions. 
# - In order to achieve an in-depth and more accurate analysis, domain knowledge is required to better understand data,  terminologies and contexts. 

# ### 0. Analysis Plan
# 1. data ingestion
# 2. data cleaning  
# 3. define/update right data science questions
# 4. data exploring & DV (Iteration between 3 and 4)
# 5. build, train and refine ML models to answer DS questions
# 6. conclusions
# 7. future work
# 8. acknowledges & references

# ### 1. Data ingestion, load dataset 
# #### Note: use 'conda install posix' in Anaconda Prompt window to enable linux shell commands in Jupyter in Windows OS host

# In[ ]:


#!ls crop


# In[ ]:


#files = !ls crop/*Soybean*
all = os.listdir(path='../input/')
files = [fname for fname in all if 'Soybean' in fname]
files


# #### 1.1 Reading Multiple CSV Files into one Python Pandas Dataframe
# ##### Ref. https://stackoverflow.com/questions/21149920/pandas-import-multiple-csv-files-into-dataframe-using-a-loop-and-hierarchical-i

# In[ ]:


path_prefix='../input/'
soybean = pd.concat([pd.read_csv(path_prefix+f) for f in files], keys=files)
#soybean.head()


# In[ ]:


close = pd.read_csv(path_prefix+'soybean_JUL14.txt')
#close.head()


# In[ ]:


nearby = pd.read_csv(path_prefix+'soybean_nearby.txt')
#nearby.head()


# #### 1.2 A quick glimpse of data set statistics

# In[ ]:


soybean.shape


# In[ ]:


soybean.columns


# In[ ]:


#soybean.dtypes


# In[ ]:


soybean.describe().transpose()[:27]


# In[ ]:


soybean.head(5)


# In[ ]:


#soybean.tail(5)


# In[ ]:


close.shape


# In[ ]:


close.columns


# In[ ]:


close.dtypes


# In[ ]:


close.describe().transpose()


# In[ ]:


close.head(3)


# In[ ]:


#close.tail(3)


# ##### It looks like soybean WSDE has a span of 12 years (2007-2018) data whereas close price has just 5 years (10 -14).

# In[ ]:


nearby.shape


# In[ ]:


nearby.columns


# In[ ]:


#nearby.dtypes


# In[ ]:


nearby.describe().transpose()


# In[ ]:


nearby.head(5)


# In[ ]:


#nearby.tail(5)


# ##### It looks like soybean WSDE has a span of 12 years (2007-2018) data whereas close price has just 5 years (10 -14) and nearby price has 10 years (08-17) data.  Keep this difference in mind for later analysis.

# ### 2. Data preprocessing (cleaning and transforming if needed)

# #### 2.1 Data cleaning 
# - any np.nan values in the raw dataset?
# - any duplicate records?
# - any invalid format?

# In[ ]:


#any dataset containing np.nan values?
sum(soybean.isna().any(axis=1))


# In[ ]:


sum(close.isna().any(axis=1))


# In[ ]:


sum(nearby.isna().any(axis=1))


# ##### Good, no NULL values in the dataset.  Next, let's see if any duplicate records exist?

# In[ ]:


#it's unclear why the dataset has lots of duplicate dates? 
#It might be due to the way the script is aggregating the raw data from the USDA website
print(soybean.shape[0], len(soybean.Date.unique()))


# In[ ]:


print('before drop duplicates: ', soybean.shape[0])
soybean.drop_duplicates(inplace=True)
print('after drop duplicates: ', soybean.shape[0])


# ##### It means although we have duplicate values in the 'Date' column of soybean df, but values in other columns are different so no 'duplicate' records with identical values for all columns exist. Due to the lack of domain knowledge / background info on why several records of the same date exist and which one is correct, we have to live up with it and might average them out later as the best effort to curate it.

# In[ ]:


print('before drop duplicates: ', close.shape[0])
close.drop_duplicates(inplace=True)
print('after drop duplicates: ', close.shape[0])


# In[ ]:


print('before drop duplicates: ', nearby.shape[0])
nearby.drop_duplicates(inplace=True)
print('after drop duplicates: ', nearby.shape[0])


# ##### No duplicate records from all 3 dfs were detected and dropped.

# In[ ]:


#reset df index
soybean.reset_index(drop=True, inplace=True)


# #### 2.2 Data transforming
# - Any time object need to be parsed into datetime?
# - Datasets have been loaded as Panda DataFrame, a good format ready for analysis
# - Later on if we want to do clustering or use some ML algo. sensitive to feature scaling, normalisation is needed to preproess the data

# In[ ]:


#parse Date object type to datetime type
soybean.Date = pd.to_datetime(soybean.Date)


# In[ ]:


#soybean.dtypes['Date']


# In[ ]:


#add 'year' and 'month' new columns to the df to facilitate following DV and analysis
#ref.: Extracting just Month and Year from Pandas Datetime column (Python)
#https://stackoverflow.com/questions/25146121/extracting-just-month-and-year-from-pandas-datetime-column-python
soybean['year'] = soybean.Date.dt.year
soybean['month'] = soybean.Date.dt.month


# In[ ]:


soybean.loc[:3, ['Date','year','month']]


# ### 3. Defining/updating DS Questions 
# At this stage, I will first define my initial DS questions and ask hypothesis, and then explore the datasets. Via exploring the actual data I might need to come back to update or change my questions in multiple interations. Finally I will have confidence that I have asked the right DS questions in the first place.   
# 
# #### Q1: Is there any strong correlation / dependancy of US soybean export on China's deman / import?
# Note: after fast iteration between step 3 and 4, the question is refined to 'What are main factors impacting US Soybean export'? to make it more general and interesting to a broad audience. 
# 
# #### Q2: Which years have similar profiles of crop yield, supply, demand situations? 
# Note: after iteration I decided to make it more specific, focusing on 'Which years have similar profiles of US Soybean production, export and import'?  
# 
# #### Q3: Can we use crop's yield, production, supply, demand and other features to predict market price? 
# 
# <br>
# <br>

# ### 4. Exploratory Study and Data Visualisation

# #### 4.1 Visualise feature correlation (US exports vs China imports)

# In[ ]:


plt.scatter(soybean['United States Exports'], soybean['China Imports'])
plt.xlabel('US Exports')
plt.ylabel('China Imports')
plt.title('US Soybean Exports vs. China Imports')


# ##### It is easy to see an evident relationship between US soybean export and China import over years.

# In[ ]:


plt.scatter(soybean['United States Exports'], soybean['Japan Imports'])
plt.xlabel('US Exports')
plt.ylabel('Japan Imports')
plt.title('US Soybean Exports vs. Japan Imports')


# ##### It can be seen that there is almost no correlation between US soybean export and Japan import, which implies for soybean China is heavily relying on US supply, whereas Japan is independent of US soybean supply.

# #### 4.2 Show time trends of soybean yield, grow and harvest areas or import/export (i.e. temporary-depedent patterns) 

# In[ ]:


#since data contains different number of months for different years, I use the average over months per year as a common factor for comparision
soybean_yearly = soybean.groupby('year').mean()
#drop 'month' column 
soybean_yearly.drop(axis=1, labels=['month'], inplace=True)
#print(soybean_yearly.shape)
#soybean_yearly


# In[ ]:


#pd.unique(soybean_yearly.index)


# #### Show time trends of one variable using bar plot

# In[ ]:


#option A
soybean_yearly.Yield.plot(kind='bar',figsize=(6,4))
#option B
#plt.bar(soybean_yearly.index, soybean_yearly['Yield'])
plt.xlabel('Year')
plt.ylabel('Yield')
plt.title('Soybean Yearly Yield')


# #### Show time trends of multiple variables (of same unit) using plt.plot

# In[ ]:


#option A
def plot_trends1 (df,f1, f2, title):
    #set figure style
    linestyle = ['b--','g-s']
    linewidth = 1.8
    fig = plt.gcf()
    fig.set_size_inches(8, 6)   
    
    #plot multi plots in the same figure 
    #option A (preferred)
    plt.plot(df[f1], linestyle[0], linewidth=linewidth, label=f1)
    plt.plot(df[f2], linestyle[1], linewidth=linewidth, label=f2)
    #option B
    #df[f1].plot(color='r',marker='.', label=f1)
    #df[f2].plot(color='b', marker='*',label=f2)
    
    #set figure annotation info (xlabel, ylabel, title, legend, etc.)
    plt.xlabel('Year')
    plt.title(title+f1+' & ' +f2)  
    plt.legend(loc='best')
    plt.show()


# In[ ]:


plot_trends1(soybean_yearly,'Area Planted','Area Harvested', 'Soybean Yearly ')


# In[ ]:


#Option B
def plot_trends1B (df,f1,f2, title): 

    #set figure style
    color=['green','coral']
    linewidth = 1.8
    fig = plt.gcf()
    fig.set_size_inches(8, 6)

    #plot multi plots in the same figure 
    ax = df[f1].plot(linewidth=linewidth, marker='*', color=color[0])
    df[f2].plot(linewidth=linewidth, marker='s', color=color[1])
    
    #set figure annotation info (xlabel, ylabel, title, legend, xticks, xticklabels with rotation, etc.)
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.index, rotation=45) 
    plt.xlabel('Year')
    plt.title(title+f1+' & ' +f2)  
    plt.legend(loc='best')    
    plt.show()


# In[ ]:


plot_trends1B(soybean_yearly,'Area Planted','Area Harvested', 'Soybean Yearly ')


# ##### Observation: no surprise to see area harvested is proportional to area planted and a little bit less due to various causes on a yearly basis.

# #### Show time trends of multiple variables (of different units) using plt.plot

# In[ ]:


#multiple plots in same figure with one axis via python matplotlib
#ref.: https://stackoverflow.com/questions/43179027/multiple-plots-in-same-figure-with-one-axis-via-python-matplotlib
def plot_trends2 (df,f1,f2,f3,title): 
    #color=['blue','green','red']
    linestyle=['r-.','b--','g-*']
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    fig.set_size_inches(8, 6)

    ax.plot(df[f2], linestyle[1], label=f2)
    ax.plot(df[f3], linestyle[2], label=f3)
    ax2.plot(df[f1], linestyle[0],label=f1)

    ax.set_title(title+f1+' vs. '+f2+' & '+f3)

    ax.legend(loc=2)
    ax2.legend(loc=4)
    plt.show()


# In[ ]:


plot_trends2(soybean_yearly,'Yield','Area Planted','Area Harvested','Soybean Yearly ')


# In[ ]:


plot_trends2(soybean_yearly,'Production','Area Planted','Area Harvested','Soybean Yearly ')


# In[ ]:


plot_trends2(soybean_yearly,'United States Exports','Area Planted','Area Harvested','Soybean Yearly ')


# In[ ]:


plot_trends2(soybean_yearly,'China Exports','Area Planted','Area Harvested','Soybean Yearly ')


# In[ ]:


plot_trends2(soybean_yearly,'China Imports','Area Planted','Area Harvested','Soybean Yearly ')


# ##### Observation: 
# - It can be seen that the soybean yield / production and (US) exports closely depends on and follows the areas planted & harvested in an annual trend. 
# - Very interestingly to look at China: its soybean export has slumped after 2009 and in the meantime its soybean import keeps constant increase in the last 12 years regardless of global yield, which implies China has a huge and lasting demand on soybe but it now heavily relies on global supply to meet the needs. 

# #### 4.3 Similarly we can show month-by-month trends of soybean yield, grow and harvest areas

# In[ ]:


soybean_monthly = soybean.groupby(['year','month']).mean()


# In[ ]:


#soybean_monthly.shape


# In[ ]:


#soybean_monthly.head(3) # this is a multi-index (i.e. multi index levels) df


# In[ ]:


#soybean_monthly.index.names


# In[ ]:


#soybean_monthly.index.levels


# In[ ]:


#soybean_monthly.index.labels


# In[ ]:


#soybean_monthly.index.get_level_values('year')


# In[ ]:


#soybean_monthly.index.get_level_values('month')


# In[ ]:


def plot_trends1B_multi_index (df,f1,f2,title): 

    #set figure style
    color=['green','coral']
    marker=['*','s']
    linewidth = 1.8
    fig = plt.gcf()
    fig.set_size_inches(8, 6)

    #plot multi plots in the same figure 
    #ax = df.Yield.plot(linewidth=linewidth, color=color[0])
    ax = df[f1].plot(linewidth=linewidth, marker=marker[0], color=color[0])
    df[f2].plot(linewidth=linewidth, marker=marker[1], color=color[1])
    
    #set figure annotation info (xlabel, ylabel, title, legend, xticks, xticklabels with rotation, etc.)
    dur = np.arange(len(df.index.labels[0])) # [0,....129]
    #print(dur)
    every10sampling = (dur%10 == 0) #boolean vector for filtering, 1/10 sampling rate
    #print(every10sampling)
    lessdur = dur[every10sampling]
    #print(lessdur)
    
    ax.set_xticks(lessdur)
    
    lvv = df.index.get_level_values('year') #for each row, get its corresponding level value (specified by level name) in a multi index (levels) df!  
    #print(lvv)
    lesslvv = lvv[every10sampling]
    #print(lesslvv)
    
    #ax.set_xticklabels(df.index.labels, rotation=45) #Notice the ; (remove it and see what happens !)
    ax.set_xticklabels(lesslvv, rotation=45)
    
    plt.xlabel('Year')
    plt.title(title+f1+' & ' +f2)  
    plt.legend(loc='best')    
    plt.show()


# In[ ]:


plot_trends1B_multi_index(soybean_monthly,'Area Planted','Area Harvested','Soybean Monthly ')


# In[ ]:


plot_trends1B_multi_index(soybean_monthly, 'United States Exports', 'China Imports','Soybean Monthly ')


# ##### Observation: 
# - Similar to the yearly trend, the monthly trend of the area harvested is closely coupled with the area planted
# - China import is also historically positively correlated to US export trend, but in recent years China import growth rate is becoming higher than US export increase rate, which implies China also possibly sources its soybean supply from other parts of the world

# In[ ]:


def plot_trends2_multi_index (df,f1,f2,f3,title): 
    color=['blue','green','red']
    marker=['*','s','d']
    linewidth=1.8
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    fig.set_size_inches(8, 6)

    ax.plot(range(len(df.index)),df[f2], linewidth=linewidth, marker=marker[1],color=color[1], label=f2)
    ax.plot(range(len(df.index)),df[f3], linewidth=linewidth, marker=marker[2],color=color[2], label=f3)
    ax2.plot(range(len(df.index)),df[f1], linewidth=linewidth, marker=marker[0], color=color[0],  label=f1)

    dur = np.arange(len(df.index.labels[0])) # [0,....129]
    #print(dur)
    every10sampling = (dur%10 == 0) #boolean vector for filtering, 1/10 sampling rate
    #print(every10sampling)
    lessdur = dur[every10sampling]
    #print(lessdur)
    
    ax.set_xticks(lessdur)
    
    lvv = df.index.get_level_values('year') #for each row, get its corresponding level value (specified by level name) in a multi index (levels) df!  
    #print(lvv)
    lesslvv = lvv[every10sampling]
    #print(lesslvv)
    
    #ax.set_xticklabels(df.index.labels, rotation=45) #Notice the ; (remove it and see what happens !)
    ax.set_xticklabels(lesslvv, rotation=45)
    
    ax.set_title(title+f1+' vs. '+f2+' & '+f3)

    ax.legend(loc=2)
    ax2.legend(loc=4)
    plt.show()


# In[ ]:


plot_trends2_multi_index(soybean_monthly,'Yield','Area Planted','Area Harvested','Soybean Monthly ')


# In[ ]:


plot_trends2_multi_index(soybean_monthly,'Production','Area Planted','Area Harvested','Soybean Monthly ')


# In[ ]:


plot_trends2_multi_index(soybean_monthly,'United States Exports','Area Planted','Area Harvested','Soybean Monthly ')


# In[ ]:


plot_trends2_multi_index(soybean_monthly,'China Imports','Area Planted','Area Harvested','Soybean Monthly ')


# ##### Observation:  similar to yearly trends, soybean crop yield/production and exports roughly follows area planted & harvested whereas China imports is in stable growth insensitive to the soybean global yield/production. 

# #### 4.4 People said commodity price is in reverse proportional relationship with its supply. Can we visualise it? 

# In[ ]:


#tranform nearby dataset's dates field to datetime and add year and month fields
nearby.dates = pd.to_datetime(nearby.dates)
nearby['year'] = nearby.dates.dt.year
nearby['month']= nearby.dates.dt.month
#nearby.head(3)


# In[ ]:


#group and average nearby price per year, month
nearby_monthly = nearby.groupby(['year','month']).mean()
#nearby_monthly.head(3)


# In[ ]:


#nearby_monthly.shape


# #### Merge two python (pandas) dataframes by index
# Three options: use merge, join or cancat see: https://stackoverflow.com/questions/40468069/python-pandas-merge-two-dataframes-by-index
#             

# In[ ]:


#Option A: use df1.merge(df2, ...) default is inner join
merged0 = nearby_monthly.merge(soybean_monthly, how='inner', left_index=True, right_index=True)


# In[ ]:


#print(nearby_monthly.shape, soybean_monthly.shape,merged0.shape)


# In[ ]:


#Option B: use df1.join(df1, ...) default is left join
merged = nearby_monthly.join(soybean_monthly, how = 'inner', lsuffix='_x')


# In[ ]:


#print(nearby_monthly.shape, soybean_monthly.shape,merged.shape)


# In[ ]:


#Option C: use concat([df1,df2], axis=1) default is full outer join
merged2 = pd.concat([nearby_monthly,soybean_monthly], join='inner', axis=1)


# In[ ]:


#print(nearby_monthly.shape, soybean_monthly.shape,merged2.shape)


# In[ ]:


#merged.head(5)


# In[ ]:


#merged.tail(5)


# In[ ]:


plot_trends2_multi_index(merged, 'nearby_close', 'Area Planted','Area Harvested', 'Soybean Monthly ')


# ##### Observation:  
# It can be seen nearby_close price is roughly in reverse proportation with area planted&harveste (which corresponds to yield quantity):  
# The higher supply, the lower the price, and vice versa.

# In[ ]:


#group and average nearby price per year
nearby_yearly = nearby.groupby('year').mean()
nearby_yearly.drop(axis=1,labels=['month'],inplace=True)
#nearby_yearly.head(3)


# In[ ]:


#merged with soybean yearly data
merged_yearly = nearby_yearly.merge(soybean_yearly, how='inner', left_index=True, right_index=True)


# In[ ]:


#print(nearby_yearly.shape, soybean_yearly.shape, merged_yearly.shape)


# In[ ]:


#merged_yearly


# In[ ]:


plot_trends2(merged_yearly,'nearby_close','Area Planted','Area Harvested','Soybean Yearly ')


# ##### Observation:  When fluctuation is averaged and smoothed out over year, it is clearer to see that market price is indeed in inverse porportation to the grow areas and quantity of supply

# ### 5 Modelling & Predictive Analysis
# In this section I am going to build statistical and ML models to answer the questions worth potential biz values, such as what factors drive the US soybean export growth, what years have similar situations for soybean yie 

# #### Q1. What are main factors impacting US Soybean export in the history?
# 
# From the section 4 exploratory study and data visualisation, we have seen a strong correlation between US export and China import of soybean.  Is there anything else driving the US soybean export, can we quantify it? 

# In[ ]:


# create a function for plotting a series with index and numeric values

def plot_series(ser, y_label,title):  
    color='m'
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.ylabel(y_label)
    plt.title(title)
    
    ax = ser.plot(linewidth=3.3, color=color)
    ax.set_xticks(range(len(ser)))
    ax.set_xticklabels(ser.index,rotation=90)
    plt.show()


# In[ ]:


plot_series(soybean.corr()['United States Exports'].sort_values(ascending=False)[1:11],'United States Exports','Top 10 Correlations')


# #### Show top 10 features / factors that are relevant to US soybean export

# In[ ]:


top10 = soybean.corr()['United States Exports'].sort_values(ascending=False)[0:10]
top10


# In[ ]:


sub_df = soybean[top10.index]
#sub_df


# In[ ]:


def draw_imshow(df):
    plt.imshow(df, cmap=plt.cm.Blues, interpolation='nearest')
    plt.colorbar()
    tick_marks = [i for i in range(len(df.columns))]
    plt.xticks(tick_marks, df.columns, rotation='vertical')
    plt.yticks(tick_marks, df.columns)


# In[ ]:


draw_imshow(sub_df.corr())


# #### It can be seen:
# 1. US soybean exports heavily depends on its own production (obviously) and world exports & imports, which implies a healthy international trade is important to US soybean export prospective
# 2. US soybean exports are also closely correlated to China imports and domestic consumption, which implies China could be one of the most important buyers of American soybean
# 3. US soybean exports are relevent to the world domestic demand, which implies US soybean exports to many places in the world

# #### Q2. 'Which years have similar profiles of US Soybean production, export and import?
# I use unsupervised learning: clustering KMeans to group similar years together

# #### Select features of interest for clustering

# In[ ]:


USs = soybean_yearly.columns[soybean_yearly.columns.str.contains('United States')]
#USs


# In[ ]:


#subset a df only containing us soybean columns 
df_us = soybean_yearly [USs]
#df_us.head()


# #### Do feature scaling as clustering KMeans sensitive to feature of different units / magnitudes 

# In[ ]:


ss = StandardScaler()
X = ss.fit_transform(df_us)
#print(df_us.shape, X.shape)


# In[ ]:


pd.DataFrame(X, columns=USs).describe().transpose()


# #### Build the KMeans model with training data

# In[ ]:


#set a range of no. of clusters to explore
nc = range(2,11) #2~10

wsse_list = []
km_list = []

for i in nc:
    km=KMeans(n_clusters=i)
    _ = km.fit(X)
    wsse = km.inertia_
    wsse_list.append(wsse)
    km_list.append(km)
#print(wsse_list)
#print(km_list)   


# #### Use data-driven approach (e.g. elbow chart) to find suitable n_clusters value for tradeoff
# - Several methods to select n_clusters, if someone has domain knowledge, that will come in handy to choose a number
# - W/o domain knowledge and context, I will apply data-driven approach to decide the n_clusters value

# In[ ]:


plt.plot(nc,wsse_list)
plt.xlabel('no. of cluster')
plt.ylabel('WSSE (Within-Cluster Sum of Square Errors)')
plt.title('Elbow Curve for Selecting n_clusters value')


# #### From the elbow curve it can be seen when n_clusters increased more than 6, there is no much gain in terms of WSSE reduction. So I choose 6 as the n_clusters for the final model.

# In[ ]:


final_model = km_list[nc.index(6)]
final_model


# #### Show cluster centroids

# In[ ]:


uscentroids = final_model.cluster_centers_
#print(uscentroids.shape, uscentroids)


# #### Show cluster labels

# In[ ]:


print(final_model.labels_)


# In[ ]:


pd.value_counts(final_model.labels_, sort=False)


# #### Visualise the cluster centroids 

# In[ ]:


# create a df of centroids with labels

def centroids_df(names, centroids):
    df = pd.DataFrame(centroids, columns=names)
    df['label'] = df.index
    return df


# In[ ]:


def draw_parallel(df):
    colors = ['r','y', 'g', 'c', 'b', 'm','k']    
    ax = plt.figure(figsize=(12,9)).gca().axes.set_ylim([-3,+3])
    plt.xticks(rotation=90) 
    parallel_coordinates(df, 'label', color = colors, marker='s')


# In[ ]:


centroidsDF = centroids_df(USs, uscentroids)
#centroidsDF


# In[ ]:


draw_parallel(centroidsDF)


# In[ ]:


year_label = zip(df_us.index,final_model.labels_)
#year_label


# In[ ]:


pd.DataFrame(list(year_label), columns=['Year','Label']).transpose()


# #### Analysis:
# - It can be seen groups  (year 2007 ~ 2008) have very similar profiles 
# - The group  (year 2009 ~ 2013) is quite similar to group (2007 - 2008) except in 'United States Beginning Stocks' 
# - The group (year 2015 ~ 2018) are very similar to each other  
# - The group  (year 2014) is an interesting group, similar to group (year 2009 - 2013) in tersm of US DomesticCrush, DomesticTotal, Exports and Ending Stocks and similar to group (year 2015 - 2018)  in terms of US BeginningStocks, Production and Imports. Implies it might be a trasitional year (adjustment period) between two stable patterns 
# - Interestingly, that group patterns matches the soybean yearly yield and market price profiles:  
# 
#     - 2007 - 08: supply started to increase and price started to drop  
#     - 2009 - 13: supply and price were dramatically fluctuating in reverse directions  
#     - 2015 - 18: supply was being in constant increase and price keeps decreasing 

# #### Q3. Can we use crop's yield, production, supply, demand and other features to predict market price? 
# 
# To get numeric prediction according to a set of features, the supervised ML Regression models are used here.

# #### I will set nearby_close as the target to predict and use all other features as predictor candidates in the model 

# In[ ]:


data = merged.copy().drop(axis=1, labels=('nearby_close'))
#data.head()


# In[ ]:


labels = merged['nearby_close']
#labels.head()


# #### Divide data set into train and test sub sets

# In[ ]:


data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=.2, random_state=7)


# In[ ]:


#print(data_train.shape, data_test.shape, label_train.shape, label_test.shape)


# #### A. Train a Linear Regression model with training set and learning algo

# In[ ]:


lm = LinearRegression()
lm.fit(data_train,label_train)


# #### Display which features are most relevant to predict nearby_close price

# In[ ]:


#np.array(zip(data.columns,lm.coef_)).reshape(-1,2)


# In[ ]:


#option A
revf = pd.DataFrame(list(zip(data.columns,lm.coef_)),columns=['feature','coef'])
#option B
#revf = pd.DataFrame(np.array(zip(data.columns,lm.coef_)).reshape(-1,2),columns=['feature','coef'])
revf.coef = revf.coef.astype(np.float64)
#revf


# In[ ]:


#pandas - sort by absolute value without changing the data
#ref.: https://stackoverflow.com/questions/30486263/pandas-sort-by-absolute-value-without-changing-the-data

#top 10 most important features for lm model prediction
revf.reindex(revf.coef.abs().sort_values(ascending=False).index)[:10]


# #### Use fit lm model to predict target price

# In[ ]:


label_pred_lm = lm.predict(data_test)
#print(label_pred_lm.shape, type(label_pred_lm))
#print(label_pred_lm)


# #### Use RMSE to assess the linear regression model accuracy 

# In[ ]:


RMSE_lm = sqrt(mean_squared_error(y_true = label_test, y_pred = label_pred_lm))


# In[ ]:


print(RMSE_lm, label_test.mean(), label_test.std())


# #### Visualise how well the trained Linear Regressor predicts target

# In[ ]:


obs_pred_df_lm = pd.DataFrame({'observ':label_test,'pred':label_pred_lm})


# In[ ]:


opdf_lm = obs_pred_df_lm.reset_index()
#opdf_lm


# In[ ]:


#How to sort a dataFrame in python pandas by two or more columns?
#Ref.: https://stackoverflow.com/questions/17141558/how-to-sort-a-dataframe-in-python-pandas-by-two-or-more-columns
sorted_opdf_lm = opdf_lm.sort_values(['year','month'], ascending=[True,True])


# In[ ]:


sorted_opdf_lm['year_month']=list(zip(sorted_opdf_lm['year'].values,sorted_opdf_lm['month'].values))


# In[ ]:


sorted_opdf_lm.reset_index(drop=True, inplace=True)


# In[ ]:


def plot_pred_vs_obs (df,f1,f2,title): 

    #set figure style
    color=['green','coral']
    marker=['*','s']
    linewidth = 1.8
    fig = plt.gcf()
    fig.set_size_inches(8, 6)

    #plot multi plots in the same figure 
    ax = df[f1].plot(linewidth=linewidth, marker=marker[0], color=color[0])
    df[f2].plot(linewidth=linewidth, marker=marker[1], color=color[1])
    
    #set figure annotation info (xlabel, ylabel, title, legend, xticks, xticklabels with rotation, etc.)
    ax.set_xticks(range(len(df.index)))
    ax.set_ylim([0,1600])
    
    #ax.set_xticklabels(df.index.labels, rotation=45) #Notice the ; (remove it and see what happens !)
    ax.set_xticklabels(df.year_month, rotation=60)
    
    plt.xlabel('(Year, Month)')
    plt.ylabel('nearby_close')
    plt.title(title+f1+' vs. ' +f2)  
    plt.legend(loc='best')    
    plt.show()


# In[ ]:


plot_pred_vs_obs(sorted_opdf_lm,'observ','pred', 'LM Nearby_close ' )


# #### It can be seen Linear Regression did a not too bad job of predicting nearby_close price which is close to the actual observed value.

# #### B. Now let's use Decision Tree Regressor to learn from training data for prediction

# In[ ]:


dt = DecisionTreeRegressor(random_state=123)
dt.fit(data_train,label_train)


# #### Display which features are most relevant to predict nearby_close price

# In[ ]:


feature_relevance = pd.Series(dt.feature_importances_, index=data.columns)


# In[ ]:


#top 10 most important features for DT model prediction
feature_relevance.sort_values(ascending=False)[:10]


# #### Use the tree model trained to predict nearby_close target

# In[ ]:


label_pred_dt = dt.predict(data_test)
#print(label_pred_dt.shape, type(label_pred_dt))
#print(label_pred_dt)


# #### Use RMSE to assess the decision tree regression model accuracy 

# In[ ]:


RMSE_dt = sqrt(mean_squared_error(y_true = label_test, y_pred = label_pred_dt))
RMSE_dt


# In[ ]:


pd.DataFrame({'target mean':label_test.mean(),'target std.':label_test.std(),'Linear regression RMSE':RMSE_lm,'Decisition Tree RMSE':RMSE_dt},index=['Compare'])


# #### Visualise how well Decision Tree Regressor predicts target

# In[ ]:


obs_pred_df_dt = pd.DataFrame({'observ':label_test,'pred':label_pred_dt})
#print(obs_pred_df_dt.head(3))
opdf_dt = obs_pred_df_dt.reset_index()
#print(opdf_dt.head(3))
sorted_opdf_dt = opdf_dt.sort_values(['year','month'], ascending=[True,True])
#print(sorted_opdf_dt.head(3))
sorted_opdf_dt['year_month']=list(zip(sorted_opdf_dt['year'].values,sorted_opdf_dt['month'].values))
#print(sorted_opdf_dt.head(3))
sorted_opdf_dt.reset_index(drop=True, inplace=True)
#print(sorted_opdf_dt.head(3))


# In[ ]:


plot_pred_vs_obs(sorted_opdf_dt,'observ','pred', 'DT Nearby_close ' )


# #### Decision Tree regressor did an even better job of predicting nearby_close values compared to linear regressor accuracy based on test set evaluation

# #### C. Extension: as per reader's feedback, let's use data at time t-1 and label at time t to train and predict nearby price

# In[ ]:


#let's use the whole common period except the last month as new data: 2008-02 ~ 2017-11
#For example: use the data at time t-1 (year: 2008 month: 02) to predict the nearby price at year: 2008 month: 03  
data2 = data[:-1]
print(data.shape, data2.shape)


# In[ ]:


#let's use the whole common period except the first month as new label: 2008-03 ~ 2017-12
labels2 = labels[1:]
print(labels.shape,labels2.shape)


# In[ ]:


#train / rest set split
data_train2, data_test2, label_train2, label_test2 = train_test_split(data2, labels2, test_size=.2, random_state=7)
print(data_train2.shape, data_test2.shape, label_train2.shape, label_test2.shape)


# In[ ]:


#Re-train the DT model using 'new' data
dt2 = DecisionTreeRegressor(random_state=123)
dt2.fit(data_train2,label_train2)


# ##### Display which features are most relevant to predict nearby_close price

# In[ ]:


feature_relevance2 = pd.Series(dt2.feature_importances_, index=data.columns)
#top 10 most important features for DT model prediction
feature_relevance2.sort_values(ascending=False)[:10]


# ##### Use the tree model trained to predict nearby_close target

# In[ ]:


label_pred_dt2 = dt2.predict(data_test2)
print(label_pred_dt2.shape, type(label_pred_dt2))
print(label_pred_dt2)


# ##### Use RMSE to assess the decision tree regression model accuracy 

# In[ ]:


RMSE_dt2 = sqrt(mean_squared_error(y_true = label_test2, y_pred = label_pred_dt2))
RMSE_dt2


# In[ ]:


pd.DataFrame({'Decisition Tree RMSE  (data T to predict price T)':RMSE_dt,'Decisition Tree RMSE (data T-1 to predict price T)':RMSE_dt2},index=['Compare'])


# ##### Visualise how well Decision Tree Regressor predicts target

# In[ ]:


obs_pred_df_dt2 = pd.DataFrame({'observ':label_test2,'pred':label_pred_dt2})
#print(obs_pred_df_dt.head(3))
opdf_dt2 = obs_pred_df_dt2.reset_index()
#print(opdf_dt.head(3))
sorted_opdf_dt2 = opdf_dt2.sort_values(['year','month'], ascending=[True,True])
#print(sorted_opdf_dt.head(3))
sorted_opdf_dt2['year_month']=list(zip(sorted_opdf_dt2['year'].values,sorted_opdf_dt2['month'].values))
#print(sorted_opdf_dt.head(3))
sorted_opdf_dt2.reset_index(drop=True, inplace=True)
#print(sorted_opdf_dt.head(3))


# In[ ]:


plot_pred_vs_obs(sorted_opdf_dt,'observ','pred', 'DT Nearby_close ' )


# #### It can be seen that if we attempt to use data at time t-1 to predict future price at time t, our prediction accuracy degraded significantly (RMSE increased from 45.8 to 125.5). It is likely because for regression model, the further we do extrapolate prediction into the future the less accurate our prediction could be.

# ### 6 Tentative conclusions 
# - There is strong dependcies between soybean yield/production and exports/import, as well as between US soybean export and other countries' import (e.g. China)
# - US soybean biz has gone through 4~6 typical periods,from starting to grow (before 2008), to big flutuations of both yield and prices (2009-2013), to adjustment (2014) and then the stable increasing stage (after 2015)
# - It is reliable to use soybean growth areas, yield/productions and export/import profiles to predict the nearby_close market price,  which is of practical value for financial purpose

# ### 7. Future work
# - Find root cause to global soybean production changes over yearly - why nowadays soybean price is dropping but the production keeps growing?
# - Grasp context, background info and domain knoweldge to find other hidden patterns, trends and agriculture-commercial relationships

# ### 8. Aknowledges & References
# - Thanks Ainslie for collecting the raw data with scripts. W/o it, this analysis cannot be done.
# - Thanks for all those great contributors on Stack Overflow for useful tips for various problems (references embedded).
# - Some plotting and analysis refers to UCSD 'Python for Data Science' course on EDX, credits given.

# In[ ]:




