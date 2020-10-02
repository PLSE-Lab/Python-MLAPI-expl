#!/usr/bin/env python
# coding: utf-8

# # Exploratory Analysis
# 
# ## Table of Contents
# 
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Gathering Data</a></li>
# <li><a href="#eda">Assessing Data</a></li>
# <li><a href="#eda">Cleaning Data</a></li>
# <li><a href="#eda">Exploratory analysis</a></li>        
# </ul>

# ## Introduction
# 
# > The dataset has been chosen from the options provided by Udacity.The U.S. Department of Transportation's (DOT) Bureau of Transportation Statistics (BTS) tracks the on-time performance of domestic flights operated by large air carriers. Summary information on the number of on-time, delayed, canceled and diverted flights appears in DOT's monthly Air Travel Consumer Report, published about 30 days after the month's end, as well as in summary tables posted on this website. BTS began collecting details on the causes of flight delays in June 2003. Summary statistics and raw data are made available to the public at the time the Air Travel Consumer Report is released.
# 
# >The datset used here reports flights in the United States, including carriers, arrival and departure delays,in 1987.
# 

# In[ ]:


# Importing relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Gathering data

# In[ ]:


# Reading the unzipped file into a pandas dataframe   
df=pd.read_csv('../input/1987.csv')


# ## Assessing data

# In[ ]:


# First few columns of the dataset
df.head(5)


# In[ ]:


# Info of the dataset
df.info()


# In[ ]:


#Descriptive statistics for the dataset
df.describe()


# ### Quality issues
# 
# - Numerous columns with null values viz. TaxiIn,TaxiOut,TailNum etc.
# - Null values in columns of ArrTime.
# - Null values in ArrDelay and DepDelay.
# - An entire column for representing Year.
# - Erroneous datatype of DayOfWeek,DayofMonth,Month,FlightNum columns.
# 
# ### Tidiness issues
# 
# - Improper representation of time in ArrTime,CRSArrTime,DepTime,CRSDepTime columns.

# ## Cleaning data

# In[ ]:


# Creating a copy of dataset before cleaning
df_new=df.copy()


# #### Define:
# 
# > Dropping all columns with complete null values as no information can be conveyed from them and they only occupy memory.

# #### Code

# In[ ]:


df_new=df_new.drop(['AirTime','TaxiIn','TaxiOut','Diverted','CarrierDelay','CancellationCode','WeatherDelay',
             'NASDelay','SecurityDelay','LateAircraftDelay','TailNum'],axis=1)


# #### Test

# In[ ]:


df_new.describe()


# #### Define:
#  
# > Dropping records with null ArrTime as no information can be conveyed also it cannnot be used for analysis.

# #### Code

# In[ ]:


df_new.drop(df_new[df_new.ArrTime.isnull()].index,inplace=True)


# #### Test

# In[ ]:


sum(df_new.ArrTime.isnull())


# In[ ]:


sum(df_new.ArrDelay.isnull())


# In[ ]:


sum(df_new.DepDelay.isnull())


# In[ ]:


sum(df_new.DepTime.isnull())


# In[ ]:


df_new.info()


# #### Define:
#  
# > Representing the Arrtime,CRSArrTime,DepTime,CRSDepTime columns in hh:mm:ss format.

# In[ ]:


df_new.sample(5)


# #### Code

# In[ ]:


#Converting the columns into hh:mm:ss format 
s = df_new['CRSArrTime'].astype(int).astype(str).str.zfill(4)
df_new['arrtime'] = s.str[:2] + ':' + s.str[2:] + ':00'
df_new['Schduled_Arr'] = pd.to_timedelta(df_new['arrtime'])
s = df_new['ArrTime'].astype(int).astype(str).str.zfill(4)
df_new['arrtime'] = s.str[:2] + ':' + s.str[2:] + ':00'
df_new['Actual_arr'] = pd.to_timedelta(df_new['arrtime'])
s = df_new['CRSDepTime'].astype(int).astype(str).str.zfill(4)
df_new['arrtime'] = s.str[:2] + ':' + s.str[2:] + ':00'
df_new['Schduled_dep'] = pd.to_timedelta(df_new['arrtime'])
s = df_new['DepTime'].astype(int).astype(str).str.zfill(4)
df_new['arrtime'] = s.str[:2] + ':' + s.str[2:] + ':00'
df_new['Actual_dep'] = pd.to_timedelta(df_new['arrtime'])


# In[ ]:


df_new.sample(5)


# In[ ]:


# Dropping the previous columns
df_new=df_new.drop(['DepTime','arrtime','CRSDepTime','ArrTime','CRSArrTime'],axis=1)


# In[ ]:


df_new.head(5)


# #### Define:
# 
# > Converting the DayOfWeek,DayofMonth,Month,FlightNum columns into appropriate type.

# #### Code

# In[ ]:


df_new['Month']=df_new['Month'].astype(str)
df_new['DayofMonth']=df_new['DayofMonth'].astype(str)
df_new['DayOfWeek']=df_new['DayOfWeek'].astype(str)


# In[ ]:


df_new['FlightNum']=df_new['FlightNum'].astype(str)


# #### Test

# In[ ]:


df_new.info()


# #### Define:
# 
# > As already conveyed in the beginning of notebook that only 1987 year dataset has been taken into account due to memory conctraint, so dropping the year column.

# #### Code

# In[ ]:


df_new=df_new.drop(['Year'],axis=1)


# #### Test

# In[ ]:


df_new.info()


# ### Combining the two delays arrival delay and departure delay columns into a single column total delay.

# In[ ]:


df_new['Total_Delay']=df_new['ArrDelay']+df_new['DepDelay']


# #### Define:
# 
# > Actual elapsed time represents the time of travel which cannot be negative so dropping off all the records having negative actual elapsed time.

# #### Code

# In[ ]:


df_new.drop(df_new[(df_new['ActualElapsedTime'] <= 0)].index,inplace=True)
# Resetting the index
df_new=df_new.reset_index(drop=True)


# #### Test

# In[ ]:


df_new.query('ActualElapsedTime <=0')


# #### Define:
# 
# > Dropping the cancelled column as the cleaned dataset does not contain any cancelled flight.

# #### Code

# In[ ]:


df_new=df_new.drop(['Cancelled'],axis=1)


# #### Test

# In[ ]:


df_new.columns


# In[ ]:


# Saving the cleaned data with a new csv file name.
df_new.to_csv('1987_modified.csv')


# ### Exploratory Analysis

# #### Plotting the distribution of Total_Delay

# In[ ]:


plt.figure(figsize=[8,5])
bins=np.arange(0,df_new['Total_Delay'].max()+10,100)
plt.hist(data=df_new,x='Total_Delay',bins=bins);


# > From the above histogram it is clear that most of the delays are distributed around zero.

# ### Total Delay vs UniqueCarrier

# In[ ]:


plt.figure(figsize=[8,5])
sb.barplot(data=df_new,x='UniqueCarrier',y='Total_Delay');


# > From the above barplot it clear that PS plane carrier has major contribution the delays followed by CO and AS.

# ### Violinplot for the same above plot for a better insight.

# In[ ]:


#plt.figure(figsize=[20,20])
sb.violinplot(data=df_new,x='UniqueCarrier',y='Total_Delay',color=sb.color_palette()[0],inner='quartile');
#


# > The violin plot clearly indicates that median for each carrier is approximately zero.
# 
# > 'CO' carrier has the most no of outliers.

# ### Pie chart depicting which carrier constitutes major proportion in the dataset.

# In[ ]:


b=df_new['UniqueCarrier'].value_counts()
b.plot(kind='pie',figsize=(20,10),legend=True)
plt.legend(loc=6,bbox_to_anchor=(1.0,0.5));


# > From the above pie chart it is clearly evident that DL carrier has most flights in our dataset and PA(1) the least.

# ### Scatterplot for Distance vs Total Delay

# In[ ]:


plt.figure(figsize=[8,5])
sb.regplot(data=df_new,x='Total_Delay',y='Distance',fit_reg=False);


# In[ ]:


df_new['Distance'].corr(df_new['Total_Delay'])


# > The scatterplot above does not depict any clear relation between distance and total delay as the data points are present as clusters.The correlation coefficient is also calculated above,which clearly shows that theere's hardly any correlation between distance and total delay.
# 
# > The majority of delays are clustered between 0 to 3000 miles distance.

# ### Scatterplot for Distance vs Actual Elapsed time in minutes.

# In[ ]:


sb.regplot(data=df_new,y='Distance',x='ActualElapsedTime',fit_reg=False)


# In[ ]:


df_new['Distance'].corr(df_new['ActualElapsedTime'])


# > The scatterplot for distance vs actual elapsed time does depict some sort of linear relation.
# 
# > The correlation coefficient calculated above also has a high value of 0.97 which shows that there's strong positive correlation between distance and actual elapsed time.

# ### Performing regression analysis for the above scatter plot and plotting the line of best fit.

# In[ ]:


# Importing releavant libraries
import statsmodels.api as sm
# Declaring the dependent and the independent variables
line_reg=df_new.drop(df_new[df_new.Distance.isnull()].index)
line_reg=line_reg.reset_index(drop=True)
x1=line_reg['Distance']
y=line_reg['ActualElapsedTime']


# #### Regression itself

# In[ ]:


x=sm.add_constant(x1)
results=sm.OLS(y,x).fit()
results.summary()


# > The p values clearly indicate that the coeficients are significant.

# > Plotting the regression line on the initial scatter

# In[ ]:


sb.set()
plt.scatter(x1,y)
y1=0.1207*x1+30.8584
fig=plt.plot(x1,y1,lw=4,c='red',label='regression line')
plt.show()


# In[ ]:


sb.regplot(data=df_new,y='Distance',x='ArrDelay',fit_reg=True);


# In[ ]:


df_new['Distance'].corr(df_new['ArrDelay'])


# > The scatterplot above does not depict any clear relation between distance and arrival delay as the data points are present as clusters.The correlation coefficient is also calculated above,which clearly shows that theere's hardly any correlation between distance and total delay.
# 
# > Also the scatter plot is identical to that of distance vs total delay.

# In[ ]:


sb.regplot(data=df_new,y='Distance',x='DepDelay');


# In[ ]:


df_new['Distance'].corr(df_new['ArrDelay'])


# > The scatterplot above does not depict any clear relation between distance and departure delay as the data points are present as clusters.The correlation coefficient is also calculated above which close to zero,which clearly shows that there's hardly any correlation between distance and total delay.
# 
# > Also the scatter plot is identical to that of distance vs total delay.

# ### Scatterplot for distance vs total delay on month basis

# Here Month = 1 refers to January Month =2 as February and so on upto Month =12 as December.

# In[ ]:


g=sb.FacetGrid(data=df_new,col='Month',margin_titles=True)
g.map(plt.scatter,'Distance','Total_Delay')
g.add_legend();


# In[ ]:


df_new[(df_new['Month'] == "10")]['Total_Delay'].corr(df_new[(df_new['Month'] == "10")]['Distance'])


# In[ ]:


df_new[(df_new['Month'] == "11")]['Total_Delay'].corr(df_new[(df_new['Month'] == "11")]['Distance'])


# In[ ]:


df_new[(df_new['Month'] == "12")]['Total_Delay'].corr(df_new[(df_new['Month'] == "12")]['Distance'])


# > For each of the above scatter plot correlation coefficient has been calculated.
# 
# > The correlation coefficient for each of them is quite close to zero which depicts that there is no relation between diatance and total delay.

# ### Barplot for distribution of unique carrier with total on monthly basis

# In[ ]:


g=sb.FacetGrid(data=df_new,col='Month',col_wrap=1,margin_titles=True,sharex=False,sharey=False,size=7)
g.map(sb.barplot,'UniqueCarrier','Total_Delay')
g.add_legend();


# > The plot above shows that most of the delays in the month of October and November are of flights belonging to PS.
# Only in the month of December CO carrier has the majority of delays.
# 
# > Also one interesting fact that the out of the three months Month 11 i.e. November has the most number of delays.

# ### Scatterplot for distance vs total delay grouped by day of the week
# 
# > Here day of week =1 depicts Monday similarly upto 7 where 7 depicts Sunday.

# In[ ]:


g=sb.FacetGrid(data=df_new,col='DayOfWeek',margin_titles=True,col_wrap=3,height=5,sharex=False,sharey=False)
g.map(plt.scatter,'Distance','Total_Delay')
g.add_legend();


# In[ ]:


print(df_new[(df_new['DayOfWeek'] == "1")]['Total_Delay'].corr(df_new[(df_new['DayOfWeek'] == "1")]['Distance']))
print(df_new[(df_new['DayOfWeek'] == "2")]['Total_Delay'].corr(df_new[(df_new['DayOfWeek'] == "2")]['Distance']))
print(df_new[(df_new['DayOfWeek'] == "3")]['Total_Delay'].corr(df_new[(df_new['DayOfWeek'] == "3")]['Distance']))
print(df_new[(df_new['DayOfWeek'] == "4")]['Total_Delay'].corr(df_new[(df_new['DayOfWeek'] == "4")]['Distance']))
print(df_new[(df_new['DayOfWeek'] == "5")]['Total_Delay'].corr(df_new[(df_new['DayOfWeek'] == "5")]['Distance']))
print(df_new[(df_new['DayOfWeek'] == "6")]['Total_Delay'].corr(df_new[(df_new['DayOfWeek'] == "6")]['Distance']))
print(df_new[(df_new['DayOfWeek'] == "7")]['Total_Delay'].corr(df_new[(df_new['DayOfWeek'] == "7")]['Distance']))


# > For each of the above scatter plot correlation coefficient has been calculated.
# 
# > The correlation coefficient for each of them is quite close to zero which depicts that there is no relation between diatance and total delay.

# ## Cluster Analysis

# #### Performing cluster analysis on distance vs total delay

# In[ ]:


from sklearn.cluster import KMeans


# > Separating the dataset into two clusters initially

# In[ ]:


#Creating a copy of the datset
x=df_new.copy()
# Dropping all columns with categorical values
x=x.drop(['Schduled_dep','Actual_dep','Actual_arr','Schduled_Arr','Dest','Origin','UniqueCarrier'],axis=1)
# Dropping the rows with null values of distance
x=x.drop(x[x.Distance.isnull()].index)
x=x.reset_index(drop=True)


# In[ ]:


# creating a k-means object with 2 clusters
kmeans=KMeans(2)
# fit the data
identified_clusters=kmeans.fit_predict(x)
# predicting the cluster for each observation
x['cluster']=identified_clusters


# In[ ]:


# creating a scatter plot based on two features (distance and total delay)
plt.scatter(data=x,x='Total_Delay',y='Distance',c='cluster',cmap='rainbow')
plt.show();


# #### Using Elbow Method for obtaining an optimum number of clusters.

# In[ ]:


# WCSS is the within cluster sum of squares elbow method used for calculating the optimum number of clusters for the given scatter plot
wcss=[]
for i in range(1,8):
    kmeans=KMeans(i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[ ]:


no=range(1,8)
plt.plot(no,wcss);


# > As it is evident from the above graph seven is the optimal number of clusters for which within cluster sum of squares is minimum(WCSS). 
# 
# > Thereby the scatter plot with seven clusters is plotted below.

# In[ ]:


# creating a k-means object with 7 optimum clusters 
kmeans=KMeans(7)
# fit the data
identified_clusters=kmeans.fit_predict(x)
# predicting the cluster for each observation
x['cluster']=identified_clusters
# creating a scatter plot based on two features (distance and total delay)
plt.scatter(data=x,x='Total_Delay',y='Distance',c='cluster',cmap='rainbow')
plt.show();


# ### Websites referred
# 
# > https://stackoverflow.com/
# 
# > https://pandas.pydata.org/
