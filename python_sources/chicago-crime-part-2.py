#!/usr/bin/env python
# coding: utf-8

# # Data Analysis of Chicago Crime Dataset(2012 - 2017)

# ## Importing the dependencies

# In[ ]:


#importing numpy for mathematical operation
import numpy as np

#importing the pandas for handling the data 
import pandas as pd

#importing matplotlib for plotting purpose
import matplotlib.pyplot as plt

#importing seaborn for drawing the countplot and barplot
import seaborn as sns


# ## Reading in the data ##

# In[ ]:


dataset = pd.read_csv('../input/Chicago_Crimes_2012_to_2017.csv')
dataset.head(3)


# ## Drawing in the heatmap for seeing the amount of null values present in the data ##

# In[ ]:


plt.figure(figsize = (10, 10))
sns.heatmap(dataset.isnull())


# In[ ]:


#creting copy of data
df = dataset.copy()

#removing the null values
df = df.dropna()

#removing the unnecessary data for the notebook
df = df.drop(columns = ['Unnamed: 0', 'Block', 'Case Number', 'IUCR', 'X Coordinate', 'Y Coordinate','Updated On','Year','FBI Code', 'Beat', 'Ward','Community Area', 'District'], axis = 1)
df.head(3)


# ## Converting the data into python readable format

# In[ ]:


get_ipython().run_cell_magic('time', '', "df.Date = pd.to_datetime(df.Date, format = '%m/%d/%Y %I:%M:%S %p')")


# In[ ]:


#checking changes done are according to our wish or not
df.head(3)


# In[ ]:


get_ipython().run_cell_magic('time', '', '#assigning date as index of dataset as we will need it in later section for drwing beautiful heatmaps\ndf.index = df.Date')


# In[ ]:


df.head(3)


# ## Extracting description and location_description which are less prone to crime

# In[ ]:


list_of_description = list(df['Description'].value_counts().index[20:])
list_of_location_description = list(df['Location Description'].value_counts().index[20:])


# In[ ]:


#above extracted values are assigned to 'OTHER' category
df.loc[df['Description'].isin(list_of_description), df.columns == 'Description'] = 'OTHER'
df.loc[df['Location Description'].isin(list_of_location_description), df.columns == 'Location Description'] = 'OTHER'


# In[ ]:


df['Description'] = pd.Categorical(df['Description'])
df['Location Description'] = pd.Categorical(df['Location Description'])


# ## Drawing plot of crime after sampling them on monthly basis

# In[ ]:


plt.figure(figsize = (12,3))
df['Date'].resample('M').size().plot(legend = False, color = 'red')
plt.title('Crimes Monthly basis (2012 - 2017)')
plt.xlabel('Month', fontsize = 15)
plt.ylabel('Number of Crimes', fontsize = 15)
plt.show()


# ## Drawing plots of crime after sampling them on daily basis

# In[ ]:


plt.figure(figsize = (12, 4))
df['Date'].resample('M').size().rolling(12).sum().plot(legend = False, color = 'red')
plt.title('Moving Average Crimes Monthly basis (2012 - 2017)')
plt.xlabel('Month', fontsize = 15)
plt.ylabel('Number of Crimes', fontsize = 15)
plt.show()


# ## Drawing moving average plot of crime after sampling on monthly basis

# In[ ]:


plt.figure(figsize = (12, 4))
df['Date'].resample('D').size().rolling(365).sum().plot(legend = False, color = 'red')
plt.title('Moving Average Crimes Daily basis (2012 - 2017)')
plt.xlabel('Month', fontsize = 15)
plt.ylabel('Number of Crimes', fontsize = 15)
plt.show()


# ## Creating a table which consists of types of crimes as columns 

# In[ ]:


crimes_count_date = df.pivot_table('ID', aggfunc=np.size, columns='Primary Type', index=df.index.date, fill_value=0)


# ## Plotting moving average of different types of crimes as subplots

# In[ ]:


crimes_count_date.rolling(365).sum().plot(figsize=(25, 40), subplots=True, layout=(-1, 3), sharex=False, sharey=False)
plt.show()


# ## Drawing bargraph classifying crimes according to day of week

# In[ ]:


plt.figure(figsize = (8, 5))
day = ['Saturday', 'Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday','Sunday']
df.groupby([df.index.dayofweek]).size().plot(kind = 'barh')
plt.title('Number of Crimes Day Wise')
plt.ylabel('Day of Week', fontsize = 14)
plt.yticks(np.arange(7), day)
plt.xlabel('Number of Crimes', fontsize = 13)
plt.show();


# ## Drawing bargraph classifying crimes according to month of year

# In[ ]:


plt.figure(figsize = (8, 6))
day = ['December', 'November', 'October', 'September', 'August', 'July', 'June', 'May',  'April', 'March', 'February', 'January']
df.groupby([df.index.month]).size().plot(kind = 'barh')
plt.title('Number of Crimes Month Wise', fontsize = 16)
plt.ylabel('Month of Year', fontsize = 14)
plt.yticks(np.arange(12), day)
plt.xlabel('Number of Crimes', fontsize = 13)
plt.show();


# ## Drawing countplot showing different types of crimes in descending order

# In[ ]:


plt.figure(figsize = (15, 12))
df.groupby(df['Primary Type']).size().sort_values(ascending = True).plot(kind = 'barh')
plt.title('Types of Crimes in Descending Order', fontsize = 15)
plt.xlabel('Number of Crimes', fontsize = 15)
plt.ylabel('Types of Crimes', fontsize = 15)
plt.show()


# ## Drawing Countplot showing crime incident Location Description

# In[ ]:


plt.figure(figsize = (10, 7))
df.groupby(df['Location Description']).size().sort_values(ascending = True).plot(kind = 'barh')
plt.title('Location Decription of Crimes in Descending Order', fontsize = 15)
plt.xlabel('Number of Crimes', fontsize = 15)
plt.ylabel('Location Description', fontsize = 15)
plt.show()


# ## Creating different pivot table of dataset for drawing heatmap

# In[ ]:


hour_by_location = df.pivot_table(values = 'ID', index = 'Location Description', columns = df.index.hour, aggfunc = np.size).fillna(0)
hour_by_type     = df.pivot_table(values = 'ID', index = 'Primary Type', columns = df.index.hour, aggfunc = np.size).fillna(0)
hour_by_week     = df.pivot_table(values = 'ID', index = df.index.hour, columns = df.index.weekday_name, aggfunc = np.size).fillna(0)
hour_by_week     = hour_by_week.T

day_by_location  = df.pivot_table(values = 'ID', index = 'Location Description', columns = df.index.dayofweek, aggfunc = np.size).fillna(0)
day_by_type      = df.pivot_table(values = 'ID', index = 'Primary Type', columns = df.index.dayofweek, aggfunc = np.size).fillna(0)
type_by_location = df.pivot_table(values = 'ID', index = 'Location Description', columns = 'Primary Type', aggfunc = np.size).fillna(0)


# ## Function for creating heatmap, doing clustering which makes them look nicer

# In[ ]:


from sklearn.cluster import AgglomerativeClustering as AC

def scale_df(df,axis=0):
    '''
    A utility function to scale numerical values (z-scale) to have a mean of zero
    and a unit variance.
    '''
    return (df - df.mean(axis=axis)) / df.std(axis=axis)

def plot_hmap(df, ix=None, cmap='bwr'):
    '''
    A function to plot heatmaps that show temporal patterns
    '''
    if ix is None:
        ix = np.arange(df.shape[0])
    plt.imshow(df.iloc[ix,:], cmap=cmap)
    plt.colorbar(fraction = 0.03)
    plt.yticks(ix, df.index[ix])
    plt.xticks(np.arange(df.shape[1]))
    plt.grid(False)
    plt.show()
    
def scale_and_plot(df, cmap = 'bwr', ix = None):
    '''
    A wrapper function to calculate the scaled values within each row of df and plot_hmap
    '''
    df_marginal_scaled = scale_df(df.T).T
    if ix is None:
        ix = AC(4).fit(df_marginal_scaled).labels_.argsort() # a trick to make better heatmaps
    cap = np.min([np.max(df_marginal_scaled.as_matrix()), np.abs(np.min(df_marginal_scaled.as_matrix()))])
    df_marginal_scaled = np.clip(df_marginal_scaled, -1*cap, cap)
    plot_hmap(df_marginal_scaled, ix=ix, cmap = cmap)
    
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# In[ ]:


sns.clustermap(scale_df(hour_by_type.T).T)


# ## Heatmap showing intensity of different types of Crime according to hour of day

# In[ ]:


plt.figure(figsize = (15, 12))
scale_and_plot(hour_by_type, 'Purples')


# ## Heatmap showing intensity of Crime at particular location according to hour of day

# In[ ]:


plt.figure(figsize = (10, 15))
scale_and_plot(hour_by_location, 'bwr')


# ## Heatmap showing intensity of Crime at particular day according to hour of day

# In[ ]:


plt.figure(figsize = (12, 7))
scale_and_plot(hour_by_week, 'Spectral')


# ## Heatmap showing intensity of Crime at different locations according to day of week

# In[ ]:


plt.figure(figsize = (4, 8))
scale_and_plot(day_by_location, 'Greens')


# ## Heatmap showing intensity of different types of Crimes according to day of week 

# In[ ]:


plt.figure(figsize = (9, 13))
scale_and_plot(day_by_type, 'Blues')


# ## Heatmap showing intensity of different types of Crimes according to different locations

# In[ ]:


df = normalize(type_by_location)
ix = AC(3).fit(df.T).labels_.argsort() # a trick to make better heatmaps
plt.figure(figsize=(10, 15))
plt.imshow(df.T.iloc[ix,:], cmap='PuBuGn')#Reds
plt.colorbar(fraction=0.03)
plt.xticks(np.arange(df.shape[0]), df.index, rotation='vertical')
plt.yticks(np.arange(df.shape[1]), df.columns)
plt.title('Normalized location frequency for each crime')
plt.grid(False)
plt.show()


# In[ ]:




