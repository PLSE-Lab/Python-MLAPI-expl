#!/usr/bin/env python
# coding: utf-8

# # A General View in the 911 Call in the US

# For this capstone project we will be analyzing some 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). The data contains the following fields:
# 
# * lat : String variable, Latitude
# * lng: String variable, Longitude
# * desc: String variable, Description of the Emergency Call
# * zip: String variable, Zipcode
# * title: String variable, Title
# * timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# * twp: String variable, Township
# * addr: String variable, Address
# * e: String variable, Dummy variable (always 1)
# 

# ## Data and Setup

# ____
# ** Import numpy and pandas **

# In[103]:


import numpy as np
import pandas as pd


# ** Import visualization libraries and set %matplotlib inline. **

# In[104]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[105]:


import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# ** Read in the csv file as a dataframe called df **

# In[106]:


df = pd.read_csv('../input/911.csv')


# ** Check the info() of the df **

# In[107]:


df.info()


# ** Check the head of df **

# In[108]:


len(df)


# ** What are the top 5 zipcodes for 911 calls? **

# In[109]:


df['zip'].value_counts().head(5)


# ** What are the top 5 townships (twp) for 911 calls? **

# In[110]:


df['twp'].value_counts().head(5)


# ** Take a look at the 'title' column, how many unique title codes are there? **

# In[111]:


len(df['title'].value_counts())


# ## Creating new features

# ** In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.** 
# 
# **For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. **

# In[112]:


df['Reason'] = df['title'].apply(lambda x: x[:x.find(':')])
df.dropna(inplace=True)


# **Round the latitude and longtitude for geographical distribtuion purpose**
# ****
# **Reduce the length of the dataset in the geographical level from around 86 thousand and use the groupby function to group them to 1.5 thousand**

# In[113]:


dt=df
dt['lat']=dt['lat'].round(2)
dt['lng']=dt['lng'].round(2)
statedist=dt.pivot_table(index=['lat','lng'],columns='Reason',values='e',aggfunc=np.sum)
statedist.reset_index(inplace=True)
statedist.fillna(value=0,inplace=True)
statedist['Distribution']=statedist['EMS'].name+statedist['EMS'].astype(str)+                        '<br>'+statedist['Fire'].name+statedist['Fire'].astype(str)+                        '<br>'+statedist['Traffic'].name+statedist['Traffic'].astype(str)
statedist.columns.rename('',inplace=True) #rename the column 

print('df:',len(df),'','statedist:',len(statedist))
statedist.head()


# **Visualize the distribution of the 911 emergency call in US**

# In[114]:


data = [ dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = statedist['lng'],
        lat = statedist['lat'],
        text = statedist['Distribution'],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.8,
            symbol = 'circle',

            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            )

        ))]

layout = dict(
        title = 'Emergence Call Geographical Distribution',
        #colorbar = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            showlakes = True,

            lakecolor = "rgb(255, 255, 255)",
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

fig = dict( data=data, layout=layout )
iplot(fig)


# ** What is the most common Reason for a 911 call based off of this new column? **

# In[115]:


df['Reason'].value_counts()


# ** Now use seaborn to create a countplot of 911 calls by Reason. **

# In[116]:


sns.countplot(x='Reason',data=df,palette='viridis')


# ** Use [pd.to_datetime](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) to convert the column from strings to DateTime objects. **

# In[117]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# ** Nnow grab specific attributes from a Datetime object by calling them. For example:**
# 
#     time = df['timeStamp'].iloc[0]
#     time.hour
# 
# **Use Jupyter's tab method to explore the various attributes. Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week. Then create these columns based off of the timeStamp column**

# In[118]:


df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['Month'] = df['timeStamp'].apply(lambda x: x.month)
df['Day of Week'] = df['timeStamp'].apply(lambda x: x.dayofweek)


# **Use the .map() with this dictionary to map the actual string names to the day of the week: **
# 
#     dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

# In[119]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)


# ** Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column. **

# In[120]:


sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# ** Now do the same for Month:**

# In[121]:


sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# > ** Something strange about the Plot? **

# In[122]:


# It is missing some months! 9,10, and 11 are not there.


# **** It was missing some Months, let's see if we can maybe fill in this information by plotting the information in another way, possibly a simple line plot that fills in the missing months, in order to do this, we'll need to do some work with pandas...****

# ** Now create a gropuby object called byMonth, where I group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame. **

# In[123]:


byMonth = df.groupby('Month').count()
byMonth.head()


# ** Now create a simple plot off of the dataframe indicating the count of calls per month. **

# In[124]:


# Could be any column
byMonth['twp'].plot()


# ![](http://)** Use seaborn's lmplot() to create a linear fit on the number of calls per month. It needs to reset the index to a column. **

# In[125]:


sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())


# **Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method. ** 

# In[126]:


df['Date']=df['timeStamp'].apply(lambda t: t.date())


# ** Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**

# In[127]:


df.groupby('Date').count()['twp'].plot(figsize=(12,6))
plt.tight_layout()


# ** Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call**

# In[128]:


df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot(figsize=(12,6))
plt.title('Traffic')
plt.tight_layout()


# In[129]:


df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot(figsize=(12,6))
plt.title('Fire')
plt.tight_layout()


# In[130]:


df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot(figsize=(12,6))
plt.title('EMS')
plt.tight_layout()


# 
# ** Now let's move on to creating  heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. **

# In[131]:


dayHour = df.pivot_table(index='Day of Week',columns='Hour',values='e',aggfunc=np.sum)
dayHour.head()


# ** Now create a HeatMap using this new DataFrame. **

# In[132]:


plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')


# ** Now create a clustermap using this DataFrame. **

# In[133]:


sns.clustermap(dayHour,cmap='viridis')


# ** Now repeat these same plots and operations, for a DataFrame that shows the Month as the column. **

# In[134]:


dayMonth = df.pivot_table(index='Day of Week',columns='Month',values='e',aggfunc=np.sum)
dayMonth.head()


# In[135]:


plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='viridis')


# In[136]:


sns.clustermap(dayMonth,cmap='viridis')

