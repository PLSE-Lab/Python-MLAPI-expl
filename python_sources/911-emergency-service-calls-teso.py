#!/usr/bin/env python
# coding: utf-8

# # DATA ANYALYSIS AND VISUALIZATION WITH PYTHON
# #### <div align='right'>By Dr. Kuleafenu Joachim </div>
# #### <div align='right'>Email kuleafenujoachim@gmail.com</div>

# # 911 Calls Capstone Project 

# ### THIS DATASET IS FROM THE ONE OF THE BIGGEST DATA SCIENCE PLATFORM, KAGGLE!

# ### With this grand walk-through project we will dive depeer into:
#    ##### * How to wrangle data set with pandas library
#    ##### * Do scientific and statistical analysis with numpy library
#    ##### * How to visualize data with matplotlib and seaborn capabilities
#    ##### * How to use scikit learn library to enhance data visualization
#    ##### * and alot more!!!

# You can download this data set from [Kaggle](https://www.kaggle.com/mchirico/montcoalert).
# 
# A little info of the data set:
# 
# The data contains the following fields:
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

# In[ ]:


# The script below to add borders to dataframe


# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<style type='text/css'>\ntable.dataframe th, table.dataframe td{\n    border: 1px  black solid !important;\n    color: black !important;\n}\n</style>")


# # 1). Lets kick start by importing various libraries

# ___
# ** Import numpy and pandas **

# In[ ]:


import numpy as np
import pandas as pd


# ** Import visualization libraries and set %matplotlib inline. **

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# ** Read in the csv file as a dataframe called df **

# In[ ]:


df = pd.read_csv('../input/911-calls/911.csv')


# # 2).Check the distribution of the data set

# **use the info() function to get information about the various fields**

# In[ ]:


df.info()


# **The describe() function helps you get basic statical knowledge about the data**

# In[ ]:


df.describe(include='all')


# ### some observations:
#        **There are nine different columns**    
#        **lat,lng and zip are of float data type**    
#        **e has integer type and all the rest are of string type**    
#        **The total 911 calls recorded is 99492**    
#        **The frequent appearing town is LOWER MERION**    

# # 3). Now let check the head of the data and explore more!

# ** This is the firt 5 rows of the data set **

# In[ ]:


df.head(5)


# ### We are going to explore the data to answer some questions like:
#    **What are the top zip codes for 911 calls?**  
#    **What are the top towships for 911 calls?**  
#    **How many unique title codes are there?**  

# ## Basic Questions

# ** top 5 zipcodes for 911 calls? **

# In[ ]:


df['zip'].value_counts().head(5)


# ** What are the top 5 townships (twp) for 911 calls? **

# In[ ]:


df['twp'].value_counts().head()


# ** Take a look at the 'title' column, how many unique title codes are there? **

# In[ ]:


df['title'].nunique()


# ## Creating new features

# ** In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.** 
# 
# **For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. **

# In[ ]:


df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])


# ** What is the most common Reason for a 911 call based off of this new column? **

# In[ ]:


for i ,j in df['Reason'].value_counts().items():
    print(str(j)+' people call for '+str(i)+' representing '+str(np.round((j/df.shape[0])*100,2))+'%')


# ** Now we use seaborn to create a countplot of 911 calls by Reason. **

# In[ ]:


# sns.countplot(x='Reason',data=df,palette='viridis',hue='Reason')
# plt.legend(loc=1,bbox_to_anchor=(0.9,0.5,0.4,0.5))
fig,ax = plt.subplots()
num,names = df['Reason'].value_counts().values, df['Reason'].value_counts().index
ax.pie(num,labels=names,autopct='%1.2f%%',shadow=True,radius=2,startangle=90);


# ### Some Observations:
#    - The call for Emergency reasons is the highest,followed by Traffic then Fire

# ___
# ** Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column? **

# In[ ]:


type(df['timeStamp'].iloc[0])


# ** You should have seen that these timestamps are still strings. Use [pd.to_datetime](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) to convert the column from strings to DateTime objects. **

# In[ ]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# ** You can now grab specific attributes from a Datetime object by calling them. For example:**
# 
#     time = df['timeStamp'].iloc[0]
#     time.hour
# 
# **You can use Jupyter's tab method to explore the various attributes you can call. Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week. You will create these columns based off of the timeStamp column.**

# In[ ]:


df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)


# In[ ]:


from datetime import datetime as dt


# ** Notice how the Day of Week is an integer 0-6. We will use the .map() with this dictionary to map the actual string names to the day of the week: **
# 
#     dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

# In[ ]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['Day of Week'] = df['Day of Week'].map(dmap)


# ** Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column. **

# In[ ]:


sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)


# # Some observations:
#    - The plot above give almost similar frequency among the day of the week  
#    - Emergency calls lead all of the days followed by Traffic then Fire outbreak  
#    - Careful observation reveals that on Sundays all of the 3 posibilities decreases in number

# ** Now do the same for Month:**

# In[ ]:


# m_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
# df['Month']=df['Month'].map(m_map)


# In[ ]:


sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)


# ** Did you notice something strange about the Plot? **

# In[ ]:


# It is missing some months! 9,10, and 11 are not there.


# **You should have noticed it was missing some Months, let's see if we can maybe fill in this information by plotting the information in another way, possibly a simple line plot that fills in the missing months, in order to do this, we'll need to do some work with pandas...**  
# 
# **Now create a groupby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame.**

# In[ ]:


byMonth = df.groupby('Month').count()
byMonth.head()


# ** Now create a simple plot of the dataframe indicating the count of calls per month. **

# In[ ]:


# Could be any column
byMonth['twp'].plot()


# In[ ]:


# byMonth = df.groupby('Month').count()
# by_week = df.groupby('Day of Week').count()
# fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5),sharey=False)

# axes[0].plot('Month','twp',data=byMonth.reset_index())
# axes[1].plot('Day of Week','twp',data=by_week.reset_index())

# # x1 = sns.lineplot('Month','twp',data=byMonth.reset_index())
# # x2 = sns.lineplot('Day of Week','twp',data=by_week.reset_index(),sharey=x1)
# axes[0].set_xlabel('Month')
# axes[0].set_ylabel('Num of Calls')


# ** Now lets use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column. **

# In[ ]:


sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())


# **Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method. ** 

# In[ ]:


df['Date']=df['timeStamp'].apply(lambda t: t.date())
df['Date'] = pd.to_datetime(df['Date'])


# ** Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**

# In[ ]:


plt.figure(figsize=(10,5))
df.groupby('Date').count()['twp'].plot()
plt.tight_layout()


# ### Hypothesis:
#    - Lower Merion Makes the most 911 calls, we can predict that since the most frequent reason for all township is emergency purposes,most people from lower merion also call for emergency reasons.

# In[ ]:


plt.figure(figsize=(10,5))
df.groupby('twp').count().sort_values('lat',ascending=False)
df[df['twp'] == 'LOWER MERION']
sns.countplot(x='Reason',data=df[df['twp'] == 'LOWER MERION'],hue='Reason')
plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.7, 0.9))


# ### some observation:
#    - It turn out to be that people from Lower Merion call to report Traffic more than emergency reasons

# ** Now we will be creating three similar line plots each for a 911 call reason**  
# ** We can do this by using the pandas data visualization capabilities**

# In[ ]:


df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# In[ ]:


df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()


# In[ ]:


df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# ### Some observations:
#    - For the fire visualization, march recorded the highest fire incident according to the number of calls received for
#     that reason.
#    - From the Emergency chart, the 911 calls reduced significantly in the month of May and prior to the end of August.
#    - Between the month of January and February, high rate of Traffic incidents were recorded.This is obvious because it will be xmas,
#     and many people will be on holidays which may create tension on the various streets.

# ____
# ** Now let's move on to creating  heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but I would recommend trying to combine groupby with an [unstack](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.unstack.html) method. Reference the solutions if you get stuck on this!**

# In[ ]:


dayHour = df.groupby(['Day of Week','Hour']).count()['twp'].unstack()


# ** Now create a HeatMap using this new DataFrame. **

# In[ ]:


plt.figure(figsize=(12,6))
cmap = sns.cubehelix_palette(light=2, as_cmap=True)
sns.heatmap(dayHour,cmap=cmap)


# ** Now create a clustermap using this DataFrame. **

# In[ ]:


sns.clustermap(dayHour,cmap=cmap)


# ### Some Observations:
#    - People mostly call the police at the 16th and 17th hour of the day.This is the time most workers closes from their jobs
#    - Sundays and Saturdays were recorded the least number of calls throughout the period.
#     

# ** Now we will repeat these same plots and operations, for a DataFrame that shows the Month as the column. **

# In[ ]:


# dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth = pd.pivot_table(df,index='Day of Week',values='Reason',columns='Month',aggfunc='count')


# In[ ]:


plt.figure(figsize=(17,6))
sns.heatmap(dayMonth,cmap=cmap,annot=False)


# ### Some Observation:
#    - The heatmap clearly shows that significant number of people call on Saturdays in the month of January

# In[ ]:


# df[df['Month'] == 1]['Reason'].value_counts().plot()


# In[ ]:


# plt.figure(figsize=(12,10))
# month_day = df.groupby(['Month','Day of Week']).count().reset_index()
# month_day[['Month','Day of Week','e']].plot('Month',figsize=(15,9))
# fig = plt.gcf()
# fig.figsize\
df.head()


# - We will create a new feature called incident, the title column is made up of two information per row that is before
# and after the colon(:).
# - We will use the lambda expression to extract the 'incident' from the title column to form a new column.
# - After the creation of the column, we will then use the seaborn heatmap to visualize incident against month.

# In[ ]:


df['incident'] = df['title'].apply(lambda title: title.split(':')[1])


# In[ ]:


plt.figure(figsize=(12,12))
heat_reason = df.groupby(['incident','Month']).count()['Reason'].unstack()
cm = sns.cubehelix_palette(light=2, as_cmap=True)
sns.heatmap(heat_reason,cmap=cm)


# ### Observation:
#    - The dip colored horizontal bar is vehicle accident incident. This means that above all the reasons for the 911 calls,
#     vehicle accident appears to be to most frequent recorded one.
#    - Most of the calls were recorded in January,May and June.
#    - Disabled Vehicle is the second most recorded incident.Disabled vehicle probably causes Traffic so the reason for this
#    will probably be 'Traffic'.

# In[ ]:


# Lets silent the system warning by putting it on silence
import warnings as wn
wn.filterwarnings('ignore')


# In[ ]:


We now create a new column called Problem 


# In[ ]:


df['Problem'] = df['title'].apply(lambda x: x.split(':')[1])


# # We now want to get the top ten problems reported by the callers in each category be it in Emergency,Traffic and Fire

# In[ ]:


# We can do this by creating a function to extract the data and draw a count plot for all the three titles
# NB: We will set the default to Emergency

def top_ten(title='EMS'):
    # Using boolean masking to get only emergency title
    emerg_df = df[df['Reason'] == title]
    top_10_list = list(emerg_df.groupby('incident').count().sort_values('twp',ascending=False).head(10).index)

    top_df = emerg_df['incident'].apply(lambda x: x in top_10_list)
    
    # Creating a countplot for the top ten problems
    fig,ax = plt.subplots(figsize=(10,5))
    ax = sns.countplot('incident',data=emerg_df[top_df])
    
    # set title
    ax.set_title(title,fontdict={'fontsize':15})
    ax.set_xlabel('Incident',fontsize=15)
    
    # We now use the ax.get_xticklabels() to get the tick labels of the x axis
    label = [i for i in ax.get_xticklabels()]
    ax.set_xticklabels([lab.get_text() for lab in label])
    
    # We can use this for loop to prevent the labels from overlapping
    for xlabel in ax.get_xticklabels():
        xlabel.set_rotation(90)
    plt.show()
    print('\n\n')
        
# Invoking to draw a countplot for Emergency
top_ten()
# For Traffic
top_ten('Fire')
# For Fire
top_ten('Traffic')


# ### Some Observations:
#    - Taking the Emergency first,Cardiac,Respiration and Fall victim are the most frequently calls receive.
#    - Taking the Fire, 'Fire Alarm' is the most reported.
#    - Taking the Traffic it is observed that Vehicle accident causes traffic than any other reason.

# ### We are going to take one incident with the highest frequency in all the three categories and use seaborn heatmap to visualize it.
# ### we want to know the time and day this happens

# In[ ]:


# This function will take care of the plot, the default parameter is respiratory emergency
def create_heatmap(problem=' RESPIRATORY EMERGENCY'):
    (df['incident'].value_counts().sort_values(ascending=False)).index
    resp_df = df[df['incident'] == problem].groupby(['Day of Week','Hour']).count()['twp'].unstack()
    
    cm = sns.cubehelix_palette(light=2, as_cmap=True)
    
    fig,ax = plt.subplots(figsize=(12,6))
    ax = sns.heatmap(resp_df,cmap=cm)
    ax.set_title(problem,fontdict={'fontsize':15})
    ax.set_xlabel('Hour',fontdict={'fontsize':15})
    ax.set_ylabel('Day of Week',fontdict={'fontsize':15})
    plt.show()
    print('\n\n')

# Create heatmap for Respiratory emergency
create_heatmap()
# Create heatmap for Vehicle Accident
create_heatmap(' VEHICLE ACCIDENT -')
# Create heatmap for Fire Alarm
create_heatmap(' FIRE ALARM')


# ### Some Observations:
#    - Mondays,Tuesdays and Wednesdays in the hour of 9 and 10 are the period that people call the most to report for Respiratory Cases.
#    - From the heatmap Vehicle accident happens mostly on Thursdays in the 17th hour.
#    - Fire Alarm occurs the most on friday the hour of 11 am and thursdays the hour of 13.

# # Now lets do some times series analyses with our data

# **We are now going to restructure our data using the pandas pivot_table().**  
# **we will make the date the index and title each being a column.**

# In[ ]:


# df[df['title'] == 'EMS: ASSAULT VICTIM']
piv_df = pd.pivot_table(data=df,index='Date',columns='title',values='e',aggfunc=np.sum)
piv_df = piv_df.resample(rule='W',how=np.sum).reset_index()
piv_df.head()


# ### Lets create function to create our line plots 
# ### we use scikit learn library to create line of best fits in the graph

# In[ ]:


from sklearn import linear_model
import matplotlib.lines as mlines

def single_plot(category='EMS: ASSAULT VICTIM'):

    
    fig, ax = plt.subplots()

    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)  



    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left() 
    plt.xticks(fontsize=12) 



    # Build Linear Fit
    Y = piv_df[category].values.reshape(-1,1)
    X=np.arange(Y.shape[0]).reshape(-1,1)
    model = linear_model.LinearRegression()
    model.fit(X,Y)
    m = model.coef_[0][0]
    c = model.intercept_[0]
    ax.plot(piv_df['Date'],model.predict(X), color='blue',
             linewidth=2)
    blue_line = mlines.Line2D([], [], color='blue', label='Linear Fit: y = %2.2fx + %2.2f' % (m,c))
    

    
    # Robustly fit linear model with RANSAC algorithm
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),random_state=23)
    model_ransac.fit(X, Y)
    mr = model_ransac.estimator_.coef_[0][0]
    cr = model_ransac.estimator_.intercept_[0]
    ax.plot(piv_df['Date'],model_ransac.predict(X), color='green',
             linewidth=2)
    green_line = mlines.Line2D([], [], color='green', label='RANSAC Fit: y = %2.2fx + %2.2f' % (mr,cr))


    
    ax.legend(handles=[blue_line,green_line], loc='best')
    

    ax.plot_date(piv_df['Date'], piv_df[category],'k')
    ax.plot_date(piv_df['Date'], piv_df[category],'ro')


    ax.set_title(category)
    fig.autofmt_xdate()
    plt.show()
    print('\n')


    
def embeded_plot(cat1='EMS: ASSAULT VICTIM',cat2='EMS: VEHICLE ACCIDENT'):
    
    fig, ax = plt.subplots()

    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)  



    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left() 
    plt.xticks(fontsize=12) 

    

    ax.plot_date(piv_df['Date'], piv_df[cat1],'k')
    ax.plot_date(piv_df['Date'], piv_df[cat1],'ro')


    ax.plot_date(piv_df['Date'], piv_df[cat2],'g')
    ax.plot_date(piv_df['Date'], piv_df[cat2],'bo')


    
    
# Build Linear Fit
    
    # cat 1
    Y = piv_df[cat1].values.reshape(-1,1)
    X=np.arange(Y.shape[0]).reshape(-1,1)
    model = linear_model.LinearRegression()
    model.fit(X,Y)
    m = model.coef_[0][0]
    c = model.intercept_[0]
    ax.plot(piv_df['Date'],model.predict(X), color='black',
             linewidth=2)
    
    black_line = mlines.Line2D([], [], color='black', marker='o',markerfacecolor='darkred',
                               markersize=7,
                               label='%s, y = %2.2fx + %2.2f' % (cat1,m,c))
  
    # cat 2
    Y = piv_df[cat2].values.reshape(-1,1)
    X=np.arange(Y.shape[0]).reshape(-1,1)
    model = linear_model.LinearRegression()
    model.fit(X,Y)
    m = model.coef_[0][0]
    c = model.intercept_[0]
    ax.plot(piv_df['Date'],model.predict(X), color='green',
             linewidth=2)
    
    green_line = mlines.Line2D([], [], color='green',marker='o',markerfacecolor='blue',
                          markersize=7, label='%s, y = %2.2fx + %2.2f' % (cat2,m,c))
  
 
    
    ax.set_title(cat1 + ' vs ' + cat2)
    ax.legend(handles=[green_line,black_line], loc='best')

    fig.autofmt_xdate()
    plt.show()
    print('\n')
    
# Create some plots
single_plot('EMS: RESPIRATORY EMERGENCY')
single_plot('EMS: NAUSEA/VOMITING')
single_plot('EMS: CARDIAC EMERGENCY')
single_plot('EMS: FALL VICTIM')
single_plot('EMS: HEMORRHAGING')
single_plot('EMS: ALLERGIC REACTION')

embeded_plot(cat1='EMS: ASSAULT VICTIM',cat2='EMS: VEHICLE ACCIDENT')


# # If you have come so far, then congratulations!!!

# references  
# [github-911-calls walkthrough](https://www.kaggle.com/mchirico/dataset-walk-through-911)

# **Continue exploring the Data however you see fit!**
# 
