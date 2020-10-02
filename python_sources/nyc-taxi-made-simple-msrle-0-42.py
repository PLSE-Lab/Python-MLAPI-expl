#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt # to handle dates and time
from datetime import datetime, timedelta, date

from functools import reduce

# data visualization
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#import cufflinks as cf
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


# sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import Imputer, OneHotEncoder, LabelBinarizer, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb


#  ## Custom Functions

# In[3]:


def mean_cross_score(x):
    print("Accuracy: %0.3f +/- %0.3f" % (np.mean(x), np.std(x)))


# In[4]:


class DataFrameSelector(BaseEstimator, TransformerMixin): # to select dataframes in a pipeline.
    def __init__(self, attributes): 
        self.attributes = attributes
    def fit(self, df, y=None): 
        return self
    def transform(self, df):
        return df[self.attributes].values


# In[5]:


def RemoveOutliers(df,cols,n_sigma): # keep only instances that are within p\m n_sigma in columns cols
    new_df = df.copy()
    for col in cols:
        new_df = new_df[np.abs(new_df[col]-new_df[col].mean())<=(n_sigma*new_df[col].std())]
    print('%i instances have been removed' %(df.shape[0]-new_df.shape[0]))
    return new_df


# In[6]:


def my_pipeline(df, func_list):
    new_df = df.copy()
    return reduce(lambda x, func: func(x), func_list, new_df)


# ## Loading Data

# In[7]:


train_df = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')
test_df = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')

holidays_df = pd.read_csv('../input/nyc2016holidays/NYC_2016Holidays.csv', sep=';')


# In[8]:


# Geo locations taken from Google maps
NYC = np.array([-74.0059,40.7128]) # google maps coordinates of NYC

fifth_ave = np.array([0.58926996811979,0.8079362008674332]) # versor of Fifth Av. digitized from google maps
ort_fifth_ave = np.array([-0.8079362008674332,0.58926996811979]) # orthogonal versor

EastRiver = np.array([-73.955921,40.755157])
HudsonRiver = np.array([-74.012226,40.755677])
LeftBound = np.array([-74.020485,40.701463])
RightBound = np.array([-73.932614,40.818593])


# ## Initial Exploration of the Data

# In[9]:


train_df.info()


# There are 1.5 millions instances in this dataset, which means it is fairly big. Notice also that there are no *null* values in the training set. Given the size of this set I think it is fair to assume that also the test set won't have any null value, but you never know...
# 
# The labels are pretty much self explenatory, so we want go into details. 
# 
# Notice also that there are 4 objects that need to be transformed for ML purposes:
# - **id**: this will probably be useless
# - **pickup_datetime** and **dropoff_datetime**: this can be transformed in with the *datetime* library
# - **store_and_fwd_flag**: this attribute flags whether the instance was uploaded immidiately or not. I don't know if it will be usefull or not...
# 
# Lets do some consistency check on the train set

# In[10]:


train_df['id'].value_counts().shape


# This matches the total number if instances, so each instance is unique.
# 
# What about the **store_and_fwd_flag**?

# In[11]:


train_df['store_and_fwd_flag'].value_counts()


# There are just 8000 instances flagged 'Yes' out of 1.5 millions. This suggests that this attribute will probably be useless for our ML. However, we should check whether this instances are peculiar in some respect.  

# In[13]:


train_df.describe()


# Lets take a better look at the **vendor_id** and **passenger_count** attributes:

# In[14]:


plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
data = train_df['vendor_id'].value_counts().sort_index()
data.plot(kind='bar')
plt.xlabel('vendor_id')
plt.ylabel('events')

plt.subplot(1,2,2)
data = train_df['passenger_count'].value_counts().sort_index()
data.plot(kind='bar')
plt.xlabel('passenger_count')

plt.show()


# Notice that *value_counts()* givs only the non-null values... so apparently there were rides with zero passengers!
# Lets keep in mind this when we exclude outliers.

# This is probably going to be one of the most eye catching plots. Lets look at the pickup location:

# In[15]:


plt.figure(figsize=(10,10))

plt.scatter(x=train_df['pickup_longitude'].values,y=train_df['pickup_latitude'].values, marker='^',s=1,alpha=.3)
plt.xlim([-74.1,-73.7])
plt.ylim([40.6, 40.9])
plt.axis('off')

#plt.scatter(x=train_df['dropoff_longitude'].values,y=train_df['dropoff_latitude'].values, marker='v',s=1,alpha=.1)
#plt.xlim([-74.05,-73.75])
#plt.ylim([40.6, 40.9])
#plt.axis('off')

plt.show()


# One million rides... 
# 
# Even without a map beneath one can recognize Manhattan, Brooklyn and the airports JFK and La Guardia!
# In fact, from this plot one already can see that most of the rides happen in Manhatthan. We'll check this quantitatively later. 
# 
# There are also some weird instances of cabs going into the ocean! and a couple of NY cabs in SF! outliers...
# We could remove them more carefully later, but instead lets just remove the 3 $\sigma$ tails.

# In[16]:


clean_att = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']
train_df_clean = RemoveOutliers(train_df,clean_att,5)


# Next we have **trip_duration**, that is what we have to predict. Notice that the competition calculates the MSR Log Error. So we can directly take the log of **trip_duration**.

# In[17]:


train_df_clean['trip_duration'] = np.log(train_df_clean['trip_duration'])


# In[18]:


plt.figure(figsize=(8,6))

plt.hist(train_df_clean['trip_duration'], bins=100)
#plt.yscale('log', nonposy='clip')
plt.xlabel('Trip duration (log)')
plt.ylabel('events')

plt.show()


# The duration is a nice bell-shaped distribution in log space, however there is some small peak around log(duration)=11. Lets clean a bit more the data by removing the outliers

# In[19]:


clean_att = ['trip_duration']
train_df_clean = RemoveOutliers(train_df_clean,clean_att,5)


# This is the result after cleaning

# In[20]:


plt.figure(figsize=(8,6))

plt.hist(train_df_clean['trip_duration'], bins=100)
#plt.yscale('log', nonposy='clip')
plt.xlabel('Trip duration (log)')
plt.ylabel('events')

plt.show()


# ## Data Preparation for ML

# Most ML algorithm and sklearn class works with numberical attributes only, so we need to transform the object-type attributes.
# Lets create a list of numerical and categorical attributes

# In[21]:


num_att = [f for f in train_df_clean.columns if train_df_clean.dtypes[f] != 'object']
cat_att = [f for f in train_df_clean.columns if train_df_clean.dtypes[f] == 'object']
print("-"*10+" numerical attributes "+"-"*10)
print(num_att)
print('')
print("-"*10+" categorical attributes "+"-"*10)
print(cat_att)


# ### Trip Duration

# First lets run a consistency check on the data

# In[22]:


train_df_clean['pickup_datetime'] = pd.to_datetime(train_df_clean['pickup_datetime'])
train_df_clean['dropoff_datetime'] = pd.to_datetime(train_df_clean['dropoff_datetime'])

delta_t = np.log((train_df_clean['dropoff_datetime']-train_df_clean['pickup_datetime']).dt.total_seconds())
print("Number of wrong trip durations: %i" %train_df_clean[np.round(delta_t,5)!=np.round(train_df_clean['trip_duration'],5)].shape[0])


# The train_set is consitent, so we can drop the dropoff time and split the dataframes in X (the features) and Y (what we have to predict)

# In[23]:


X_train = train_df_clean.drop(['id','dropoff_datetime','trip_duration'], axis=1)
Y_train = train_df_clean['trip_duration'].copy()

X_test = test_df.drop(['id'], axis=1)
X_test_id = test_df['id'].copy()


cat_att = ['store_and_fwd_flag']
date_att = ['pickup_datetime']
coord_att = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']


# ### Categorical attributes

# In[24]:


def FlagEncoder(df):
    new_df = df.copy()
    new_df['store_and_fwd_flag'] = (new_df['store_and_fwd_flag']=='Y')*1
    return new_df


# ### Datetime Attributes

# Here we'll use the holidays dataset to add a 'weekend' attribute. We'll also transform the pick_datetime into 'day' (i.e. the day of the week) and 'time' (the time of the pickup).

# In[25]:


holidays = (pd.to_datetime('2016 '+ holidays_df['Date']))


# In[26]:


def DateAttributes(df):
    new_df = df.copy()
    new_df['pickup_datetime'] = pd.to_datetime(new_df['pickup_datetime'])
    new_df['day'] = new_df['pickup_datetime'].dt.weekday
    new_df['weekend'] = 1*((new_df['day']>=5)|(new_df['pickup_datetime'].dt.date.isin(holidays.dt.date.values)))
    new_df['time'] = np.round(new_df['pickup_datetime'].dt.time.apply(lambda x: x.hour + x.minute/60.0),1)
    return new_df


# ### Numerical Attributes

# #### Distance
# 
# From Physics 101, distance = speed * time, so we should try to find the distance and the speed of each ride to help the ML algorith. This is probably the most difficult part of the game.
# 
# We will define two kind of distances called L1 and L2. The L1 distance is the typical distance you experience in Mahatthan going from point A to point B. The L2 distance is just the Euclidean distance. 
# 
# I've seen many people using very complex definition of distances on this competition, for example the Haversine distance. This is the distance on a a great circle from a point A on a sphere of radius R to another point B. I think this is quite an overdoing. The size of NYC compared to the radius of the Earth is minuscle. so the error that one makes in treating the Earth as flat is more that acceptable. 
# 
# So the easy option is just to use longitude and latitude as coordinates on a plane! No Haversine needed! We only have to express latitude and longitude with the same (arbitrary) unit of measurement. In fact, I've already used this approximation in the plot above. 

# In[27]:


def distance(coords): #  L1 and L2 distances (in arbitrary units)
    units = np.array([np.cos(np.radians(NYC[1])),1]) # multiply by 111.2 to get km 
    picks = np.split(coords.transpose(),2)[0].transpose()*units
    drops = np.split(coords.transpose(),2)[1].transpose()*units
    x1 = np.dot(picks,fifth_ave*units)
    y1 = np.dot(picks,ort_fifth_ave*units)    
    x2 = np.dot(drops,fifth_ave*units)
    y2 = np.dot(drops,ort_fifth_ave*units)
    dist_L1 = abs(x1-x2) + abs(y1-y2)
    dist_L2 = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return [dist_L1,dist_L2]


# In[28]:


def DistanceAttribute(df):
    new_df = df.copy()
    coords = new_df[coord_att].values
    new_df['dist_L1'] = distance(coords)[0]
    new_df['dist_L2'] = distance(coords)[1]
    return new_df


# In[29]:


# Adding attributes
pipe_list = [FlagEncoder, DateAttributes, DistanceAttribute]

X_train_prepared = my_pipeline(X_train,pipe_list)
X_test_prepared = my_pipeline(X_test,pipe_list)


# But how many rides actually happen within Manhattan?
# 
# To answer this question one could use a clustering algorith. but we'll make it simpler here: we'll just define a strip on the map and call it Manhattan. Here it is:

# In[30]:


X=X_train_prepared

c1 = (X['pickup_longitude']>LeftBound[0])&(X['pickup_longitude']<RightBound[0])
c2 = (X['pickup_longitude']>LeftBound[0])&(X['dropoff_longitude']<RightBound[0])
c3 = ((X['pickup_longitude']-EastRiver[0])*ort_fifth_ave[0]+(X['pickup_latitude']-EastRiver[1])*ort_fifth_ave[1])>0
c4 = ((X['pickup_longitude']-HudsonRiver[0])*ort_fifth_ave[0]+(X['pickup_latitude']-HudsonRiver[1])*ort_fifth_ave[1])<0
c5 = ((X['dropoff_longitude']-EastRiver[0])*ort_fifth_ave[0]+(X['dropoff_latitude']-EastRiver[1])*ort_fifth_ave[1])>0
c6 = ((X['dropoff_longitude']-HudsonRiver[0])*ort_fifth_ave[0]+(X['dropoff_latitude']-HudsonRiver[1])*ort_fifth_ave[1])<0

Manhattan_df = X[c1&c2&c3&c4&c5&c6]
Y_Manhattan = Y_train.values[c1&c2&c3&c4&c5&c6]
print('Percentage of trips within Manhattan: %.2f' %(1.*Manhattan_df.shape[0]/X.shape[0])) 


# Almost all rides are within Manhattan! The L1 distance will be a quite accurate estimate of the true distance for these instances. 
# 
# What about JFK and La Guardia?

# In[31]:


JFK = np.array([-73.779148,40.653416])
LaGuardia = np.array([-73.873890,40.775341])


# In[32]:


X=X_train_prepared

c1 = (X['pickup_longitude']>(JFK[0]-0.05))&(X['pickup_longitude']<(JFK[0]+0.05))
c2 = (X['pickup_latitude']>(JFK[1]-0.05))&(X['pickup_latitude']<(JFK[1]+0.05))
c3 = (X['dropoff_longitude']>(JFK[0]-0.05))&(X['dropoff_longitude']<(JFK[0]+0.05))
c4 = (X['dropoff_latitude']>(JFK[1]-0.05))&(X['dropoff_latitude']<(JFK[1]+0.05))

JFK_df = X[(c1&c2)|(c3&c4)]
Y_JFK = Y_train.values[(c1&c2)|(c3&c4)]
print('Percentage of trips to/from JFK: %.2f' %(1.*JFK_df.shape[0]/X.shape[0])) 


# In[33]:


X=X_train_prepared

c1 = (X['pickup_longitude']>(LaGuardia[0]-0.02))&(X['pickup_longitude']<(LaGuardia[0]+0.02))
c2 = (X['pickup_latitude']>(LaGuardia[1]-0.02))&(X['pickup_latitude']<(LaGuardia[1]+0.02))
c3 = (X['dropoff_longitude']>(LaGuardia[0]-0.02))&(X['dropoff_longitude']<(LaGuardia[0]+0.02))
c4 = (X['dropoff_latitude']>(LaGuardia[1]-0.02))&(X['dropoff_latitude']<(LaGuardia[1]+0.02))

LaGuardia_df = X[(c1&c2)|(c3&c4)]
Y_LaGuardia = Y_train.values[(c1&c2)|(c3&c4)]
print('Percentage of trips to/from La Guardia: %.2f' %(1.*LaGuardia_df.shape[0]/X.shape[0])) 


# That is interesting La Guardia as twice as many rides as JFK, but from [this link](http://http://laguardiaairport.com/about-us/facts-and-statistics/) and [this link](http://https://www.panynj.gov/airports/pdf/stats/JFK_DEC_20016.pdf), JFK has twice as many passengers as La Guardia. SO people going to La Guardia are more inclined to take a Taxi... if you have been to La Guardia you know what I'm talking about.

# In[34]:


# Dataframe consolidation
Manhattan_df = DistanceAttribute(Manhattan_df)
JFK_df = DistanceAttribute(JFK_df)
LaGuardia_df = DistanceAttribute(LaGuardia_df)


# Lets now look at how well the distance attribute predict the trip duration. A scatter plot is a good starting point.

# In[35]:


plt.figure(figsize=(8,8))

plt.scatter(x=X_train_prepared['dist_L2'].values,y=np.exp(Y_train).values,s=1,alpha=0.1)
plt.xlim([0,0.3])
plt.ylim([0, 6000])
#plt.axis('off')
plt.xlabel('dist_L2')
plt.ylabel('Trip duration (log)')

plt.show()


# This is another interesting plot. One can see a bulk of instances with large scatter at distance L2 less than 0.1, and then 3 clear spikes: one at 0 distance (outliers?), one at 0.12 and one at 0.22. Notice that a spike in the distance means a fixed prefered length of travelling... I already know where people is going!

# In[36]:


plt.figure(figsize=(8,8))

plt.scatter(x=X_train_prepared['dist_L2'].values,y=np.exp(Y_train),s=1,alpha=0.1)
plt.scatter(x=Manhattan_df['dist_L2'].values,y=np.exp(Y_Manhattan),s=1,alpha=0.1)
plt.scatter(x=JFK_df['dist_L2'].values,y=np.exp(Y_JFK),s=1,alpha=0.1)
plt.scatter(x=LaGuardia_df['dist_L2'].values,y=np.exp(Y_LaGuardia),s=1,alpha=0.1)

plt.xlim([0,0.3])
plt.ylim([0, 6000])
plt.xlabel('dist_L2')
plt.ylabel('Trip duration')

plt.show()


# The data divides well in three clusters: rides within Manhattan and rides to/from the airport. This make around the 90% of the data. so lets define new attribute with this information encoded a la OneHot. Furthermore lets clean the spike at zero distance.

# In[37]:


c1 = X_train_prepared['dist_L1']>0.0001
c2 = X_train_prepared['dist_L2']>0.0001

X_train = X_train[(c1&c2)]
Y_train = Y_train[c1&c2]


# In[38]:


def RideScope(df):
    X = df.copy()
    
    # Manhatthan only
    c1 = (X['pickup_longitude']>LeftBound[0])&(X['pickup_longitude']<RightBound[0])
    c2 = (X['dropoff_longitude']>LeftBound[0])&(X['dropoff_longitude']<RightBound[0])
    c3 = ((X['pickup_longitude']-EastRiver[0])*ort_fifth_ave[0]+(X['pickup_latitude']-EastRiver[1])*ort_fifth_ave[1])>0
    c4 = ((X['pickup_longitude']-HudsonRiver[0])*ort_fifth_ave[0]+(X['pickup_latitude']-HudsonRiver[1])*ort_fifth_ave[1])<0
    c5 = ((X['dropoff_longitude']-EastRiver[0])*ort_fifth_ave[0]+(X['dropoff_latitude']-EastRiver[1])*ort_fifth_ave[1])>0
    c6 = ((X['dropoff_longitude']-HudsonRiver[0])*ort_fifth_ave[0]+(X['dropoff_latitude']-HudsonRiver[1])*ort_fifth_ave[1])<0

    X['M&M'] = (c1&c2&c3&c4&c5&c6)*1
    
    # JFK
    c1 = (X['pickup_longitude']>(JFK[0]-0.05))&(X['pickup_longitude']<(JFK[0]+0.05))
    c2 = (X['pickup_latitude']>(JFK[1]-0.05))&(X['pickup_latitude']<(JFK[1]+0.05))
    c3 = (X['dropoff_longitude']>(JFK[0]-0.05))&(X['dropoff_longitude']<(JFK[0]+0.05))
    c4 = (X['dropoff_latitude']>(JFK[1]-0.05))&(X['dropoff_latitude']<(JFK[1]+0.05))
    
    X['JFK'] = ((c1&c2)|(c3&c4))*1
    
    #LaGuardia
    c1 = (X['pickup_longitude']>(LaGuardia[0]-0.02))&(X['pickup_longitude']<(LaGuardia[0]+0.02))
    c2 = (X['pickup_latitude']>(LaGuardia[1]-0.02))&(X['pickup_latitude']<(LaGuardia[1]+0.02))
    c3 = (X['dropoff_longitude']>(LaGuardia[0]-0.02))&(X['dropoff_longitude']<(LaGuardia[0]+0.02))
    c4 = (X['dropoff_latitude']>(LaGuardia[1]-0.02))&(X['dropoff_latitude']<(LaGuardia[1]+0.02))

    X['LaG'] = ((c1&c2)|(c3&c4))*1
    
    return X


# In[39]:


pipe_list = pipe_list + [RideScope]

X_train_prepared = my_pipeline(X_train,pipe_list)
X_test_prepared = my_pipeline(X_test,pipe_list)


# Everybody knows that a ride from JFK is unpredictable, but can we understand the dispersion better? It surely depends on the traffic along the way, which is probably correlated with the time of the pickup. Lets check.

# In[40]:


plt.figure(figsize=(8,8))

sc = plt.scatter(JFK_df['dist_L2'].values,np.exp(Y_JFK),c=JFK_df['time'],s=1,cmap=plt.get_cmap('jet'),alpha=0.5)
plt.xlim([0,0.3])
plt.ylim([0, 6000])
cb = plt.colorbar(sc)
plt.xlabel('dist_L2')
plt.ylabel('Trip duration')
cb.set_label('Time of the day')

plt.show()


# As expected, trips that start at night have much less scatter than trips that happen during the day. So better to include the time in the attribute for ML. 

# #### Speed
# 
# The previous plot suggesed that the time of the day is (obviously) important. So lets take a look at the average speed of each ride vs the time of the day

# In[41]:


def Speeds(df,Y):
    new_df = df.copy()
    new_df['speed_L1'] = new_df['dist_L1']/np.exp(Y)
    new_df['speed_L2'] = new_df['dist_L2']/np.exp(Y)
    return new_df


# In[42]:


speed_df = Speeds(X_train_prepared,Y_train)

weekday_speed = speed_df.groupby(['weekend','time']).agg(['mean','std'])[['speed_L1','speed_L2']].loc[0].reset_index()
weekend_speed = speed_df.groupby(['weekend','time']).agg(['mean','std'])[['speed_L1','speed_L2']].loc[1].reset_index()


# In[43]:


plt.figure(figsize=(6,6))

plt.errorbar(weekday_speed['time'],weekday_speed['speed_L2']['mean'],yerr=0, label='weekday')
plt.errorbar(weekend_speed['time'],weekend_speed['speed_L2']['mean'],yerr=0, label='weekend')
plt.ylim([0, 0.000125])

#plt.errorbar(weekday_speed['time'],weekday_speed['speed_L2']['mean'],yerr=weekday_speed['speed_L2']['std'])
#plt.errorbar(weekend_speed['time'],weekend_speed['speed_L2']['mean'],yerr=weekend_speed['speed_L2']['std'])
#plt.ylim([0, 0.000125])
plt.ylim([0, 0.0001])
plt.xlabel('time')
plt.ylabel('speed_L2')
plt.legend()

plt.show()


# Notice the speed plateau during 9 -19 of weekdays and the lower speed at night during weekend.

# We can define a speed attribute (actually two, wrt L1 and L2) only on the train set, since there is no known duration on the test. Therfore we need a rule to define it also for the test set. 'time' and 'weekend' seems to obvious selection rules.

# In[44]:


speeds = speed_df.groupby(['weekend','time']).mean()[['speed_L1','speed_L2']].reset_index()


# In[45]:


def SpeedAttribute(df):
    new_df = df.copy()
    new_df = pd.merge(new_df, speeds, how='left', on=['weekend','time'])
    return new_df


# In[46]:


pipe_list = pipe_list + [SpeedAttribute]

X_train_prepared = my_pipeline(X_train,pipe_list)
X_test_prepared = my_pipeline(X_test,pipe_list)


# ## Attributes selection and scaling

# In[47]:


X_train_prepared.columns


# In[48]:


num_att = ['passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','dist_L1','dist_L2','speed_L1','speed_L2','time']
OneHot = ['weekend','M&M','JFK','LaG']


# In[49]:


num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_att)),
    ('imputer', Imputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

OneHot_pipeline = Pipeline([
    ('selector', DataFrameSelector(OneHot)),
])


# In[50]:


full_pipeline = FeatureUnion(transformer_list=[
           ("num_pipeline", num_pipeline),
           ("cat_pipeline", OneHot_pipeline),
])


# In[51]:


X_train_scaled = full_pipeline.fit_transform(X_train_prepared)
X_test_scaled = full_pipeline.transform(X_test_prepared)


# ## Model Selection and Training

# In[52]:


#tree_reg = DecisionTreeRegressor()

#scores = cross_val_score(tree_reg, X_train_scaled, Y_train, cv=3, scoring='neg_mean_squared_error')
#mean_cross_score(-scores)


# In[53]:


rnd_reg = RandomForestRegressor()

scores = cross_val_score(rnd_reg, X_train_scaled, Y_train, cv=3, scoring='neg_mean_squared_error')
mean_cross_score(-scores)


# In[60]:


rnd_reg = RandomForestRegressor()
rnd_reg.fit(X_train_scaled, Y_train)


# ## Submission

# In[63]:


Y_test_pred = rnd_reg.predict(X_test_scaled)

submission = pd.DataFrame({
        "id": X_test_id,
        "trip_duration": np.exp(Y_test_pred)
    })
submission.to_csv('submission2.csv', index=False)


# **This notebook scored on 0.42 on the leaderboard.**
# 
# Notice however that the score I get here is much better thatn the one on the leaderboard. This should be investigate. Also one could performe some gridsearch to tune the model a check for overfitting. 
# 
# Nevertheless I think the result is fairly good: with just a few feature we can predict quite well the trip duration. 
# A lot more could be done. Especially crossing with other dataset. 
# 
# For example there are dataset of traffic in NYC given by the DOT. I tried to use them, but the measurement are geographically to sparse to be usefull, but maybe there are better dataset. 
# 
# The weather might be relevant as well
# 
# **Anyway, if you like the job done here and want to continue, fork and upvote!**
# 

# In[ ]:




