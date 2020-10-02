#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import the packages we need
import numpy as np
import pandas as pd
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import preprocessing


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# get our train and test data
train_raw = pd.read_csv("../input/bigquery-geotab-intersection-congestion/train.csv")
test_raw  = pd.read_csv("../input/bigquery-geotab-intersection-congestion/test.csv")


# ## Exploring the data
# ### We know we have four cities, how many intersections do we have in each?

# In[ ]:


city_ints = train_raw[['IntersectionId','City']].drop_duplicates().groupby(['City']).count()

plt.figure()
sns.barplot(city_ints.index, city_ints.IntersectionId)
plt.xlabel('City')
plt.ylabel('# of intersections')
plt.show()


# ### What does the stop time profile look like for each city (weekdays only)?

# In[ ]:


city_vol_hour = train_raw.loc[train_raw.Weekend == 0]
city_vol_hour = city_vol_hour[['City','Hour','TotalTimeStopped_p50']]                        .groupby(['City','Hour']).mean().reset_index()


# In[ ]:


vol_plot = sns.catplot(x="Hour", y="TotalTimeStopped_p50", col="City", col_wrap = 2,
                       data=city_vol_hour, kind = "bar", color = "black")


# In[ ]:


city_vol_hour_we = train_raw.loc[train_raw.Weekend == 1]
city_vol_hour_we = city_vol_hour_we[['City','Hour','TotalTimeStopped_p50']]                        .groupby(['City','Hour']).mean().reset_index()


# ### and at weekends?

# In[ ]:


vol_plot = sns.catplot(x="Hour", y="TotalTimeStopped_p50", col="City", col_wrap = 2,
                       data=city_vol_hour_we, kind = "bar", color = "red")


# ### What does the average median stop period look like for each city & hour (weekdays only)?

# In[ ]:


city_vol_hour_week = train_raw.loc[train_raw.Weekend == 0]
city_vol_hour_week = city_vol_hour_week[['City','Month','Hour','TotalTimeStopped_p50']]                        .groupby(['City','Month','Hour']).mean().reset_index()


# In[ ]:


vol_plot = sns.catplot(x="Hour", y="TotalTimeStopped_p50", col="Month", col_wrap = 2,
            hue="City", data=city_vol_hour_week, kind = "bar")


# ### Intersection data by city seems to be fairly incomplete for January and May. It is missing altogether for Feb-Apr.
# * Based on this very crude metric, it seems at initial glance that Atlanta is the most 'congested'
# * However this could be due to the fact we have less intersections (and they are just closer to the center of town)
# * This may explain why Chicago seems to have less congestion (since it has the most intersections and likely spans a greater area
# * Boston seems to be relatively busier during the day, but tails off at night compared to the other cities (in particular Philadelphia)
# 

# ### We will eventually predict two variables: 'Distance to First Stop' & 'Total Time Stopped', what does the correlation of the medians look like by city?

# In[ ]:


corr_grid = sns.FacetGrid(train_raw[['City','TotalTimeStopped_p50','DistanceToFirstStop_p50']], hue = 'City' 
                          ,col = 'City', col_wrap = 2)
corr_grid = corr_grid.map(plt.scatter,'TotalTimeStopped_p50','DistanceToFirstStop_p50')


# ### A short total time stopped but with a long distance to first stop likely means there is a very long phase on the junction (these are usually bigger junctions)
# * There is a long queue at the junction, but when the lights turn green, we get through in one go.
# * It looks like Atlanta and Philadelphia have more of these junctions
# 

# ### What does junction complexity look like for each city?

# In[ ]:


all_data = pd.concat([train_raw, test_raw], join = 'inner')
junc_complex = all_data[['City','IntersectionId','EntryHeading','ExitHeading']]                    .drop_duplicates().groupby(['City','IntersectionId']).count().reset_index()


# In[ ]:


complexity_grid = sns.FacetGrid(junc_complex[['City','EntryHeading']], col = 'City', col_wrap = 2, hue = "City")
complexity_grid = complexity_grid.map(plt.hist, 'EntryHeading', density = True)
complexity_grid.set_axis_labels("Possible Junction Decisions", "Density")


# #### Our most complex junctions...

# In[ ]:


junc_complex.loc[junc_complex.EntryHeading > 15]


# ## Cleaning the data and feature preparation

# ### Do we have any missing values?

# In[ ]:


all_data.isnull().sum()


# #### We are only missing the street names, which we argue we can drop for this prediction, since we know the intersection id and the entry and exit heading.
# 
# This also means that some entries 'Path' are of the form 'Unknown_...'. We will also drop this column based on inituition
# > 

# In[ ]:


# Give the missing road names the NoName tag
all_data.fillna('NoName', inplace = True)


# In[ ]:


def day_periods(row):
    
    if row.Weekend == 1:
        if row.Hour >= 11 and row.Hour <= 19:
            return 'we_day'
        if row.Hour >= 20 or row.Hour <= 8:
            return 'we_night'
        else:
            return 'we_morning'
        
    else:
        if row.Hour >= 15 and row.Hour <= 18:
            return 'eve_rush'
        if row.Hour >= 19 and row.Hour <= 22:
            return 'evening'
        if row.Hour <= 5 or row.Hour >= 23:
            return 'night'
        if row.Hour >= 6 and row.Hour <= 9:
            return 'mor_rush'
        else:
            return 'day'


# In[ ]:


all_data['DayPeriod'] = all_data.apply(day_periods, axis = 1)


# ### Cyclical variables
# 
# The month and hour variables are cyclical in that December (12) is prior to January (1) and similarly for hour 23 and hour 0. We would like to transform our variables such that we can reflect this. We will keep these variables as numeric in our prediction (i.e. we won't treat them as categorical).
# 
# $$ H(t) = \sin \Big(\frac{ \pi t}{12} \Big)$$
# 
# $$ M(t) = \sin \Big(\frac{ \pi t}{12} \Big)$$
# 
# Keeping t between $0$ and $\pi$ in monts means, the feature is maximized in the summer, minimized in the winter and spring and autumn are treated roughly the same.

# In[ ]:


all_data['Hour']  = round( np.sin( all_data['Hour'] * np.pi / 12.0 ),4)
all_data['Month'] = round( np.sin( all_data['Month'] * np.pi / 12.0 ),4)


# ### Type of Road Change at Intersection
# 
# We add features which tell us if we are moving between big/small roads as defined in 
# https://en.wikipedia.org/wiki/Types_of_road
# 
# We also add whether the path continues along the same road or not

# In[ ]:


big_roads = ['Highway','Expressway','Parkway']

def junc_type(row):
    if any(x in str(row.EntryStreetName) for x in big_roads) and any(x in str(row.ExitStreetName) for x in big_roads):
        return 'BigBig'
    elif any(x in str(row.EntryStreetName) for x in big_roads) and all(x not in str(row.ExitStreetName) for x in big_roads):
        return 'BigSmall'
    elif all(x not in str(row.EntryStreetName) for x in big_roads) and any(x in str(row.ExitStreetName) for x in big_roads):
        return 'SmallBig'
    else:
        return 'SmallSmall'
    
def straight_on(row):
    if row.EntryStreetName == 'NoName' or row.ExitStreetName == 'NoName':
        return 0
    if row.EntryStreetName == row.ExitStreetName:
        return 1
    else:
        return 0


# In[ ]:


all_data['RoadChange'] = all_data.apply(junc_type, axis = 1)
all_data['StraightOn'] = all_data.apply(straight_on, axis = 1)


# ### Longitude and Latitude
# 
# We would like some consistent frame of reference for measuring the position of each junction. It makes sense to find the centrepoint of each city and measure radial distance.
# 

# In[ ]:


city_centers = pd.DataFrame({'Atlanta':[33.748151,-84.387840],
                             'Philadelphia': [39.952732,-75.165198],
                             'Chicago': [41.878136,-87.630822],
                             'Boston':[42.360046,-71.060006]
                            })

def add_radial_dist(df, city_centers):
    
    df['radialDist'] = df.apply(lambda x: np.sqrt( (city_centers[x.City][0] - x.Latitude )**2 + 
                                                     (city_centers[x.City][1] - x.Longitude )**2
                                                 ), axis = 1 )
    
def shift_long_lat(df, city_centers):
    df['Latitude'] = df.apply(lambda x: x.Latitude - city_centers[x.City][0], axis = 1)
    df['Longitude'] = df.apply(lambda x: x.Longitude - city_centers[x.City][1], axis = 1)


# In[ ]:


add_radial_dist(all_data,city_centers)
shift_long_lat(all_data,city_centers)


# ### Junction Complexity

# In[ ]:


junc_complex.head()
junc_complex = junc_complex[['City','IntersectionId','EntryHeading']]
junc_complex.columns = ['City','IntersectionId','Complexity']


# In[ ]:


all_data = all_data.merge(junc_complex, how = "left", on = ['IntersectionId','City'])


# ### Let's combine Entry and Exit directions into a JuncPath Variable
# 
# This will give us 8C2 = 28 columns when we one-hot encode these

# In[ ]:


all_data['JuncPath'] = all_data['EntryHeading'] + all_data['ExitHeading']


# ### Unique IntersectionId
# The IntersectionId is only use for each city, we will construct a unique id and label encode them.

# In[ ]:


all_data['Intersection'] = all_data['IntersectionId'].astype(str) + all_data['City']


# In[ ]:


le = preprocessing.LabelEncoder()
le.fit(list(all_data['Intersection'].values))
all_data['Intersection'] = le.transform(list(all_data['Intersection'].values))


# ## Prediction

# In[ ]:


predict_cols = ['Intersection','Latitude','Longitude','Hour','Weekend','Month', 'City', 'DayPeriod', 'radialDist',
                'Complexity', 'JuncPath', 'RoadChange', 'StraightOn']


# In[ ]:


all_x = all_data[predict_cols]


# In[ ]:


all_x['Weekend'] = all_x['Weekend'].astype(str)


# In[ ]:


all_x = pd.get_dummies(all_x)


# In[ ]:


x_train = all_x[:len(train_raw)]
x_test  = all_x[len(train_raw):]

output_cols = ['TotalTimeStopped_p20' ,'TotalTimeStopped_p50','TotalTimeStopped_p80',
                     'DistanceToFirstStop_p20','DistanceToFirstStop_p50','DistanceToFirstStop_p80']

y_train = train_raw[output_cols]


# In[ ]:


rf = ensemble.RandomForestRegressor(n_estimators = 500)


# In[ ]:


rf.fit(x_train, y_train)
predictions = rf.predict(x_test)

pred_series = pd.Series(list(predictions) )
all_preds   = pd.Series.explode(pred_series)


# In[ ]:


sub_file = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv"
sub_file["Target"] = all_preds.values
sub_file.to_csv('submission.csv', index=False)

