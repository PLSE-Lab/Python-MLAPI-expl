#!/usr/bin/env python
# coding: utf-8

# # Project Summary
# 
# The objective of this project is to predict forest cover type in the 4 wilderness areas within Roosevelt National Forest of northern colorado. Each observation is based on 30m x 30m patch. The training set contains features and the Cover_Type. The test set contains only features and the objective is to predict Cover_Type.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from bokeh.core.properties import value
from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from bokeh.models import ColumnDataSource
from bokeh.transform import jitter
import matplotlib.image as mpimg

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

output_notebook()


# # Loading training and test datasets

# In[ ]:


train_file_path = '../input/learn-together/train.csv'
test_file_path = '../input/learn-together/test.csv'

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)


# # Seperating the target from the training dataframe

# In[ ]:


y = train_df.Cover_Type


# # Exploratory Data Analysis

# In[ ]:


train_df.describe()


# The predictors for this dataset are cover types. 
# 
# 1. Spruce/Fir
# 2. Lodgepole Pine
# 3. Ponderosa Pine
# 4. Cottonwood/Willow
# 5. Aspen
# 6. Douglas-fir
# 7. Krummholz

# ## Evaluating Soil Types

# In[ ]:


cols_list = train_df.columns.values.tolist()


# In[ ]:


soilType_features = cols_list[15:-1]

#for feature in soilType_features:
#    print(feature+' : '+str(train_df[feature].unique()))
       
train_df_T = train_df[soilType_features].apply(pd.value_counts).fillna(0).transpose().astype(int)

train_df_T.columns = ['absence','presence']

train_df_T


# In[ ]:


ax = train_df_T[['absence','presence']].plot(kind='bar', title ="Soil Types Absence/Presence", figsize=(20, 10), legend=True, fontsize=12, width=0.5)
ax.set_xlabel("SoilTypes", fontsize=12)
ax.set_ylabel("Counts", fontsize=12)
plt.show(ax)


# The distribution of values for the soil types mentioned below is lopsided
# 
# * SoilType7
# * SoilType15
# 
# So I am going to eliminate these features from the training and test dataset that we are going to use in modeling and predicting.Now lets determine how some of the other features other than soil types impact forest cover types.

# ## Does Elevation impact forest cover type

# In[ ]:


max_ct = max(pd.Series(train_df['Cover_Type']).tolist())
elevation_df = train_df[['Elevation', 'Cover_Type']]

x_val = pd.Series(elevation_df['Elevation']).tolist()
y_val = pd.Series(elevation_df['Cover_Type']).tolist()
    
source = ColumnDataSource(data=dict(x=x_val,y=y_val))

p1 = figure(plot_width=800, plot_height=800, y_range=(0,max_ct+1),
           title="Does elevation effect forest cover types?")

p1.circle(x='x', y=jitter('y', width=0.5, range=p1.y_range),  source=source, alpha=0.3)

p1.x_range.range_padding = 0
p1.ygrid.grid_line_color = None

show(p1)


# The scatter plot reveals that forest covers change with elevation. Some observations based on the output
# 
# * Krummholz cover type approximately starts at an altitude of 2800 meters
# * Aspen cover type is restricted to 2500 and 3000 meters
# * Cottonwood/Willow cover type is restricted to 2000 and 2500 meters

# ## Does Aspect impact forest cover type
# 
# Wikipedia article on Aspect <https://en.wikipedia.org/wiki/Aspect_(geography)> clearly states the impact **aspect** has on vegetation.

# In[ ]:


max_ct = max(pd.Series(train_df['Cover_Type']).tolist())
aspect_df = train_df[['Aspect', 'Cover_Type']]

x_val = pd.Series(aspect_df['Aspect']).tolist()
y_val = pd.Series(aspect_df['Cover_Type']).tolist()
    
source = ColumnDataSource(data=dict(x=x_val,y=y_val))

p_at = figure(plot_width=800, plot_height=800, y_range=(0,max_ct+1),
           title="Does aspect effect forest cover types?")

p_at.circle(x='x', y=jitter('y', width=0.5, range=p_at.y_range),  source=source, alpha=0.3)

p_at.x_range.range_padding = 0
p_at.ygrid.grid_line_color = None

show(p_at)


# ## Does slope impact forest cover types

# In[ ]:


slope_df = train_df[['Slope', 'Cover_Type']]

x_val = pd.Series(slope_df['Slope']).tolist()
y_val = pd.Series(slope_df['Cover_Type']).tolist()
    
source = ColumnDataSource(data=dict(x=x_val,y=y_val))

p_sl = figure(plot_width=800, plot_height=800, y_range=(0,max_ct+1),
           title="Does slope effect forest cover types?")

p_sl.circle(x='x', y=jitter('y', width=0.5, range=p_sl.y_range),  source=source, alpha=0.3)

p_sl.x_range.range_padding = 0
p_sl.ygrid.grid_line_color = None

show(p_sl)


# The Viz reveals that
# 
# * Spruce/Fir (1) and Lodgepole Pine (2) cover types get sparse at Slope > 40 degrees
# * Douglas-fir (6) cover type does not exist at Slope values greater than 42 degrees
# * Almost all cover types get sparse from 40 degrees

# ## Does horizontal distance to hydrology impact covertype

# In[ ]:


hdth_df = train_df[['Horizontal_Distance_To_Hydrology', 'Cover_Type']]

x_val = pd.Series(hdth_df['Horizontal_Distance_To_Hydrology']).tolist()
y_val = pd.Series(hdth_df['Cover_Type']).tolist()
    
source = ColumnDataSource(data=dict(x=x_val,y=y_val))

p_hdth = figure(plot_width=800, plot_height=800, y_range=(0,max_ct+1),
           title="Does horizontal distance to hydrology affect forest cover types?")

p_hdth.circle(x='x', y=jitter('y', width=0.5, range=p_hdth.y_range),  source=source, alpha=0.3)

p_hdth.x_range.range_padding = 0
p_hdth.ygrid.grid_line_color = None

show(p_hdth)


# As you can see from the Viz, as the horizontal distance to nearest surface water body increases forest cover is impacted.

# ## Does vertical distance to hydrology impact covertype

# In[ ]:


vdth_df = train_df[['Vertical_Distance_To_Hydrology', 'Cover_Type']]

x_val = pd.Series(vdth_df['Vertical_Distance_To_Hydrology']).tolist()
y_val = pd.Series(vdth_df['Cover_Type']).tolist()
    
source = ColumnDataSource(data=dict(x=x_val,y=y_val))

p_vdth = figure(plot_width=800, plot_height=800, y_range=(0,max_ct+1),
           title="Does vertical distance to hydrology affect forest cover types?")

p_vdth.circle(x='x', y=jitter('y', width=0.5, range=p_vdth.y_range),  source=source, alpha=0.3)

p_vdth.x_range.range_padding = 0
p_vdth.ygrid.grid_line_color = None

show(p_vdth)


# ## Does horizontal distance to roadways impact covertype

# In[ ]:


hdtr_df = train_df[['Horizontal_Distance_To_Roadways', 'Cover_Type']]

x_val = pd.Series(hdtr_df['Horizontal_Distance_To_Roadways']).tolist()
y_val = pd.Series(hdtr_df['Cover_Type']).tolist()
    
source = ColumnDataSource(data=dict(x=x_val,y=y_val))

p_hdtr = figure(plot_width=800, plot_height=800, y_range=(0,max_ct+1),
           title="Does horizontal distance to roadways affect forest cover types?")

p_hdtr.circle(x='x', y=jitter('y', width=0.5, range=p_hdtr.y_range),  source=source, alpha=0.3)

p_hdtr.x_range.range_padding = 0
p_hdtr.ygrid.grid_line_color = None

show(p_hdtr)


# ## Does horizontal distance to firepoints impact covertype
# 
# Lets plot Horizontal_Distance_To_Fire_Points and CoverType and see if there is any prominent pattern.

# In[ ]:



hdtf_df = train_df[['Horizontal_Distance_To_Fire_Points', 'Cover_Type']]

x_val = pd.Series(hdtf_df['Horizontal_Distance_To_Fire_Points']).tolist()
y_val = pd.Series(hdtf_df['Cover_Type']).tolist()
    
source = ColumnDataSource(data=dict(x=x_val,y=y_val))

p_hdtf = figure(plot_width=800, plot_height=800, y_range=(0,max_ct+1),
           title="Does horizontal distance to firepoints affect forest cover types?")

p_hdtf.circle(x='x', y=jitter('y', width=0.5, range=p_hdtf.y_range),  source=source, alpha=0.3)

p_hdtf.x_range.range_padding = 0
p_hdtf.ygrid.grid_line_color = None

show(p_hdtf)


# ## Evaluating wilderness areas

# In[ ]:


wa_features = ['Wilderness_Area1',
'Wilderness_Area2',
'Wilderness_Area3',
'Wilderness_Area4']

wa_df_T = train_df[wa_features].apply(pd.value_counts).fillna(0).transpose().astype(int)

wa_df_T.columns = ['absence','presence']

wa_df_T

#train_df['Wilderness_Area1'].isnull().values.any(), train_df['Wilderness_Area2'].isnull().values.any(),test_df['Wilderness_Area3'].isnull().values.any(),test_df['Wilderness_Area4'].isnull().values.any()


# # Feature and Model selection
# 
# In the following sections we will look into the steps listed below
# 
# * finding interesection of columns between training and test dataframes
# * removing some columns as previous runs revealed they do not improve the model performance in any way
# * using K-Fold and Stratified K-Fold cross validation techniques along with MinMaxScaler feature scaler

# In[ ]:


# finding common features between training and test dataframes

matching_cols = train_df.columns.intersection(test_df.columns)
matching_cols_list = matching_cols.tolist()

# Features that will be removed from the training and test dataframes

cols_remove = ['Id','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Soil_Type7','Soil_Type15']

for col in cols_remove:
    matching_cols_list.remove(col)


# In[ ]:


X = train_df[matching_cols_list]

# Checking for any NA values

X.isna().sum()


# In[ ]:


# Transforming the test dataframe

test_X = test_df[matching_cols_list]


# # Cross-Validation

# ## Defining Classification Models

# ### KNeighboursClassifier

# In[ ]:


def knc(train_X, train_y):
    clf = KNeighborsClassifier(n_neighbors=10)
    scores = cross_val_score(clf, train_X, train_y, cv=2)
    return scores.mean()


# ### RandomForestClassifier

# In[ ]:


def rfc(train_X, train_y):
    clf = RandomForestClassifier(random_state=0, n_estimators=200,min_samples_split=2,max_depth=None)
    scores = cross_val_score(clf, train_X, train_y, cv=5)
    return scores.mean()


# ### DecisionTreeClassifier

# In[ ]:


def dtc(train_X, train_y):
    clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
    scores = cross_val_score(clf, train_X, train_y, cv=5)
    return scores.mean()


# ### ExtraTreesClassifier

# In[ ]:


def etc(train_X, train_y):
    clf = ExtraTreesClassifier(n_estimators=200, max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, train_X, train_y, cv=5)
    return scores.mean()


# ## K-Fold Cross-Validation

# In[ ]:


kf = KFold(n_splits=3, random_state=0, shuffle=False)

X_kf = train_df[matching_cols_list]

scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
scaler.fit(X_kf)
X_kf=scaler.transform(X_kf)

len(X_kf)


# In[ ]:


test_kf_X = test_df[matching_cols_list]
scaler.fit(test_kf_X)
test_kf_X=scaler.transform(test_kf_X)

len(test_kf_X)


# In[ ]:


for train_index, test_index in kf.split(X_kf):
    #print("TRAIN:", train_index, "TEST:", test_index)
    train_kf_X, val_kf_X = X_kf[train_index], X_kf[test_index]
    train_kf_y, val_kf_y = y[train_index], y[test_index]
    knc_kf_scores = knc(train_kf_X, train_kf_y)
    rfc_kf_scores = rfc(train_kf_X, train_kf_y)
    dtc_kf_scores = dtc(train_kf_X, train_kf_y)
    etc_kf_scores = etc(train_kf_X, train_kf_y)
    print(knc_kf_scores, rfc_kf_scores, dtc_kf_scores, etc_kf_scores)


# ## Stratified K-Fold Cross-Validation

# In[ ]:


X_sk = train_df[matching_cols_list]
y_sk = y

skf = StratifiedKFold(n_splits=3)

scaler_sk = MinMaxScaler(feature_range=(0, 1), copy=True)
scaler_sk.fit(X_sk)
X_sk = scaler_sk.transform(X_sk)

len(X_sk)


# In[ ]:


test_sk_X = test_X
scaler_sk.fit(test_sk_X)
test_sk_X=scaler_sk.transform(test_sk_X)

len(test_sk_X)


# In[ ]:


for train_index, test_index in skf.split(X_sk, y_sk):
    train_X_sk, test_X_sk = X.loc[train_index], X.loc[test_index]
    train_y_sk, test_y_sk = y_sk[train_index], y_sk[test_index]
    knc_sk_scores = knc(train_X_sk, train_y_sk)
    rfc_sk_scores = rfc(train_X_sk, train_y_sk)
    dtc_sk_scores = dtc(train_X_sk, train_y_sk)
    etc_sk_scores = etc(train_X_sk, train_y_sk)
    print(knc_sk_scores, rfc_sk_scores, dtc_sk_scores, etc_sk_scores)


# # Predicting target
# 
# The Stratified K-Fold ExtraTreesClassifier model produced the highest cross validation score. So we will be using ExtraTreesClassifier model for predicting the target.

# In[ ]:


clf = ExtraTreesClassifier(random_state=0, n_estimators=200,min_samples_split=2,max_depth=None)

clf.fit(X_sk, y_sk)
test_preds = clf.predict(test_sk_X)

output = pd.DataFrame({'Id': test_df.Id, 'Cover_Type': test_preds})

output.to_csv('submission_vs.csv', index=False)

