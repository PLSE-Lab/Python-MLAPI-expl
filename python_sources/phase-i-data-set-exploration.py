#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction
# 
# ## 1.1. Data provenance
# The available data set was made publicly available by [UnknownClass](http://www.kaggle.com/nphantawee) at [www.kaggle.com/nphantawee/pump-sensor-data](http://www.kaggle.com/nphantawee/pump-sensor-data). The data set comprises a collection of 52 sensors' measurements of a small town water pump system. 
# 
# ## 1.2. Porposed approach
# The author of the data set has reported ([www.kaggle.com/nphantawee/pump-sensor-data](http://www.kaggle.com/nphantawee/pump-sensor-data)) that the system had 7 system failures over the course of one year, which caused serious living problems to some families. Thus, the problem consists on predicting when will the next failure occur. To this end, a two-phase approach is proposed. First, an exploration of the provided data will take place. The purpose of such exploration is threefold: (1) to get familiar with the data, (2) to assess the _quality_ of data and decide which methodology to employ accordingly, (3) to determine the features of interest. At the end of this first phase, it is expected to have a clean, ready to work data set, which will be used throughout the second phase. In this phase, the predetermined methodology will be implemented to tackle the prediction problem.
# 
# # 2. Methodology
# 
# ## 2.1. Data exploration
# To get a first glimpse of the sensors' measurements, one can do the following.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/sensor.csv")
print(df.shape)
df.head()


# The previous data frame shows that, in fact, there are 52 sensors (one per each column) which recorded data at a 1min time-sample. Since there are 220320 recordings, it follows that the present data set represents a 153 day period. Moreover, the measurements have different scales.

# To check if there are any NaN values, one can do the following.

# In[ ]:


df.isnull().values.any()


# Since there exist NaN values, it is important to inpect the impact of eliminating them. One can do that as follows.

# In[ ]:


print('Deletting rows:')
print('Shape before elimination:', df.shape)
df_flag = df.dropna(axis=0, how='any')
print('Shape after elimination:', df_flag.shape,'\n')

print('Deletting collumns:')
print('Shape before elimination:', df.shape)
df_flag = df.dropna(axis=1, how='any')
print('Shape after elimination:', df_flag.shape)


# In[ ]:


df_flag.head()


# Hence, there are NaN values in every row and collumn of interest. In order to use data, the only thing on can do is replace NaN values with some other value. Since that at this phase one is only interested in getting to know the available data, a simple way of proceeding is to normalise data and replace NaN values by 0. Moreover, one can also partition data in features (X) and labels (y) using the [Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer) method, as follows.

# In[ ]:


X = df.iloc[:,2:54].fillna(0)
normalizer_scaler = preprocessing.Normalizer(norm='max')
X = normalizer_scaler.fit_transform(X.transpose())
X = pd.DataFrame(X.transpose())

y = df['machine_status']

X.head()


# In turn, y is a categorical feature. Thus it needs to be one hot encoded, as follows.

# In[ ]:


one_hot = pd.get_dummies(y)
one_hot.head()


# Up until now, one has the following data:
# * **X:** normalized sensors' measurements -- shape: (220320,51),
# * **y:** categoriacl labels, corresponding to the machine status (*Normal*, *Broken*, *Recovering*) -- shape: (220320,1),
# * **one_hot:** a one-hot enconded data frame of **y**.
# 
# To get a graphical, more clear representation of the data to be processed, consider the next figures represent the trends of each sensor measurement.

# In[ ]:


fig, axes = plt.subplots(figsize=(20, 20), dpi=120, nrows=7, ncols=2)

ax0 = X.iloc[::1500,0:4].plot(ax=axes[0,0])
ax0.set_xlim([0,220320])
ax0.grid()
ax0.set_xlabel('Time [minutes]')

ax1 = X.iloc[::1500,4:8].plot(ax=axes[0,1])
ax1.set_xlim([0,220320])
ax1.grid()
ax1.set_xlabel('Time [minutes]')

ax2 = X.iloc[::1500,8:12].plot(ax=axes[1,0])
ax2.set_xlim([0,220320])
ax2.grid()
ax2.set_xlabel('Time [minutes]')

ax3 = X.iloc[::1500,12:16].plot(ax=axes[1,1])
ax3.set_xlim([0,220320])
ax3.grid()
ax3.set_xlabel('Time [minutes]')

ax4 = X.iloc[::1500,16:20].plot(ax=axes[2,0])
ax4.set_xlim([0,220320])
ax4.grid()
ax4.set_xlabel('Time [minutes]')

ax5 = X.iloc[::1500,20:24].plot(ax=axes[2,1])
ax5.set_xlim([0,220320])
ax5.grid()
ax5.set_xlabel('Time [minutes]')

ax6 = X.iloc[::1500,24:28].plot(ax=axes[3,0])
ax6.set_xlim([0,220320])
ax6.grid()
ax6.set_xlabel('Time [minutes]')

ax7 = X.iloc[::1500,28:32].plot(ax=axes[3,1])
ax7.set_xlim([0,220320])
ax7.grid()
ax7.set_xlabel('Time [minutes]')

ax8 = X.iloc[::1500,32:36].plot(ax=axes[4,0])
ax8.set_xlim([0,220320])
ax8.grid()
ax8.set_xlabel('Time [minutes]')

ax9 = X.iloc[::1500,36:40].plot(ax=axes[4,1])
ax9.set_xlim([0,220320])
ax9.grid()
ax9.set_xlabel('Time [minutes]')

ax10 = X.iloc[::1500,40:44].plot(ax=axes[5,0])
ax10.set_xlim([0,220320])
ax10.grid()
ax10.set_xlabel('Time [minutes]')

ax11 = X.iloc[::1500,44:48].plot(ax=axes[5,1])
ax11.set_xlim([0,220320])
ax11.grid()
ax11.set_xlabel('Time [minutes]')

ax12 = X.iloc[::1500,48:52].plot(ax=axes[6,0])
ax12.set_xlim([0,220320])
ax12.grid()
ax12.set_xlabel('Time [minutes]')

plt.tight_layout()


# As can be seen there are a pattern being captured by the sensors (e.g. measurements 0, 4, 11). In turn, there are signals that are very noisy and seem to follow no trend in particular (e.g. measurements 40 -- 51). 
# 
# The problem at hand consists in determining when will the machine fail. Another way of addressing the problem is to determine when will the machine change from **Normal** to any other status.
# 
# For computational and problem tractability reasons, one can reduce the number of relevant features, i.e. one can choose the best $k$ features in which the prediction will be based upon. The assumption here is that having 51 simultaneous and independent measurements of the same system produces redundant data. For many practical reasons it may be important to have redundant sources of information (e.g. security, maintenance, etc.). However for the problem at hand, redundancy produces nothing but noise. Therefore, to choose the best $k$ features one can make use of the scikit learn's [SelectKBest](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html) method, as follows.

# In[ ]:


# Instanciate selector
selector = SelectKBest(chi2, k=10) # select k = 11 < 51

# Fit it to data
X_fitted = selector.fit_transform(X, one_hot['NORMAL'])

# Determine k-best features
mask = selector.get_support() #list of booleans
new_features = [] # The list of your K best features

feature_names = list(X.columns.values)

for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)

X_fitted = X.iloc[:,new_features]
X_fitted.head()


# The 10 features of interest are presented below.

# In[ ]:


fig, axes = plt.subplots(figsize=(20, 5), dpi=120, nrows=1, ncols=2)

ax0 = X_fitted.iloc[::1500,0:5].plot(ax=axes[0])
ax0.set_xlim([0,220320])
ax0.set_ylim([0,1])
ax0.grid()
ax0.set_xlabel('Time [minutes]')

ax1 = X_fitted.iloc[::1500,5:10].plot(ax=axes[1])
ax1.set_xlim([0,220320])
ax1.set_ylim([0,1])
ax1.grid()
ax1.set_xlabel('Time [minutes]')

plt.tight_layout


# As can be seen, it is clear that most measurements (except signal 13) follow the same trend. Therefore, one may focus on a dataset comprising signals 0 to 12 to develop a prediction model.
# 
# The final data set is presented below, including a superposition of the machine status.

# In[ ]:


Y = one_hot['NORMAL']
data = pd.concat([X_fitted.iloc[:, 0:8],Y], axis=1, sort=False)

# Data visualisation imposing machine status information
fig, axes = plt.subplots(figsize=(20, 5), dpi=120, nrows=1, ncols=2)

ax0 = data.iloc[::1500, 0:4].plot(ax=axes[0])
ax0 = data.iloc[::1500, -1].plot(drawstyle="steps", ax=axes[0])
ax0.set_xlim([0,220320])
ax0.grid()
ax0.set_xlabel('Time [minutes]')

ax1 = data.iloc[::1500,4:8].plot(ax=axes[1])
ax1 = data.iloc[::1500, -1].plot(drawstyle="steps",ax=axes[1])
ax1.set_xlim([0,220320])
ax1.grid()
ax1.set_xlabel('Time [minutes]')

plt.tight_layout


# Data shows that 6 (not 7) failures were recorded. 
# 
# Given the problem at hand and the data available, the best approach is to use recurrent neural networks (RNN), since the primordial goal is to be able to make predictions about the system's future beahaviour. To this end, a data set will be constructed using the following data partitioning:
# 
# * Each sensor's measurement will be broken into 6 subseries, where the breaking point is the moment when machine status switches to **NOT NORMAL**. Such a partitioning results in $6 \times 8 = 48$ time-series.
# 
# * Then, the 48 time-series will be further divided into **Train** and **Test** sets.
# 
# The next section focuses on the data set creation.

# ## 2.2. Data set creation for regression problem
# 
# In the previous section, a data exploration took place, from which a set of features of interest were determined. Moreover, based on the data a data set creation methodology was devised. This secion focuses on building such data set.
# 
# Recall that during the exploration NaN values were substituted by 0 and data was normalized without great discussion. Such data processing was performed to enable an easy familirisation with data. However, to build a data set one does not want to corrupt data, i.e. one must be very carefull of its pre-processing since the model building methodology is very sensitive to such data manipulations.
# 
# Therefore, the data set creation will have a bottoms-up approach, i.e. starting with the raw data, features will be selected and manipulated individually and aggregated at the end.
# 
# ### 2.2.1. Data imputation
# Recall fom the previous section that the data set at hand had NaN values in every row and collumn. To address this problem, invalid values were substituted by 0. However, that is a poor approach since the system being studied is a continous one. Forcing the time series to switch from a given value to 0 intermittently corrupts the signal, creating patterns that may be learnt by the RNN which would be counter-productive. Thus, the best way to address the problem is to somehow infer from a nieghberhood of values what is the most likely one to appear in place of NaN.
# 
# As a starting point, consider this example. Consider the series defined as follows,
# 
# $$\{1,2,3,4,5,NaN,7,8,9\}.$$
# 
# Any person would be able to infer NaN should be replaced by 6. Several methods were developed to teach a machine to infer the same. Arguably, the most simple one consists on averaging, as follows.
# 
# Let $x_i$ represent the sequence element in position $i$. Consider the situation in which a NaN value is at position $i$, i.e. $x_i = NaN$. Then, one can infer is value as follows,
# 
# $$x_i = \frac{x_{i-1}+x_{i+1}}{2}.$$
# 
# On can employ such a method resorting to scikit learn's method [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer), as follows.

# In[ ]:


# instanciate imputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

new_data = imp.fit_transform(df.iloc[:,2:54])  # new_data originates from raw data
new_data = pd.DataFrame(new_data)


# Note, however, that the previous method recquires $x_i$ and $x_{i+1}$, i.e. it is assumed the previous and the next values of a given NaN are always available. This excludes situations in which there are more than 2 consecutive NaN values. Since in such a big collection of data one can not visualise all data entries, one must assume such situations might occur at some point in the data set.
# 
# To deal with problem, one must resort to more flexible and robust imputation methods, such as interpolations. The interested reader is referred to [Wikipedia](http://en.wikipedia.org/wiki/Interpolation). To employ such methods on the data set in question, one can make use of the pandas' [interpolate](http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html#pandas.DataFrame.interpolate) methods. Different methods of interpolation are available, e.g. linear, quadratic, polynomial, etc. To decide which one to choose, one must *see* the actual signal one wants to keep and employ the method that most resembles its trend.
# 
# The figures below represent a chunck of various of the features of interes.

# In[ ]:


# Data visualisation imposing machine status information
fig, axes = plt.subplots(figsize=(20, 5), dpi=120, nrows=2, ncols=2)

ax0 = df.iloc[0:50,2].plot(ax=axes[0,0], legend = 'sensor_00')
ax0.set_xlim([0,50])
ax0.grid()
ax0.set_xlabel('Time [minutes]')

ax1 = df.iloc[0:50,6].plot(ax=axes[0,1], legend = 'sensor_04')
ax1.set_xlim([0,50])
ax1.grid()
ax1.set_xlabel('Time [minutes]')

ax2 = df.iloc[0:50,8].plot(ax=axes[1,0], legend = 'sensor_06')
ax2.set_xlim([0,50])
ax2.grid()
ax2.set_xlabel('Time [minutes]')

ax3 = df.iloc[0:50,9].plot(ax=axes[1,1], legend = 'sensor_07')
ax3.set_xlim([0,50])
ax3.grid()
ax3.set_xlabel('Time [minutes]')

plt.tight_layout


# The measurements seem to follow a linear trend between points, thus one can exploit this information and perform data imputation resorting to linear interpolation, as follows.

# In[ ]:


def interpol(X):
    X_interpolled = X.interpolate(method='linear')
    return X_interpolled


# In[ ]:


# new_data originates from raw data
new_data = df.iloc[:,2:54].astype(float)

# each independent signal corresponds to a given collumn
for i in range(0,52):
    feat = interpol(new_data.iloc[:,i])
    new_data.iloc[:,i] = feat

new_data.head()


# ### 2.2.2. Data partitioning
# After data imputation, once can proceed to partition data. To this end, one starts by keeping only the features of interest (i.e. sensors 0, 4, 6, 7, 8, 9, 10, and 11) and drop the rest. Then, one performes feature normalization to bring all values into the range $[0,1]$. After, one truncates the time-series following the machine status information.
# 
# Starting by dropping unused features, one can proceed as follows.

# In[ ]:


new_data = new_data[['sensor_00','sensor_04','sensor_06','sensor_07','sensor_08','sensor_09','sensor_10','sensor_11']]
new_data.isnull().values.any()


# One can repeat the normalisation process, as follows.

# In[ ]:


#normalizer_scaler = preprocessing.Normalizer(norm='max')
new_data = normalizer_scaler.fit_transform(new_data.transpose())
new_data = pd.DataFrame(new_data.transpose())
new_data = pd.concat([new_data, Y], axis=1, sort=False)
new_data.columns = ['sensor_00','sensor_04','sensor_06','sensor_07','sensor_08','sensor_09','sensor_10','sensor_11','machine_status']
new_data.head()


# Before proceeding to truncate data, one can save the new data set (new_data) for future research avenues, as follows.

# In[ ]:


new_data.to_csv("sensor_new_data.csv")


# The last step in data prepocessing consists on truncating each time-series (i.e. feature) according to the machine status. Namely, one is interested in partitioning data as shown in the following figure.
# 
# ![](http://eduardogfma.github.io/Articles/kaggle/pumpDataFig/fig.png)
# 
# Although it seems information is being lost, since the resulting time-series are truncated, one gains two things: (1) more data, i.e. after partitioning 8 time-series one gets 48 new ones; (2) it is guaranteed that both the event that one wants to predict (i.e. failure) and the normal behaviour of the system (i.e. NORMAL status) are reapresented in the data that will be used to train, validate, and test the predictioni model.
# 
# To this end, function, **find_gaps**, will be defined in order to determine gaps between the indexes that correspond to a normal status of the pump system. Then, this information is used to create a look-up table (**fail_ind**), which consists of a 6x2 array in which each row corresponds chuncks of data (as seen in the figure above), and columns represente the starting and finishing indexes of the fail status. After, resorting to **series_partitioning** one can initiate the partitioning process.
# 
# The implementation of the presented methodology is presented below.

# In[ ]:


def find_gaps(numbers, gap_size):
    adjacent_differences = [(y - x) for (x, y) in zip(numbers[:-1], numbers[1:])]
    # If adjacent_differences[i] > gap_size, there is a gap of that size between
    # numbers[i] and numbers[i+1]. We return all such indexes in a list - so if
    # the result is [] (empty list), there are no gaps.
    return [i for (i, x) in enumerate(adjacent_differences) if x > gap_size]


# In[ ]:


null_ind = Y[Y == 0].index  # get null indexes
v = find_gaps(null_ind, 1)  # vector with null indexes gaps
v


# In[ ]:


# create gaps look-up table of the form (gap start, gap finish)
fail_ind = np.zeros((6,2)).astype(int)

for i in range(0,len(v)):
    if i == 0:
        fail_ind[i,0] = null_ind[0:v[i]][0]
        fail_ind[i,1] = null_ind[0:v[i]][-1]
    else:
        fail_ind[i,0] = null_ind[v[i-1]+1:v[i]][0]
        fail_ind[i,1] = null_ind[v[i-1]+1:v[i]][-1]

fail_ind


# In[ ]:


# each time-series will be chuncked into 6 new ones
def series_partitioning(feature, table):
    ts1 = feature.iloc[0:table[0][1]]
    ts2 = feature.iloc[table[0][1]:table[1][1]]
    ts3 = feature.iloc[table[1][1]:table[2][1]]
    ts4 = feature.iloc[table[2][1]:table[3][1]]
    ts5 = feature.iloc[table[3][1]:table[4][1]]
    ts6 = feature.iloc[table[4][1]:table[5][1]]
    
    return ts1,ts2,ts3,ts4,ts5,ts6


# In[ ]:


s00_1,s00_2,s00_3,s00_4,s00_5,s00_6 = series_partitioning(new_data['sensor_00'], fail_ind)


# In[ ]:


plt.figure(figsize=(20, 5), dpi=120)
plt.plot(s00_1.iloc[::1500], label ='s00_1')
plt.plot(s00_2.iloc[::1500], label ='s00_2')
plt.plot(s00_3.iloc[::1500], label ='s00_3')
plt.plot(s00_4.iloc[::1500], label ='s00_4')
plt.plot(s00_5.iloc[::1500], label ='s00_5')
plt.plot(s00_6.iloc[::1500], label ='s00_6')
Y.iloc[::1500].plot(drawstyle="steps")
plt.ylim(0,1)
plt.xlim(0,150000)
plt.legend()
plt.grid()
plt.tight_layout()


# In[ ]:


s04_1,s04_2,s04_3,s04_4,s04_5,s04_6 = series_partitioning(new_data['sensor_04'], fail_ind)
s06_1,s06_2,s06_3,s06_4,s06_5,s06_6 = series_partitioning(new_data['sensor_06'], fail_ind)
s07_1,s07_2,s07_3,s07_4,s07_5,s07_6 = series_partitioning(new_data['sensor_07'], fail_ind)
s08_1,s08_2,s08_3,s08_4,s08_5,s08_6 = series_partitioning(new_data['sensor_08'], fail_ind)
s09_1,s09_2,s09_3,s09_4,s09_5,s09_6 = series_partitioning(new_data['sensor_09'], fail_ind)
s10_1,s10_2,s10_3,s10_4,s10_5,s10_6 = series_partitioning(new_data['sensor_10'], fail_ind)
s11_1,s11_2,s11_3,s11_4,s11_5,s11_6 = series_partitioning(new_data['sensor_11'], fail_ind)
y_1,y_2,y_3,y_4,y_5,y_6 = series_partitioning(new_data['machine_status'], fail_ind)


# Since each partition has different lengths, one can group different sensors' time-series by partition, as follows.

# In[ ]:


sxx_1 = pd.concat([s00_1, s04_1, s06_1, s07_1, s08_1, s09_1, s10_1, s11_1, y_1], axis=1, sort=False)
sxx_1.columns = ['sensor_00','sensor_04','sensor_06','sensor_07',
                 'sensor_08','sensor_09','sensor_10','sensor_11','machine_status']
sxx_1.head()


# In[ ]:


sxx_2 = pd.concat([s00_2, s04_2, s06_2, s07_2, s08_2, s09_2, s10_2, s11_2, y_2], axis=1, sort=False)
sxx_2.columns = ['sensor_00','sensor_04','sensor_06','sensor_07',
                 'sensor_08','sensor_09','sensor_10','sensor_11','machine_status']

sxx_3 = pd.concat([s00_3, s04_3, s06_3, s07_3, s08_3, s09_3, s10_3, s11_3, y_3], axis=1, sort=False)
sxx_3.columns = ['sensor_00','sensor_04','sensor_06','sensor_07',
                 'sensor_08','sensor_09','sensor_10','sensor_11','machine_status']

sxx_4 = pd.concat([s00_4, s04_4, s06_4, s07_4, s08_4, s09_4, s10_4, s11_4, y_4], axis=1, sort=False)
sxx_4.columns = ['sensor_00','sensor_04','sensor_06','sensor_07',
                 'sensor_08','sensor_09','sensor_10','sensor_11','machine_status']

sxx_5 = pd.concat([s00_5, s04_5, s06_5, s07_5, s08_5, s09_5, s10_5, s11_5, y_5], axis=1, sort=False)
sxx_5.columns = ['sensor_00','sensor_04','sensor_06','sensor_07',
                 'sensor_08','sensor_09','sensor_10','sensor_11','machine_status']

sxx_6 = pd.concat([s00_6, s04_6, s06_6, s07_6, s08_6, s09_6, s10_6, s11_6, y_6], axis=1, sort=False)
sxx_6.columns = ['sensor_00','sensor_04','sensor_06','sensor_07',
                 'sensor_08','sensor_09','sensor_10','sensor_11','machine_status']


# Having completed the partitioning process, one can save each series of data as follows.

# In[ ]:


sxx_1.to_csv("sxx_1_data.csv")
sxx_2.to_csv("sxx_2_data.csv")
sxx_3.to_csv("sxx_3_data.csv")
sxx_4.to_csv("sxx_4_data.csv")
sxx_5.to_csv("sxx_5_data.csv")
sxx_6.to_csv("sxx_6_data.csv")


# # 3. Conclusion
# 
# This work focused on the first of a two phase approach to a failure rediction problem. The devised methodologies focused on data exploration. Two important results were achieved: 
# 
# 1. the features of interest were determined, i.e. the measurements performed by sensors 0, 4, 6, 7, 8, 9, 10, 11;
# 
# 2. a set of clean, ready to work data sets, which will be used throughout the second phase were also constructed (please refer to sxx_1.csv, sxx_2.csv, sxx_3.csv, sxx_4.csv, sxx_5.csv, sxx_6.csv). 
# 
# Moreover, an extra data set (sensor_new_data.csv) was also built, which consists on the 8 features of interes, plus the binary label (machine_status, i.e. either NORMAL, or FAILURE).

# In[ ]:




