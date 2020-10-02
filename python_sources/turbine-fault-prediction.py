#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(20,20)})
import random 
from datetime import datetime, timedelta
from bisect import *
from collections import defaultdict

get_ipython().run_line_magic('matplotlib', 'inline')

random.seed(123)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

dataframe_faultdata = pd.read_csv('/kaggle/input/fault_data.csv')
dataframe_scadadata = pd.read_csv('/kaggle/input/scada_data.csv')
dataframe_statusdata = pd.read_csv('/kaggle/input/status_data.csv')


# In[ ]:


#Short shape of data set
print('The size of the fault data is: %d rows and %d columns' % dataframe_faultdata.shape)
print('The size of the scda data is: %d rows and %d columns' % dataframe_scadadata.shape)
print('The size of the status is: %d rows and %d columns' % dataframe_statusdata.shape)


# In[ ]:


dataframe_scadadata.columns


# In[ ]:


#Show data of all column in data set
for x in dataframe_scadadata.columns:
    print(dataframe_scadadata[x].name, '\t',dataframe_scadadata[x].dtype)


# In[ ]:


dataframe_scadadata.head()


# In[ ]:


dataframe_scadadata['WEC: ava. windspeed'].value_counts(bins=10)


# In[ ]:


dataframe_scadadata.shape[0]


# In[ ]:


#Show missing value in all column on dataset
miss_num = dataframe_scadadata.shape[0] - dataframe_scadadata.count()
print(miss_num)


# No missing data in our dataset, Luckily !!
# BTW, we can impute/ drop the NA or missing value be below method. 

# In[ ]:


#Replace missing value with dropout/ mode
dataframe_scadadata_noNA = dataframe_scadadata.dropna()

#Replace missing value with mode
dataframe_scadadata_mode = dataframe_scadadata.fillna({'WEC: ava. windspeed':dataframe_scadadata['WEC: ava. windspeed'].mode()[0]})


# In[ ]:


dataframe_scadadata_mode


# In[ ]:


#Plot histogram of average windspeed
dataframe_scadadata['WEC: ava. windspeed'].value_counts(bins=20).plot(kind='bar')


# In[ ]:


np.log(dataframe_scadadata['WEC: ava. windspeed'] + 1).hist(bins=50)


# In[ ]:


histrogram = np.log(dataframe_scadadata['WEC: ava. windspeed'] + 1)
histrogram


# In[ ]:


dataframe_scadadata['WEC: ava. windspeed'].hist(bins=50)


# In[ ]:



corr = dataframe_scadadata.corr()

ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right');


# In[ ]:


dataframe_scadadata.corr()


# In[ ]:


#Check some pair of correlation
dataframe_scadadata[['WEC: ava. windspeed', 'WEC: ava. Rotation']].corr()


# In[ ]:


dataframe_scadadata[['WEC: ava. windspeed', 'Blade A temp.']].corr()


# In[ ]:


dataframe_scadadata[['WEC: ava. windspeed', 'WEC: ava. Rotation', 'Blade A temp.']].corr()


# In[ ]:


#Plot pair of high correaltion windspeed and rpm of turbine
plt.scatter(dataframe_scadadata['WEC: ava. windspeed'], dataframe_scadadata['WEC: ava. Rotation'])


# In[ ]:


dataframe_statusdata.head(10)


# In[ ]:


#Pre define error codes on Main Status column
errorcodes=['62','60','228','80','9']
error_rows = dataframe_statusdata[dataframe_statusdata['Main Status'].isin(errorcodes)]


# In[ ]:


error_rows['Main Status'].value_counts()


# In[ ]:


for col in dataframe_statusdata.columns:
    print(dataframe_statusdata[col].name, ':\t', dataframe_statusdata[col].dtype)


# In[ ]:


#All error code on main status and sub status
error_rows[['Main Status', 'Sub Status', 'Status Text']].drop_duplicates().sort_values(['Main Status', 'Sub Status'])


# In[ ]:


#Change datetime to datetime data type
dataframe_scadadata['DateTime'] = pd.to_datetime(dataframe_scadadata['DateTime'])
dataframe_scadadata.head()


# In[ ]:


#Find range of date time in dataframe 
print('Earliest Date:',min(dataframe_scadadata['DateTime']))
print('Latest Date: ',max(dataframe_scadadata['DateTime']))


# In[ ]:


#Set start date and stop date
start = datetime(2014,5,2)
stop = datetime(2015,4,9)


# In[ ]:


#Construct list of 30 minute interval as time delta object.
tm = start 

half_hour = timedelta(minutes = 30)
dt_list = []

while tm < stop : 
    dt_list.append(tm)
    tm += half_hour


# In[ ]:


#dt_list contain list of time from starting to ending with time step = 30 minutes interval
len(dt_list)


# In[ ]:


dt_list[0:10]


# In[ ]:


#Create function to return index of date time in Scada data frame that match with 30 minutes interval
def MostRecentIndex(timelist, tm):
    i = bisect_right(timelist, tm)-1
    return i 


# In[ ]:


dataframe_scadadata_M = dataframe_scadadata.as_matrix()
raw_dts = list(dataframe_scadadata['DateTime'])
raw_dts[129:133]


# In[ ]:


dataframe_scadadata_M


# In[ ]:


print(MostRecentIndex(raw_dts, dt_list[0]))


# In[ ]:


standardized_rows =[]
for tm in dt_list:
    i = MostRecentIndex(raw_dts, tm)
    standardized_rows.append(dataframe_scadadata_M[i])


# In[ ]:


standardized_rows


# In[ ]:


standardized_cols = list(dataframe_scadadata.columns)


# In[ ]:


print('Number of records in raw feature dataset:', len(dataframe_scadadata))
print('Number of records in standardize feature dataset:', len(standardized_rows))


# In[ ]:


# Create a Features data-frame for export
dataframe_features = pd.DataFrame(standardized_rows,columns=standardized_cols)
dataframe_features['DateTime'] = dt_list

#save to intermediate data store
LOCALFILE = '/kaggle/features_base.csv'
dataframe_features.to_csv(LOCALFILE,index=False)


# In[ ]:


#Fault Data Set on Wind turbine
#convert DateTime column to proper data type
dataframe_faultdata['DateTime'] = pd.to_datetime(dataframe_faultdata['DateTime'])

#display head of data-frame
dataframe_faultdata.head()


# In[ ]:


#print only unique value of fault code
fault_names = sorted(set(dataframe_faultdata['Fault']))
print(fault_names)


# In[ ]:


time_frames = [24,48,72]
tf_labels = [str(a) for a in time_frames] # TimeFrame Labels _ change numeric to string


# In[ ]:


tf_labels


# In[ ]:



dates = list(dataframe_faultdata['DateTime'])
#convert data-frame to matrix
dataframe_faultdata_M = dataframe_faultdata.as_matrix()
outcomes = []

for dt1 in dt_list:
    outcome = [dt1]
    index1 = bisect_left(dates,dt1) #bisect = return in index for the closest one

    for tf,tf_label in zip(time_frames,tf_labels): # zip = sum of 2 tables in dictionary
        dd = defaultdict(int)
        dt2 = dt1+timedelta(hours=tf)
        index2 = bisect_right(dates,dt2)
        m2 = dataframe_faultdata_M[index1:index2]
        if len(m2)>0:
            fault_labels = list(m2[:,-1])
            for label in set(fault_labels):
                dd[label] = fault_labels.count(label)

        for label in fault_names:
            count = dd[label]
            outcome.append(count)

    outcomes.append(outcome)


# In[ ]:


fault_names


# In[ ]:


print('\n'.join(map(str, outcomes[0:10])))


# In[ ]:


columns = ['DateTime']
for tf_label in tf_labels :
    for name in fault_names : 
        col = name+'-'+tf_label
        columns.append(col)

    print(columns)


# In[ ]:


dataframe_outcomes = pd.DataFrame(outcomes, columns = columns)
dataframe_outcomes.head()


# In[ ]:


dataframe_features.head()


# In[ ]:


dataframe_outcomes.head()


# In[ ]:


dataframe_outcomes_bool = pd.DataFrame(dataframe_outcomes)
Vars = list(dataframe_outcomes_bool.columns)[1:]
for var in Vars:
    dataframe_outcomes_bool[var] = [int(bool(a)) for a in dataframe_outcomes_bool[var]]

dataframe_outcomes_bool.head()


# In[ ]:


LOCALFILE_COUNTS = '/kaggle/outcomes_counts.csv'
LOCALFILE_BOOL = '/kaggle/outcomes_bool.csv'

dataframe_outcomes.to_csv(LOCALFILE_COUNTS,index=False)
dataframe_outcomes_bool.to_csv(LOCALFILE_BOOL,index=False)


# In[ ]:


dataframe_features['DateTime'] = pd.to_datetime(dataframe_features['DateTime'])
dataframe_outcomes['DateTime'] = pd.to_datetime(dataframe_outcomes['DateTime'])


# In[ ]:


print('Features Data-frame Shape:', dataframe_features.shape)
print('Outcomes Data-frame Shape:', dataframe_outcomes.shape)


# In[ ]:


print('Q: Are the DateTime values the same?')
print('A:', list(dataframe_features['DateTime']) == list(dataframe_outcomes['DateTime']))


# In[ ]:


dataframe_outcomes.sum()


# In[ ]:


print('Variable: Count: Pct%:')
dataframe_length = len(dataframe_outcomes)
columns = list(dataframe_outcomes.columns)[1:] #Aggregate Ocurrences
print('Variable:\tCount:\tPct%:')
dataframe_length = len(dataframe_outcomes)
columns = list(dataframe_outcomes.columns)[1:] 
for col in columns: 
    occurrences = [bool(a) for a in dataframe_outcomes[col]]
    occurrence_count = sum(occurrences)
    occurrence_percentage = round(100.0*occurrence_count/dataframe_length,2)
    print(col, '\t\t', occurrence_count, '\t',occurrence_percentage)
    


# In[ ]:


YVar = 'WEC: ava. windspeed'
XVar = 'DateTime'

YVals = dataframe_features[YVar]
XVals = dataframe_features[XVar]

sns.set_style("darkgrid")
plt.figure(figsize=(12, 6))
plt.plot(XVals, YVals)
adf = plt.gca().get_xaxis().get_major_formatter()
adf.scaled[1.0] = '%m-%d'
plt.ylabel(YVar)
plt.xlabel(XVar)
plt.show()


# In[ ]:


YVar = 'Rectifier cabinet temp.'
XVar = 'DateTime'

YVals = dataframe_features[YVar]
XVals = dataframe_features[XVar]

sns.set_style("darkgrid")
plt.figure(figsize=(12, 6))
plt.plot(XVals, YVals)
adf = plt.gca().get_xaxis().get_major_formatter()
adf.scaled[1.0] = '%m-%d'
plt.ylabel(YVar)
plt.xlabel(XVar)
plt.show()


# In[ ]:


YVars = ['Yaw inverter cabinet temp.', 'Fan inverter cabinet temp.']

dt1 = datetime(2014, 6, 1)
dt2 = datetime(2014, 6, 20)

time_frame = [bool(dt1 <= t <= dt2) for t in dataframe_features['DateTime']]
dataframe_subset = dataframe_features[time_frame]

sns.set_style("darkgrid")
plt.figure(figsize=(12, 6))

XVar = 'DateTime'
XVals = dataframe_features[XVar]
for YVar in YVars:
    YVals = dataframe_features[YVar]
    plt.plot(XVals, YVals)

adf = plt.gca().get_xaxis().get_major_formatter()
adf.scaled[1.0] = '%m-%d'

plt.title('Comparing Multiple Variables', size=14)
plt.ylabel('Temp - Celsius')
plt.xlabel(XVar)
plt.legend(loc='upper left')
plt.show()


# In[ ]:


dataframe_features.corr()


# In[ ]:


var = 'WEC: ava. windspeed'
vals = dataframe_features[var]
num_bins = 25
plt.figure(figsize=(8, 5))
plt.hist(vals, num_bins, normed=1, facecolor='blue', alpha=0.5)
plt.title(var, size=14)
plt.ylabel('Frequency')
plt.xlabel('Wind Speed (m/s)')
plt.show()


# In[ ]:


var = 'WEC: ava. windspeed'
vals = dataframe_features[var]
num_bins = 25
plt.figure(figsize=(8, 5))
sns.distplot(vals, color='b', bins=num_bins, hist_kws={'alpha': 0.3})
plt.title(var, size=14)
plt.ylabel('Frequency')
plt.xlabel('Wind Speed (m/s)')
plt.show()


# In[ ]:


print(vals.describe())


# In[ ]:


dataframe_subset.hist(figsize=(20, 20), bins=25, xlabelsize=8, ylabelsize=8)
plt.show()


# In[ ]:


column_subset = [
    'WEC: ava. windspeed',
    'WEC: ava. Rotation',
    'WEC: ava. Power',
    'WEC: Production kWh',
    'Tower temp.',
    'Control cabinet temp.',
    'Transformer temp.',
    'Stator temp. 1',
    'Ambient temp.',
]
dataframe_subset = dataframe_features[column_subset]

dataframe_subset = dataframe_subset.select_dtypes(include=['float64', 'int64'])

dataframe_subset.hist(
    figsize=(20, 20), bins=25, xlabelsize=8, ylabelsize=8) #figsize=(16, 20)
plt.show()


# In[ ]:


YVars = ['Stator temp. 1']
XVars = ['WEC: ava. windspeed']
sns.pairplot(data=dataframe_subset, x_vars=XVars, y_vars=YVars, size=5, aspect=1.5)
plt.show()


# In[ ]:


YVars = column_subset
XVars = column_subset
width = len(column_subset)

for i in range(0, len(XVars), width):
    sns.pairplot(data = dataframe_subset, x_vars = XVars[i: i + width], y_vars = YVars)


# In[ ]:


import pandas as pd
import numpy as np
import warnings
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
#Seed the random method to ensure repeatability in our nondeterministic methods.
random.seed(7654)

#Suppress warnings
warnings.filterwarnings("ignore")


# In[ ]:


dataframe_features['DateTime'] = pd.to_datetime(dataframe_features['DateTime'])


# In[ ]:


columns_to_remove = set(['Time', 'Error'])
filtered_columns = [a for a in dataframe_features.columns if a not in columns_to_remove]
dataframe_features = dataframe_features[filtered_columns]


# In[ ]:


base_vars = list(dataframe_features.columns)[1:]
temp_vars = [v for v in base_vars if 'temp.' in v.lower()]
non_temp_vars = [v for v in base_vars if v not in temp_vars]


# In[ ]:


temp_vars


# In[ ]:





# In[ ]:


dataframe_temps = deepcopy(dataframe_features[temp_vars])
print('Base Vars:', len(base_vars))
print('Temp. Vars:', len(temp_vars))
print('Non-Temp:', len(non_temp_vars))


# In[ ]:


dataframe_temps.head()


# In[ ]:


temps_M = dataframe_temps.as_matrix()
pca = PCA()
pca.fit(temps_M) #addtemp_M to PCA 
components_M = pca.transform(temps_M)


# In[ ]:


N = 10 # number of component features to generate 
explained = list(pca.explained_variance_ratio_)[:N]
total = round(100.0*sum(explained),2)

print('Total Variance Explained')
print('by', str(N),'components:',str(total)+'%')
print('')
print('% Explained by PCA Component:')

comp_vars = ['C'+str(n)+'_Temp' for n in range(1, N+1)]
for var, val in zip(comp_vars, explained):
    per = round(100.0*val,2)
    print('', var,':', str(per)+'%')
    


# In[ ]:




