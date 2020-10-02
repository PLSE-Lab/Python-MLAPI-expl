#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Data Acquisition/Loading
# 
# First I will load the data of all the machines in a single dataframe by concatenating individual dataframes corresponding to each csv for each of the machines. I have concatenated the data both, column wise and row wise and stored the result in 2 different dataframes, so that any one can be used whenever required.

# In[ ]:


from glob import glob
import pandas as pdlib

def produceCombinedCSV(list_of_files):
   
   # Consolidate all CSV files into one object
   result_df = pdlib.concat([pdlib.read_csv(file).add_prefix(str(list_of_files.index(file)) + '_') for file in list_of_files], axis=1).T.drop_duplicates().T
   return result_df

# Move to the path that holds our CSV files
csv_file_path = '/kaggle/input/'

# List all CSV files in the working dir
file_pattern = "csv"
#list_of_files = [file for file in glob('*.{}'.format(file_pattern))]
list_of_files = glob(csv_file_path + "*.csv")
print(list_of_files)

df_consolidated_columnwise = produceCombinedCSV(list_of_files)


# In[ ]:


df_consolidated_columnwise.rename(columns={'0_Unnamed: 0': 'DateTime'}, inplace=True)
df_consolidated_columnwise.set_index('DateTime', inplace=True)
df_consolidated_columnwise.head()


# In[ ]:


from glob import glob
import pandas as pdlib
from os import chdir
def produceCombinedCSV(list_of_files):
   
   # Consolidate all CSV files into one object
   result_df = pdlib.concat([pdlib.read_csv(file) for file in list_of_files], keys=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])
   # Convert the above object into a csv file and export
   result_df.to_csv('/kaggle/working/ConsolidateOutput_rowwise.csv')

# Move to the path that holds our CSV files
csv_file_path = '/kaggle/input/'
chdir(csv_file_path)

# List all CSV files in the working dir
file_pattern = "csv"
list_of_files = [file for file in glob('*.{}'.format(file_pattern))]
print(list_of_files)

produceCombinedCSV(list_of_files)


# In[ ]:


import pandas as pd
df_consolidated_rowwise = pd.read_csv("/kaggle/working/ConsolidateOutput_rowwise.csv", index_col=[0]).drop('Unnamed: 1', axis=1)
df_consolidated_rowwise.rename(columns={'Unnamed: 0.1': 'DateTime'}, inplace=True)
df_consolidated_rowwise.head()


# We'll now plot the features of one machine's reading, to check the waveforms of the 4 signals.

# In[ ]:


from matplotlib import pyplot
# load dataset
values = df_consolidated_columnwise.values
# specify columns to plot
groups = [0, 1, 2, 3]
i = 1
# plot each column
pyplot.figure(figsize=(24,12))
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(df_consolidated_columnwise.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()


# From the above plot, we can see that the values in all fields, follow a certain pattern upto a certain point in time, after which a sudden disruption in the pattern is observed, post which, the signal goes 0.
# 
# Also, the entire signal reading is loaded with a lot of noisy measurements as indicated by the high-value, vertical readings. We'll remove those to better view the waveforms.

# ### Noise Removal

# In[ ]:


sample = df_consolidated_rowwise.loc[0].copy()


# In[ ]:


sample_df = sample[(sample['0'] < 100) & (sample['0'] > -100) & (sample['1'] < 100) & (sample['1'] > -100) & (sample['2'] < 100) & (sample['2'] > -100) & (sample['3'] < 100) & (sample['3'] > -100)].copy()
sample_df.head()


# ### Plot for clean values

# In[ ]:


from matplotlib import pyplot
# load dataset
values = sample_df.values
# specify columns to plot
groups = [1,2,3,4]
i = 1
# plot each column
pyplot.figure(figsize=(24,12))
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.xticks(np.arange(0, 3000, 100)) 
    pyplot.title(sample_df.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()


# Since the signal contains both negative and positive values, we'll square the signal values and use the squared value for our analysis.

# In[ ]:


sample_df['0_sqr'] = np.square(sample_df['0'])
sample_df['1_sqr'] = np.square(sample_df['1'])
sample_df['2_sqr'] = np.square(sample_df['2'])
sample_df['3_sqr'] = np.square(sample_df['3'])

sample_df['DateTime'] = pd.to_datetime(sample_df['DateTime'])
sample_df.set_index('DateTime', inplace=True)
sample_df.head()


# Now we will try using a couple of methods/algorithms to identify the fault inception date/time. We will try using - 
# 
# 1. SARIMAX Model
# 2. Custom Function
# 3. LSTM

# ## 1. SARIMAX
# 
# Since our data contains a time-dependent signal, it can be modelled using Autoregressive models. In AR models, present value is modeled as a function of past values and noise. We will be using the SARIMAX model here, because as shown by the ACF plot below, our signal is dependent on the past values and also shows some seasonality (a repeating pattern of 12 periods approximately). Also since the signal is non-stationary, we need to use a time-differenced model which explains use of the I (integrated) in SARIMAX.

# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(sample_df['0_sqr'], lags=50, alpha=1)


# In[ ]:


import statsmodels.api as sm
import statsmodels.tsa.api as smt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import matplotlib.pyplot as plt

train, test = sample_df['0_sqr'].iloc[0:70], sample_df['0_sqr'].iloc[70:1000]
#train_log, test_log = np.log10(train), np.log10(test)
my_order = (0,0,0)
my_seasonal_order = (1, 1, 1, 12)


# In[ ]:


history = [x for x in train]
predictions = list()
predict_log=list()
for t in range(len(test)):
    model = sm.tsa.SARIMAX(history, order=my_order, seasonal_order=my_seasonal_order,enforce_stationarity=False,enforce_invertibility=False)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    predict_log.append(output[0])
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
print('predicted=%f, expected=%f' % (output[0], obs))
#error = math.sqrt(mean_squared_error(test_log, predict_log))
#print('Test rmse: %.3f' % error)
# plot
figsize=(24, 12)
plt.figure(figsize=figsize)
pyplot.plot(sample_df['0_sqr'].iloc[70:1000].index, test,label='Actuals')
pyplot.plot(sample_df['0_sqr'].iloc[70:1000].index, predictions, color='red',label='Predicted')
pyplot.legend(loc='upper right')
pyplot.show()


# So, while SARIMAX works well at predicting signals so that discrepancies in signal values can be identified by subtracting from the predicted signal, it works for single variables/signals only. Using this, we'll have to create a different SARIMAX model for each of the signals of each machine. This can be very expensive process-wise so we will assess other options first.

# ## Windowed MAX and First order Differences

# Here, we will follow a set of steps to specifically identify incoherent patterns in the signal values.
# 1. First we will square the signal values to transform negative values to positive
# 
# 2. Then we will take a rolling max of squared signal values. This will enable to get constant signal values for normal patterns, which will rise or fall if a different pattern emerges in the original signal.
# 
# 3. Then we will take the first order difference of the Windowed MAX values which should largely remain 0 (since the max for normal mode of operation would be the same due to the signal following a constant pattern), and should change only for incoherent signal patterns/disturbances indicating the commencement of fault.
# 
# 4. Then we will filter the first order differences by taking a threshold for noise tolerance (this threshold is set to be the mean of 15 largest values of the differenced values) and get the datetime of the first such value which breaches the threshold.
# 
# 5. These steps will be computed for all the signals (0, 1, 2, 3) and the corresponding dates identified by filtering would be returned by our function
# 
# 

# In[ ]:


def finding_first_fault(df):
    #Noise Removal
    sample_df = df[(df['0'] < 100) & (df['0'] > -100) & (df['1'] < 100) & (df['1'] > -100) & (df['2'] < 100) & (df['2'] > -100) & (df['3'] < 100) & (df['3'] > -100)].copy()
    
    #Squaring the Waveforms
    sample_df['0_sqr'] = np.square(sample_df['0'])
    sample_df['1_sqr'] = np.square(sample_df['1'])
    sample_df['2_sqr'] = np.square(sample_df['2'])
    sample_df['3_sqr'] = np.square(sample_df['3'])

    #Setting Index
    sample_df['DateTime'] = pd.to_datetime(sample_df['DateTime'])
    sample_df.set_index('DateTime', inplace=True)

    #Windowed MAX
    sample_df['0_max'] = sample_df['0_sqr'].rolling(72).max()
    sample_df['1_max'] = sample_df['1_sqr'].rolling(72).max()
    sample_df['2_max'] = sample_df['2_sqr'].rolling(72).max()
    sample_df['3_max'] = sample_df['3_sqr'].rolling(72).max()
    
    #Removal of Blanks (Initial values of the window)
    sample_df.dropna(inplace=True)
    
    #First order difference of Rolling MAX
    sample_df['0_change'] = sample_df['0_max'].diff()
    sample_df['1_change'] = sample_df['1_max'].diff()
    sample_df['2_change'] = sample_df['2_max'].diff()
    sample_df['3_change'] = sample_df['3_max'].diff()
    
    fault_date_0 = sample_df[(sample_df['0_change']>=sample_df['0_change'].nlargest(15).mean()) | (sample_df['0_change']<=sample_df['0_change'].nsmallest(15).mean())].index[0]
    fault_date_1 = sample_df[(sample_df['1_change']>=sample_df['1_change'].nlargest(15).mean()) | (sample_df['1_change']<=sample_df['1_change'].nsmallest(15).mean())].index[0]
    fault_date_2 = sample_df[(sample_df['2_change']>=sample_df['2_change'].nlargest(15).mean()) | (sample_df['2_change']<=sample_df['2_change'].nsmallest(15).mean())].index[0]
    fault_date_3 = sample_df[(sample_df['3_change']>=sample_df['3_change'].nlargest(15).mean()) | (sample_df['3_change']<=sample_df['3_change'].nsmallest(15).mean())].index[0]
    
    date_list = [fault_date_0, fault_date_1, fault_date_2, fault_date_3]
    
    fault_induction_date = min(date_list)
    
    return date_list


# In[ ]:


machine_nos = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]


df_machine_and_fault = pd.DataFrame(columns=['Machine No.', 'Fault_0', 'Fault_1', 'Fault_2', 'Fault_3', 'Fault_Inception_Date'])
df_machine_and_fault['Machine No.'] = machine_nos
for machine_no in machine_nos:
    df_machine_and_fault.iloc[machine_no,1:5] = finding_first_fault(df_consolidated_rowwise.loc[machine_no])
    
#We will now select the earliest date of the dates identified for all 4 signals, since we want to be able to identify the fault at the earliest.

#Through deductive analysis, we will be excluding the Signal 2 from this because signal 2 has been found to contain 
#values that breach the set threshold, even before the fault has set in, i.e. in the normal mode of operation itself. 
#These can be considered as outliers, and since the other signals are contributing enough to the fault identification 
#this signal can be excluded.

df_machine_and_fault['Fault_Inception_Date'] = df_machine_and_fault[['Fault_0', 'Fault_1', 'Fault_3']].min(axis=1)
df_machine_and_fault


# In[ ]:




