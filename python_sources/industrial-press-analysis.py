#!/usr/bin/env python
# coding: utf-8

# 
# **Industrial Press Analysis **
# 
# Hi,
# 
# 
# The Dataset "Press_data" is a collection of real energy consumption data of one aluminium press.
# The number of cycles is the supervised variable.
# The task is to find some correlations between energy consumptions and number of cycles.
# 
# For each press two energy meters are installed:
# 
# - One energy meter of the overall press (p2T_xx)
# - One Energy meter of the OVEN  (p2O_xx)
# 
# Sample time
# -----------------
# 30 s
# 
# Available Parameters:
# --------------------------
# 'DateTime',  
# 'p2T_133',    Press Active Power [W]
# 
# 'p2T_134',    Press Reactive Power [VAr]
# 
# 'p2T_135',    Press Power Factor
# 
# 'p2O_133',    Oven Active Power [W]
# 
# 'p2O_134',    Oven Reactive Power [VAr]
# 
# 'p2O_135',    Oven Power Factor
# 
# 'p2T_222d',   Press energy [Wh] (differential value over sample time)
# 
# 'p2O_222d',   Oven energy [Wh] (differential value over sample time)
# 
# 'plc1_1107d'  Press number or cycles [.] (differential value over sample time)
# 
# 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O import numpy as np # linear algebra(e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Load dataset 

# In[2]:


filename = '/df_see_press2_30s_20190422_h11m36.csv'
df = pd.read_csv("../input" +filename, sep=';',index_col=False, header=0);  
#Adapt Datetime values to matplotlib 
from datetime import  datetime
df['DateTime1']= df['DateTime'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

#Calcolate "Oil Pump Active Power" and Oil Pump Reactive Power"  (as difference of power  beetwwn press's and oven's energies)
df['p2P_133']  = df['p2T_133'] - df['p2O_133'] # Oil Pump Active Power
df['p2P_134']  = df['p2T_134'] - df['p2O_134'] # Oil Pump Reactive Power
df['p2P_222d']  = df['p2T_222d'] - df['p2O_222d'] # Oil Pump Reactive Power

print('df.shape: ', df.shape)

df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# DateTime range

# In[5]:


print('Min DateTime: ', df['DateTime1'].min())
print('Max DateTime: ', df['DateTime1'].max())


# Let's look inside some  parameters:
# - Oven Active Power -> it is correated to the quantity of melted and processed aluminium
# - Oil Pump Active Power (it is the difference between Press Active Power and Oven Active Power). 
# - Number of cycles
# 
# 

# In[6]:


import matplotlib.pyplot  as plt
import matplotlib.dates  as mdates

years     = mdates.YearLocator()   # every year
months    = mdates.MonthLocator()  # every month
days      = mdates.DayLocator()
hours      = mdates.HourLocator()
majorFmt     = mdates.DateFormatter('%m/%d/%Y') # %H:%M:%S')
minorFmt     = mdates.DateFormatter('%H:%M') 

n_plots = 6
fig, ax = plt.subplots(n_plots,1)

fig.autofmt_xdate()
for i in range(n_plots):
    ax[i].xaxis.set_major_formatter(majorFmt)
    ax[i].xaxis.set_major_locator(days)
    ax[i].xaxis.set_minor_formatter(minorFmt)

ax[0].set_ylabel('Press [W]')
ax[1].set_ylabel('Oven [W]')
ax[2].set_ylabel('Oil Pump [W]')
ax[3].set_ylabel('Oil Pump [Var]')
ax[4].set_ylabel('Oil Pump [Wh]')
ax[5].set_ylabel('Number of cycles')
ax[5].set_xlabel('time')

s_   = slice(0,len(df),1)
x    =   mdates.date2num(df['DateTime1'])[s_]
y0   =   df['p2T_133'][s_]
y1   =   df['p2O_133'][s_] 
y2   =   df['p2P_133'][s_]
y3   =   df['p2P_134'][s_]
y4   =   df['p2P_222d'][s_]
y5   =   df['plc1_1107d']

ax[0].plot(x,y0)
ax[1].plot(x,y1)
ax[2].plot(x,y2)
ax[3].plot(x,y3)
ax[4].plot(x,y4)
ax[5].plot(x,y5)
fig.set_size_inches(20,15)
plt.show()


# **Correlation matrix **
# 
# It seems that the best correlation is between number of cycles and the oil pump...
# 
# -> plc1_1107d = press cycles
# 
# It has some sense because the **oil pump** is the actuator that puts in movement the press.
# 
# -> p2P_133 = oil pump  Active Power 
# -> p2P_134 = oil pump  Reactive Power 
# 
# Among the Active Power and the Reactive Power the higher correlation is with the oil pump Reactive Power. Why?
# The oven is a "pure resistive" load, with no reactive power.
# Instead the oil pump is put in rotation by an asyncronous motor which absorbs a lot of reactive power. The reactive power of the pumps is equal of the press reactive power:
# 
#  ->  p2T_134 = p2P_134
# 

# In[7]:


print(df.columns)
#plt.matshow(df[['p2T_133', 'p2O_133', 'p2T_222d', 'p2O_222d', 'plc1_1107d']].corr())
flds = ['p2T_133', 'p2T_134','p2O_133', 'p2O_134', 'p2P_133','p2P_134', 'plc1_1107d']
flds = ['p2T_222d', 'p2O_222d',  'p2P_222d', 'p2P_134','plc1_1107d']
corr= df[flds].corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# Plot the pairwise relationships in the dataset.

# import seaborn as sns
# sns.pairplot(df[flds])
# 
# 

# In[ ]:


import seaborn as sns
sns.pairplot(df[flds])


# **Generator of input and output arrays**
# 
# AI_generator() is a generator which gives as output
#  - tensor sample() of shape (batches, lookback//step , n. data_in_parameters )
#  - tensor targets() of shape (batches)

# In[ ]:


def AI_generator( data_in,
                  data_out,              #data matrix
                  lookback,              #steps in the past
                  delay,                 #steps in the future
                  min_index, max_index,  #range of delimiters for validation  and test data
                  shuffle = False,      #if or not shuffle data timebased
                  batch_size = 128,      # number of samples per batch; samples = lookback//batch_Size
                  step = 1,              #periods in timesteps, at which you sample data
                  target_sum = True,     #sum of output data
              ):
    #Generator used to slice data in the AI model
    
    if max_index is None:
            max_index   =   len(data_in) -1
    
    i = min_index + lookback
    
    while 1:
        if shuffle: 
            rows = np.random.randint(min_index+lookback,  max_index, size = batch_size)
        else:
            if i+batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i+batch_size, max_index))
            i += len(rows)
        
        samples = np.zeros((len(rows),
                            lookback // step,
                            data_in.shape[-1]))
        targets = np.zeros ((len(rows),))
        
        for j, _ in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data_in[indices,:]
            
            #Two output options: actual number of cycles or summation of number of cycles
            if target_sum:
                targets[j] = sum([data_out[x] for x in range(rows[j]-lookback+delay, rows[j]+delay)])
                #print('sum_target %s %s: ', (j,targets[j]))
            else:
                targets[j] = data_out[rows[j] + delay]
                
        yield samples, targets
        
        


# Model
# Input and output tensors

# In[ ]:


float_data_in  = df[['p2T_133', 'p2T_134','p2O_133', 'p2O_134', 'p2P_133','p2P_134']].values
float_data_out = df[['plc1_1107d']].values
print('float_data_in.shape: ',float_data_in.shape)
print('float_data_out.shape: ',float_data_out.shape)
print('float_data_in:\n ', float_data_in)
print('float_data_out:\n ', float_data_out)


# In[ ]:


#Dictionary of configuration parameters
config_batchData = {
                'lookback'    : 10,          #number of points back for each batch of data
                'step'        : 1,          #numbers of rows step for each sample. samples = lookback//steps
                'delay'       : 0,          #delay from input data to temperature (one day)
                'shufle'      : False,      #shufle input order
                'batch_size'  : 4,          #number of data batches
                'target_sum'  : True,      #summarize last lookback data
                'steps_epochs': 64,        #64
                'epochs_'     : 100,
                'train_rate'  : 0.8,
                'val_rate'    : 0.15,
}


lookback        = config_batchData['lookback']     #number of rows for each batch of data
step            = config_batchData['step']          #numbers of rows step for each sample. samples = lookback//steps
delay           = config_batchData['delay']          #delay from input data to temperature (one day)
batch_size      = config_batchData['batch_size']         #number of data batches
target_sum      = config_batchData['target_sum']     #summarize last column data
steps_epochs    = config_batchData['steps_epochs']  #64
epochs_         = config_batchData['epochs_']
shufle          = config_batchData['shufle']


# In[ ]:



config_batchData = {
                'lookback'    : 10,          #number of points back for each batch of data
                'step'        : 1,          #numbers of rows step for each sample. samples = lookback//steps
                'delay'       : 0,          #delay from input data to temperature (one day)
                'shufle'      : False,      #shufle input order
                'batch_size'  : 4,          #number of data batches
                'target_sum'  : True,      #summarize last lookback data
                'steps_epochs': 64,        #64
                'epochs_'     : 100,
                'train_rate'  : 0.8,
                'val_rate'    : 0.15,
}


lookback        = config_batchData['lookback']     #number of rows for each batch of data
step            = config_batchData['step']          #numbers of rows step for each sample. samples = lookback//steps
delay           = config_batchData['delay']          #delay from input data to temperature (one day)
batch_size      = config_batchData['batch_size']         #number of data batches
target_sum      = config_batchData['target_sum']     #summarize last column data
steps_epochs    = config_batchData['steps_epochs']  #64
epochs_         = config_batchData['epochs_']
shufle          = config_batchData['shufle']


# In[ ]:


from keras import models, layers, Input
from keras.optimizers import RMSprop
model = models.Sequential()

if 1==1:
    #simple model with no activation function
    #no memory, 
    input_tensor = Input(shape=(lookback//step, float_data_in.shape[-1]))
    x = layers.Flatten()(input_tensor)
    x = layers.Dense(64,activation='relu')(x)
    x = layers.Dense(32,activation='relu')(x)
    output_tensor = layers.Dense(1)(x)
    model = models.Model(input_tensor, output_tensor)
    

if 1==0:
    model.add(layers.Conv1D(
                            32,3 ,
                            activation='relu',
                            input_shape = (None,  float_data_in.shape[-1])))

    
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(
                                32, 5,
                                activation='relu',
                                ))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.GRU(32, 
                             dropout = 0.1,
                             recurrent_dropout=0.5,
                             ))                  
    model.add(layers.Dense(1))
    
model.summary()


# In[ ]:


print('config_batchData: \n', config_batchData)

model.compile(  optimizer = RMSprop(), #lr=1e-4
                loss = 'mse', #'sparse_categorical_crossentropy', #'mse' 
                metrics = ['mae']
              )  

print('steps_epochs: ', steps_epochs)
print('epochs_: ', epochs_)
print('val_gen: ', val_gen)
print('val_steps: ', val_steps)

history = model.fit_generator(train_gen,
                              steps_per_epoch = steps_epochs, 
                              epochs=   epochs_,
                              validation_data = val_gen,
                              validation_steps = val_steps,
                              )


# In[ ]:


import matplotlib.pyplot as plt
loss_train   = history.history['loss']
loss_val     = history.history['val_loss']
epochs       = range(1,len(loss_train) + 1)

plt.plot(epochs, loss_train, 'bo', label = 'Training loss')
#plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.plot(epochs, loss_val, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# In[ ]:



#save model
model_name = 'model_p2_1.h5'
model.save(model_name) 



