#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import gc


# In[ ]:


tr = pd.read_csv('../input/train.csv', dtype={'acoustic_data': 'int16', 'time_to_failure': 'float64'})


# In[ ]:


earthquake_list = [-1, 5656573, 50085877, 104677355, 138772452, 187641819, 
                   218652629, 245829584, 307838916, 338276286, 375377847, 
                   419368879, 461811622, 495800224, 528777114, 585568143, 621985672, 629145479]


# In[ ]:


JUMP_AMOUNT = 40000

list_of_acoustics = []
list_of_earthquakes = []
list_of_ttfs = []

for i in tqdm(range(len(earthquake_list)-1)):
    jumper = 0
    # While you still have the ability to jump forward in your earthquake
    while jumper < (earthquake_list[i+1] - earthquake_list[i] - 150000):
        # Append numpy array of length 150,000 to your training set
        list_of_acoustics.append(tr['acoustic_data'].iloc[(earthquake_list[i]+1+jumper):(earthquake_list[i]+1+jumper+150000)].values)
        
        # Append which earthquake you're recording
        list_of_earthquakes.append(i)
        
        # Append the time to failure
        list_of_ttfs.append(tr['time_to_failure'].iloc[earthquake_list[i]+1+jumper+150000])
        
        # Add the JUMP_AMOUNT
        jumper += JUMP_AMOUNT
        
    # Now, jumper is too much, so let's manually add the last 150,000 before the time to failure
    # To capture this special moment
    list_of_acoustics.append(tr['acoustic_data'].iloc[(earthquake_list[i+1]+1-150000):(earthquake_list[i+1]+1)])
    list_of_earthquakes.append(i)
    list_of_ttfs.append(tr['time_to_failure'].iloc[earthquake_list[i+1]])


# In[ ]:


del tr
gc.collect()


# In[ ]:


training_set = np.vstack(list_of_acoustics)


# In[ ]:


training_set.shape


# In[ ]:


list_of_earthquakes = np.array(list_of_earthquakes)
list_of_ttfs = np.array(list_of_ttfs)


# In[ ]:


training_set = training_set.reshape(training_set.shape[0],training_set.shape[1],1)


# In[ ]:


list_of_ttfs = list_of_ttfs.reshape(list_of_ttfs.shape[0], 1, 1)


# In[ ]:


from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, Lambda, concatenate, Flatten
from keras.optimizers import Adam


# In[ ]:


# convolutional layer parameters
n_filters = 4
filter_width = 10
dilation_rates = [2**i for i in range(4)] 


# In[ ]:


# define an input history series and pass it through a stack of dilated causal convolutions. 
history_seq = Input(shape=(None, 1))
x = history_seq


# In[ ]:


for dilation_rate in dilation_rates:
    x = Conv1D(filters=n_filters,
               kernel_size=filter_width, 
               padding='causal',
               dilation_rate=dilation_rate)(x)


# In[ ]:


x = Dense(128, activation='relu')(x)
x = Dropout(.2)(x)
out = Dense(1)(x)


# In[ ]:


model = Model(history_seq, out)


# In[ ]:


model.summary()


# In[ ]:


first_n_samples = 15000
batch_size = 16
epochs = 10


# In[ ]:


model.compile(Adam(), loss='mean_absolute_error')


# In[ ]:


history = model.fit(training_set[:first_n_samples], np.array(list_of_ttfs)[:first_n_samples],
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data = (training_set[15000:], np.array(list_of_ttfs)[15000:]))


# In[ ]:




