#!/usr/bin/env python
# coding: utf-8

# First lets import wizardhat, ble2lsl and the muse2016 device

# In[ ]:


import ble2lsl
from ble2lsl.devices import muse2016
from wizardhat import acquire, transform

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Next, lets set up a virtual (dummy) streamer with the ble2lsl library.

# In[ ]:


#if we had a device with us, we would use:
#streamer = ble2lsl.Streamer(muse2016)
#but if you're debugging or learning, use the dummy streamer with the command below:
streamer = ble2lsl.Dummy(muse2016)


# Lets take a look at some of the methods and properties of this streamer object

# In[ ]:


#After writing streamer. you can use the tab key to see a list of properties and methods that streamer has,
#for example, streamer.subscriptions shows all the subscribed data streams that the streamer object has picked up
#from the device.
streamer.subscriptions


# Now that you're connected to the device and it is streaming,  set up a data receiver object from wizardhat's receiver class. 

# In[ ]:


receiver = acquire.Receiver()


# Again, let's take a look to see what infromation is under the hood

# In[ ]:


receiver.buffers


# In[ ]:


receiver.ch_names


# In[ ]:


receiver.buffers['EEG'].data


# In[ ]:


receiver.buffers['EEG'].data.shape
#the default window for seeing data is 10 seconds. You can change that when you call acquire.Receiver(window=15) etc


# In[ ]:


receiver.buffers['EEG'].unstructured
#this version of the data has no labels and is just a pure numpy matrix


# In[ ]:


receiver.buffers['EEG'].get_timestamps()


# Let's try the record method and record some data

# In[ ]:


our_first_recording = receiver.record(5)
#wait 5 seconds after running this command 


# Now the variable calledo our_first_recording is a receiver instance that has all the usual properties and methods that our first receiver had. we can access the data, but this time instead of changing over a window, it will stay static and have a total of 5 seconds of data

# In[ ]:


our_first_recording.buffers['EEG'].data
#notice it only goes up to 5 seconds


# Next up, plotting.
# WizardHat has a Plot class built with a library called Bokeh, but we haven't added jupyter notebook support yet, so for now we will use another popular library called Matplotlib

# In[ ]:


channel_to_view = 'TP9'
samples_to_view = 2000


# In[ ]:


raw = receiver.buffers['EEG'].data[channel_to_view][-samples_to_view:]
time_raw = receiver.buffers['EEG'].data['time'][-samples_to_view:]

plt.subplots(figsize=(20,5))
plt.plot(time_raw,raw)


# Let's implement some filters on the data we collected and see what it looks like after filtering

# In[ ]:


lo_cut = 20
hi_cut = 50

filter = transform.Bandpass(receiver.buffers['EEG'],lo_cut,hi_cut)


# In[ ]:


raw = receiver.buffers['EEG'].data[channel_to_view][-samples_to_view:]
time_raw = receiver.buffers['EEG'].data['time'][-samples_to_view:]
filt = filter.buffer_out.data[channel_to_view][-samples_to_view:]
time_filt = filter.buffer_out.data['time'][-samples_to_view:]


# In[ ]:


plt.subplots(figsize=(20,5))
plt.plot(time_raw,raw)
plt.plot(time_filt,filt)


plt.xlabel('time (s)',fontsize=20)
plt.ylabel('voltage (mV)',fontsize=20)
plt.legend(['Raw signal','Filtered Signal'],fontsize=20)


# Now we will use PSD transformer  to generate the frequency power spectrum of our data 

# In[ ]:


pre_filter = transform.PSD(receiver.buffers['EEG'])
post_filter = transform.PSD(filter.buffer_out)


# Let's see if our filters worked by looking at the PSD of pre and post filter data

# In[ ]:


timestamp_to_view = pre_filter.buffer_out.get_timestamps(1)
pre_filter_data = pre_filter.buffer_out.data[['time',channel_to_view]]
post_filter_data = post_filter.buffer_out.data[['time',channel_to_view]]

psd_raw = pre_filter_data[pre_filter_data['time']==timestamp_to_view]
psd_filt = post_filter_data[post_filter_data['time']==timestamp_to_view]
psd_time = np.arange(0,len(psd_raw[channel_to_view].T))


# In[ ]:


plt.subplots(figsize=(20,5))
plt.plot(psd_time,psd_raw[channel_to_view].T)
plt.plot(psd_time,psd_filt[channel_to_view].T)


plt.xlabel('Freq (Hz)',fontsize=20)
plt.ylabel('Power',fontsize=20)
plt.axvline(x=lo_cut,color='red',linestyle='--')
plt.axvline(x=hi_cut,color='red',linestyle='--')
plt.legend(['Raw signal','Filtered Signal'],fontsize=20)
plt.title(f'Bandpass from {lo_cut} Hz to {hi_cut} Hz',fontsize=20)


# 
