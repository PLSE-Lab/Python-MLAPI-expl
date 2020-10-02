#!/usr/bin/env python
# coding: utf-8

# # Intro to raw spike data and the upper limit on model performance.
# 
# ## Intro
# We are now providing the raw data from which the data you have been competing on was derived. The raw data consists of every trial an image was shown to a neuron and exactly when the neuron spiked relative to the onset of that stimulus. One advantage to having the original trials is that you can get an estimate of the variability from trial-to-trial which in turn can give you a sense of how much noise vs signal there is in the data, which in turn can help you estimate how well you can perform on predicting a given neurons responses. There might be other advantages to having this finer grained dataset but its up to you to find them at the very least you may find the data interesting! So now I'll give you a tour of this new dataset then see if we can estimate the ceiling on performance. 
# 
# ### Loading spike data
# The spike data is stored in '.nc' files which are used by the xarray package for loading and saving data. xarray is a great package for exploring data when it is multidimensional e.g. it was largely developed for climate science where data is collected over latitude, longtidude, altitude and many measurements are made at each of those positions e.g. temperature, humidity, etc. This format works well for neural data where the dimensions are time, trials, neurons, stimuli, lets take a look:
# 

# In[ ]:


import xarray as xr #importing xarray
import pandas as pd # importing pandas to make comparison to processed data
import numpy as np
import os 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

#load the processed data you originally got
df = pd.read_csv('../input/train.csv')
df.index = df.Id#turn the first col to index
df = df.iloc[:,1:]#get rid of first col that is now index
stim = np.load('../input/stim.npy')

da = xr.open_dataset('../input/spike_data.nc')['resp']#read in the spike data (1gb)
print(da)


# So this is a DataArray of of all the spike data on the top line it list of the data dimensions and the number of items along that dimension. So there are 551 stimuli, up to 17 trials, 18 units, 800 discrete time points (each is a milisecond). In the coordinate listed below the top line the dimensions with  a  asterisk are indexing dimensions (meaning you use these to index) and without a asterisk are auxillary simply giving a description of the data (iso is the quality of isolation from 0 to 4, and stim_t is the time of the stimuli with respect to the beginning of the experiment).  Now lets look at some spike data:

# In[ ]:


plt.figure(figsize=(10,4))
plt.subplot(121)
plt.imshow(stim[50]);plt.xticks([]);plt.yticks([]);
plt.subplot(122)
#isel selects based on the order of the data (0-n) and sel based on the labels of the data.
#nans fill up where data was not collected so for this cell 5 trials were collected so the other 
#trials are filled with nans so I drop them along the trial axis.
da.isel(unit=4).sel(stim=50).dropna('trial').plot();#the 4th unit, the 50th stimulis


# So when the image on the left was shown the neuron had these responses across trials and time on the left. For the data on the left the time axis (labeled t) is seconds since the stimuli presentation where negative numbers are before and positive after. Each yellow line is the time when a spike arrived blue is when no spike arrived. You can see that the number of spikes increases rapidly after about 0.05 seconds post stimulus, this is the result of the time it takes for response from the retina to get to V4. 
# 
# One thing to notice is that the number of spikes is clearly different from trial to trial on some only one spike and on others a dozen. Thus any estimate of the expected number of spikes is going to be a noisy estimate. Since you are fitting a noisy estimate it is unlikely for you to be able to capture all the variance but how bad is it? Lets do a simulation! First I will work through the theory then the code feel free to skip to the bottom if you just want to see the results.
# 
# It is typicaly to model the number of spikes a neuron emits given a stimulus as Poisson distributed:
# $$Y_i \sim P(\lambda_i)$$
# Where $Y_i$ is the number of spikes in response to the $ith$ of $m$ stimuli and it is Poisson distributed with mean $\lambda_i$. 
# For a Poisson random variable the mean is equal to the variance: $$E[P(\lambda_i)] = Var[P(\lambda_i)] = \lambda_i$$ which is alot of variability!
# A typical experiment involves showing each of the $m$ stimuli $n$ times then averaging the response to reduce this variability and in addition it is heteroscedastic (different stimuli give responses with different variability):
# $$\bar{Y}_i = \frac{1}{n} \sum_j^n Y_{i,j} \sim \frac{1}{n} P(n \lambda_i)$$
# By performing a square root transformation and invoking the CLT the problem becomes easier. First lets see what happens when we take the square root of our responses:
# $$E[\sqrt{P(\lambda)}] \approx \sqrt{\lambda}$$ this is a reasonable approximation
# $$Var[\sqrt{P(\lambda)}] \approx \frac{1}{4}$$ the variance becomes a known constant thus the square root is a variance stabilizing transformation.
# 
# so $$\bar{Y}_i = \frac{1}{n} \sum_j^n \sqrt{Y_{i,j}} \sim \frac{1}{n} \sum_j^n{\sqrt{P(\lambda_i)}}$$
# 
# $$E[\bar{Y}_i] = \frac{1}{n} \sum_j^n E[\sqrt{Y_{i,j}}] \approx \sqrt{\lambda_i}   $$
# $$Var[\bar{Y}_i] = \frac{1}{n^2} \sum_j^n Var[\sqrt{Y_{i,j}}] \approx \frac{1}{4n}   $$
# 
# Finally invoking the CLT we approximate the average with normal random variables and have:
# $$\bar{Y}_i \sim N(\sqrt{\mu_i}, \frac{1}{4n})$$
# Great so we are in very friendly terrain a normal random variably with known variance! This is after a fair amount of approximations and assumptions the worst of which I would say is assuming the neurons are poisson distributed as often you will find their variance is higher than their mean. But as a first pass it should give us a sense of which neurons have good SNR and which don't.
# 
# So our simulation plan will be to take the average of the square root of responses then simulate the response to each stimulus as $N(\bar{Y_i}, \frac{1}{4n})$ and then with the 'perfect model' $\bar{Y_i}$ (the true means of our simulated responses) see what a typical r value is to it.

# In[ ]:


da_p = da.sel(t=slice(.05, .35))#grab only the times from 50ms to 350 ms post stimulus
da_p = da_p.transpose('unit', 'stim', 'trial', 't')#transpose for ease

#the overall array is 'ragged' units can have different number of stim, 
#and each stim can have different number of trials so we make a list of non-ragged arrays
units = [unit.dropna('stim',how='all').dropna('trial',how='any') for unit in da_p]
#get number of spikes and take sqrt
units = [unit.sum('t')**0.5 for unit in units]
#get number of trials for each unit
ns = [len(unit.coords['trial']) for unit in units]
#average number of spikes
m_units = [unit.mean('trial') for unit in units]


# In[ ]:


#simulation
sim_results = pd.DataFrame(np.zeros((len(ns), 2)), columns=['mean', 'sd'])
nsims = 500
for i, n, m_unit in zip(range(len(ns)), ns, m_units): 
    perfect_model = m_unit.values#perfect model is the mean to be used in the simulation
    sims = np.random.normal(loc=perfect_model,#use sample means as true means in sim
                            scale=(4*n)**-0.5,#variance is scaled by number of trials
                            size=(nsims,len(perfect_model)))#make n simulations
    #get simulated fraction variance explained by fitting the true means to noisy simulations.
    sim_r = [np.corrcoef(perfect_model, sim)[0,1]**2 for sim in sims]
    #store results
    sim_results['mean'][i] = np.mean(sim_r)
    sim_results['sd'][i] = np.std(sim_r)
sim_results
plt.errorbar(x=range(len(ns)), y=sim_results['mean'], yerr=2*sim_results['sd']);
plt.grid();plt.xlabel('Unit');plt.ylabel(r'$R^2$ upper limit');plt.ylim(0,1);
plt.title('Theoretical optimal model performance on neural data');


# So there you have it there are clearly some neurons responses which are fundamentally more predictable than others. I have been optimistic in my estimate of variance and many neurons may be more variable than what you see here in the simulation. But this should give you a sense looking at the quality of your fits which neurons you could do better on and which you might want to give up on (i.e. the first, second, tenth, and last unit are more difficult to predict).  You'll notice the low units are a subset of the units which did not perform well on the initial model in the Intro notebook. You could improve on this by estimating the variance instead of assuming its relation to the mean. If you are curious what makes some neurons easier than others it essentially comes down to dynamic range: did you have a set of stimuli over which the neuron had very different mean responses. In the worst case scenario the neuron barely responds on average to any stimuli and you end up only having trial-to-trial variability. One goal for a better model of these neurons is to more efficiently find stimuli to modulate neural responses thus increasing the signal to noise ratio and make model performance depend on the model and not trial-to-trial noise.
# 
# What other insights can you get from having the original trials and even the spike times!?

# 
