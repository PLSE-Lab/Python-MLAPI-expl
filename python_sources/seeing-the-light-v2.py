#!/usr/bin/env python
# coding: utf-8

# ##Data exploration. 
# The dataset look at the day/night variation of protein expression in certain neurons. In this notebook we have explored some basic properties. Contents:
# 
#  1. Load the data
#  2. Plot how mean intensity varies over time
#  3. Compute periods numerically
#  4. Take a closer look at the subregions

# In[ ]:


import numpy as np # linear algebra
import matplotlib.pyplot as plt # graphs
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.ndimage.filters import uniform_filter # to smooth images

from sklearn.linear_model import LinearRegression

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# ## 1. Load the data. 
# I reshape some arrays to make them work with scikit-learn.

# In[ ]:


data = np.loadtxt('../input/data141110.csv', delimiter=',', skiprows=1)
image_no=data[:,0].reshape(-1,1)
frame_no=data[:,1].reshape(-1,1)
time_hrs=data[:,2].reshape(-1,1)
nb_images = data.shape[0]


# In[ ]:


flo_image_1 = np.load('../input/flo_image_1.npz')
flo_image_2 = np.load('../input/flo_image_2.npz')
image_ids = np.concatenate([flo_image_1['image_ids'], flo_image_2['image_ids']])
images = np.concatenate([flo_image_1['image_stack'], flo_image_2['image_stack']])
del flo_image_1, flo_image_2


# First a simple plot to show that there are exactly ten images each hour. This means that we will only really need 'images' for the analysis, and not the other data arrays.

# In[ ]:


plt.plot(np.arange(nb_images), time_hrs)


# And let's just plot one image to see what we're working with:

# In[ ]:


fig = plt.figure(figsize = (12,12))
plt.imshow(images[100], cmap='hot')


# As can be seen in the dataset description, this is a section of mouse brain. At the bottom is the optic nerve (cannot be seen). The two bright ovals are the suprachiasmatic nuclei and the vertical line at the top is the third ventricle. The light corresponds to a certain protein whose expression varies over the day/night cycle,  and our goal is to characterize this variation over time.

# ##2. Plot intensity over time
# Now we take the mean intensity over the whole image and plot it against time.

# In[ ]:


mean_values = np.mean(images, axis=(1,2))
plt.plot(time_hrs, mean_values, 'b')


# We can see that there is a linear trend towards higher values late in the week. To estimate this part, we use linear regression.

# In[ ]:


model = LinearRegression()
model.fit(time_hrs, mean_values)
plt.plot(time_hrs, model.predict(time_hrs), 'r')
plt.plot(time_hrs, mean_values, 'b')


# In[ ]:


mean_linear = model.predict(time_hrs)
norm_values = mean_values - mean_linear
plt.plot(time_hrs, norm_values, 'b')


# We see that the overall shape is periodic, but the graph also suggests a discontinuity in the data at 40 hours and 75 hours. We cannot really tell what happens there, but for a more detailed analysis it might be a good idea to analyse time < 40, and time > 75 separately. In this notebook, we will just apply a high pass filter when needed.

# We collect the linear regression in a function for later use.

# In[ ]:


def remove_linear_trend(time_series):
    x_axis = np.arange(len(time_series)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_axis, time_series)
    return time_series - model.predict(x_axis)


# ##3. Compute the periods
# A fast way to get the dominant period is the Fast Fourier Transform FFT: just take the frequency that gives the maximum value in the graph below:

# In[ ]:


spectrum = np.fft.rfft(norm_values)
freq = np.fft.rfftfreq(len(norm_values), 1/10.)  #because there are ten image frames per hour
plt.plot(1.0 / freq[3:], abs(spectrum[3:len(freq)]))


# This is however only approximate: we cannot this easily tell if the period is really 24 hrs or, say, 23 hours. Autocorrelation gives a more exact answer, here also showing a peak at 24 hours.

# In[ ]:


acorr = plt.acorr(norm_values, maxlags=300)


# Getting a more exact value for the period would be nice, in fact, this is possible if we combine the two methods and compute autocorrelation with the fft:

# In[ ]:


def filtered_acorr(time_series, high_pass=None, unbiased=True):
    """
    high_pass is a short 1D float array, 
    by which the low-frequency amplitudes are multiplied,
    e.g. high_pass = [0,0,0,0,0,0]
    """
    N = len(time_series)
    norm_values = remove_linear_trend(time_series)
    spectrum = np.fft.fft(norm_values, n=2*N)
    if high_pass is not None:
        spectrum[0] *= high_pass[0]
        for i in range(len(high_pass)):
            spectrum[i] *= high_pass[i]
            spectrum[-i] *= high_pass[i]
    acorr = np.real(np.fft.ifft(spectrum * np.conj(spectrum))[:N])
    if unbiased:
        return acorr / (N - np.arange(N))
    else:
        return acorr / N


# In[ ]:


def get_period(acorr):
    """
    Returns the index with largest acorr value, 
    after the first zero crossing.
    There are of course more sophisticated methods of doing this.
    """
    negative_periods = np.where(acorr <= 0.0)
    if negative_periods[0].size == 0:
        return 0
    first_zero = np.min(negative_periods)
    return first_zero + np.argmax(acorr[first_zero:])


# In[ ]:


print(get_period(filtered_acorr(mean_values)[:300]))


# This means that the dominant period for the whole dataset is 24.3 hours. 

# ## 4. Compare the smaller regions:
# We now divide the area into smaller squares to see what the local period is.

# In[ ]:


def get_grid_periods(images, box_size, max_period=300, high_pass=np.zeros((10,)), unbiased=False):
    """
    periods, acorrs = get_grid_periods(images, box_size, max_period=300, 
                                    high_pass=np.zeros((10,)), unbiased=False)
    Divides the image domain into small boxes of size box_size = (h,w) and computes
    the period over each of these small boxes."""
    h,w = box_size
    rows = images.shape[1] // h
    cols = images.shape[2] // w
    acorrs = np.empty((rows, cols, max_period), dtype = "float32")
    periods = np.empty((rows,cols, ), dtype = "int")
    for i in range(rows):
        for j in range(cols):
            time_series = np.mean(images[:, i*h:(i+1)*h, j*w:(j+1)*w], axis=(1,2))
            acorrs[i ,j] = filtered_acorr(time_series, high_pass = high_pass, 
                                          unbiased=unbiased)[:max_period]
            periods[i ,j] = get_period(acorrs[i, j])
    return periods, acorrs


# In[ ]:


periods, acorrs = get_grid_periods(images, (32, 32), max_period=1000, unbiased=False)
plt.imshow(periods, cmap='hot')
print(periods)


# This suggests that the period is slightly smaller (around 23 hours), in the center of the two bright nuclei, and close to 25 hours at the periphery. I am not sure that this is true, for several reasons:
# 
#  1. It doesn't really make much sense that light-sensitive neurons in a realistic setting vary in anything else than 24 hours. Over time, this would cause them to be completely out of phase with the day/night cycle.
#  2. As will be seen in the section below, there is still much noise/artifact in the data which would confuse our simple functions, even with the filters on. 

# We also collected the autocorrelations for subregions. Looking at a few of them, we see that they vary greatly in amplitude, but the period seems to be correct.

# In[ ]:


plt.plot(acorrs[13,5], color ='k')
plt.plot(acorrs[3,8], color='r')
plt.plot(acorrs[5,1], color='y')


# The results above suggest that all regions have period close to 24 hours. We can now look at some images separated by 12 hours and see if the light intensity varies as expected:

# In[ ]:


fix, ax = plt.subplots(1,8, figsize=(12,4))
idx = [80, 200, 320, 440, 560, 740, 860, 980]
for j in range(8):
    ax[j].set_axis_off()
    ax[j].imshow(images[idx[j],:,:] - mean_linear[idx[j]], cmap='hot', vmin=-800, vmax=6400)
        


# This does indeed show some day/night variation, but it also shows how the vertical line in the middle becomes gradually more intense, maybe explaining the linear trend we subtracted previously.  This can be seen more clearly if we instead select time frames corresponding to daylight:

# In[ ]:


fig, ax = plt.subplots(1,4,figsize=(12,12))
ax[0].imshow(images[100], cmap='hot', vmin=1500, vmax=6000)
ax[1].imshow(images[320], cmap='hot', vmin=1500, vmax=6000)
ax[2].imshow(images[1050], cmap='hot', vmin=1500, vmax=6000)
ax[3].imshow(images[1530], cmap='hot', vmin=1500, vmax=6000)


# I would guess that these slower changes have technical reasons such as diffusion/excretion of a contrast agent. I'll end this notebook here. I learnt a lot doing this, hope you enjoyed it as well!

# In[ ]:




