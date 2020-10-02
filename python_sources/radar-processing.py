#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# This notebook provides some examples of the various levels of data processing for airborne radar sounding data. You scroll through the rendered notebook for the lecture notes, code snippets, and plots, or click on "Fork" in the upper right hand corner of your screen to open an interactive session where you can adjust parameters and run the code snippets yourself.

# In[ ]:


# Load packages for processing 

import math                  # basic math operations
import numpy as np           # data matrix manipulation
import scipy.io as sio       # matlab file reads
import matplotlib as plt     # plots
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import netcdf as nc      # read netcdf files for UTIG data load

# Sets the size of all inline plots
plt.rcParams['figure.figsize'] = [20, 12]

# Load physical constants and site locations
c = 3e8                           # Speed of light in a vaccuum 
n_ice = math.sqrt(3.17)           # index of refraction for glacial ice

# Reference trace locations for the radargrams 
lat_rough = 75.1525
lon_rough = -56.202423
lat_smooth = 77.121733
lon_smooth = -50.458014

# Load raw radar data example
tmp1 = sio.loadmat('../input/greenland-mcords3/RawRadarExample.mat', struct_as_record = False, squeeze_me = True)
raw = tmp1['data'][3000:7000:1,::]

# Load the pulse compressed data examples
tmp2 = sio.loadmat('../input/greenland-mcords3/rds_20140426_01_041.mat', struct_as_record = False, squeeze_me = True)
pc_rough = tmp2['seg']
tmp3 = sio.loadmat('../input/greenland-mcords3/rds_20140508_01_061.mat', struct_as_record = False, squeeze_me = True)
pc_smooth = tmp3['seg'] 

# Load the SAR processed data examples
tmp4 = sio.loadmat('../input/greenland-mcords3/sar_20140426_01_041.mat', struct_as_record = False, squeeze_me = True)
sar_rough = tmp4['seg']
tmp5 = sio.loadmat('../input/greenland-mcords3/sar_20140508_01_061.mat', struct_as_record = False, squeeze_me = True)
sar_smooth = tmp5['seg']

get_ipython().run_line_magic('reset_selective', '-f tmp # clear temporary variables from memory')


# In[ ]:


# Sets the size of the inline plot
plt.rcParams['figure.figsize'] = [20,12]

# Plot raw radargram

fig1, ax1 = plt.subplots()
raw_im = ax1.imshow(raw, cmap = plt.cm.Blues, aspect = 'auto', vmin = -1000, vmax = 1000)
ax1.set_title("Raw Radar Data", fontsize = 24, fontweight = 'bold')
ax1.set_xlabel("Trace Number", fontsize = 18, fontweight = 'bold')
ax1.set_ylabel("Fast Time Sample Number", fontsize = 18, fontweight = 'bold')


# **Raw Radar Data**
# 
# The first thing we'll take a look at is the raw radar data with no processing applied other than the onboard real-time processing that occurs on the airplane. The main reason for looking at this data is to understand what you would get if you ask a radar engineer for their raw data. As you can see, raw data is pretty much useless for science. In fact, about all we can tell from this radargram is that there were some reflections from something that happened at some time. If we want to perform even a basic visual analysis of the ice sheet cross-section, we will need to process the data further. 

# In[ ]:


# Plot a single trace from the raw radargram 

plt.plot(raw[:,1000])
plt.xlabel('Fast Time Sample Number', fontsize = 18, fontweight = 'bold')
plt.ylabel('Amplitude', fontsize = 18, fontweight = 'bold')
plt.title('Single Trace - Raw Radar Data')
plt.show()


# Looking a single radar trace (which represents a single transmit and receive event, or a depth vs power profile at a single point on the ice sheet) does not offer any additional information.

# **Pulse Compression**
# 
# The most basic level of data processing required to form an interpretable radargram is pulse compression. In this processing step, we cross-correlate a copy of the transmitted radar pulse with each trace in the radargram. This process concentrates the energy from each reflection event into a sharp peak at the appropriate time delay. As a result, pulse compression dramatically improves our resolution along the range/depth axis and also greatly increases the signal to noise ratio (SNR). The range resolution will be proportional to the inverse of the radar's bandwidth (so high bandwidth = good resolution) and the improvements in SNR will be proportional to the time-bandwidth product (so large bandwidths and long pulse lengths will result in the greatest increase in SNR).

# In[ ]:


# Plot the rough topography pulse compressed data

depth_rough = 0.5*(pc_rough.Time - pc_rough.Time[150])*(c/n_ice)
box = np.array([np.amin(pc_rough.Along_Track)/1000, np.amax(pc_rough.Along_Track)/1000, depth_rough[1000], depth_rough[0]])

fig2, ax2 = plt.subplots()
pcr_im = ax2.imshow(10*np.log10(np.square(np.abs(pc_rough.Data[0:1000:1,::]))), cmap = plt.cm.Blues, aspect = 'auto', vmin = -180, vmax = -40,                     extent = box)
ax2.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
ax2.set_xlabel('Distance Along Track (km)', fontsize = 18, fontweight = 'bold')
ax2.set_title('Pulse Compressed Data - Rough Topography', fontsize = 24, fontweight = 'bold')
fig2.colorbar(pcr_im, ax = ax2)


# We first look at the pulse compressed data for an area along the margins of Northwest Greenland which exhibits fairly rough bed and surface topography. After pulse compression, we can clearly distinguish the ice sheet surface and bed. However, many of the englacial details are hidden by noise and overlapping hyperbola tails from rough surface scatterers. The true geometry of the bed is also hard to distingiush because of the hyperbola tails from rough bed scattering. 

# In[ ]:


# Plot a representative trace from the rough topography pulse compressed data

ind_rough_pc = np.argmin(np.abs(pc_rough.Latitude - lat_rough + pc_rough.Longitude - lon_rough))

plt.plot(depth_rough, np.asarray(10*np.log10(np.square(np.absolute(pc_rough.Data[:,ind_rough_pc])))))
plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')
plt.title('Single Trace - Pulse Compressed Data - Rough Topography', fontsize = 24, fontweight = 'bold')
plt.show()


# A single trace from pulse compressed data is difficult to interprete. We might be tempted to guess that the oscillations in the trace between the surface and bed are representative of englacial features, but we know from the full radargrams this is really just noise. We can see very clearly in this plot that the noise variance is quite high. The true bed reflection is also difficult to pick out, due to the long rise and fall time from extra energy folding in from off-nadir scattering angles. 

# In[ ]:


# Plot the smooth topography pulse compressed data

depth_smooth = 0.5*(pc_smooth.Time - pc_smooth.Time[98])*(c/n_ice)
box = np.array([np.amin(pc_smooth.Along_Track)/1000, np.amax(pc_smooth.Along_Track)/1000, depth_smooth[1080], depth_smooth[0]])

fig2, ax2 = plt.subplots()
pcr_im = ax2.imshow(10*np.log10(np.square(np.abs(pc_smooth.Data[0:1080:1,::]))), cmap = plt.cm.Blues, aspect = 'auto', vmin = -200, vmax = -50,                     extent = box)
ax2.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
ax2.set_xlabel('Distance Along Track (km)', fontsize = 18, fontweight = 'bold')
ax2.set_title('Pulse Compressed Data - Smooth Topography', fontsize = 24, fontweight = 'bold')
fig2.colorbar(pcr_im, ax = ax2)


# This next radargrams example is from an interior region of Northwest Greenland with low surface and bed relief and slow flowing ice. The pulse compressed data performs much better in this region. We can clearly distingush the surface and the bed, along with englacial layers. However, there is still some noise in the near surface, and a significant gap above the bed where no layers exceed the noise floor. 

# In[ ]:


# Plot a representative trace from the smooth topography pulse compressed data

ind_smooth_pc = np.argmin(np.abs(pc_smooth.Latitude - lat_smooth + pc_smooth.Longitude - lon_smooth))

plt.plot(depth_smooth, np.asarray(10*np.log10(np.square(np.absolute(pc_smooth.Data[:,ind_smooth_pc])))))
plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')
plt.title('Single Trace - Pulse Compressed Data - Smooth Topography', fontsize = 24, fontweight = 'bold')
plt.show()


# Plotting a single trace from this topographically smooth area, we can see the surface and bed exhibit much sharper and more distinguishable returns compared to the rough area near the margins. However, noise variance is still high and it is still difficult to distinguish englacial layer returns from noise in the single trace. 

# **When to Use Pulse Compressed Data**
# 
# You might typically choose to use data which has only been pulse compressed when you are concerned with the scattering behavior of the ice sheet. This includes most types of sub-resolution roughness or scattering character analyses. Example would include analyzing the angular scattering function from the doppler spectrum or calculating coherence length (Oswald and Gogineni, 2008). A second reason might be to preserve a sufficient number of measurements over approximately the same location in the ice sheet in order to conduct some kind of statistical analysis of the scattering properties. Radio Statistial Reconnaissance (Grima, 2012) would be once such example.

# **Unfocused Data**
# 
# The next level of processing is often referred to as coherent summation, stacking, or unfocused SAR data processing. The idea is simple. In order to improve our SNR, we add consecutive complex valued traces within some window together. In the example below, we use 20 coherent summations, so 20 traces are added together to form one. Because we are adding the complex valued traces, the reflections from flat horiztonally continuous features will add constructively. On the other hand, noise and clutter, such as the hyperbola tails, will add destructively between traces. So coherent summation will improve both our SNR and our signal to clutter ratio (SCR).

# In[ ]:


# Implements coherent summation 

window = 20

width_rough = int(np.floor(pc_rough.Data.shape[1]/window))
uf_rough = {}
uf_rough["Data"] = np.empty((pc_rough.Data.shape[0], width_rough), dtype = np.complex64)
uf_rough["Latitude"] = np.empty(width_rough)
uf_rough["Longitude"] = np.empty(width_rough)
uf_rough["Along_Track"] = np.empty(width_rough)
uf_rough["Time"] = pc_rough.Time

for k in range(width_rough):
    uf_rough["Data"][:,k] = np.sum(pc_rough.Data[:,k*window:(k+1)*window:1],axis=1)
    uf_rough["Latitude"][k] = np.mean(pc_rough.Latitude[k*window:(k+1)*window:1])
    uf_rough["Longitude"][k] = np.mean(pc_rough.Longitude[k*window:(k+1)*window:1])
    uf_rough["Along_Track"][k] = np.mean(pc_rough.Along_Track[k*window:(k+1)*window:1])

width_smooth = int(np.floor(pc_smooth.Data.shape[1]/window))
uf_smooth = {}
uf_smooth["Data"] = np.empty((pc_smooth.Data.shape[0], width_smooth), dtype = np.complex64)
uf_smooth["Latitude"] = np.empty(width_smooth)
uf_smooth["Longitude"] = np.empty(width_smooth)
uf_smooth["Along_Track"] = np.empty(width_smooth)
uf_smooth["Time"] = pc_smooth.Time

for k in range(width_smooth):
    uf_smooth["Data"][:,k] = np.sum(pc_smooth.Data[:,k*window:(k+1)*window:1],axis=1)
    uf_smooth["Latitude"][k] = np.mean(pc_smooth.Latitude[k*window:(k+1)*window:1])
    uf_smooth["Longitude"][k] = np.mean(pc_smooth.Longitude[k*window:(k+1)*window:1])
    uf_smooth["Along_Track"][k] = np.mean(pc_smooth.Along_Track[k*window:(k+1)*window:1])


# In[ ]:


# Plot the rough topography unfocused data

depth_rough_uf = 0.5*(uf_rough["Time"] - uf_rough["Time"][150])*(c/n_ice)
box = np.array([np.amin(uf_rough["Along_Track"])/1000, np.amax(uf_rough["Along_Track"])/1000, depth_rough_uf[1000], depth_rough_uf[0]])

fig2, ax2 = plt.subplots()
pcr_im = ax2.imshow(10*np.log10(np.square(np.abs(uf_rough["Data"][0:1000:1,::]))), cmap = plt.cm.Blues, aspect = 'auto', vmin = -150, vmax = -30,                     extent = box)
ax2.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
ax2.set_xlabel('Distance Along Track (km)', fontsize = 18, fontweight = 'bold')
ax2.set_title('Unfocused Data - Rough Topography', fontsize = 24, fontweight = 'bold')
fig2.colorbar(pcr_im, ax = ax2)


# We can see significant improvement in the radargram from the topographically rough area. Most of the hyperbola tails are gone and englacial layers are now clearly identifiable. We can start to distinguish the true shape of the bed now that some of the off-nadir scattering has canceled out. You may note, however, many regions where the englacial layers fade out or disappear. This illustrates one of the primary disadvantages of coherent summation - while it is very effective for improving the SNR of flat features in the ice, it can actually make the SNR of steeply sloped layers worse, since the reflections are not in phase between traces due to the slope. 

# In[ ]:


# Plot the smooth topography unfocused data

depth_smooth_uf = 0.5*(uf_smooth["Time"] - uf_smooth["Time"][98])*(c/n_ice)
box = np.array([np.amin(uf_smooth["Along_Track"])/1000, np.amax(uf_smooth["Along_Track"])/1000, depth_smooth_uf[1080], depth_smooth_uf[0]])

fig2, ax2 = plt.subplots()
pcr_im = ax2.imshow(10*np.log10(np.square(np.abs(uf_smooth["Data"][0:1080:1,::]))), cmap = plt.cm.Blues, aspect = 'auto', vmin = -150, vmax = -30,                     extent = box)
ax2.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
ax2.set_xlabel('Distance Along Track (km)', fontsize = 18, fontweight = 'bold')
ax2.set_title('Unfocused Data - Smooth Topography', fontsize = 24, fontweight = 'bold')
fig2.colorbar(pcr_im, ax = ax2)


# While the improvements are less dramatic for in the smooth bed region, we can still see that the layers in the near surface are more clearly defined, now that some of the firn clutter has been eliminated, and there are a few deep layers near the bed which are now visible above the noise floor. One of the things we should note is that radargrams still look quite "grainy". This effect is known as speckle and is the result of coherent power fluctuations between traces. 

# In[ ]:


# Plot a representative trace from the rough topography unfocused data

ind_rough_uf = np.argmin(np.abs(uf_rough["Latitude"]- lat_rough + uf_rough["Longitude"] - lon_rough))

plt.plot(depth_rough_uf, np.asarray(10*np.log10(np.square(np.absolute(uf_rough["Data"][:,ind_rough_uf])))))
plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')
plt.title('Single Trace - Unfocused Data - Rough Topography', fontsize = 24, fontweight = 'bold')
plt.show()


# Here is an individual radar trace from the same location as the previous example in the topographically rough region near the margins. We can see that the bed echo tail falls off more quickly now, there is less clutter power in teh near surface, and we can clearly resolve the surface multiple reflection.

# In[ ]:


# Plot a representative trace from the smooth topography unfocused data

ind_smooth_uf = np.argmin(np.abs(uf_smooth["Latitude"]- lat_smooth + uf_smooth["Longitude"] - lon_smooth))

plt.plot(depth_smooth_uf, np.asarray(10*np.log10(np.square(np.absolute(uf_smooth["Data"][:,ind_smooth_uf])))))
plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')
plt.title('Single Trace - Unfocused Data - Smooth Topography', fontsize = 24, fontweight = 'bold')
plt.show()


# In the smooth, interior region, we see a distinct improvement in the abruptness and SNR of the bed echo, as well as a number of englacial layers which are well above the noise floor. 

# **When to Use Unfocused Data**
# 
# Purely unfocused data might be appropriate if you need to analyze features with low signal to noise ratio but wish to preserve the phase of the signal. Some of the analyses mentioned under pulse compression could still be accomplished with data that uses a small amount of coherent summation, though it tends to limit the resolution in many cases. 

# **Multi-Loooking**
# 
# Similar to coherent summation, we can also peform incoherent averaging of adjacent traces, which is often called multi-looking. Now, instead of adding complex valued traces, we first calculate the power (square of the absolute values of the trace) and then add the traces. Multi-looking is particularly effective at reducing noise variance and smoothing out speckle. 

# In[ ]:


# Implement Multi-Looking

window = 10

width_rough = int(np.floor(uf_rough["Data"].shape[1]/window))
ml_rough = {}
ml_rough["Data"] = np.empty((uf_rough["Data"].shape[0], width_rough))
ml_rough["Latitude"] = np.empty(width_rough)
ml_rough["Longitude"] = np.empty(width_rough)
ml_rough["Along_Track"] = np.empty(width_rough)
ml_rough["Time"] = uf_rough["Time"]

for k in range(width_rough):
    ml_rough["Data"][:,k] = np.sum(np.square(np.abs(uf_rough["Data"][:,k*window:(k+1)*window:1])),axis=1)
    ml_rough["Latitude"][k] = np.mean(uf_rough["Latitude"][k*window:(k+1)*window:1])
    ml_rough["Longitude"][k] = np.mean(uf_rough["Longitude"][k*window:(k+1)*window:1])
    ml_rough["Along_Track"][k] = np.mean(uf_rough["Along_Track"][k*window:(k+1)*window:1])

width_smooth = int(np.floor(uf_smooth["Data"].shape[1]/window))
ml_smooth = {}
ml_smooth["Data"] = np.empty((uf_smooth["Data"].shape[0], width_smooth))
ml_smooth["Latitude"] = np.empty(width_smooth)
ml_smooth["Longitude"] = np.empty(width_smooth)
ml_smooth["Along_Track"] = np.empty(width_smooth)
ml_smooth["Time"] = uf_smooth["Time"]

for k in range(width_smooth):
    ml_smooth["Data"][:,k] = np.sum(np.square(np.abs(uf_smooth["Data"][:,k*window:(k+1)*window:1])),axis=1)
    ml_smooth["Latitude"][k] = np.mean(uf_smooth["Latitude"][k*window:(k+1)*window:1])
    ml_smooth["Longitude"][k] = np.mean(uf_smooth["Longitude"][k*window:(k+1)*window:1])
    ml_smooth["Along_Track"][k] = np.mean(uf_smooth["Along_Track"][k*window:(k+1)*window:1])


# In[ ]:


# Plot the rough topography multi-looked data

depth_rough_ml = 0.5*(ml_rough["Time"] - ml_rough["Time"][150])*(c/n_ice)
box = np.array([np.amin(ml_rough["Along_Track"])/1000, np.amax(ml_rough["Along_Track"])/1000, depth_rough_ml[1000], depth_rough_ml[0]])

fig2, ax2 = plt.subplots()
pcr_im = ax2.imshow(10*np.log10(ml_rough["Data"][0:1000:1,::]), cmap = plt.cm.Blues, aspect = 'auto', vmin = -140, vmax = -10,                     extent = box)
ax2.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
ax2.set_xlabel('Distance Along Track (km)', fontsize = 18, fontweight = 'bold')
ax2.set_title('Multi-Looked Data - Rough Topography', fontsize = 24, fontweight = 'bold')
fig2.colorbar(pcr_im, ax = ax2)


# We can see that multi-looking has smoothed out the radargram and made it much less grainy. However, we are also begining to lose some along-track resolution due to summing so many traces, so the radargram is not as sharp. When deciding how many coherent and incoherent summations to perform, it can be important to think about what final spacing between traces is acceptable for your application and what the intrinsic resolution of your system is. In general, resolution in the along-track direction for radar sounders is limited by the diameter of the first Fresnel zone. So as long as your final spacing between traces is less than that value, you have not degraded your resolution. 

# In[ ]:


# Plot the smooth topography multi-looked data

depth_smooth_ml = 0.5*(ml_smooth["Time"] - ml_smooth["Time"][98])*(c/n_ice)
box = np.array([np.amin(ml_smooth["Along_Track"])/1000, np.amax(ml_smooth["Along_Track"])/1000, depth_smooth_ml[1080], depth_smooth_ml[0]])

fig2, ax2 = plt.subplots()
pcr_im = ax2.imshow(10*np.log10(ml_smooth["Data"][0:1080:1,::]), cmap = plt.cm.Blues, aspect = 'auto', vmin = -140, vmax = -10,                     extent = box)
ax2.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
ax2.set_xlabel('Distance Along Track (km)', fontsize = 18, fontweight = 'bold')
ax2.set_title('Multi-Looked Data - Smooth Topography', fontsize = 24, fontweight = 'bold')
fig2.colorbar(pcr_im, ax = ax2)


# The effects are similar for topographically smooth area. 

# In[ ]:


# Plot a representative trace from the rough topography multi-looked data

ind_rough_ml = np.argmin(np.abs(ml_rough["Latitude"]- lat_rough + ml_rough["Longitude"] - lon_rough))

plt.plot(depth_rough_ml, np.asarray(10*np.log10(ml_rough["Data"][:,ind_rough_ml])))
plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')
plt.title('Single Trace - Multi-Looked Data - Rough Topography', fontsize = 24, fontweight = 'bold')
plt.show()


# The single-trace view is where the effects of multi-looking are most evident. We can see that the noise variance has been signficantly reduced and we can be much more confident in selecting specific peaks which exceed the noise floor and may be evidence of specific reflections in the ice sheet.

# In[ ]:


# Plot a representative trace from the smooth topography multi-looked data

ind_smooth_ml = np.argmin(np.abs(ml_smooth["Latitude"]- lat_smooth + ml_smooth["Longitude"] - lon_smooth))

plt.plot(depth_smooth_ml, np.asarray(10*np.log10(ml_smooth["Data"][:,ind_smooth_ml])))
plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')
plt.title('Single Trace - Multi-Looked Data - Smooth Topography', fontsize = 24, fontweight = 'bold')
plt.show()


# Similar properties are evident in the smooth region and englacial layers are clearly distinguishable in the trace, even near the bed where they approach the noise floor. 

# **When to Use Multi-Looked Data**
# 
# For any analysis which does not require phase information, multi-looking is probably desirable. Examples include bed reflectivity analysis and many of the empirical attenuation techniques mentioned in our earlier session. 

# **Synthetic Aperture Processing**
# 
# The final level of processing is synthetic aperture (SAR) processing. There are a number of algorithms which can be used, but the overarching concept is as follows. Because the aircraft is moving as it collects each trace, energy which comes from behind or infront of the airplane, rather than from directly below it, will experience some Doppler shift. Therefore, we can distinguish where the scattered energy arrived from based on frequency and use some SAR algorithm to migrate is back to the appropriate location in the radargram. As a result, SAR processing will improve our along-track resolution proportional to inverse of the aperture length over which we migrate the energy, improve out SNR proportional to the length of the aperture, and improve our SCR by migrating hyperbola tails back to their scattering center. We can also apply multi-looking after SAR processing to get the final improvement in noise variance. 

# In[ ]:


# Plot the rough topography SAR focused data

rough_left = np.argmin(np.abs(sar_rough.Latitude - pc_rough.Latitude[0] + sar_rough.Longitude - pc_rough.Longitude[0]))
rough_right = np.argmin(np.abs(sar_rough.Latitude - pc_rough.Latitude[-1] + sar_rough.Longitude - pc_rough.Longitude[-1]))

depth_rough_sar = 0.5*(sar_rough.Time - sar_rough.Time[101])*(c/n_ice)
box = np.array([np.amin(sar_rough.Along_Track[rough_left:rough_right])/1000, np.amax(sar_rough.Along_Track[rough_left:rough_right])/1000, depth_rough_sar[1000], depth_rough_sar[0]])

fig2, ax2 = plt.subplots()
pcr_im = ax2.imshow(10*np.log10(sar_rough.Data[0:1000:1,rough_left:rough_right]), cmap = plt.cm.Blues, aspect = 'auto', vmin = -180, vmax = -40,                     extent = box)
ax2.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
ax2.set_xlabel('Distance Along Track (km)', fontsize = 18, fontweight = 'bold')
ax2.set_title('Focused Data - Rough Topography', fontsize = 24, fontweight = 'bold')
fig2.colorbar(pcr_im, ax = ax2)


# Looking at the SAR focused radargram for the topographically rough area, we can see that this processing makes a significant difference. The true geometry of the bed is fairly clear and englacial layers are evident. Compared to the radargram formed by coherent summation, the englacial layers are still resolved for a greater range of slopes. 

# In[ ]:


# Plot the smooth topography SAR focused data

smooth_left = np.argmin(np.abs(sar_smooth.Latitude - pc_smooth.Latitude[0] + sar_smooth.Longitude - pc_smooth.Longitude[0]))
smooth_right = np.argmin(np.abs(sar_smooth.Latitude - pc_smooth.Latitude[-1] + sar_smooth.Longitude - pc_smooth.Longitude[-1]))

depth_smooth_sar = 0.5*(sar_smooth.Time - sar_smooth.Time[102])*(c/n_ice)
box = np.array([np.amin(sar_smooth.Along_Track[smooth_left:smooth_right])/1000, np.amax(sar_smooth.Along_Track[smooth_left:smooth_right])/1000, depth_smooth_sar[1080], depth_smooth_sar[0]])

fig2, ax2 = plt.subplots()
pcr_im = ax2.imshow(10*np.log10(sar_smooth.Data[0:1080:1,smooth_left:smooth_right]), cmap = plt.cm.Blues, aspect = 'auto', vmin = -180, vmax = -40,                     extent = box)
ax2.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
ax2.set_xlabel('Distance Along Track (km)', fontsize = 18, fontweight = 'bold')
ax2.set_title('Focused Data - Smooth Topography', fontsize = 24, fontweight = 'bold')
fig2.colorbar(pcr_im, ax = ax2)


# The difference from SAR processing in the topographically smooth area is much less noticeable and does not provided significant improvements over simple coherent summation. This is because what SAR processing does best is deal with the migration of energy which was scattered at off-nadir angles. But in a smooth region, the majority of the energy comes from direct reflection below the radar. For an area like this, the benefits of SAR processing may not be worth the additional computational complexity and time required. 

# In[ ]:


# Plot a representative trace from the rough topography multi-looked data

ind_rough_sar = np.argmin(np.abs(sar_rough.Latitude- lat_rough + sar_rough.Longitude - lon_rough))

plt.plot(depth_rough_sar, np.asarray(10*np.log10(sar_rough.Data[:,ind_rough_sar])))
plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')
plt.title('Single Trace - Focused Data - Rough Topography', fontsize = 24, fontweight = 'bold')
plt.show()


# We can see in the single trace that full SAR processing combines all of the improvements from each of the individual processing step we considered previously.

# In[ ]:


# Plot a representative trace from the smooth topography SAR focused data

ind_smooth_sar = np.argmin(np.abs(sar_smooth.Latitude- lat_smooth + sar_smooth.Longitude - lon_smooth))

plt.plot(depth_smooth_sar, np.asarray(10*np.log10(sar_smooth.Data[:,ind_smooth_sar])))
plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')
plt.title('Single Trace - Focused Data - Smooth Topography', fontsize = 24, fontweight = 'bold')
plt.show()


# Again, the improvements are less dramatic in smooth regions of the ice sheet. 

# **When to Use SAR Focused Data**
# 
# All of the analyses for which you would use multi-looked data can also be be accomplished using SAR focused data. But, as we saw, SAR focusing may be an unecessary computational expense in well-behaved regions of the ice sheet.
# 
# Where you should almost always use SAR data is for questions of large scale geometry because the migration process will ensure that the point of maximum return in the radargram is as near to the true scattering center as possible. This includes measurements of ice thickness, feature geometry, and large scale bed or surface topography.

# **System Comparison**
# 
# We will now look at three of the major airborne deep radar sounders in operation today: the Multichannel Coherent Depth Sounder (MCoRDS) operated by the University of Kansas Center for the Remote Sensing of Ice Sheets (CReSIS), the High Capability Radar Sounder (HiCARS) from the University of Texas Institute of Geophysics, and the Polarimetric Radar Airborne Science Instrument (PASIN) from the British Antarctic Survey. We will use Dome C, Antarctica as a comparison area.
# 
# The CReSIS data offers good depth and along-track resolution and good radiometric balance. The modern system operates at 195 MHz with a typical range resolution in ice of around 3 meters (along track resolution will varying by processing technique). You may notice some areas of receiver fading where the aircraft banked on a tight turn, as well as intereference streaks. This is one of the prices of a highly sensitive instrument with wide bandwidth, high dynamic range, and a more directive antenna array. However, this is about the worst you will see in published CReSIS data - in general, their results are quite reliable.
# 
# All processed data from CReSIS is available from their public website: https://data.cresis.ku.edu/. You can download kml files of all flight lines, processed radargram images, as well as MATLAB files with both coherently summed and SAR processed radar data. Raw data is available on request - you just have to ship them a hard drive since the data volume is typically quite large. Additionally, they make available their MATLAB processing toolbox upon request. They also provide extensive documentation on the data files, instruments, and parameters in their README files: ftp://data.cresis.ku.edu/data/rds/rds_readme.pdf. 

# In[ ]:


# Load and plot CReSIS data for system comparison

cresis = sio.loadmat('../input/system-comparison/CReSIS_20131127_01_029.mat', struct_as_record = False, squeeze_me = True)

lat = -75.090606
lon = 123.297632

ilat = np.argmin(np.abs(cresis["Latitude"] - lat + cresis["Longitude"] - lon))
d = 0.5*(cresis["Time"] - cresis["Time"][75])*(c/n_ice)
box = np.array([0, cresis["Data"].shape[1], d[1500], d[0]])

fig, ax = plt.subplots()
cresis_im = ax.imshow(10*np.log10(cresis["Data"][0:1500:1,::]), cmap = 'Blues', aspect = 'auto', vmin = -170, vmax = -40, extent = box)
ax.set_xlabel('Trace Number', fontsize = 18, fontweight = 'bold')
ax.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
ax.set_title('Dome C CReSIS Data', fontsize = 24, fontweight = 'bold')
fig.colorbar(cresis_im, ax = ax)


# In[ ]:


# Plot a single trace for CReSIS

plt.plot(d, np.asarray(10*np.log10(cresis["Data"][:,1000])))
plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')
plt.title('Single Trace - Dome C CReSIS Data', fontsize = 24, fontweight = 'bold')
plt.show()


# HiCARS data from the University of Texas operates at 60 MHz, which means it is generally less susceptible for rough surface scattering losses and may perform better near the ice sheet margins. It has a somewhat lower range resolution than MCoRDS, typically about 6 meters in ice. The radiometric balance and stability is also quite reliable. 
# 
# There is less HiCARS data publically available. Flight lines collected through the NASA Operation IceBridge are available through the OIB data portal: https://nsidc.org/icebridge/portal/map. This comprises entirely Antarctic data, with the most significant coverage in East Antarctica, particularly around Dome C and Dome Law. The NetCDF files provided through the OIB portal have all required metadata embedded in them. UTIG has collected significantly more data throughout Antarctica, particularly in the Admundsen Sea Embayment, but this data is not available publically, although data sharing within a specific collaboration may be possible. UTIG website: http://www-udc.ig.utexas.edu/external/facilities/aero/.

# In[ ]:


# Load and plot UTIG data for system comparison

utig = nc.netcdf_file('../input/system-comparison/UTIG_IR1HI1B.nc', 'r')
latitude = np.asarray(utig.variables["lat"][:].copy())
longitude = np.asarray(utig.variables["lon"][:].copy())
time = np.asarray(utig.variables["fasttime"][:].copy())
power_high = np.transpose(np.asarray(utig.variables["amplitude_high_gain"][:,:].copy()))
power_low = np.transpose(np.asarray(utig.variables["amplitude_low_gain"][:,:].copy()))
utig.close()

ilat2 = np.argmin(np.abs(latitude - lat + longitude - lon))
d2 = 0.5*(time - time[170])*(1e-6)*(c/n_ice)
box = np.array([0, power_high.shape[1], d2[2550], d2[0]])

fig, ax = plt.subplots()
cresis_im = ax.imshow(power_high[0:2550:1,::], cmap = 'Blues', aspect = 'auto', vmin = 40, vmax = 140, extent = box)
ax.set_xlabel('Trace Number', fontsize = 18, fontweight = 'bold')
ax.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
ax.set_title('Dome C UTIG Data', fontsize = 24, fontweight = 'bold')
fig.colorbar(cresis_im, ax = ax)


# The HiCARS data format is an excellent example of the use of the use of separate high and low gain receive channels. In order to avoid surface saturation while also improving sensitivty to deep reflectors, systems often use a high and low gain channel. The high gain channel will saturate at the surface, but has better SNR at the bed. The low gain channel collects the surface well but suffers at depth. UTIG provides these two channels separately in their data. CReSIS uses a similar scheme (but with three channel - high, medium, and low - that all use different chirp pulse lengths) and then splices the data together in processing before publishing it. 

# In[ ]:


# Plot a single trace for UTIG

plt.plot(d2, power_high[:,1000])
plt.plot(d2, power_low[:,1000] + 40)
plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')
plt.title('Single Trace - Dome C UTIG Data', fontsize = 24, fontweight = 'bold')
plt.show()


# PASIN operates at a center frequency of 150 MHz with a typical in ice range resolution of 6-8 meters. This instrument actually uses two different pulse types to collected the near-surface and deep ice. The near-surface pulse is a short impulse, while the deep sounding pulse is a typical chirped waveform. BAS frequently uses some form of exponential gain with depth to improve the signal to noise ratio of the bed in post-processing. As a result, their data typically resolves the bed very well but may not be as radiometrically reliable in the middle of the ice column. 
# 
# PASIN data is generally not available publically - access requires coordination and collaboration with BAS. One exception would be the SAR processed radargrams for the Institute and Moller Ice Streams which can be found here: https://ramadda.data.bas.ac.uk/repository/entry/show/?entryid=8a975b9e-f18c-4c51-9bdb-b00b82da52b8. However, the data record is fairly extensive and covers a number of regions which have not been well surveyed by other systems, such as parts of the Gamburtsev Mountains, regions around Taylor and Talos Domes, and Institute and Moller Ice Streams. The BAS website with a list of surveyed regions is here: https://secure.antarctica.ac.uk/data/aerogeo/. 

# In[ ]:


# Load and plot BAS data for system comparison

bas = sio.loadmat('../input/system-comparison/BAS_SectionW36.mat', struct_as_record = False, squeeze_me = True)

lat = -75.090606
lon = 123.297632

ilat3 = np.argmin(np.abs(bas["lat"] - lat + bas["lon"] - lon))
Fs = 22e6
del_z = 0.5*(1/Fs)*(c/n_ice)
d3 = np.multiply(del_z, range(bas["abit"].shape[0])) - 114.805
box = np.array([0, bas["abit"].shape[1], d3[-1], d3[0]])

fig, ax = plt.subplots()
cresis_im = ax.imshow(10*np.log10(bas["abit"]), cmap = 'Blues', aspect = 'auto', vmin = 0, vmax = 85, extent = box)
ax.set_xlabel('Trace Number', fontsize = 18, fontweight = 'bold')
ax.set_ylabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
ax.set_title('Dome C BAS Data', fontsize = 24, fontweight = 'bold')
fig.colorbar(cresis_im, ax = ax)


# In[ ]:


# Plot a single trace for BAS

plt.plot(d3, 10*np.log10(bas["abit"][:,500]))
plt.xlabel('Depth (meters)', fontsize = 18, fontweight = 'bold')
plt.ylabel('Power(dB)', fontsize = 18, fontweight = 'bold')
plt.title('Single Trace - Dome C BAS Data', fontsize = 24, fontweight = 'bold')
plt.show()

