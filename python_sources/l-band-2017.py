#!/usr/bin/env python
# coding: utf-8

# # Breakthrough Listen: L-Band Data

# <img style="float: left;" src="https://seti.berkeley.edu/img/andromedasmall.jpg" width="700px"/>

# Since the invention of radio technology, there have been efforts to look upwards and listen for signs of extra-terrestrial intelligence. [Even as far back as 1896](https://en.wikipedia.org/wiki/Search_for_extraterrestrial_intelligence#Early_work), Nikola Tesla claimed to have detected strange repeating signals from Mars. Later, in 1924, efforts were made to listen in on Mars again, this time with a 36 hour "National Radio Silence Day" in the US. As the decades rolled by, researchers began to consider other parts of the electromagnetic spectrum for alien communications, such as microwaves, and in 1960, [Frank Drake](https://en.wikipedia.org/wiki/Frank_Drake) (of the famous Drake Equation) created "Project Ozma". This was a study of the stars Tau Ceti and Epsilon Eridani using a radio telescope at Green Bank, West Virginia. He observed the stars at and around 1.42GHz, specifically, a 400KHz band using a receiver with a bandwidth of 100Hz. This frequency range was chosen due to something dubbed the ["water hole"](https://en.wikipedia.org/wiki/Water_hole_(radio)). This is a section of the electromagnetic spectrum that occurs between the natural signals from interstellar hydrogen and interstellar hydroxyl ions. Not only is this a relatively quiet part of the radio spectrum, it has been posited that any extra-terrestial civilisation may use these as markers for interstellar communication. As the scientist Barney Oliver once said, *"Where shall we meet our neighbors?"* he asked. *"At the water-hole, where species have always gathered."*
# 
# Efforts continued throughout 50s, 60s and 70s, with the famous ["Wow!"](https://en.wikipedia.org/wiki/Wow!_signal) signal being detected in 1977 by Ohio State University's Big Ear radio telescope. This signal, seen by some as the strongest candidate for an alien signal to date, has never been seen again. This period also attracted the attention of a young [Carl Sagan](https://en.wikipedia.org/wiki/Carl_Sagan), who would go on to play a major part in future searches.
# 
# Throughout the 80s and 90s, technological advances led to equipment that could scan an ever increasing numbers of channels, including META (Megachannel Extra-Terrestrial Assay) in 1985 and BETA (Billion-channel Extraterrestrial Assay) in 1995. However, in 1999, strong winds damaged the radio telescope on which BETA was based and the project was discontinued.
# 
# Another 90s project, the "Microwave Observing Program" (MOP), aimed to survey 800 nearby stars using the National Radio Astronomy Observatory at Green Bank, West Virginia and the radio telescope at the Arecibo Observatory, Puerto Rico. The US Congress pulled the plug in 1995, but work recommenced with alternative funding in the form of "Project Phoenix" (headed by [Jill Tarter](https://en.wikipedia.org/wiki/Jill_Tarter)). 
# 
# Today, the Search for Extraterrestrial Intelligence (SETI) continues in many forms, including radio, microwave and optical searches. There are even efforts to look for technosignatures, such as massive alien construction projects known as [Dyson spheres](https://en.wikipedia.org/wiki/Dyson_sphere). Key sites include the Allen Telescope Array in California, the Green Bank Observatory, West Virginia, the Parkes Observatory, New South Wales, and the Automated Planet Finder, California. These last 3 are involved in a project called [**Breakthrough Listen**](https://breakthroughinitiatives.org/initiative/1), which is described as "the most comprehensive search for alien communications to date". It's aim is ambitious; the survey 1 million stars and the centres of 100 galaxies, with data made available to the public.
# 
# Learn more about Breakthrough listen in the following two papers,
# 
# 1. [The Breakthrough Listen Search for Intelligent Life: Target Selection of Nearby Stars and Galaxies](https://iopscience.iop.org/article/10.1088/1538-3873/aa5800)
# 2. [The Breakthrough Listen Search for Intelligent Life: A Wideband Data Recorder System for the Robert C. Byrd Green Bank Telescope](https://iopscience.iop.org/article/10.1088/1538-3873/aa80d2/meta)

# **Breakthrough Listen**
# 
# This kernel looks at some of that data. Specifically, data relating to candidate signals from a recent paper titled **[THE BREAKTHROUGH LISTEN SEARCH FOR INTELLIGENT LIFE: 1.1-1.9 GHZ OBSERVATIONS OF 692 NEARBY STARS](https://arxiv.org/abs/1709.03491)** by J. Emilio Enriquez et al. This study observed 692 stars across the L-band (1.1 - 1.9GHz), covering the entire water-hole (mentioned above). The researchers used a technique known as an ABACAD search. This involves first observing a target for 5 minutes (dubbed 'A'), then moving the telescope to another target and listening to that for 5 minutes (dubbed 'B'). Then, back to A, then to a third target, C, then back to A for a final listen, followed by a fourth target, D. 
# 
# The idea is that if any signals are detected that are terrestrial, they'll probably be seen in the scans of all 4 targets. This would appear as 'SIGNAL'-'SAME SIGNAL'-'SAME SIGNAL'-'SAME SIGNAL'-'SAME SIGNAL'-'SAME SIGNAL'. However, if a signal is only seen when looking at the A target, that suggests a signal specific to that point in the sky. This would appear as 'SIGNAL'-'NO SIGNAL'-'SIGNAL'-'NO SIGNAL'-'SIGNAL'-'NO SIGNAL' (or perhaps, the B, C or D targets could have a different type of signal to the A target?). Such terrestrial signals are referred to as radio-frequency interference (RFI).
# 
# These stars were observed between January 2016 and February 2017 at the Green Bank Observatory. The researches note in their paper that notch filters were used between 1.2 and 1.33GHz to exclude local radar signals.

# **The Data**
# 
# When targets are observed, the voltages obtained from the radio telescope are processed into 3 different data formats. These are,
# 
# - **High frequency resolution** (~3 Hz frequency resolution, ~18 second sample time)
# - **High time resolution** (~366 kHz frequency resolution, ~349 microsecond sample time)
# - **Medium resolution** (~3 kHz frequency resolution, ~1 second sample time)
# 
# This study alone resulted in the processing of an eye-watering 180TB of data. For this study, the first of these was used, focussing on narrowband signals that can't be produced by natural processes. The researchers were also looking for signals that drifted over time, which would be expected in non-terrestrial signals due to the Doppler shift between the transmitter and receiver. They also note that as the 692 targets were relatively close, effects that can alter signals over larger distances, such as scintillation and spectral broadening, can be ignored.
# 
# A bespoke software packaged called turboSETI is used in this paper to look for these narrowband, drifting signals. They define as hit as *"the signal with the largest signal-to-noise ratio at a given frequency channel over all the drift
# rates searched"*. They then excluded any hits for which there was a corresponding hit in the scans of the associated B, C or D targets, within a window of +-600Hz.
# 
# After all processing, removing any candidates that appear in any way to be RFI, the researchers were left with 11 'significant events'. Let's take a look at some data.
# 
# Note that there is a website associated with this work [here](https://seti.berkeley.edu/lband2017/index.html). This is the site the data was downloaded from, and it also shows images similar to those in the paper. 

# **Spectrograms**
# 
# Data is shown in this kernel as a [spectrogram](https://en.wikipedia.org/wiki/Spectrogram), or more precisely, 6 stacked spectrograms from the ABACAD search. A spectrogram is a heatmap of frequency vs time vs power.
# 
# First, let's load the required packages. Note that I'm using one called [blimpy](https://github.com/UCBerkeleySETI/blimpy) (Breakthrough Listen I/O Methods for Python), which was developed specifically for dealing with Breakthrough Listen data.

# In[ ]:


import pylab as plt
import numpy as np
import PIL
import os
from scipy import ndimage, misc
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
from blimpy import Filterbank
from blimpy import Waterfall
from blimpy.utils import db, lin, rebin, closest


# Now set the filenames,

# In[ ]:


file = '../input/hip4436/HIP4436/spliced_blc0001020304050607_guppi_57803_80733_HIP4436_0032.gpuspec.0000.h5'
file2 = '../input/hip4436/HIP4436/spliced_blc0001020304050607_guppi_57803_81079_HIP3333_0033.gpuspec.0000.h5'
file3 = '../input/hip4436/HIP4436/spliced_blc0001020304050607_guppi_57803_81424_HIP4436_0034.gpuspec.0000.h5'
file4 = '../input/hip4436/HIP4436/spliced_blc0001020304050607_guppi_57803_81768_HIP3597_0035.gpuspec.0000.h5'
file5 = '../input/hip4436/HIP4436/spliced_blc0001020304050607_guppi_57803_82111_HIP4436_0036.gpuspec.0000.h5'
file6 = '../input/hip4436/HIP4436/spliced_blc0001020304050607_guppi_57803_82459_HIP3677_0037.gpuspec.0000.h5'

filenames_list = [file,file2,file3,file4,file5,file6]


# Breakthrough listen data is typically stored and used in a **filterbank** file format. These are time series of power spectra (created by performing a [fast Fourier transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) on the raw voltage data), plus metadata. They come as either a .fil or .h5 file. Note the filename conventions,
# 
# blcNN_guppi_MMMMM_SSSSS_TTTTT_XXXX.suffix
# 
# - NN : compute nodes or banks where the data was recorded (each two-digit NN represents 187.5 MHz of bandwidth)
# - MMMMM : MJD (modified Julian date) of observation
# - SSSSS : Seconds after midnight of observation
# - TTTTT : Target name (may contain underscores)
# - XXXX : sequence # (or #s) of observation
# 
# The first file we're looking at is *'spliced_blc0001020304050607_guppi_57803_80733_HIP4436_0032.gpuspec.0000.h5'*. You can now see, from the naming convention, that this was taken on the modified Julian date of 57803 (corresponding to 19th Feb 2017), 80733 seconds after midnight (corresponding to a time of 22:25:33), a target of HIP4436, with this being the 32nd sequence of observation.
# 
# Let's first look at the file's metadata. Note I'm using 'load_data=False' to avoid loading signal data when all I'm after is the header information,

# In[ ]:


filterbank_head = Waterfall(file, load_data=False)
filterbank_head.info()


# There's a lot going on here, but the main points of interest are the start and end frequencies. Note that these numbers are incorrect in this data due to a bug in blimpy (now resolved). We need to manually adjust these for the purposes of this kernel. First, we'll load the signal data,

# In[ ]:


filterbank = Waterfall(file)


# In[ ]:


filterbank.container.f_stop


# Then make the correction,

# In[ ]:


correction = filterbank.container.f_stop - 1380.87763 #know for HIP4436
filterbank.container.f_start =  filterbank.container.f_start - correction
filterbank.container.f_stop =  filterbank.container.f_stop - correction


# In[ ]:


print('Min freq: ' + str(filterbank.container.f_start))
print('Max freq: ' + str(filterbank.container.f_stop))


# Now let's plot a spectrogram (also known as a Waterfall plot) using the signal data,

# In[ ]:


filterbank.plot_waterfall()


# We can see Frequency along the x-axis, time along the y-axis, with the pixel colour proportional to the power.
# 
# This shows some sort of signal. In the paper, they state that their detection methods found a hit at 1380.87763MHz. Let's focus on that point,

# In[ ]:


window = 0.002
correction = filterbank.container.f_stop - 1380.877634 #know for HIP4436
filterbank.container.f_start =  filterbank.container.f_start - correction
filterbank.container.f_stop =  filterbank.container.f_stop - correction
f_start = (filterbank.container.f_start + (filterbank.container.f_stop - filterbank.container.f_start)/2) - (window/2)
f_stop = f_start + window
filterbank.plot_waterfall(f_start=f_start, f_stop=f_stop)


# This looks similar to the first observation in figure 1 of the paper.
# 
# Next, I'm going to use some bits of code shared by the paper's researchers to plot a full ABACAD stacked spectrogram,

# In[ ]:


#Below is a stripped down version of the author's (J. Emilio Enriquez) code here: https://github.com/jeenriquez/Lband_seti/blob/master/analysis/plot_candidates.py

pd.options.mode.chained_assignment = None  # To remove pandas warnings: default='warn'

#------
#Hardcoded values
MAX_DRIFT_RATE = 2.0
OBS_LENGHT = 300.

MAX_PLT_POINTS      = 65536                  # Max number of points in matplotlib plot
MAX_IMSHOW_POINTS   = (8192, 4096)           # Max number of points in imshow plot
MAX_DATA_ARRAY_SIZE = 1024 * 1024 * 1024     # Max size of data array to load into memory
MAX_HEADER_BLOCKS   = 100                    # Max size of header (in 512-byte blocks)

#------


def plot_waterfall(fil, f_start=None, f_stop=None, if_id=0, logged=True,cb=False,freq_label=False,MJD_time=False, **kwargs):
    """ Plot waterfall of data
    Args:
        f_start (float): start frequency, in MHz
        f_stop (float): stop frequency, in MHz
        logged (bool): Plot in linear (False) or dB units (True),
        cb (bool): for plotting the colorbar
        kwargs: keyword args to be passed to matplotlib imshow()
    """


    fontsize=18

    font = {'family' : 'serif',
            'size'   : fontsize}

    matplotlib.rc('font', **font)

    plot_f, plot_data = fil.grab_data(f_start, f_stop, if_id)

    # Make sure waterfall plot is under 4k*4k
    dec_fac_x, dec_fac_y = 1, 1
    if plot_data.shape[0] > MAX_IMSHOW_POINTS[0]:
        dec_fac_x = plot_data.shape[0] / MAX_IMSHOW_POINTS[0]

    if plot_data.shape[1] > MAX_IMSHOW_POINTS[1]:
        dec_fac_y =  plot_data.shape[1] /  MAX_IMSHOW_POINTS[1]

    print(dec_fac_x)
        
    plot_data = rebin(plot_data, int(dec_fac_x), int(dec_fac_y))

    if MJD_time:
        extent=(plot_f[0], plot_f[-1], fil.timestamps[-1], fil.timestamps[0])
    else:
        extent=(plot_f[0], plot_f[-1], (fil.timestamps[-1]-fil.timestamps[0])*24.*60.*60, 0.0)

    this_plot = plt.imshow(plot_data,
        aspect='auto',
        rasterized=True,
        interpolation='nearest',
        extent=extent,
        cmap='viridis_r',
        **kwargs
    )
    if cb:
        plt.colorbar()

    if freq_label:
        plt.xlabel("Frequency [Hz]",fontdict=font)
    if MJD_time:
        plt.ylabel("Time [MJD]",fontdict=font)
    else:
        plt.ylabel("Time [s]",fontdict=font)

    return this_plot

def make_waterfall_plots(filenames_list,target,f_start,f_stop,ion = False,correction_in = 0,**kwargs):
    ''' Makes waterfall plots per group of ON-OFF pairs (up to 6 plots.)
    '''

    fontsize=18

    font = {'family' : 'serif',
            'size'   : fontsize}

    matplotlib.rc('font', **font)

    if ion:
        plt.ion()

    n_plots = len(filenames_list)
    fig = plt.subplots(n_plots, sharex=True, sharey=True,figsize=(10, 2*n_plots))

    fil = Waterfall(filenames_list[0], f_start=f_start, f_stop=f_stop)
    
    A1_avg = np.median(fil.data)
    A1_max = fil.data.max()
    A1_std = np.std(fil.data)

    labeling = ['A','B','A','C','A','D']

    delta_f = np.abs(f_start-f_stop)
    mid_f = np.abs(f_start+f_stop)/2.
    
    #Adjust the incorrect header data
    correction = mid_f - correction_in
    mid_f_text = mid_f - correction

    for i,filename in enumerate(filenames_list):
        plt.subplot(n_plots,1,i+1)

        fil = Waterfall(filename, f_start=f_start, f_stop=f_stop)

        this_plot = plot_waterfall(fil,f_start=f_start, f_stop=f_stop,vmin=A1_avg-A1_std*0,vmax=A1_avg+5.*A1_std,**kwargs)

        if i == 0:
            plt.title(target.replace('HIP','HIP '))

        if i < len(filenames_list)-1:
            plt.xticks(np.arange(f_start, f_stop, delta_f/4.), ['','','',''])

    #Some plot formatting.
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    
    if target == 'HIP7981':
        #f_start -= 0.3
        #f_stop  += 0.3
        factor = 1e3
        units = 'kHz'
    else:
        factor = 1e6
        units = 'Hz'
    
    plt.xticks(np.arange(f_start, f_stop, delta_f/4.),[round(loc_freq) for loc_freq in np.arange((f_start-mid_f), (f_stop-mid_f), delta_f/4.)*factor])
    plt.xlabel("Relative Frequency [%s] from %f MHz"%(units,mid_f_text),fontdict=font)

    #to plot color bar. for now.
    cax = fig[0].add_axes([0.9, 0.11, 0.03, 0.77])
    fig[0].colorbar(this_plot,cax=cax,label='Power [Arbitrary Units]')

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    plt.subplots_adjust(hspace=0,wspace=0)


# In[ ]:


window = 0.002
filterbank_head = Waterfall(file, load_data=False)
f_start = (filterbank_head.container.f_start + (filterbank_head.container.f_stop - filterbank_head.container.f_start)/2) - (window/2)
f_stop = f_start + window

make_waterfall_plots(filenames_list=filenames_list, target='HIP4436', f_start=f_start, f_stop=f_stop, ion = False, correction_in=1380.87763)


# Here we see the first figure from the paper. Let's try a few more,

# In[ ]:


file = '../input/hip7981/HIP7981/spliced_blc0001020304050607_guppi_57680_15520_HIP7981_0039.gpuspec.0000.h5'
file2 = '../input/hip7981/HIP7981/spliced_blc0001020304050607_guppi_57680_15867_HIP6917_0040.gpuspec.0000.h5'
file3 = '../input/hip7981/HIP7981/spliced_blc0001020304050607_guppi_57680_16219_HIP7981_0041.gpuspec.0000.h5'
file4 = '../input/hip7981/HIP7981/spliced_blc0001020304050607_guppi_57680_16566_HIP6966_0042.gpuspec.0000.h5'
file5 = '../input/hip7981/HIP7981/spliced_blc0001020304050607_guppi_57680_16915_HIP7981_0043.gpuspec.0000.h5'
file6 = '../input/hip7981/HIP7981/spliced_blc0001020304050607_guppi_57680_17264_HIP6975_0044.gpuspec.0000.h5'

filenames_list = [file,file2,file3,file4,file5,file6]

window = 0.2
filterbank_head = Waterfall(file, load_data=False)
f_start = (filterbank_head.container.f_start + (filterbank_head.container.f_stop - filterbank_head.container.f_start)/2) - (window/2)
f_stop = f_start + window

make_waterfall_plots(filenames_list=filenames_list, target='HIP7981', f_start=f_start, f_stop=f_stop, ion = False, correction_in=1621.24028)


# In[ ]:


file = '../input/hip65352/HIP65352/spliced_blc02030405_2bit_guppi_57459_34297_HIP65352_0027.gpuspec.0000.h5'
file2 = '../input/hip65352/HIP65352/spliced_blc02030405_2bit_guppi_57459_34623_HIP65352_OFF_0028.gpuspec.0000.h5'
file3 = '../input/hip65352/HIP65352/spliced_blc02030405_2bit_guppi_57459_34949_HIP65352_0029.gpuspec.0000.h5'
file4 = '../input/hip65352/HIP65352/spliced_blc02030405_2bit_guppi_57459_35275_HIP65352_OFF_0030.gpuspec.0000.h5'
file5 = '../input/hip65352/HIP65352/spliced_blc02030405_2bit_guppi_57459_35601_HIP65352_0031.gpuspec.0000.h5'

filenames_list = [file,file2,file3,file4,file5]

window = 0.002
filterbank_head = Waterfall(file, load_data=False)
f_start = (filterbank_head.container.f_start + (filterbank_head.container.f_stop - filterbank_head.container.f_start)/2) - (window/2)
f_stop = f_start + window

make_waterfall_plots(filenames_list=filenames_list, target='HIP65352', f_start=f_start, f_stop=f_stop, ion = False, correction_in=1522.181016)


# In[ ]:


file = '../input/hip17147/HIP17147/spliced_blc0001020304050607_guppi_57523_69379_HIP17147_0015.gpuspec.0000.h5'
file2 = '../input/hip17147/HIP17147/spliced_blc0001020304050607_guppi_57523_69728_HIP16229_0016.gpuspec.0000.h5'
file3 = '../input/hip17147/HIP17147/spliced_blc0001020304050607_guppi_57523_70077_HIP17147_0017.gpuspec.0000.h5'
file4 = '../input/hip17147/HIP17147/spliced_blc0001020304050607_guppi_57523_70423_HIP16299_0018.gpuspec.0000.h5'
file5 = '../input/hip17147/HIP17147/spliced_blc0001020304050607_guppi_57523_70769_HIP17147_0019.gpuspec.0000.h5'
file6 = '../input/hip17147/HIP17147/spliced_blc0001020304050607_guppi_57523_71116_HIP16341_0020.gpuspec.0000.h5'

filenames_list = [file,file2,file3,file4,file5,file6]

window = 0.002
filterbank_head = Waterfall(file, load_data=False)
f_start = (filterbank_head.container.f_start + (filterbank_head.container.f_stop - filterbank_head.container.f_start)/2) - (window/2)
f_stop = f_start + window

make_waterfall_plots(filenames_list=filenames_list, target='HIP17147', f_start=f_start, f_stop=f_stop, ion = False, correction_in=1379.27751)


# We've managed to take their shared data and some shared code and reproduce figures from their paper. This is a great example of a research group using an open and reproducible approach to modern scientific research.
