#!/usr/bin/env python
# coding: utf-8

# ## Signal decomposition with Fast Fourier Transforms
# 
# In a previous notebook, ["denoising algorithms"](https://www.kaggle.com/residentmario/denoising-algorithms/), I discussed algorithms that can be used to compute a smoothed/simplified version of a time series dataset. The notebook focused in large part on filters: algorithms which use some kind of probabilistic or mathematical properties of the data to determine how the data ought to be smoothed.
# 
# The **Fast Fourier Transform** or FFT is another kind of signal analysis technique. It can also be used to simplify a signal, but it differs from most filters in that it is a decomposition technique, e.g. it can be used to transform a signal into a sum of other, simpler signals.
# 
# The F in FFT is for the **Fourier series**. A Fourier series is an infinite sum of sinusodial functions with coefficients which, taken as a sum, roughly equals an input function. In other words it's a way of decomposing a function $f(x)$ into a bunch of components $a_1(t) + a_2(t) + \ldots$ in some domain $t$ that is dependent on $x$, which when taken as a whole can reconstruct $f(x)$.
# 
# Fourier series have deep mathematical significance, for instance they are related to the eigenvectors of the input matrix. However they are mathematically complex. For the purposes of application the relevant thing to know is that by computing a Fourier trainsform and then inverting it, you may collect the "dominant" signal on a given frequency band. For larger frequency bands this will equate to smoothing out the function; for small frequency bands this will equate to singling out the noise in the function.
# 
# For example:

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfrom numpy.fft import rfft, irfft, rfftfreq\nfrom scipy import fftpack\nimport pandas as pd\n\ntrain_meta_df = pd.read_csv("../input/metadata_train.csv").set_index(\'signal_id\')\ntrain = pd.read_parquet("../input/train.parquet")\ny = train_meta_df.target')


# In[ ]:


import matplotlib.pyplot as plt

def low_pass(s, threshold=1e4):
    fourier = rfft(s)
    frequencies = rfftfreq(s.size, d=2e-2 / s.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)

lf_signal_1 = low_pass(train.iloc[:, 0])
plt.plot(train.iloc[:, 0], color='lightgray')
plt.plot(lf_signal_1, color='black')


# Notice that this method:
# 1. Computes the FFT of the time-series.
# 2. Samples it along the given frequencies.
# 3. Thresholds it, removing all low-frequency Fourier series from the result.
# 
# We can also go the other way and keep only the low-frequency Fourier series:

# In[ ]:


def high_pass(s, threshold=1e7):
    fourier = rfft(s)
    frequencies = rfftfreq(s.size, d=2e-2/s.size)
    fourier[frequencies < threshold] = 0
    return irfft(fourier)

hf_signal_1 = high_pass(train.iloc[:,0], threshold=1e4)

plt.plot(hf_signal_1)


# If we sum these two signals we closely approximate the original dataset, to the degree that the difference doesn't appear on the plot:

# In[ ]:


plt.plot(train.iloc[: 0], color='black')
plt.plot(lf_signal_1 + hf_signal_1, color='lightgray')


# FFT does not compute the true infinite Fourier series (because that is infinite). Instead it computes a subsequence of coefficients for Fourier series spaces at certain frequency intervals. This is why the low-band filter isolates noise, as it includes only sinusoidals with low periodicity, whilst the high-band filter isolates the smoothed function, as it includes only sinusoidals with high periodicity. Here is a plot of our example frequency sample space:

# In[ ]:


plt.plot(rfftfreq(train.iloc[:, 0].size, d=2e-2 / train.iloc[:, 0].size))


# In[ ]:


import numpy as np

def decompose_into_n_signals(srs, n):
    fourier = rfft(srs)
    frequencies = rfftfreq(srs.size, d=2e-2/srs.size)
    out = []
    for vals in np.array_split(frequencies, n):
        ft_threshed = fourier.copy()
        ft_threshed[(vals.min() > frequencies)] = 0
        ft_threshed[(vals.max() < frequencies)] = 0        
        out.append(irfft(ft_threshed))
    return out

def plot_n_signals(sigs):
    fig, axarr = plt.subplots(len(sigs), figsize=(12, 12))
    for i, sig in enumerate(sigs):
        plt.sca(axarr[i])
        plt.plot(sig)
    plt.gcf().suptitle(f"Decomposition of signal into {len(sigs)} frequency bands", fontsize=24)


# If we decompose a complex signal into frequency bands (evenly spaced bands in this example, but you can choose whatever bands you'd like) we can see which frequencies dominate the information in the signal.

# In[ ]:


plot_n_signals(decompose_into_n_signals(train.iloc[:,0], 5))


# In this case we see that there is a lot of information at the highest frequency band (indicating an underlying macro signal), then low signal at the intermediate band, then a lot of signal at low bands. This indicates that there is a strong macro signal and strong low-frequency signal, e.g. random noise, but less intermediate-strength structure.
