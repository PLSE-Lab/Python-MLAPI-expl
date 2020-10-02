#!/usr/bin/env python
# coding: utf-8

# # Intergalactic Hello World

# ![](https://upload.wikimedia.org/wikipedia/commons/5/55/The_Arecibo_Observatory_20151101114231-0_8e7cc_c7a44aca_orig.jpg)

# In 1974, a message designed by [Frank Drake](https://en.wikipedia.org/wiki/Frank_Drake) and [Carl Sagan](https://en.wikipedia.org/wiki/Carl_Sagan) was beamed into space from the [Arecibo radio telescope](https://en.wikipedia.org/wiki/Arecibo_Observatory) in Puerto Rico. The message, made up of 1,679 binary digits (1,679 being the product of 23 and 73, each a prime number) conveyed information about DNA, the solar system, the human form and the Arecibo telescope itself. Aimed at star cluster M13, it will take 50,000 years for us to get a reply (if there is anyone listening and they fancy a chat).
# 
# Taking the binary signal from Wikipedia, let's take a look at the message,

# In[ ]:


import numpy as np
import os
import matplotlib.pyplot as plt

arecibo = '00000010101010000000000001010000010100000001001000100010001001011001010101010101010100100100000000000000000000000000000000000001100000000000000000001101000000000000000000011010000000000000000001010100000000000000000011111000000000000000000000000000000001100001110001100001100010000000000000110010000110100011000110000110101111101111101111101111100000000000000000000000000100000000000000000100000000000000000000000000001000000000000000001111110000000000000111110000000000000000000000011000011000011100011000100000001000000000100001101000011000111001101011111011111011111011111000000000000000000000000001000000110000000001000000000001100000000000000010000011000000000011111100000110000001111100000000001100000000000001000000001000000001000001000000110000000100000001100001100000010000000000110001000011000000000000000110011000000000000011000100001100000000011000011000000100000001000000100000000100000100000001100000000100010000000011000000001000100000000010000000100000100000001000000010000000100000000000011000000000110000000011000000000100011101011000000000001000000010000000000000010000011111000000000000100001011101001011011000000100111001001111111011100001110000011011100000000010100000111011001000000101000001111110010000001010000011000000100000110110000000000000000000000000000000000011100000100000000000000111010100010101010101001110000000001010101000000000000000010100000000000000111110000000000000000111111111000000000000111000000011100000000011000000000001100000001101000000000101100000110011000000011001100001000101000001010001000010001001000100100010000000010001010001000000000000100001000010000000000001000000000100000000000000100101000000000001111001111101001111000'
arecibo = ",".join([arecibo[i:i+1] for i in range(0, len(arecibo), 1)])
arecibo = np.fromstring(arecibo, dtype=int, sep=',')

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(np.array(arecibo).reshape(73,23))


# This message is a lot more sophisticated than it might first appear. See [here](https://en.wikipedia.org/wiki/Arecibo_message#Explanation) for a detailed breakdown.

# Now, imagine the scene; some distant, alien equivalents to Frank and Carl rally support to blast a radio signal into space. Maybe our solar system is targeted for some reason, or perhaps we just happen to get in the way. If such a signal was ever received, maybe at the time it was sent, the Pyramids of Giza were being built, or maybe Velociraptors were roaming the Earth. Who knows the distances such a message would have transversed.
# 
# Recall that electromagnetic radiation is caused by the acceleration of charged particles. Once the alien signal reached the earth, this electromagnetic radiation would repay the favour by in turn causing the oscillation of the charged particles wandering through the metallic structure of the Earth's radio telescopes. This incredibley subtle voltage oscillation is measured, decomposed via **Fourier analysis** (whereby a complex signal that is made up of several sinusoidal signals can be broken down into its constituent signals), and recorded.
# 
# SETI use the [Allen Telescope Array (ATA)](https://en.wikipedia.org/wiki/Allen_Telescope_Array) in San Francisco to scan the sky for signals. Specifically, [narrow-band](https://en.wikipedia.org/wiki/Narrowband) [carrier waves](https://en.wikipedia.org/wiki/Carrier_wave). These 42 dishes, each 6 meters in diameter, are all pointed together at various candidate objects. The office **setiquest** website shows which targets are being observed in real-time,

# <img src="http://setiquest.info/images/ata-obs-screen.png" width="700px"/>

# Over time, SETI have observed many signals in their data (all seemingly manmade or natural, to date) and improving the automatic detection and classification of such signals would improve the efficiency of their work.
# 
# To this end, SETI have generated a dataset of simulated signals, used primarily for a machine-learning challenge in the summer of 2017. For the participants in that challenge, they had a basic dataset (4 signal categories and well-defined signals), plus small, medium and full 'primary' datasets (7 signal categories, less-well-defined signals). This kernel uses the small primary dataset.
# 
# This kernel takes a look at the SETI data, focussing on the raw data stored in the **primary_small_v3/** folder. Note that the **primary_small/** folder contains the data converted into PNG files and broken down into training, validation and testing folders.

# First, let's import the customer ibmseti module,

# In[ ]:


import ibmseti


# Now, set the path to the raw data,

# In[ ]:


path = '../input/primary_small_v3/primary_small_v3/'


# Let's take a look at one of the files, using the ibmseti package,

# In[ ]:


primarymediumlist = os.listdir(path)
firstfile = primarymediumlist[0]
print(path + firstfile)

data_1 = ibmseti.compamp.SimCompamp(open(path + firstfile,'rb').read())
data_1.header().get("signal_classification")


# The signal classification is 'narrowbanddrd'. The 'drd' stands for 'drift rate derivative', which basically means you get a curved line. Read more about the technicalities [here](https://medium.com/ibm-watson-data-lab/using-artificial-intelligence-to-search-for-extraterrestrial-intelligence-ec19169e01af).
# 
# Let's get the spectrogram from the data and plot it,

# In[ ]:


spectrogram = data_1.get_spectrogram()

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(spectrogram,  aspect = spectrogram.shape[1] / spectrogram.shape[0])


# There's the curved line!

# There are in total 7 different classes of signals simulated by SETI. The link above gives an introduction to the different types, and there is also an open-access paper with more details [here](https://arxiv.org/abs/1803.08624). The classes have the following names,
# 
# - noise
# - squiggle
# - narrowband
# - narrowbanddrd
# - squarepulsednarrowband
# - squigglesquarepulsednarrowband
# - brightpixel
# 
# Let's take a look at examples of each,
# 

# In[ ]:


def plot_spectrogram(index_num):
    file = primarymediumlist[index_num]
    data = ibmseti.compamp.SimCompamp(open(path + file,'rb').read())
    spectrogram = data.get_spectrogram()
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title(data.header().get("signal_classification"))
    ax.imshow(spectrogram,  aspect = spectrogram.shape[1] / spectrogram.shape[0])

plot_spectrogram(0) #narrowbanddrd
plot_spectrogram(2) #narrowband
plot_spectrogram(4) #squiggle
plot_spectrogram(11) #brightpixel
plot_spectrogram(13) #noise
plot_spectrogram(14) #squarepulsednarrowband
plot_spectrogram(17) #squigglesquarepulsednarrowband


# Below is a histogram of the first example (the narrowbanddrd),

# In[ ]:


plt.hist(spectrogram)
plt.title(data_1.header().get("signal_classification"))
plt.show()


# The SETI folk over on the GitHub pages take the log of the spectrogram, which gives a better spread,

# In[ ]:


plt.hist(np.log(spectrogram))
plt.title(data_1.header().get("signal_classification"))
plt.show()


# It's also possible to create the spectrogram without the use of the ibmseti package, which then gives a bit more scope for signal processing to be done. From one of the SETI Github scripts,

# In[ ]:


complex_data = data_1.complex_data()
complex_data = complex_data.reshape(32, 6144)
complex_data = complex_data * np.hanning(complex_data.shape[1]) #This step applies something called a Hanning Window
cpfft = np.fft.fftshift( np.fft.fft(complex_data), 1)
spectrogram = np.abs(cpfft)**2

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(np.log(spectrogram),  aspect = spectrogram.shape[1] / spectrogram.shape[0])


# To my eye, the signal is now gone! I don't have a background in signal processing and need to do further reading into this aspect.
# 
# For now, I simply saved each spectrogram (the log of each individual numpy array) as a separate PNG, labelling each image with the classification from the header file. I can't run this code here, but I'll include it (commented out) in case you wish to use it on your local machine,

# In[ ]:


#png_path = ''
#zip_path = ''
#zip_list = os.listdir(zip_path)

#i = 1
#for z in zip_list:
#    zz = zipfile.ZipFile(zip_path + z)
#    primarymediumlist = zz.namelist()
#    primarymediumlist.pop(0)
#    for f in primarymediumlist:
#        i=i+1
#        temp_data = ibmseti.compamp.SimCompamp(zz.open(f, 'r').read())
#        temp_type = temp_data.header().get("signal_classification")
#        temp_spectrogram = temp_data.get_spectrogram()
#        matplotlib.image.imsave(png_path + str(i) + '_' + str(temp_type) + '.png', np.log(temp_spectrogram))
#        print("zip file: " + str(z), ", file: " + str(i))


# Once you have your PNGs, get them into train, valididation and test folders (already done in this kaggle dataset) to then apply some deep learning. See '[SETI Simulated Signals - InceptionResNetV2](https://www.kaggle.com/tentotheminus9/seti-simulated-signals-inceptionresnetv2)' for a look at this next step.

# 
