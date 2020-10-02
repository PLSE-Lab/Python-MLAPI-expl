#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center;font-size:46px">Cornell Birdcall Identification EDA</h1>
# 
# <a href="https://imgbb.com/"><img src="https://i.ibb.co/5Y96jpx/What-the-BIRD.png" alt="What-the-BIRD"></a>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchaudio
import matplotlib.pyplot as plt

import IPython.display as ipd



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from os import path



get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('pip install tinytag')


# In[ ]:


DATA_PATH = '../input/birdsong-recognition/'
AUDIO_PATH = "../input/birdsong-recognition/train_audio"


# <div style="display:flex">
#     <div>
#         <a href="https://ebird.org/species/aldfly"><img src="https://i.ibb.co/9VJTLsC/160820341.jpg" alt="160820341" border="0" width=400px></a>
#     </div>
#     <div style="margin-left:20px">
#         <h1>Aldfly</h1>
#         <h2>Description</h2>
#         <p style="width:500px; font-size:14px">Small flycatcher, extremely similar to several other species. Prefers clearings and boggy areas, often with patches of alders. Grayish-olive above and pale below with thin white eyering. Wings dark with bold white wingbars. Nearly identical to Willow Flycatcher; once considered the same species. Also compare with Least Flycatcher, which is very similar but has slightly shorter wingtips and a bolder eyering. Best identified by voice: song is a rolling, rough "freeBEER;" call is a clear "pip." Silent birds, especially in migration, often best left unidentified.</p>
#     </div>
# </div>
# 

# In[ ]:


display(ipd.Audio('../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3'))


# <div style="display:flex">
#     <div>
#          <a href="https://ebird.org/species/Ameavo"><img src="https://i.ibb.co/HzgXmcK/64807071.jpg" alt="64807071" border="0" width=400px></a>
#     </div>
#     <div style="margin-left:20px">
#         <h1>Ameavo</h1>
#         <h2>Description</h2>
#         <p style="width:500px; font-size:14px">Distinctive large shorebird with a long, thin upturned bill and lean neck. Bold black-and-white wings prominent year-round. Adults in summer have buffy-orange wash on head. Frequents wetlands where it swings its head back-and-forth in shallow water to catch small invertebrates.</p>
#     </div>
# </div>

# In[ ]:


display(ipd.Audio('../input/birdsong-recognition/train_audio/ameavo/XC133080.mp3'))


# <div style="display:flex">
#     <div>
#        <a href="https://ebird.org/species/Amebit"><img src="https://i.ibb.co/m86C9nq/37758621.jpg" alt="37758621" border="0" width=400px></a> 
#     </div>
#     <div style="margin-left:20px">
#         <h1>Amebit</h1>
#         <h2>Description</h2>
#         <p style="width:500px; font-size:14px">Stocky, brown heron found in marshes and bogs; secretive but occasionally found in the open. Most similar to juvenile night-herons. Note striped neck, plain unspotted wings, and behavior: American Bittern does not typically perch on branches. Varied diet includes fish, frogs, insects, and small mammals. Most active around dawn and dusk flying low over extensive marshes. Listen for its incredible vocalization: a resonant, booming noise produced by air sacs on the neck, "oonk-GA-loonk.</p>
#     </div>
# </div>

# In[ ]:


display(ipd.Audio('../input/birdsong-recognition/train_audio/amebit/XC127371.mp3'))


# In[ ]:


df_train = pd.read_csv(DATA_PATH + 'train.csv')


# <h1>Explore train data</h1>

# In[ ]:


df_train.head()


# <h2>Visualization of "ebird_code"</h2>

# In[ ]:


species_count = df_train['ebird_code'].value_counts(sort=True)
species_count = species_count[250:]
plt.figure(figsize=(15,7))
sns.barplot(species_count.index,species_count.values,palette=("Blues_d"), alpha=0.85)
plt.title('Birds species distribution', fontsize=18)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Birds code', fontsize=12)
plt.show()


# In[ ]:


print('Minimum samples per category = ', min(df_train.ebird_code.value_counts()))
print('Maximum samples per category = ', max(df_train.ebird_code.value_counts()))


# In[ ]:


print("Total number of labels: ",len(df_train['ebird_code'].value_counts()))
print("Labels: ", df_train['ebird_code'].unique())


# <h3>Word cloud viz for ebird_code</h3>

# In[ ]:


from wordcloud import WordCloud
wordcloud = WordCloud(background_color="white", max_font_size=50, width=800, height=500, collocations=False, max_words=200).generate(' '.join(df_train['ebird_code']))
plt.figure(figsize=(18,10))
plt.imshow(wordcloud, cmap=plt.cm.gray)
plt.title("Wordcloud for Labels in Train Curated", fontsize=25)
plt.axis("off")
plt.show()


# <hr>

# <h3>Visualization of "playback_used"</h3>

# In[ ]:


playback_used_count  = df_train['playback_used'].value_counts(sort=True)
playback_used_count = playback_used_count[:10]
plt.figure(figsize=(15,7))
sns.barplot(playback_used_count.index,playback_used_count.values,palette=("Blues_d"), alpha=0.85)
plt.title('Playback_used distribution', fontsize=18)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Playback_used', fontsize=12)
plt.show()


# In[ ]:


print('Minimum samples per category = ', min(df_train.playback_used.value_counts()))
print('Maximum samples per category = ', max(df_train.playback_used.value_counts()))


# <hr>

# <h3>Visualization of "channels"</h3>

# In[ ]:


channels_count  = df_train['channels'].value_counts(sort=True)
channels_count = channels_count[:10]
plt.figure(figsize=(15,7))
sns.barplot(channels_count.index,channels_count.values,palette=("Blues_d"), alpha=0.85)
plt.title('Channels distribution', fontsize=18)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Channels', fontsize=12)
plt.show()


# In[ ]:


print('Minimum samples per category = ', min(df_train.channels.value_counts()))
print('Maximum samples per category = ', max(df_train.channels.value_counts()))


# <hr>

# <h3>Visualization of "date"</h3>

# In[ ]:


date_count  = df_train['date'].value_counts(sort=True)
date_count = date_count[:10]
plt.figure(figsize=(15,7))
sns.barplot(date_count.index,date_count.values,palette=("Blues_d"), alpha=0.85)
plt.title('Date distribution', fontsize=18)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.show()


# In[ ]:


print('Minimum samples per category = ', min(df_train.date.value_counts()))
print('Maximum samples per category = ', max(df_train.date.value_counts()))


# <hr>

# <h3>Visualization of "pitch"</h3>

# In[ ]:


pitch_count  = df_train['pitch'].value_counts(sort=True)
pitch_count = pitch_count[:10]
plt.figure(figsize=(15,7))
sns.barplot(pitch_count.index,pitch_count.values,palette=("Blues_d"), alpha=0.85)
plt.title('Pitch distribution', fontsize=18)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Pitch', fontsize=12)
plt.show()


# In[ ]:


print('Minimum samples per category = ', min(df_train.pitch.value_counts()))
print('Maximum samples per category = ', max(df_train.pitch.value_counts()))


# <hr>

# <h3>Visualization of "duration"</h3>

# In[ ]:


duration_count  = df_train['duration'].value_counts(sort=True)
duration_count = duration_count[:30]
plt.figure(figsize=(15,7))
sns.barplot(duration_count.index,duration_count.values,palette=("Blues_d"), alpha=0.85)
plt.title('Duration distribution', fontsize=18)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Duration', fontsize=12)
plt.show()


# In[ ]:


print('Minimum samples per category = ', min(df_train.duration.value_counts()))
print('Maximum samples per category = ', max(df_train.duration.value_counts()))


# <hr>

# <h3>Visualization of "speed"</h3>

# In[ ]:


speed_count  = df_train['speed'].value_counts(sort=True)
speed_count = speed_count[:10]
plt.figure(figsize=(15,7))
sns.barplot(speed_count.index,speed_count.values,palette=("Blues_d"), alpha=0.85)
plt.title('Speed distribution', fontsize=18)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Speed', fontsize=12)
plt.show()


# In[ ]:


print('Minimum samples per category = ', min(df_train.speed.value_counts()))
print('Maximum samples per category = ', max(df_train.speed.value_counts()))


# <hr>

# <h3>Visualization of "type"</h3>

# In[ ]:


type_count  = df_train['type'].value_counts(sort=True)
type_count = type_count[:10]
plt.figure(figsize=(15,7))
sns.barplot(type_count.index,type_count.values,palette=("Blues_d"), alpha=0.85)
plt.title('Type distribution', fontsize=18)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Type', fontsize=12)
plt.show()


# In[ ]:


print('Minimum samples per category = ', min(df_train.type.value_counts()))
print('Maximum samples per category = ', max(df_train.type.value_counts()))


# <hr>

# <h3>Visualization of "sampling_rate"</h3>

# In[ ]:


sampling_rate_count  = df_train['sampling_rate'].value_counts(sort=True)
plt.figure(figsize=(15,7))
sns.barplot(sampling_rate_count.index,sampling_rate_count.values,palette=("Blues_d"), alpha=0.85)
plt.title('Sampling rate distribution', fontsize=18)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Rate (Hz)', fontsize=12)
plt.show()


# In[ ]:


print('Minimum samples per category = ', min(df_train.sampling_rate.value_counts()))
print('Maximum samples per category = ', max(df_train.sampling_rate.value_counts()))


# <hr>

# <h3>Visualization of "elevation"</h3>

# In[ ]:


elevation_count  = df_train['elevation'].value_counts(sort=True)
elevation_count = elevation_count[:10]
plt.figure(figsize=(15,7))
sns.barplot(elevation_count.index,elevation_count.values,palette=("Blues_d"), alpha=0.85)
plt.title('Elevation distribution', fontsize=18)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Elevation', fontsize=12)
plt.show()


# In[ ]:


print('Minimum samples per category = ', min(df_train.elevation.value_counts()))
print('Maximum samples per category = ', max(df_train.elevation.value_counts()))


# <hr>

# <h3>Visualization of "bitrate_of_mp3"</h3>

# In[ ]:


bitrate_of_mp3_count  = df_train['bitrate_of_mp3'].value_counts(sort=True)
bitrate_of_mp3_count = bitrate_of_mp3_count[:10]
plt.figure(figsize=(15,7))
sns.barplot(bitrate_of_mp3_count.index,bitrate_of_mp3_count.values,palette=("Blues_d"), alpha=0.85)
plt.title('Bitrate_of_mp3 distribution', fontsize=18)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Bitrate_of_mp3', fontsize=12)
plt.show()


# In[ ]:


print('Minimum samples per category = ', min(df_train.bitrate_of_mp3.value_counts()))
print('Maximum samples per category = ', max(df_train.bitrate_of_mp3.value_counts()))


# <hr>

# <h3>Visualization of "country"</h3>

# In[ ]:


country_count  = df_train['country'].value_counts(sort=True)
country_count = country_count[:10]
plt.figure(figsize=(15,7))
sns.barplot(country_count.index,country_count.values,palette=("Blues_d"), alpha=0.85)
plt.title('Countries distribution', fontsize=18)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Country', fontsize=12)
plt.show()


# In[ ]:


print('Minimum samples per category = ', min(df_train.country.value_counts()))
print('Maximum samples per category = ', max(df_train.country.value_counts()))


# <hr>
# 

# <h3>Visualization of "recordist"</h3>

# In[ ]:


recordist_count  = df_train['recordist'].value_counts(sort=True)
recordist_count = recordist_count[:10]
plt.figure(figsize=(15,7))
sns.barplot(recordist_count.index,recordist_count.values,palette=("Blues_d"), alpha=0.85)
plt.title('Recordists distribution', fontsize=18)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Recordists', fontsize=12)
plt.show()


# In[ ]:


print('Minimum samples per category = ', min(df_train.recordist.value_counts()))
print('Maximum samples per category = ', max(df_train.recordist.value_counts()))


# <hr>

# <h1>Checking other files</h1>

# **Let's check what is in 'test.csv'**

# In[ ]:


df_test = pd.read_csv(DATA_PATH + 'test.csv')


# In[ ]:


df_test.head()


# **Just have to have a look what is inside "example_test_audio_summary.csv"**

# In[ ]:


ex_test_summary = pd.read_csv(DATA_PATH + 'example_test_audio_summary.csv')


# In[ ]:


ex_test_summary.head()


# **Just have to have a look what is inside "example_test_audio_metadata.csv"**

# In[ ]:


ex_test_meta = pd.read_csv(DATA_PATH + 'example_test_audio_metadata.csv')


# In[ ]:


ex_test_meta.head()


# **Let's count check folders and files structure**
# 

# In[ ]:


files = folders = 0
FolderList = []
for _, dirnames, filenames in os.walk(AUDIO_PATH):
    files += len(filenames)
    folders += len(dirnames)
    FolderList.append(dirnames)
        
print("There are {:,} files, and {:,} folders".format(files, folders))


# <h3>Extracting metadata straight from the file</h3>

# File from "aldfly/XC134874.mp3" as an example

# In[ ]:


from tinytag import TinyTag


# In[ ]:


tag = TinyTag.get(AUDIO_PATH + "/aldfly/XC134874.mp3", image=True)


# In[ ]:


print('album:',tag.album)
print('album artist:',tag.albumartist) # album artist
print('artist name:',tag.artist)       # artist name
print('number of bytes before audio data begins:',tag.audio_offset)  # number of bytes before audio data begins
print('bitrate in kBits/s:',tag.bitrate) # bitrate in kBits/s
print('file comment', tag.comment)     # file comment
print('composer', tag.composer)        # composer
print('disc number', tag.disc)         # disc number
print('total number of discs',tag.disc_total)    # total number of discs
print('duration of the song in seconds', tag.duration)      # duration of the song in seconds
print('file size in bytes', tag.filesize)      # file size in bytes
print('genre', tag.genre)              # genre
print('samples per second', tag.samplerate)    # samples per second
print('title of the song', tag.title)  # title of the song
print('track number', tag.track)       # track number
print('total number of tracks', tag.track_total)     # total number of tracks
print('year or data', tag.year)        # year or data


# <h1>Plotting spectrograms</h1> 

# In[ ]:


from IPython.display import Audio
from matplotlib import pyplot as plt
import torchaudio
import torch

import librosa
import librosa.display


# <h3>Compare time of plotting with librosa and torchaudio</h3>

# <h3>Torchaudio wavelpot</h3>

# In[ ]:


get_ipython().run_cell_magic('time', '', 'filename = "../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3"\n\nwaveform, sample_rate = torchaudio.load(filename)\n\nplt.plot(waveform.t().numpy())')


# <h3>Librosa wavelpot</h3>

# In[ ]:


get_ipython().run_cell_magic('time', '', "x,sr = librosa.load('../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3')\nlibrosa.display.waveplot(x, sr=sr)")


# In[ ]:


librosa.feature.melspectrogram(x, sr=sample_rate, n_mels=128)


# <h3>Torchaudio Spectogram and MelSpectogram</h3>

# In[ ]:


get_ipython().run_cell_magic('time', '', "fig, ax = plt.subplots(1, 2, figsize=(16, 12))\n\nax[0].imshow(torchaudio.transforms.Spectrogram(n_fft=2000)(waveform).log2()[0,:,:].numpy(), cmap='ocean');\nax[0].set_title('Spectrogram');\nax[1].imshow(torchaudio.transforms.MelSpectrogram(n_fft=16000)(waveform).log2()[0].numpy(), cmap='inferno');\nax[1].set_title('MelSpectrogram');")


# In[ ]:


get_ipython().run_cell_magic('time', '', "X = librosa.stft(x)\nXdb = librosa.amplitude_to_db(abs(X))\nplt.figure(figsize=(14, 5))\nlibrosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')\nplt.colorbar()")

