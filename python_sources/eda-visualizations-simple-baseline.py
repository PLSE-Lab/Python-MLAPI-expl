#!/usr/bin/env python
# coding: utf-8

# ## 1. Introduction 
# 
# * There are already many projects underway to extensively monitor birds by continuously recording natural soundscapes over long periods. However, as many living and nonliving things make noise, the analysis of these datasets is often done manually by domain experts. These analyses are painstakingly slow, and results are often incomplete.
# 
# 
# * In this competition, you will identify a wide variety of bird vocalizations in soundscape recordings. Due to the complexity of the recordings, they contain weak labels. There might be anthropogenic sounds (e.g., airplane overflights) or other bird and non-bird (e.g., chipmunk) calls in the background, with a particular labeled bird species in the foreground. Bring your new ideas to build effective detectors and classifiers for analyzing complex soundscape recordings!
# 
# 
# ### Which bird is this?
# 
# ![](https://partnersinflight.org/wp-content/uploads/2017/03/Bird-Collage-PIF.png)
# 

# ## 2. Preliminaries
# 
# #### Now Let's Begin by Importing the data
# 
# 

# In[ ]:


get_ipython().system(' pip install -q pydub')


# In[ ]:


import os


import random
import seaborn as sns
import cv2
# General packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL
import IPython.display as ipd
import glob
import h5py
import plotly.graph_objs as go
import plotly.express as px
from scipy import signal
from scipy.io import wavfile
from PIL import Image
from scipy.fftpack import fft
from pydub import AudioSegment
from tempfile import mktemp

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, LinearAxis, Range1d
from bokeh.models.tools import HoverTool
from bokeh.palettes import BuGn4
from bokeh.plotting import figure, output_notebook, show
from bokeh.transform import cumsum
from math import pi

output_notebook()


from IPython.display import Image, display
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


os.listdir('../input/birdsong-recognition/')


# ## Loading Dataset
# 

# In[ ]:


BASE_PATH = '../input/birdsong-recognition'

# image and mask directories
train_data_dir = f'{BASE_PATH}/train_audio'
test_data_dir = f'{BASE_PATH}/example_test_audio'

print('Reading data...')
test_audio_metadata = pd.read_csv(f'{BASE_PATH}/example_test_audio_metadata.csv')
test_audio_summary = pd.read_csv(f'{BASE_PATH}/example_test_audio_summary.csv')

train = pd.read_csv(f'{BASE_PATH}/train.csv')
test = pd.read_csv(f'{BASE_PATH}/test.csv')
submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')


print('Reading data completed')


# ### The dataset comprises of following important files:
# 
# 
# 1. **train_audio**:  The train data consists of short recordings of individual bird calls generously uploaded by users of [xenocanto.org](https://www.xeno-canto.org/)
# 
# 2. **test_audio**:  The hidden test set audio data.
# 
# 
# 3. **test.csv**:  Only the first three rows are available for download; the full test.csv is in the hidden test set.
# 
#     * `site`: Site ID.
#     * `row_id`: ID code for the row.
#     * `seconds`: the second ending the time window, if any. Site 3 time windows cover the entire audio file and have null entries for seconds.
#     * `audio_id`: ID code for the audio file.
# 
# 4. **example_test_audio_metadata.csv**:  Complete metadata for the example test audio. These labels have higher time precision than is used for the hidden test set.
# 
# 5. **example_test_audio_summary.csv**:  Metadata for the example test audio, converted to the same format as used in the hidden test set.
# 
#     * `filename_seconds`: a row identifier.
#     * `birds`: all ebird codes present in the time window.
#     * `filename`: name of file
#     * `seconds`: the second ending the time window.
# 
# 6. **train.csv**:  A wide range of metadata is provided for the training data. The most directly relevant fields are:
# 
#     * `ebird_code`: a code for the bird species. You can review detailed information about the bird codes by appending the code to https://ebird.org/species/, such as https://ebird.org/species/amecro for the American Crow.
#     * `recodist`: the user who provided the recording.
#     * `location`: where the recording was taken. Some bird species may have local call 'dialects', so you may want to seek geographic diversity in your training data.
#     * `date`: while some bird calls can be made year round, such as an alarm call, some are restricted to a specific season. You may want to seek temporal diversity in your training data.
#     * `filename`: the name of the associated audio file.

# In[ ]:


display(train.head())
print("Shape of train_data :", train.shape)


# In[ ]:


display(test.head())
print("Shape of test :", test.shape)


# In[ ]:


display(test_audio_metadata.head())
print("Shape of test_audio_metadata :", test_audio_metadata.shape)


# In[ ]:


display(test_audio_summary.head())
print("Shape of test_audio_metadata :", test_audio_summary.shape)


# ### Checking for Null values
# 

# In[ ]:


def check_null_values(df):
    # checking missing data
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data


# **train.csv**

# In[ ]:


check_null_values(train).head(10)


# **inference**
# * column `background` has a lot of null values, almost 61%
# * column `description`, `playback_used` and `bird_seen` also has significant amount of null values.

# **test.csv**

# In[ ]:


check_null_values(test)


# **example_test_audio_metadata.csv**

# In[ ]:


check_null_values(test_audio_metadata).head(10)


# **example_test_audio_summary.csv**

# In[ ]:


check_null_values(test_audio_summary)


# ## 3. Before starting EDA let's checkout few audio samples

# In[ ]:


sample_audio = [
    'aldfly/XC134874.mp3',
    'amegfi/XC109299.mp3',
    'brebla/XC104521.mp3',
    'lewwoo/XC161334.mp3',
    'macwar/XC125970.mp3',
    'norwat/XC124175.mp3',
    'pinjay/XC153392.mp3',
    'rufhum/XC133552.mp3',
    'weskin/XC124287.mp3',
    'yetvir/XC120867.mp3'    
]


# In[ ]:


for audio in sample_audio:
    print("Audio sample of bird", audio.split('/')[0])
    display(ipd.Audio(f"{train_data_dir}/{audio}"))


# ## 4. Let's perform some EDA
# 

# ### 4.1 Plotting distribution of birds based on latitude and longitude

# In[ ]:


fig = px.scatter(data_frame=train, x='longitude', y='latitude', color='ebird_code')
fig.show()


# ### 4.2 Plotting number of samples in train_audio folder
# 
# Pulling an audio sample from each bird
# 

# In[ ]:


sample_audio = []
total = 0

bird_audio_folders = [ folder for folder in glob.glob(f'{train_data_dir}/*')]
birds_data = []

for folder in bird_audio_folders:
    # get all the wave files
    all_files = [y for y in os.listdir(folder) if '.mp3' in y]
    total += len(all_files)
    # collect the first file from each dir
    sample_audio.append(folder + '/'+ all_files[0])
    birds_data.append({'bird_name': folder.split('/')[-1], 'num_audio_samples': len(all_files)})


# In[ ]:


birds_sample_df = pd.DataFrame(data= birds_data)
# taking first 25 samples from birds_sample_df
birds_sample_df_top30 = birds_sample_df.sample(30)


# In[ ]:


import plotly.express as px
# df = px.data.tips()
fig = px.bar(birds_sample_df_top30, x="num_audio_samples", y="bird_name",color='bird_name', orientation='h',
             hover_data=["num_audio_samples", "bird_name"],
             height=800,
             title='Number of audio samples in tarin data')
fig.show()


# In[ ]:


train.ebird_code.value_counts()


# **inference**
# 
# * number of samples of each bird can vary from 9 to 100
# * most of the birds has number of recordings equal to 100
# * only a single bird named `redhea` has number of recordings less than 10

# ### 4.3 Distribution of number of recordings based on country

# In[ ]:


# displaying only the top 30 countries
country = train.country.value_counts()
country_df = pd.DataFrame({'country':country.index, 'frequency':country.values}).head(30)

fig = px.bar(country_df, x="frequency", y="country",color='country', orientation='h',
             hover_data=["country", "frequency"],
             height=1000,
             title='Number of audio samples besed on country of recording')
fig.show()


# **inference**
# * Most of the voice samples are recorded in `USA`, `Canada` and `Mexico`.
# * While some samples are also recorded in countries like `Belgium`, `Panama`, etc. That may be because of the availability of some bird species limited to these countries only.

# ### 4.4 Checking for datetime features
# 
# 

# In[ ]:


##datetime feature section is inspired from this notebook, olease upvote it too
## https://www.kaggle.com/rohanrao/birdcall-eda-chirp-hoot-and-flutter

## let's create some datafremes 

df_date = train.groupby("date")["species"].count().reset_index().rename(columns = {"species": "recordings"})
df_date.date = pd.to_datetime(df_date.date, errors = "coerce")
df_date.dropna(inplace = True)
df_date["weekday"] = df_date.date.dt.day_name()


train["hour"] = pd.to_numeric(train.time.str.split(":", expand = True)[0], errors = "coerce")
df_hour = train[~train.hour.isna()].groupby("hour")["species"].count().reset_index().rename(columns = {"species": "recordings"})


df_weekday = df_date.groupby("weekday")["recordings"].sum().reset_index().sort_values("recordings", ascending = False)


# In[ ]:


# source 1
source_1 = ColumnDataSource(df_date)
tooltips_1 = [ ("Date", "@date{%F}"), ("Recordings", "@recordings")]
formatters = { "@date": "datetime" }

v1 = figure(plot_width = 800, plot_height = 450, x_axis_type = "datetime", title = "Date of recording")
v1.line("date", "recordings", source = source_1, color = "red", alpha = 0.6)

v1.add_tools(HoverTool(tooltips = tooltips_1, formatters = formatters))

v1.xaxis.axis_label = "Date"
v1.yaxis.axis_label = "Recordings"


# source 2
source_2 = ColumnDataSource(df_hour)

tooltips_2 = [
    ("Hour", "@hour"),
    ("Recordings", "@recordings")
]

v2 = figure(plot_width = 400, plot_height = 400, tooltips = tooltips_2, title = "Hour of recording")
v2.vbar("hour", top = "recordings", source = source_2, width = 0.75, color = "blue", alpha = 0.6)

v2.xaxis.axis_label = "Hour of day"
v2.yaxis.axis_label = "Recordings"


# source 3
source_3 = ColumnDataSource(df_weekday)

tooltips_3 = [
    ("Weekday", "@weekday"),
    ("Recordings", "@recordings")
]

v3 = figure(plot_width = 400, plot_height = 400, x_range = df_weekday.weekday.values, tooltips = tooltips_3, title = "Weekday of recording")
v3.vbar("weekday", top = "recordings", source = source_3, width = 0.75, color = "blue", alpha = 0.6)

v3.xaxis.axis_label = "Day of week"
v3.yaxis.axis_label = "Recordings"

v3.xaxis.major_label_orientation = pi / 2


show(column(v1, row(v2, v3)))


# **inference**
# * Most of the recordings are taken after `year 2012`, but there are few recordings as old as of `year 1980`, they may or may not be outliers
# 
# * Majority of the recordings have taken place durin the early hours of the day (6am - 11am).
# 
# * Number of recordings taken on weekends are much greater than the number of recordings taken on weekdays.

# ### 4.5 Comparing Spectrograms for different birds
# 
# 
# A **spectrogram** is a visual representation of the spectrum of frequencies of a signal as it varies with time. When applied to an audio signal, spectrograms are sometimes called **sonographs**, **voiceprints**, or **voicegrams**. When the data is represented in a 3D plot they may be called **waterfalls**.
# 
# To know more about spectograms please visit https://en.wikipedia.org/wiki/Spectrogram
# 

# In[ ]:


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    
    freqs, _, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, np.log(spec.T.astype(np.float32) + eps)


# In[ ]:


spect_samples = [
    'aldfly/XC134874.mp3',
    'ameavo/XC133080.mp3',
    'amecro/XC109768.mp3',
    'amepip/XC111040.mp3',
    'amewig/XC150063.mp3',
    'astfly/XC109920.mp3',
    'balori/XC101614.mp3',
    'bkbmag1/XC114081.mp3',
    'bkpwar/XC133993.mp3',
    'bnhcow/XC113821.mp3',
    'btnwar/XC101591.mp3',
    'carwre/XC109026.mp3',
    'chswar/XC101586.mp3',
    'evegro/XC110121.mp3',
    'greegr/XC109029.mp3',
    'hamfly/XC122665.mp3',
    'hoomer/XC134692.mp3',
    'horlar/XC113144.mp3',
    'lesgol/XC116239.mp3',
    'macwar/XC113825.mp3',
    'norfli/XC104536.mp3',
    'orcwar/XC113131.mp3',
    'pibgre/XC109907.mp3',
    'rebnut/XC104516.mp3',
    'ruckin/XC127130.mp3'    
]


# In[ ]:


fig = plt.figure(figsize=(22,22))
plt.suptitle('comparing spectograms for different birds', fontsize=20)


for i, filepath in enumerate(spect_samples):
    # Make subplots
    plt.subplot(5,5,i+1)
    bird_name, file_name = filepath.split('/')
    plt.title(f"Bird name: {bird_name}\nfile_name: {file_name}")
    # create spectogram
    mp3_audio = AudioSegment.from_file(f'{train_data_dir}/{filepath}', format="mp3")  # read mp3
    wname = mktemp('.wav')  # use temporary file
    mp3_audio.export(wname, format="wav")  # convert to wav
    
    samplerate, test_sound  = wavfile.read(wname)
    _, spectrogram = log_specgram(test_sound, samplerate)
    plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    plt.axis('off')


# ### 4.6 Comparing Spectrograms for same bird (aldfly)
# 

# In[ ]:


aldfly_samples = [
 'aldfly/XC157462.mp3',
 'aldfly/XC318444.mp3',
 'aldfly/XC374636.mp3',
 'aldfly/XC189268.mp3',
 'aldfly/XC296725.mp3',
 'aldfly/XC167789.mp3',
 'aldfly/XC373885.mp3',
 'aldfly/XC188432.mp3',
 'aldfly/XC189264.mp3',
 'aldfly/XC154449.mp3',
 'aldfly/XC189269.mp3',
 'aldfly/XC2628.mp3',
 'aldfly/XC420909.mp3',
 'aldfly/XC179600.mp3',
 'aldfly/XC188434.mp3',
 'aldfly/XC264715.mp3',
 'aldfly/XC189262.mp3',
 'aldfly/XC139577.mp3',
 'aldfly/XC16967.mp3',
 'aldfly/XC189263.mp3',
 'aldfly/XC318899.mp3',
 'aldfly/XC193116.mp3',
 'aldfly/XC269063.mp3',
 'aldfly/XC180091.mp3',
 'aldfly/XC381871.mp3',
]


# In[ ]:


fig = plt.figure(figsize=(22,22))
plt.suptitle('comparing spectograms for same bird', fontsize=20)

for i, filepath in enumerate(aldfly_samples):
    # Make subplots
    plt.subplot(5,5,i+1)
    bird_name, file_name = filepath.split('/')
    plt.title(f"Bird name: {bird_name}\nfile_name: {file_name}")
    
    # create spectogram
    mp3_audio = AudioSegment.from_file(f"{train_data_dir}/" + filepath, format="mp3")  # read mp3
    wname = mktemp('.wav')  # use temporary file
    mp3_audio.export(wname, format="wav")  # convert to wav
    
    samplerate, test_sound  = wavfile.read(wname)
    _, spectrogram = log_specgram(test_sound, samplerate)
    
    plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    plt.axis('off')


# ### 4.7 Comparing waveforms for different birds

# In[ ]:


fig = plt.figure(figsize=(22,22))
plt.suptitle('comparing waveforms for different bird', fontsize=20)


for i, filepath in enumerate(spect_samples):
    # Make subplots
    plt.subplot(5,5,i+1)
    bird_name, file_name = filepath.split('/')
    plt.title(f"Bird name: {bird_name}\nfile_name: {file_name}")
    # create spectogram
    mp3_audio = AudioSegment.from_file(f'{train_data_dir}/{filepath}', format="mp3")  # read mp3
    wname = mktemp('.wav')  # use temporary file
    mp3_audio.export(wname, format="wav")  # convert to wav
    
    samplerate, test_sound  = wavfile.read(wname)
    plt.plot(test_sound, '-', )
    plt.axis('off')


# ### 4.8 Comparing waveforms for same bird (aldfly)

# In[ ]:


fig = plt.figure(figsize=(22,22))
plt.suptitle('comparing waveforms for aldfly bird', fontsize=20)

for i, filepath in enumerate(aldfly_samples):
    # Make subplots
    plt.subplot(5,5,i+1)
    bird_name, file_name = filepath.split('/')
    plt.title(f"Bird name: {bird_name}\nfile_name: {file_name}")
    
    # create spectogram
    mp3_audio = AudioSegment.from_file(f"{train_data_dir}/" + filepath, format="mp3")  # read mp3
    wname = mktemp('.wav')  # use temporary file
    mp3_audio.export(wname, format="wav")  # convert to wav
    
    samplerate, test_sound  = wavfile.read(wname)
    plt.plot(test_sound, '-', )
    plt.axis('off')


# ### 4.9 Plotting waveform and spectogram side by side for better comparison

# In[ ]:


duplicate_samples = []
for val in spect_samples[:5]:
    duplicate_samples.append(val)
    duplicate_samples.append(val)


# In[ ]:


fig = plt.figure(figsize=(22,22))
plt.suptitle('comparing spectograms with waveforms for same bird', fontsize=20)


for i, filepath in enumerate(duplicate_samples):
    # Make subplots    
    plt.subplot(5,2,i+1)
    bird_name, file_name = filepath.split('/')
    plt.title(f"Bird name: {bird_name}\nfile_name: {file_name}")
    # create spectogram
    mp3_audio = AudioSegment.from_file(f'{train_data_dir}/{filepath}', format="mp3")  # read mp3
    wname = mktemp('.wav')  # use temporary file
    mp3_audio.export(wname, format="wav")  # convert to wav
    
    samplerate, test_sound  = wavfile.read(wname)
    _, spectrogram = log_specgram(test_sound, samplerate)

    if i % 2 == 0:
        plt.imshow(spectrogram.T, aspect='auto', origin='lower')  
    else:
        plt.plot(test_sound, '-', )
    
    plt.axis('off')


# ## 5. Baseline submission

# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:


submission.head()


# * by just submitting `nocall` we get a LB score 0.54, which suggests 54% of data in public test set has no bird voice at all. 

# # References
# 
# * https://en.wikipedia.org/wiki/Spectrogram
# * https://www.kaggle.com/timolee/audio-data-conversion-to-images-eda
# * https://www.kaggle.com/rohanrao/birdcall-eda-chirp-hoot-and-flutter

# # END NOTES
# This notebook is work in progress. 
# I will keep on updating this kernel with my new findings and learning in order to help everyone who has just started in this competition.
# 
# **<span style="color:Red">Please upvote this kernel if you like it . It motivates me to produce more quality content :)**  
