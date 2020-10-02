#!/usr/bin/env python
# coding: utf-8

# # Data Visualization and Sound Processing Tutorial !
# 
# This notebook is created for beginners to take a look of this competition dataset and how to process sound data.  
# This is my first time to tackle with sound data competition, so I refered some other articles.
# 
# 1st public version: 04, Jul, 2020.
# 
# Here is table of contents:
# - [Library Import and Data Check](#Library-Import-and-Data-Check)
# - [Data Visualization](#Data-Visualization)
#     - [General Information and Visualization](#General-Information-and-Visualization)
#     - [Location Information and Visualization](#Location-Information-and-Visualization)
# - [Sound Processing](#Sound-Processing)
# - [Reference](#Reference)

# # Library Import and Data Check

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    # skip audio file
    if 'train_audio' in dirname:
        continue
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# For Visualization Library
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[ ]:


get_ipython().system('pip install folium')


# In[ ]:


get_ipython().system('pip install pydub')


# In[ ]:


# Create a world map to show distributions of birds
import folium
from folium.plugins import MarkerCluster
import plotly.express as px

# Machine Learning
from sklearn import model_selection

# Sound processing library
import librosa.display


# In[ ]:


# Sound Processing Library
import pydub
from io import BytesIO
from IPython.display import Audio, display
import soundfile as sf


# In[ ]:


# Data Read and Check
INPUT_DIR = "/kaggle/input/birdsong-recognition/"
train_df = pd.read_csv(INPUT_DIR + "train.csv")
test_df = pd.read_csv(INPUT_DIR + "test.csv")
train_df.head()


# In the train.csv, we have metadata of each sound records.  
# Fron now on, we'll investigate this metadata deeper and deeper.

# In[ ]:


# We can get column information by using .columns
train_df.columns


# In[ ]:


# We can get more specific information by using .info() method
# Using this method, we can get column_name, non_null count, and dtype of each column.
train_df.info()


# In[ ]:


# On the contrary, we have less columns in the test.csv
# We have site, row_id, seconds, and audio_id columns.
test_df


# In[ ]:


# Regarding example data, we take a look later.
# We should learn how to use these information for submission later.

example_audio_summary_df = pd.read_csv(INPUT_DIR + "example_test_audio_summary.csv")
example_audio_metadata_df = pd.read_csv(INPUT_DIR + "example_test_audio_metadata.csv")
example_audio_summary_df.head()


# In[ ]:


# Submission format
# We should predict birds column in each row_id.

sample_submission_df = pd.read_csv(INPUT_DIR + "sample_submission.csv")
sample_submission_df


# # Data Visualization
# 
# Fron now on, we try to visualize metadata information of train dataset.   
# First we only use train.csv information and process sound data in the next section.
# 
# ## General Information and Visualization

# In[ ]:


# countplot method of seaborn is quite useful to visualize the number of counts in one column.
sns.countplot("rating", data=train_df)
plt.title("Record Counts in each Rating")


# The highest rating 5.0 has the most records compared to others.  
# We have 0.0 rating, but it seems we don't have not so many records with rating 0.5 and 1.0.

# In[ ]:


sns.countplot("playback_used", data=train_df)
plt.title("Record Counts in playback_used flag")


# Most of the data don't use playback.

# In[ ]:


plt.figure(figsize=(18, 6))
sns.countplot("ebird_code", data=train_df)
plt.xticks(rotation=90)


# We have 264 kinds of birds in this dataset.  
# Many kinds of these birds have 100 counts, some have less than 100 counts.

# In[ ]:


sns.countplot("channels", data=train_df)


# About channel, 1 (mono) channel has a little bit more counts than 2 (stereo).

# In[ ]:


# groupby method is quite useful to calculate metrics by column data.
temp_series= train_df.groupby("date")["xc_id"].count()
temp_series


# In[ ]:


temp_series.index


# In[ ]:


# Processing date column (convert date column from object type to datetime type
# 
# Usually, we can use .astype method like below:
# df["date"] = df["date"].astype("datetime64[ns]")
# 
# but we have some illegal input in this dataset, so I arranged a little bit.

idx_list = []

for idx in temp_series.index:
    new_idx = idx
    
    # Year before 1970 is converted into 1970
    if idx[:4] <= '1970':
        new_idx = '1970-01-01' 
    
    # Month should be between 1 and 12.
    if idx[5:7] == '00':
        new_idx = new_idx[:5] + '01' + new_idx[7:]
    
    # Day should be at least 01 (Not 00)
    if idx[8:] == '00':
        new_idx = new_idx[:8] + '01'
    
    idx_list.append(new_idx)


# In[ ]:


temp_series.index = idx_list
temp_series.index = temp_series.index.astype("datetime64[ns]")
temp_series


# In[ ]:


# Now we can plot counts in each year.
temp_series.plot(figsize=(10,4))
plt.xlabel("Year")
plt.ylabel("Record Count")
plt.title("Record Count Transition")


# Recently we have more records than the past few decades.

# In[ ]:


sns.distplot(train_df["duration"])


# The distribution of this duration column in this dataset is seemed to be possion one.  
# That is, we have many short uration data in the shorter duration, and less super-long duration data.

# In[ ]:


train_df["pitch"].unique()


# In[ ]:


column_value = ['Not specified', 'both', 'increasing', 'level', 'decreasing']

fig, axs = plt.subplots(1,3, sharey=True, figsize=(8, 4))
sns.countplot(train_df["pitch"], ax=axs[0])
sns.countplot(train_df["speed"], ax=axs[1])
sns.countplot(train_df["volume"], ax=axs[2])
axs[0].set_xticklabels(column_value, rotation=90)
axs[1].set_xticklabels(column_value, rotation=90)
axs[2].set_xticklabels(column_value, rotation=90)


# In[ ]:


print(train_df["species"].unique())
print("The number of spicies is {}.".format(train_df["species"].nunique()))


# The number of spicies is 264, as we've checked when we visualized ebird_code countplot

# In[ ]:


sns.countplot(train_df["number_of_notes"])


# The number of notes in most part of the dataset is "Not-specified".  
# However we have some values with number of notes.

# In[ ]:


train_df["secondary_labels"]


# In[ ]:


sns.countplot(train_df["bird_seen"])
print("Bird seen yes is {}".
      format(len(train_df[train_df["bird_seen"] == "yes"]) / len(train_df)))


# Bird_seen of most of the record is yes. (About 76 %)

# In[ ]:


train_df["sci_name"].nunique()


# In[ ]:


train_df["location"].nunique()


# In[ ]:


train_df["sampling_rate"].unique()


# In[ ]:


order=['48000 (Hz)', '44100 (Hz)', '32000 (Hz)', '24000 (Hz)',
       '22050 (Hz)', '16000 (Hz)', '11025 (Hz)', '8000 (Hz)']
sns.countplot(train_df["sampling_rate"], order=order)
plt.xticks(rotation=60)


# Most of the sampling rate is 44100 Hz or 48000 Hz.

# In[ ]:


train_df["type"].unique()


# In[ ]:


train_df["description"]


# In[ ]:


sns.distplot(train_df["bitrate_of_mp3"].str[:-6], kde=False)


# As for bitrate of mp3 information, most of the parts is around 125,000 (bps).

# In[ ]:


sns.countplot(train_df["file_type"])


# Most of the files are .mp3 type.

# In[ ]:


train_df["background"]


# In[ ]:


sns.countplot(train_df["length"], order=["Not specified", "0-3(s)", "3-6(s)", "6-10(s)", ">10(s)"])


# ## Location Information and Visualization
# 
# This is the reference of this subsection
# - https://python-graph-gallery.com/310-basic-map-with-markers/ 
# - https://plotly.com/python/bubble-maps/

# In[ ]:


sns.distplot(train_df.loc[train_df["latitude"] != "Not specified", "latitude"])
plt.title("Distribution of Latitude")


# The latitude of the records is concentrated around between 20 and 60 degrees.

# In[ ]:


sns.distplot(train_df.loc[train_df["longitude"] != "Not specified", "longitude"])


# Regarding longitude, the most part of the records are bretween -150 and -50 degrees. (Around United States)

# In[ ]:


plt.figure(figsize=(8, 4))
sns.countplot(train_df["country"])
plt.xticks(rotation=90)


# It seems United States has huge amounts of records.

# In[ ]:


# Extract only necessesary information
temp_df = train_df.loc[(train_df["latitude"] != 'Not specified') & (train_df["longitude"] != 'Not specified'), 
                       ["country", "latitude", "longitude", "xc_id", "ebird_code"]]

# Convert to float type
temp_df["latitude"] = temp_df["latitude"].astype("float")
temp_df["longitude"] = temp_df["longitude"].astype("float")
temp_df.head()


# In[ ]:


# From now on, we plot worldmap and visualize which countries have many records. 
draw_df = temp_df.groupby("country")[["latitude", "longitude"]].mean()
draw_df = pd.concat([draw_df, temp_df.groupby("country")["xc_id"].count()], axis=1)
draw_df = draw_df.rename(columns={"xc_id":"count"})
draw_df = draw_df.reset_index()
draw_df


# In[ ]:





# In[ ]:


#empty map
world_map= folium.Map(tiles="cartodbpositron")
marker_cluster = MarkerCluster().add_to(world_map)


# In[ ]:


#for each coordinate, create circlemarker of user percent
for i in range(len(draw_df)):
        lat = draw_df.iloc[i]['latitude']
        long = draw_df.iloc[i]['longitude']
        radius= draw_df.iloc[i]["count"] / len(draw_df)
        popup_text = """Country : {}<br>
                    Counts : {}<br>"""
        popup_text = popup_text.format(draw_df.iloc[i]['country'],
                                   draw_df.iloc[i]['count']
                                   )
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True).add_to(marker_cluster)
#show the map
world_map


# In[ ]:


# Make an empty map
m = folium.Map(location=[20,0], tiles="Mapbox Bright", zoom_start=2)
 
# I can add marker one by one on the map
for i in range(len(draw_df)):
   folium.Circle(
      location=[draw_df.iloc[i]['longitude'], draw_df.iloc[i]['latitude']],
      popup=draw_df.iloc[i]['country'],
      radius=draw_df.iloc[i]['count'] / len(draw_df) * 10,
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(m)


# In[ ]:


draw_df


# In[ ]:


fig = px.scatter_geo(draw_df, lat="latitude", lon= "longitude", color="country",
                     hover_name="country", size="count",
                     projection="natural earth")
fig.update_geos(
    visible=False, resolution=50,
    showcountries=True, countrycolor="RebeccaPurple"
)
fig


# This world map is showing which countries have many records.  
# As we've seen before, United States is the most recorded place.

# In[ ]:


grid_by_bird_df = temp_df.groupby("ebird_code")[["latitude", "longitude"]].mean()
grid_by_bird_df


# In[ ]:


sns.scatterplot(y=grid_by_bird_df.iloc[:, 0], x=grid_by_bird_df.iloc[:, 1], hue=grid_by_bird_df.index)
plt.ylim([-90, 90])
plt.xlim([-180, 180])
plt.legend(bbox_to_anchor=(1.01, 1.01), ncol=10)


# In[ ]:


train_df.columns


# In[ ]:


from wordcloud import WordCloud


# In[ ]:


wordcloud = WordCloud(background_color="white")


# In[ ]:


wordcloud.generate_from_text(" ".join(list(train_df.loc[train_df["description"].notnull(), "description"])))
plt.imshow(wordcloud)


# It seems we have some records that are through high pass filter in the dataset. 

# In[ ]:





# In[ ]:


" ".join(list(train_df.loc[train_df["description"].notnull(), "description"]))


# # Sound Processing
# 
# From now on, I'll try to process sound data!  
# I refered some articles and codes to write the cells below.  
# - https://github.com/ipython-books/cookbook-2nd-code/blob/master/chapter11_image/06_speech.ipynb
# - https://dev.to/apoorvadave/environmental-sound-classification-1hhl
# - https://blog.brainpad.co.jp/entry/2018/04/17/143000 (In Japanese Only)

# In[ ]:


mp3_filename = "/kaggle/input/birdsong-recognition/train_audio/aldfly/XC134874.mp3"


# pydub is a library which can process sound data in python.

# In[ ]:


# Read sound file 
audio = pydub.AudioSegment.from_mp3(mp3_filename)

# Convert mp3 data into wave data 
wave = audio.export('_', format="wav")
wave.seek(0)
wave = wave.read()

# We get the raw data by removing first 24 bytes of the header.
X = np.frombuffer(wave, np.int16)[24:] / 2.**15


# In[ ]:


# This function allows us to play sound data in jupyter notebook

def play(x, fr, autoplay=False):
    display(Audio(x, rate=fr, autoplay=autoplay))


# In[ ]:


# We can get sampling rate by accesing like below
int(train_df.loc[train_df["xc_id"] == 134874, "sampling_rate"].str[:-4])


# In[ ]:


# This cell outputs the widget to play sound.
play(X, fr=48000, autoplay=False)


# In[ ]:


X.shape


# In[ ]:


# Sound wave plot
fr = 48000
t = np.linspace(0, len(X) / fr, len(X))
plt.plot(t, X, lw=1)


# In[ ]:


# n_mels is number of Mel bands to generate
n_mels=128
# hop_length is number of samples between successive frames.
hop_length=2068
# n_fft is length of the FFT window
n_fft=2048
# Passing through arguments to the Mel filters
S = librosa.feature.melspectrogram(y=X, sr=44000, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)

log_S = librosa.power_to_db(S)
print(log_S.shape)

plt.figure(figsize=(12, 4))
librosa.display.specshow(data=log_S, sr=44000, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()


# This plot shows the strength of each Hz in each time.  
# That is, for example, around 5 seconds, sound of 2048 ~ 4096 Hz is slightly louder than other sounds of Hz.  
# 
# This result can be used as an image, and image processing model can be used with this result.
# 
# Let's see how this plot differs from each other.  
# To do that, I define some functions.

# In[ ]:


def get_features(filename, sampling_rate):
    """
    This function returns mel-frequency cepstrum from its filename and sampling rate
    
    Parameters
    ----------
    filename : string
        target filename path
    sampling_rate : int
        target filename sampling rate

    Returns
    -------
    mfccs_scaled : np.array
        mel-frequency cepstrum 
    """

    if filename: 
        audio = pydub.AudioSegment.from_mp3(filename)

        wave = audio.export('_', format="wav")
        wave.seek(0)
        wave = wave.read()

        X = np.frombuffer(wave, np.int16)[24:] / 2.**15
    
        sr= sampling_rate

    # mfcc (mel-frequency cepstrum)
    mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    
    """
    Result Visualization Part
    """
    play(X, fr=sr, autoplay=False)
    
    # n_mels is number of Mel bands to generate
    n_mels=128
    # hop_length is number of samples between successive frames.
    hop_length=2068
    # n_fft is length of the FFT window
    n_fft=2048
    # Passing through arguments to the Mel filters
    S = librosa.feature.melspectrogram(y=X, sr=sampling_rate, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    
    log_S = librosa.power_to_db(S)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(data=log_S, sr=sampling_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
        
    return mfccs_scaled


# In[ ]:


def extract_features(target_df):
    """
    This function returns mel-frequency cepstrum result of all train dataset
    
    Parameters
    ----------
    target_df: pandas.DataFrame
        DataFrame of train dataset
    
    Returns
    -------
    features_df : pandas.DataFrame
        mel-frequency cepstrum result
    """
    features_list = []
    features_df = pd.DataFrame()
    
    for idx in target_df.index:
        if idx % 100 == 0:
            print(idx)
        
        sampling_rate = int(target_df.loc[target_df.index == idx, "sampling_rate"].str[:-4])
        bird_name = list(target_df.loc[target_df.index == idx, "ebird_code"])[0]
        xc_id = list(target_df.loc[target_df.index == idx, "xc_id"])[0]
        
        filename = INPUT_DIR + "train_audio/" + bird_name + "/XC" + str(xc_id) + ".mp3"
        
        try:
            mfccs = get_features(filename, sampling_rate)
        except Exception as e:
            print("Extraction error at {}".format(idx))
            continue
        features_list.append([mfccs, bird_name])
    
    features_df = pd.DataFrame(features_list,columns = ['feature','class_label']) 
    return features_df


# In[ ]:


ebird_idx_list = [0]
ebird_origin = "aldfly"

for idx in train_df.index:
    if train_df.loc[idx, "ebird_code"] != ebird_origin:
        ebird_idx_list.append(idx)
        ebird_origin = train_df.loc[idx, "ebird_code"]


# In[ ]:


extract_features(target_df=train_df.iloc[ebird_idx_list[:5], :])


# You can directly listen to bird call with jupyter notebook.  
# And when you click play button, you can recognize that each bird sound differs from each other.  
# Also, you can see the differences by seeing the mel spectogram plot.  
# For example, when you listen to the fifth sound, around 0:20 ~ 0:35 seconds, birds call actively.  
# The fifth mel spectogram also shows around that time, a loud sound is recorded between 1024 ~ 4096 Hz. 

# In[ ]:


def get_features(filename, sampling_rate):
    """
    This function returns mel-frequency cepstrum from its filename and sampling rate
    
    Parameters
    ----------
    filename : string
        target filename path
    sampling_rate : int
        target filename sampling rate

    Returns
    -------
    mfccs_scaled : np.array
        mel-frequency cepstrum 
    """

    if filename: 
        audio = pydub.AudioSegment.from_mp3(filename)

        wave = audio.export('_', format="wav")
        wave.seek(0)
        wave = wave.read()

        X = np.frombuffer(wave, np.int16)[24:] / 2.**15
    
        sr= sampling_rate

    # mfcc (mel-frequency cepstrum)
    mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    
   
        
    return mfccs_scaled


# In[ ]:


def extract_features(target_df):
    """
    This function returns mel-frequency cepstrum result of all train dataset
    
    Parameters
    ----------
    target_df: pandas.DataFrame
        DataFrame of train dataset
    
    Returns
    -------
    features_df : pandas.DataFrame
        mel-frequency cepstrum result
    """
    features_list = []
    features_df = pd.DataFrame()
    
    for idx in target_df.index:
        if idx % 100 == 0:
            print(idx)
        
        sampling_rate = int(target_df.loc[target_df.index == idx, "sampling_rate"].str[:-4])
        bird_name = list(target_df.loc[target_df.index == idx, "ebird_code"])[0]
        xc_id = list(target_df.loc[target_df.index == idx, "xc_id"])[0]
        
        filename = INPUT_DIR + "train_audio/" + bird_name + "/XC" + str(xc_id) + ".mp3"
        
        try:
            mfccs = get_features(filename, sampling_rate)
        except Exception as e:
            print("Extraction error at {}".format(idx))
            continue
        features_list.append([mfccs, bird_name])
    
    features_df = pd.DataFrame(features_list,columns = ['feature','class_label']) 
    return features_df


# Warning! 
# The cell below takes time!

# In[ ]:


get_ipython().run_cell_magic('time', '', 'features_df = extract_features(train_df)')


# In[ ]:


features_df.head()


# In[ ]:


# Save DataFrame and you can use it with other kernels by input this notebook output.  
# With this dataframe, you can create models like CNN.
features_df.to_pickle("train_data.pkl")


# In[ ]:


example_audio_metadata_df


# In[ ]:


example_audio_summary_df


# In[ ]:


filename = "../input/birdsong-recognition/example_test_audio/BLKFR-10-CPL_20190611_093000.pt540.mp3"


# In[ ]:


audio = pydub.AudioSegment.from_mp3(filename)

wave = audio.export('_', format="wav")
wave.seek(0)
wave = wave.read()

X = np.frombuffer(wave, np.int16)[24:] / 2.**15
    
sr= 22000


# In[ ]:


play(X, sr, False)


# # Reference
# 
# Here is the reference of this notebook.
# - https://github.com/ipython-books/cookbook-2nd-code/blob/master/chapter11_image/06_speech.ipynb
# - https://dev.to/apoorvadave/environmental-sound-classification-1hhl
# - https://blog.brainpad.co.jp/entry/2018/04/17/143000 (In Japanese Only)
# - https://python-graph-gallery.com/310-basic-map-with-markers/ 
# - https://plotly.com/python/bubble-maps/
# 
# # Acknowledgement
# 
# Thank you for reading this notebook.  
# I know my notebook isn't complete accurate, but I hope this notebook works as a clue for beginners.  
# Any comments and upvotes are very welcome, Thank you !!

# In[ ]:




