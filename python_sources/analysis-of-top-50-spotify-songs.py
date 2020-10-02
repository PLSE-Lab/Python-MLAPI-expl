#!/usr/bin/env python
# coding: utf-8

# In this kernel I have performed Exploratory Data Analysis on the **Top 50 Spotify Songs** and tried to identify relationship between thier popularity and various other features.I will use various other algorithms for predictions in future and add them in this kernel.

# I hope you find this kernel helpful and some **<font color='red'>UPVOTES</font>** would be very much appreciated

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# install pywaffle for waffle charts
get_ipython().system('pip install pywaffle')


# ### Importing required libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pywaffle import Waffle
import random

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# setting plot style for all the plots
plt.style.use('fivethirtyeight')

#accessing all the colors from matplotlib
colors=list(matplotlib.colors.CSS4_COLORS.keys())


# ### Loading the data

# In[ ]:


df = pd.read_csv('/kaggle/input/top50spotify2019/top50.csv', encoding="ISO-8859-1")
df.head()


# The **Unnamed:0** column is the same as index of the dataset, so will drop it. Also I will format the column names in order to make them easily accessable. 

# In[ ]:


#Drop the Unnamed: 0 column

df.drop('Unnamed: 0', inplace=True, axis=1)


# In[ ]:


# Renaming the columns

df.rename(columns={'Track.Name':'Track Name',
                   'Artist.Name':'Artist Name',
                   'Genre':'Genre',
                   'Beats.Per.Minute':'Beats per Minute',
                   'Energy':'Energy',
                   'Danceability':'Danceability',
                   'Loudness..dB..':'Loudness(dB)',
                   'Liveness':'Liveness',
                   'Valence.':'Valence',
                   'Length.':'Length',
                   'Acousticness..':'Acousticness',
                   'Speechiness.':'Speechiness',
                   'Popularity':'Popularity'}, inplace=True)


# In[ ]:


# let's see the dataset again
df.head()


# ### Features of the data set

# In[ ]:


df.info()


# There are **NO Null values** in the dataset. Also there are **3 categorical features** and **10 numerical features** in the dataset.

# ### Dimensions of the dataset
# 

# In[ ]:


print('Number of rows in the dataset: ',df.shape[0])
print('Number of columns in the dataset: ',df.shape[1])


# ### Basic statistical details about the dataset

# In[ ]:


df.describe().round(decimals=3)


# **The features described in the above data set are:**
# 
# **1. Count** tells us the number of NoN-empty rows in a feature.
# 
# **2. Mean** tells us the mean value of that feature.
# 
# **3. Std** tells us the Standard Deviation Value of that feature.
# 
# **4. Min** tells us the minimum value of that feature.
# 
# **5. 25%, 50%, and 75%** are the percentile/quartile of each features.
# 
# **6. Max** tells us the maximum value of that feature.

# ### Number of unique artists in the dataset

# In[ ]:


df['Artist Name'].nunique()


# ### Number of unique genre in the dataset

# In[ ]:


df['Genre'].nunique()


# ## Exploratory Data Analysis(EDA)

# First let's analyze each of the numerical features one by one.

# ### 1. Beates per Minute (Tempo)

# Beats per minute is the unit of measurement for measuring tempo. A "beat" is the standard measurement for a length of a piece of music. 
# 
# In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration

# In[ ]:


plt.figure(figsize=(8,4))
sns.distplot(df['Beats per Minute'], kde=False, bins=18,color='#3ff073', hist_kws=dict(edgecolor="black", linewidth=1))
plt.show()


# #### Minimum beats per minute

# In[ ]:


minimum_beats_per_min = df[df['Beats per Minute'] == df['Beats per Minute'].min()]
minimum_beats_per_min[['Track Name', 'Artist Name', 'Genre', 'Beats per Minute']].reset_index().drop('index', axis=1)


# #### Maximum beats per minute
# 

# In[ ]:


maximum_beats_per_min = df[df['Beats per Minute'] == df['Beats per Minute'].max()]
maximum_beats_per_min[['Track Name', 'Artist Name', 'Genre', 'Beats per Minute']].reset_index().drop('index', axis=1)


# The distribution of beats per minute is positively skewed and most of the values lie between 85 and 100

# ### 2. Energy
# 

# Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.

# In[ ]:


plt.figure(figsize=(8,4))
sns.distplot(df['Energy'], kde=False, bins=15,color='red', hist_kws=dict(edgecolor="k", linewidth=1))
plt.show()


# #### Minimum Energy

# In[ ]:


minimum_energy = df[df['Energy'] == df['Energy'].min()]
minimum_energy[['Track Name', 'Artist Name', 'Genre', 'Energy']].reset_index().drop('index', axis=1)


# #### Maximum Energy

# In[ ]:


maximum_energy = df[df['Energy'] == df['Energy'].max()]
maximum_energy[['Track Name', 'Artist Name', 'Genre', 'Energy']].reset_index().drop('index', axis=1)


# ### 3. Danceability
# 

# Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable

# In[ ]:


plt.figure(figsize=(8,4))
sns.distplot(df['Danceability'], kde=False, bins=15,color='violet', hist_kws=dict(edgecolor="black", linewidth=1))
plt.show()


# #### Maximum Danceability

# In[ ]:


maximum_danceability = df[df['Danceability'] == df['Danceability'].max()]
maximum_danceability[['Track Name', 'Artist Name', 'Genre', 'Danceability']].reset_index().drop('index', axis=1)


# #### Minimum Danceability
# 

# In[ ]:


minimum_danceability = df[df['Danceability'] == df['Danceability'].min()]
minimum_danceability[['Track Name', 'Artist Name', 'Genre', 'Danceability']].reset_index().drop('index', axis=1)


# ### 4. Loudness(dB)

# Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.

# In[ ]:


plt.figure(figsize=(8,4))
sns.distplot(df['Loudness(dB)'], kde=False, bins=15,color='aqua', hist_kws=dict(edgecolor="black", linewidth=1))
plt.show()


# #### Minimum Loudness

# In[ ]:


minimum_loudness = df[df['Loudness(dB)'] == df['Loudness(dB)'].min()]
minimum_loudness[['Track Name', 'Artist Name', 'Genre', 'Loudness(dB)']].reset_index().drop('index', axis=1)


# #### Maximum Loudness

# In[ ]:


maximum_loudness = df[df['Loudness(dB)'] == df['Loudness(dB)'].max()]
maximum_loudness[['Track Name', 'Artist Name', 'Genre', 'Loudness(dB)']].reset_index().drop('index', axis=1)


# ### 5. Liveness

# Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live. 

# In[ ]:


plt.figure(figsize=(8,4))
sns.distplot(df['Liveness'], kde=False, bins=15,color='darkorchid', hist_kws=dict(edgecolor="black", linewidth=1))
plt.show()


# #### Minimum Liveness
# 

# In[ ]:


minimum_Liveness = df[df['Liveness'] == df['Liveness'].min()]
minimum_Liveness[['Track Name', 'Artist Name', 'Genre', 'Liveness']].reset_index().drop('index', axis=1)


# #### Maximum Liveness

# In[ ]:


maximum_Liveness = df[df['Liveness'] == df['Liveness'].max()]
maximum_Liveness[['Track Name', 'Artist Name', 'Genre', 'Liveness']].reset_index().drop('index', axis=1)


# ### 6. Valence
# 

# A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).

# In[ ]:


plt.figure(figsize=(8,4))
sns.distplot(df['Valence'], kde=False, bins=15,color='darkgreen', hist_kws=dict(edgecolor="black", linewidth=1))
plt.show()


# #### Minimum Valence

# In[ ]:


minimum_Valence = df[df['Valence'] == df['Valence'].min()]
minimum_Valence[['Track Name', 'Artist Name', 'Genre', 'Valence']].reset_index().drop('index', axis=1)


# #### Maximum Valence

# In[ ]:


maximum_Valence = df[df['Valence'] == df['Valence'].max()]
maximum_Valence[['Track Name', 'Artist Name', 'Genre', 'Valence']].reset_index().drop('index', axis=1)


# ### 7. Length

# Describes the duration of the song in seconds.

# In[ ]:


plt.figure(figsize=(8,4))
sns.distplot(df['Length'], kde=False, bins=15,color='m', hist_kws=dict(edgecolor="black", linewidth=1))
plt.show()


# #### Minimum Length

# In[ ]:


minimum_Length = df[df['Length'] == df['Length'].min()]
minimum_Length[['Track Name', 'Artist Name', 'Genre', 'Length']].reset_index().drop('index', axis=1)


# #### Maximum Length

# In[ ]:


maximum_Length = df[df['Length'] == df['Length'].max()]
maximum_Length[['Track Name', 'Artist Name', 'Genre', 'Length']].reset_index().drop('index', axis=1)


# ### 8. Acousticness

# A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. 

# In[ ]:


plt.figure(figsize=(8,4))
sns.distplot(df['Acousticness'], kde=False, bins=15,color='darkblue', hist_kws=dict(edgecolor="black", linewidth=1))
plt.show()


# #### Minimum Acousticness

# In[ ]:


minimum_Acousticness = df[df['Acousticness'] == df['Acousticness'].min()]
minimum_Acousticness[['Track Name', 'Artist Name', 'Genre', 'Acousticness']].reset_index().drop('index', axis=1)


# #### Maximum Acousticness
# 

# In[ ]:


maximum_Acousticness = df[df['Acousticness'] == df['Acousticness'].max()]
maximum_Acousticness[['Track Name', 'Artist Name', 'Genre', 'Acousticness']].reset_index().drop('index', axis=1)


# ### 9. Popularity
# 

# The higher the popularity value the more popular a given song is.

# In[ ]:


plt.figure(figsize=(8,4))
sns.distplot(df['Popularity'], kde=False, bins=15,color='orange', hist_kws=dict(edgecolor="black", linewidth=1))
plt.show()


# #### Minimum Popularity
# 

# In[ ]:


minimum_Popularity = df[df['Popularity'] == df['Popularity'].min()]
minimum_Popularity[['Track Name', 'Artist Name', 'Genre', 'Popularity']].reset_index().drop('index', axis=1)


# #### Maximum Popularity

# In[ ]:


maximum_Popularity = df[df['Popularity'] == df['Popularity'].max()]
maximum_Popularity[['Track Name', 'Artist Name', 'Genre', 'Popularity']].reset_index().drop('index', axis=1)


# ### Number of songs in each genre
# 

# In[ ]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(16,8))
sns.countplot(x='Genre', data = df, linewidth=2, edgecolor='black')
plt.xticks(rotation=90)
plt.show()


# ### Wafflte Chart of all the Genres

# In[ ]:


calculated = df.Genre.value_counts()
sns.set_style('darkgrid')
fig = plt.figure(figsize=(13,8),
    FigureClass=Waffle, 
    rows=5, 
    values=list(calculated.values),
    labels=list(calculated.index),
                 legend={'loc': 'upper left', 'bbox_to_anchor': (1.1, 1)},
                 edgecolor='black',
                 colors= random.sample(colors,21),
)


# **Dance pop** has the most number of songs (8) in the top 50 category followed by pop (7) and Latin (5)

# ### Heatmap of correlation between various features

# In[ ]:


plt.figure(figsize=(12,8))
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr,mask=mask, annot=True, linewidths=1, cmap='YlGnBu')
plt.show()


# Only two pairs of features 1. Beats per Minute & Speechiness and 2. Energy & Loudness(dB) have a correlation value of more than 0.5

# ### Pairplot of all the features

# In[ ]:


sns.set_style('whitegrid')
sns.pairplot(df)
plt.show()


# ### Number of times an artist's name appears in the Top 50 songs list

# In[ ]:


plt.figure(figsize=(20,8))
plt.style.use('fivethirtyeight')
sns.countplot(x=df['Artist Name'],data=df, linewidth=2, edgecolor='black')
plt.title('Number of times an artist appears in the top 50 songs list')
plt.xticks(rotation=90)
plt.show()


# **Ed Sheeran has the most(4) number of songs in the top 50 category**

# ### Boxplot of various numerical features vs. Genre

# #### 1. Energy vs. Genre
# 

# In[ ]:


plt.figure(figsize=(20,6))
sns.boxplot(x='Genre', y='Energy', data = df, linewidth=2)
plt.xticks(rotation=90)
plt.show()


# Dance Pop has the most variability in Energy levels ranging from the lowest 32 to the highest value 88. However Reggaton Flow as the highest median value among all the genres.
# 
# Songs with more energy level are energetic tracks feel fast, loud, and noisy. 

# #### 2. Beats per minute vs. Genre

# In[ ]:


plt.figure(figsize=(20,6))
sns.boxplot(x='Genre', y='Beats per Minute', data = df, linewidth=2)
plt.xticks(rotation=90)
plt.show()


# Here Latin has the most variability in values ranging from as low as 92 to as high as 176. Country Rap however has the highest median value i.e. 145

# #### 3. Danceability vs. Genre

# In[ ]:


plt.figure(figsize=(20,6))
sns.boxplot(x='Genre', y='Danceability', data = df, linewidth=2)
plt.xticks(rotation=90)
plt.show()


# Canadian Hip Hop has the highest median value of danceability. Edm and Pop however have the most variability in their danceability values.
# 
# Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity.

# #### 4. Loudness vs. Genre

# In[ ]:


plt.figure(figsize=(20,6))
sns.boxplot(x='Genre', y='Loudness(dB)', data = df, linewidth=2)
plt.xticks(rotation=90)
plt.show()


# Pop and Canadian Hip Hop have the most variability in their Loudness values. Brostep however have the highest median value for Loudness.

# #### 5. Liveness vs. Genre

# In[ ]:


plt.figure(figsize=(20,6))
sns.boxplot(x='Genre', y='Liveness', data = df, linewidth=2)
plt.xticks(rotation=90)
plt.show()


# **Latin** has the highest variability in their Liveness values. **Brostep** however has the highest median value for liveness. 
# 
# Liveness detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8(80%) provides strong likelihood that the track is live.

# #### 6. Valence vs. Genre

# In[ ]:


plt.figure(figsize=(20,6))
sns.boxplot(x='Genre', y='Valence', data = df, linewidth=2)
plt.xticks(rotation=90)
plt.show()


# **Dance Pop**, **Pop** and **Dfw rap** have the most variability in their values. **Boy Band** has the highest median value for valence.
# 
# Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).

# #### 7. Length vs. Genre

# In[ ]:


plt.figure(figsize=(20,5))
sns.boxplot(x='Genre', y='Length', data = df, linewidth=2)
plt.xticks(rotation=90)
plt.show()


# **Reggaeton Flow** has the highest median value for length. It also has the highest maximum and minimum values of length indicating that Reggaton Flow songs usually last more than 5 minutes.

# #### 8. Acousticness vs. Genre
# 

# In[ ]:


plt.figure(figsize=(20,6))
sns.boxplot(x='Genre', y='Acousticness', data = df, linewidth=2)
plt.xticks(rotation=90)
plt.show()


# **Dance Pop** has the most variability in their acousticness values. However Australian Pop has the highest median value(probably because there's just one song present in that category). An outlier is also present in the Pop genre with an unusually high value.

# #### 9. Speechiness vs. Genre

# In[ ]:


plt.figure(figsize=(20,6))
sns.boxplot(x='Genre', y='Speechiness', data = df, linewidth=2)
plt.xticks(rotation=90)
plt.show()


# #### 10. Popularity vs. Genre

# In[ ]:


plt.figure(figsize=(20,6))
sns.boxplot(x='Genre', y='Popularity', data = df, linewidth=2)
plt.xticks(rotation=90)
plt.show()


# **dfw rap** has the highest maximum, minimum as well as median popularity values

# ### Relationsip between Popularity and other Numerical features

# In[ ]:


colors=list(matplotlib.colors.CSS4_COLORS.keys())
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18,10))
axes = axes.flatten()

numeric_cols = list(df.select_dtypes([np.number]).columns[:-1])         #selecting all the numeric columns except Popularity
plt.tight_layout(pad=2)
for i, j in enumerate(numeric_cols):
    axes[i].scatter(x=df[j], y=df['Popularity'], color= random.choice(colors), edgecolor='black')
    axes[i].set_xlabel(j)
    axes[i].set_ylabel('Popularity')


# **Dance Pop** and **Latin** display the most variability in their speechiness values. **Electropop** however represents the highest median value for speechiness.
# 
# Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words

# ### What's next?
# 
# 1. Plotting and exploring relations between Popularity and other numerical features
# 
# 2. Finding the characterstics of songs with the most popularity values.
# 
# **Suggestions are welcome**
# <font color='red'>UPVOTE</font> if you found the notebook useful.
