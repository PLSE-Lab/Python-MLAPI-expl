#!/usr/bin/env python
# coding: utf-8

# # Predective Model on Spotify Classification

# Hi All, welcome to my first notebook in Kaggle. <br/>
# This model is about predict the music whether the user like or not within the playist. <br/>
# I am very apperiacted that everyone could comment this notebook. This is the way to make the rookie improve. <br/><br/>
# ### **I am Still Learning - MICHELANGELO**

# ### Data Summary: 

# This dataset is provided by kaggle user GeorgeMcIntire.<br/>
# I am impreseed that he is the pioneer that who changes his professional from journalism to data science. <br/>
# His story gives me the encouragement to improve through data science journey.<br/> <br/>
# 
# At first, he writes the code to access the Spotify API to obtain the music.<br/>
# There is a lot of music attributes from the API result. For example, tempo & time_signature.<br/>
# So that, he selected the music from his playlist in order to create the dataset.
# 

# In[ ]:


#Import the neccessary library for the task
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import simplefilter # ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Read the dataset from the kaggle
dataset = pd.read_csv('/kaggle/input/spotifyclassification/data.csv', index_col = 0)


# Let's take a look the first 5 rows and last 5 rows of the dataset.

# In[ ]:


dataset.head()


# In[ ]:


dataset.tail()


# Wow there's a lot of different type of features in the dataset. But **how many features and rows** in this dataset?

# In[ ]:


print('Dataset: ', dataset.shape[0], 'Rows', dataset.shape[1], 'Features')


# There are total 2017 rows and 16 features include the class label in this dataset. Which means that there are 2017 pieces of music in the playist.<br/>
# Let's take a look that how many music that the user like.
# 

# In[ ]:


dataset['target'].value_counts()


# We can see that there is 1020 music the user favor music from the dataset. After that, we might need to know the features name.

# In[ ]:


dataset.columns.values


# ### Categorizing

# * Class Labels: target
# * Interval: acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness,valence 
# * Time: duration_ms
# * Numerical: tempo
# * Ordinal: key, time_signature
# * Binary: mode
# * String: song_title & artist
# 
# 
# I realzied that there are many features have the similarity.<br/>
# For example, instrumentalness is the measure to define the music is instrumental music or not. And, speechiness used to measure the music is vocable. Both features are used to define the music type. Therefore, these 2 features have the similarity.Besides, danceability and energy both features are used to define the music is energetic or not.<br/> <br/>
# 
# Meanwhile, we grouping these features into 4 groups. Which are 
# 1. SpeechinessMusic
# 2. Energetic
# 3. MusicAttribute
# 4. Environment
# 
# 
# 

# In[ ]:


SpeechinessMusic = dataset[['instrumentalness','speechiness']]
Energetic = dataset[['danceability','energy']]
MusicAttribute = dataset[['tempo','mode','key','time_signature']]
Environment = dataset[['acousticness','liveness','loudness']]

print(SpeechinessMusic.head(2))
print(Energetic.head(2))
print(MusicAttribute.head(2))
print(Environment.head(2))


# We use the **describe** function to define the summary of the dataset.<br/>

# ### **1. Numerical Features**

# In[ ]:


dataset.describe()


# #### Assumption: 
# 
# 1. According to the SpeechinessMusic group description, There is only 13% and 9 % for the average mean among the datasets. <br/>
#    It seems like it dont so contribute so much for the modeling. Therefore, we will consider to drop this group of features.<br/><br/>
# 2. Energetic group has the different result with the SpeechinessMusic. Both features have more than half average mean values. Which are 61 % for danceability & 68% for energy features. So that we decided these 2 features into the modeling. <br/> <br/>
# 3. We can see that the average mean of tempo is 120, which means that most of the music are in allegro speed music. After that, key feature indicate that the overall key of the music. Most of the music are in concert F major, 5 based on the standard pitch notation.<br/><br/>
# 4. Mode, this feature indicates that whether the music is in major key or minor key. Major Key means that the music based on major scale.<br/>According to the description result, there are nearly 61% that the music is on major key.<br/><br/>
# 5. Time_siganture, the features used to indicate how many beats in a measure. We can see that the average mean of the feature is 3.968.It's almost near to 4. <br/>We are able to know half of the music are in 4/4 time signature.<br/><br/>
# 6. The environment factor is the important to adjust the user preference in the music.<br/>Morever,acousticness and loudness both feature have the not so good performance. Only have 17% and 19% of the average mean value. But there's 25% music which have the 92% that is in live recording,<br/> <br/>
# 7. According to the documentation user provide, loudness is the feature for the quality of the music. Range from -60db to 0. It seems like the distrubution looks equally.<br/><br/>
# 

# ### **2. Categorical Features**

# In[ ]:


dataset.describe(include = 'O')


# #### Assumption: 
# 1. There are only 1956 unique value for song title feature, which means there's possible null/duplicate song title in the dataset.
# 2. Same as song title feature, there are only 1343 unique artist, possible that there are repeated artist in the dataset.
# 3. River has appeared 3 times as song title.
# 4. From the description, we know that Drake is the most favor artist of the user.There is total 16 music in the dataset.

# We need to know whether the null values / duplicate values in the dataset.

# **Null Values**

# In[ ]:


#Check the null value for the string variable
print('song_title:' ,dataset['song_title'].isnull().sum())
print('artist:' ,dataset['artist'].isnull().sum())


# Good to know that there is none of null values exist in the dataset.

# **Duplicated values**

# In[ ]:


#Check how many of duplicate values in song_title & artist features

def DuplicatedFunction(data,column):
    result = data[column].duplicated().sum()
    return result

print('Duplicate Values:' ,DuplicatedFunction(dataset,'song_title'))
print('Duplicate Values:' ,DuplicatedFunction(dataset,'artist'))


# 1. We can know that there are 61 song title is duplicated in the dataset.<br/>
# 2. Artist feature is different with song_title. Although the duplicated values is huge, but we can assumed that 1 artist may have different music in the dataset.<br/><br/>
# 
# #### **We are not going to check the duplicated values from the numeric variable. Because every value of the music attribute might be same as other music as well.**

# # Analyse the features by visualization

# At first, we start to analyse the MusicAttribute group. Which are **mode**, **key** and **time_signature**.<br/>
# We are not going to analyse the tempo at this moment. Due to tempo is the numeric feature, and other features are ordinal/binary.

# #### 1. Mode

# In[ ]:


print(dataset[['mode','target']].groupby(['mode']).mean().sort_values(by = 'target', ascending = False))


# In[ ]:


sns.factorplot('mode','target', data = dataset)
plt.show()


# #### 2. Key

# In[ ]:


dataset[['key','target']].groupby('key').mean().sort_values(by = 'target', ascending = False)


# In[ ]:


Explode = [.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1]

f, ax = plt.subplots(figsize = (7,10))
dataset[['key','target']].groupby('key').mean().plot.pie(subplots = True, explode = Explode, autopct = '%.2f%%',ax = ax)
plt.legend(loc = 'lower left') 
plt.show()


# In[ ]:


sns.factorplot('key','target', data = dataset)
plt.show()


# #### 3. Time Signature

# In[ ]:


dataset[['time_signature','target']].groupby(['time_signature']).mean().sort_values(by = 'target', ascending = False)


# In[ ]:


sns.factorplot('time_signature','target', data = dataset)
plt.show()


# ## Analyse the numeric features by visualization

# #### 1. instrumentalness  2. speechiness

# In[ ]:


f, ax = plt.subplots(2,2, figsize = (10,10))
dataset[dataset['target'] == 0].instrumentalness.plot.hist(bins = 10, ax = ax[0,0])
ax[0,0].set_title('target = 0 | instrumentalness')
dataset[dataset['target'] == 1].instrumentalness.plot.hist(bins = 10, ax = ax[0,1])
ax[0,1].set_title('target = 1 | instrumentalness')
dataset[dataset['target'] == 0].speechiness.plot.hist(bins = 10, ax = ax[1,0])
ax[1,0].set_title('target = 0 | speechiness')
dataset[dataset['target'] == 1].speechiness.plot.hist(bins = 10, ax = ax[1,1])
ax[1,1].set_title('target = 1 | speechiness')
plt.show()


# #### 3. danceability 4. energy

# In[ ]:


f, ax = plt.subplots(2,2,figsize = (10,10))
dataset[dataset['target'] == 0].danceability.plot.hist(bins = 10, ax = ax[0,0])
ax[0,0].set_title('target = 0 | danceability')
dataset[dataset['target'] == 1].danceability.plot.hist(bins = 10, ax = ax[0,1])
ax[0,1].set_title('target = 1 | danceability')
dataset[dataset['target'] == 0].energy.plot.hist(bins = 10, ax = ax[1,0])
ax[1,0].set_title('target = 0 | energy')
dataset[dataset['target'] == 1].energy.plot.hist(bins = 10, ax = ax[1,1])
ax[1,1].set_title('target = 1 | energy')

plt.show()


# #### 5. tempo

# In[ ]:


f,ax = plt.subplots(1,2,figsize = (10,5))
dataset[dataset['target'] == 0].tempo.plot.hist(bins = 10, ax = ax[0])
ax[0].set_title('target = 0 | tempo')
dataset[dataset['target'] == 1].tempo.plot.hist(bins = 10, ax = ax[1])
ax[1].set_title('target = 1 | tempo')

plt.show()


# #### 6. acousticness 7. liveness

# In[ ]:


f,ax = plt.subplots(2,2,figsize = (10,10))
dataset[dataset['target'] == 0].acousticness.plot.hist(bins = 10, ax = ax[0,0])
ax[0,0].set_title('target = 0 | acousticness')
dataset[dataset['target'] == 1].acousticness.plot.hist(bins = 10, ax = ax[0,1])
ax[0,1].set_title('target = 1 | acousticness')
dataset[dataset['target'] == 0].liveness.plot.hist(bins = 10, ax = ax[1,0])
ax[1,0].set_title('target = 0 | liveness')
dataset[dataset['target'] == 1].liveness.plot.hist(bins = 10, ax = ax[1,1])
ax[1,1].set_title('target = 1 | liveness')

plt.show()


# #### 8. loudness

# In[ ]:


f,ax = plt.subplots(1,2,figsize = (10,5))
dataset[dataset['target'] == 0].loudness.plot.hist(bins = 10, ax = ax[0])
ax[0].set_title('target = 0 | loudness')
dataset[dataset['target'] == 1].loudness.plot.hist(bins = 10, ax = ax[1])
ax[1].set_title('target = 1 | loudness')

plt.show()


# At the last, we will create a heat map to find out the correalation between the features & class labels.

# In[ ]:


f,ax = plt.subplots(figsize = (10,10)) #the size of the heat map
sns.heatmap(dataset.corr(), annot = True, fmt = '.2g', cmap = 'RdYlGn', ax= ax) #annot: values, fmt: decimal points of values
sns.set(font_scale = 0.75) #the font size of the value in the heat map
plt.xlabel('Features')
plt.show()


# # Feature Engineering

# There are few steps we need to do in this part. First of all, we will drop the features which is not contribute so much for the modeling. <br/>
# Secondly, most of the features are in numerical type, and the predective model only accept the binary data type. <br/>
# Therefore, we may use the binning method to create the new feature which is binary/ordinal data type. After that, we will replace the numerical features to binary features.<br/>

# ####  * **Drop the features**
# 
# We decided drop the song_tile,artist & duration_ms features from the dataset, As we thought that these features doesn't contribute so much for the modeling.<br/>

# In[ ]:


print('The Dimension of the dataset before drop the features:', dataset.shape)
dataset = dataset.drop(['song_title','artist','duration_ms'], axis = 1)
print('The Dimension of the dataset after drop the features:', dataset.shape)


# #### * **Create the new features by binning method**
# 
# At first, we start to create the **range features** based on the group we created before. <br/><br/>
# 
# * SpeechinessMusic: instrumentalness & speechiness<br/>
# * Energetic: danceability & energy<br/>
# * MusicAttribute: tempo, mode, key, time_signature<br/>
# * Environment: acousticness, liveness, loudness<br/>
# * Valence
# 
# It used to define the range of the value of the features.<br/>
# After that, we are binning the data to the new feature based on the range features.<br/>
# P/S. We are not going to conver the **mode**, **key**,**time_siganature** and **target** to ordinal features. Because there are already the binary/ordinal features.<br/>

# In[ ]:


#1. instrumentalness
dataset['InstrumentalnessBand'] = pd.cut(dataset['instrumentalness'],4)
dataset[['InstrumentalnessBand','target']].groupby('InstrumentalnessBand',as_index = False).mean().sort_values(by = 'InstrumentalnessBand', ascending = True)


# In[ ]:


dataset['instrumentalness2'] = 0
dataset.loc[dataset['instrumentalness'] <= 0.244,'instrumentalness2'] = 0
dataset.loc[(dataset['instrumentalness'] > 0.244) & (dataset['instrumentalness'] <= 0.488), 'instrumentalness2'] = 1
dataset.loc[(dataset['instrumentalness'] > 0.488) & (dataset['instrumentalness'] <= 0.732), 'instrumentalness2'] = 2
dataset.loc[dataset['instrumentalness'] > 0.732, 'instrumentalness2'] = 3


# In[ ]:


#2. speechiness
dataset['SpeechinessBand'] = pd.cut(dataset['speechiness'],4)
dataset[['SpeechinessBand','target']].groupby('SpeechinessBand',as_index = False).mean().sort_values(by = 'SpeechinessBand', ascending = True)


# In[ ]:


dataset['speechiness2'] = 0
dataset.loc[dataset['speechiness'] <= 0.221, 'speechiness2'] = 0
dataset.loc[(dataset['speechiness'] > 0.221) & (dataset['speechiness'] <= 0.42), 'speechiness2'] = 1
dataset.loc[(dataset['speechiness'] > 0.42) & (dataset['speechiness'] <= 0.618), 'speechiness2'] = 2
dataset.loc[dataset['speechiness'] > 0.618, 'speechiness2'] = 3


# In[ ]:


#3. danceability
dataset['DanceabilityBand'] = pd.cut(dataset['danceability'],4)
dataset[['DanceabilityBand','target']].groupby('DanceabilityBand',as_index = False).mean().sort_values(by = 'DanceabilityBand', ascending = True)


# In[ ]:


dataset['danceability2'] = 0
dataset.loc[dataset['danceability'] <= 0.338, 'danceability2'] = 0
dataset.loc[(dataset['danceability'] > 0.338) & (dataset['danceability'] <= 0.553), 'danceability2'] = 1
dataset.loc[(dataset['danceability'] > 0.553) & (dataset['danceability'] <= 0.769), 'danceability2'] = 2
dataset.loc[dataset['danceability'] > 0.769, 'danceability2'] = 3


# In[ ]:


#4. energy
dataset['EnergyBand'] = pd.cut(dataset['energy'],4)
dataset[['EnergyBand','target']].groupby('EnergyBand',as_index = False).mean().sort_values(by = 'EnergyBand', ascending = True)


# In[ ]:


dataset['energy2'] = 0
dataset.loc[dataset['energy'] <= 0.261, 'energy2'] = 0
dataset.loc[(dataset['energy'] > 0.261) & (dataset['energy'] <= 0.506), 'energy2'] = 1
dataset.loc[(dataset['energy'] > 0.506) & (dataset['energy'] <= 0.752), 'energy2'] = 2
dataset.loc[dataset['energy'] > 0.752, 'energy2'] = 3


# In[ ]:


#5. acousticness
dataset['AcousticnessBand'] = pd.cut(dataset['acousticness'],4)
dataset[['AcousticnessBand','target']].groupby('AcousticnessBand',as_index = False).mean().sort_values(by = 'AcousticnessBand', ascending = True)


# In[ ]:


dataset['acousticness2'] = 0
dataset.loc[dataset['acousticness'] <= 0.249, 'acousticness2'] = 0
dataset.loc[(dataset['acousticness'] > 0.249) & (dataset['acousticness'] <= 0.498), 'acousticness2'] = 1
dataset.loc[(dataset['acousticness'] > 0.498) & (dataset['acousticness'] <= 0.746), 'acousticness2'] = 2
dataset.loc[dataset['acousticness'] > 0.746, 'acousticness2'] = 3


# In[ ]:


#6. liveness
dataset['LivenessBand'] = pd.cut(dataset['liveness'],4)
dataset[['LivenessBand','target']].groupby('LivenessBand', as_index = False).mean().sort_values(by = 'LivenessBand', ascending = True)


# In[ ]:


dataset['liveness2'] = 0
dataset.loc[dataset['liveness'] <= 0.256,'liveness2'] = 0
dataset.loc[(dataset['liveness'] > 0.256) & (dataset['liveness'] <= 0.494),'liveness2'] = 1
dataset.loc[(dataset['liveness'] > 0.494) & (dataset['liveness'] <= 0.731),'liveness2'] = 2
dataset.loc[dataset['liveness'] > 0.731, 'liveness2'] = 3


# In[ ]:


#7. loudness
dataset['LoudnessBand'] = pd.cut(dataset['loudness'], 4)
dataset[['LoudnessBand','target']].groupby('LoudnessBand').mean()


# In[ ]:


dataset['loudness2'] = 0
dataset.loc[dataset['loudness'] <= -24.9, 'loudness2'] = 0
dataset.loc[(dataset['loudness'] > -24.9) & (dataset['loudness'] <= -16.702), 'loudness2'] = 1
dataset.loc[(dataset['loudness'] > -16.702) & (dataset['loudness'] <= -8.504), 'loudness2'] = 2
dataset.loc[dataset['loudness'] > -8.504, 'loudness2'] = 3


# In[ ]:


#8. tempo

dataset['TempoBand'] = pd.cut(dataset['tempo'],4)
dataset[['TempoBand','target']].groupby('TempoBand',as_index = False).mean().sort_values(by = 'TempoBand', ascending = True)


# In[ ]:


dataset['tempo2'] = 0
dataset.loc[dataset['tempo'] <= 90.727, 'tempo2'] = 0
dataset.loc[(dataset['tempo'] > 90.727) & (dataset['tempo'] <= 133.595), 'tempo2'] = 1
dataset.loc[(dataset['tempo'] > 133.595) & (dataset['tempo'] <= 176.463), 'tempo2'] = 2
dataset.loc[ dataset['tempo'] > 176.463, 'tempo2'] = 3


# In[ ]:


#9. valence
dataset['valenceband'] = pd.cut(dataset['valence'], 4)
dataset[['valenceband','target']].groupby('valenceband').mean().sort_values(by = 'valenceband')


# In[ ]:


dataset['valence2'] = 0
dataset.loc[dataset['valence'] <= 0.274, 'valence2'] = 0
dataset.loc[(dataset['valence'] > 0.274) & (dataset['valence'] <= 0.513), 'valence2'] = 1
dataset.loc[(dataset['valence']> 0.513) & (dataset['valence'] <= 0.753), 'valence2'] = 2
dataset.loc[dataset['valence'] > 0.753, 'valence2'] = 3


# In[ ]:


dataset.head()


# In[ ]:


#Drop the range features
dataset = dataset.drop(['InstrumentalnessBand','SpeechinessBand','DanceabilityBand','EnergyBand',
                        'AcousticnessBand','LivenessBand','LoudnessBand','TempoBand','valenceband'], axis = 1)

dataset.columns


# In[ ]:


#Drop all the numerical features without process through binning method

dataset = dataset.drop(['acousticness','danceability','energy','instrumentalness',
                        'liveness','loudness','speechiness','tempo','valence'],axis = 1)

dataset.columns


# In[ ]:


#Rename the binning features
dataset = dataset.rename(columns = {'instrumentalness2':'instrumentalness','speechiness2': 'speechiness', 'danceability2': 'danceability',
                                   'energy2': 'energy','acousticness2':'acousticness', 'liveness2':'liveness', 'loudness2':'loudness',
                                   'tempo2': 'tempo', 'valence2': 'valence'})
dataset.columns


# In[ ]:


#Change the time_signature features from numerical type features to Int type features

dataset['time_signature'] = dataset['time_signature'].astype(int)
dataset.head()


# In[ ]:


dataset.describe()


# In[ ]:


#drop the features which doesn't have the good result in average mean
dataset = dataset.drop(['instrumentalness','speechiness','acousticness','liveness'], axis = 1)
print('The dimension of the dataset after drop the features: ', dataset.shape)


# #### Create the Dummy Variables from the features, except the target class labels.

# In[ ]:


df_key = pd.get_dummies(dataset['key'])
df_time_signature = pd.get_dummies(dataset['time_signature'])
df_danceability = pd.get_dummies(dataset['danceability'])
df_energy = pd.get_dummies(dataset['energy'])
df_loudness = pd.get_dummies(dataset['loudness'])
df_tempo = pd.get_dummies(dataset['tempo'])
df_valence = pd.get_dummies(dataset['valence'])

dummy_variables = pd.concat([df_key,df_time_signature,df_danceability,df_energy,df_loudness,df_tempo,df_valence], axis = 1)
dataset = pd.concat([dataset,dummy_variables], axis = 1)
print('The dimension of the dataset after create the dummy variables: ', dataset.shape)


# In[ ]:


#Replace the numerical features by dummy variables, but the target class labels
dataset = dataset.drop(['key','time_signature','danceability','energy','loudness','tempo','valence'], axis = 1)
print('The dimension of the dataset after drop the numerical features: ', dataset.shape)


# # Model, predict & evaluation

# In[ ]:


#Import the library we need to use for the following step

from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


# In[ ]:


#1. Create the X_train without the target class label & Y_train (target)
X_train = dataset.drop('target', axis = 1)
Y_train = dataset['target']

#2. Split the X_Train & Y_train into training set & testing set by train_test_split function
x_train,x_test,y_train,y_test = train_test_split(X_train,Y_train,test_size = 0.2, random_state = 0)


# As we can see that, we extract 404 observations (20%) from the dataset to create the testing set.

# In[ ]:


print('the dimension of the x_train: ', x_train.shape)
print('the dimension of the x_test: ', x_test.shape)


# In[ ]:


#3. Fit the model into the training set

#i. Logistic Regression
log = LogisticRegression()
log.fit(x_train,y_train)
log_y_pred = log.predict(x_test)
log_result_train = round(log.score(x_train,y_train)*100,2)

#ii. Gaussian Naive Bayes
NB = GaussianNB()
NB.fit(x_train,y_train)
NB_y_pred = NB.predict(x_test)
NB_result_train = round(NB.score(x_train,y_train)*100,2)

#iii. Decision Tree
DT = DecisionTreeClassifier()
DT.fit(x_train,y_train)
DT_y_pred = DT.predict(x_test)
DT_result_train = round(DT.score(x_train,y_train)*100, 2)

#iv. K-Nearest Neighbors (K-NN)
KNN = KNeighborsClassifier()
KNN.fit(x_train,y_train)
KNN_y_pred = KNN.predict(x_test)
KNN_result_train = round(KNN.score(x_train,y_train)*100,2)

print('1. Logistic Regression: ', log_result_train)
print('2. Gaussian Naive Bayes: ', NB_result_train)
print('3. Decision Tree Classifier: ', DT_result_train)
print('4. K-NN: ', KNN_result_train)


# In[ ]:


#4. Fit the model into the testing dataset


#i. Logistic Regression
log_result_test = round(log.score(x_test,y_test)*100,2)

#ii. Gaussian Naive Bayes
NB_result_test = round(NB.score(x_test,y_test)*100,2)

#iii. Decision Tree
DT_result_test = round(DT.score(x_test,y_test)*100,2)

#iv. K-Nearest Neighbors
KNN_result_test = round(KNN.score(x_test,y_test)*100,2)

print('1. Logistic Regression: {}'.format(log_result_test))
print('2. Gaussian Naive Bayes: {}'.format(NB_result_test))
print('3. Decision Tree: {}'.format(DT_result_test))
print('4. K-NN: {}'.format(KNN_result_test))


# **Result:**
# 
# According to the performance of model from training dataset, we know that the performance are looks great. Especially the Decision Tree and K-NN.<br/>
# But we suprised that the performance of model from testing dataset have the big contrast with training dataset.<br/>

# In this stage, we will apply the K-fold cross validation method to evaluate the performance of model. K-fold cross validation splits the dataset into **K** random number of subsets to let the model fit into. We set the K number : 38 as we have the 38 features in the both dataset.

# In[ ]:


#5. Apply K-fold Cross Validation method into the model (testing data)

#1. Logistic Regression
Kfold = KFold(n_splits = 10)
logregScore = cross_val_score(log,x_test,y_test, cv = Kfold)
avglogregScore = np.mean(logregScore)

#2. Gaussien Naive Bayes
NBScore = cross_val_score(NB,x_test,y_test, cv = Kfold)
avgNBScore = np.mean(NBScore)

#3. Decision Tree Classifier
DTScore = cross_val_score(DT, x_test,y_test, cv =Kfold)
avgDTScore = np.mean(DTScore)

#4. K-NN
KNNScore = cross_val_score(KNN, x_test,y_test, cv = Kfold)
avgKNNScore = np.mean(KNNScore)
#for i in range(len(logregScore)): 
  #  print(i+1, 'Logistic Regression:',logregScore[i])
    
print('1. Logistic Regression: ', round(avglogregScore*100,2))
print('2. Gaussian Naive Bayes:  ', round(avgNBScore*100,2))
print('3. Decision Tree Classifier: ', round(avgDTScore*100,2))
print('4. K-NN: ', round(avgKNNScore*100,2))


# **Result:**
# <br/> We can see that the performance of model have slightly increased. Logistic Regression & KNN model have the bigger contrast performance with the previous result. It proves that K-fold cross validation method is useful to enhance the performance of model.

# In[ ]:


#6. Create the confusion matrix table for the performance of model

f, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize = (12,3))
#1. Logistic Regression
LogregCM = confusion_matrix(y_test,log_y_pred)
sns.heatmap(LogregCM, annot = True, fmt = 'd', vmin = 0, vmax = 150,cmap = 'viridis', ax = ax1)
#Annot: the value of the heatmap
#fmt: the decimal point of value of heatmap
#vmin, vmax: the limits of the colorbar
ax1.set_title('Logistic Regression')
ax1.set_xlabel('Features')

#2. Gaussian Naive Bayes
NBCM = confusion_matrix(y_test,NB_y_pred)
sns.heatmap(NBCM, annot = True, fmt = 'd', vmin = 0, vmax = 150, cmap = 'YlGnBu', ax = ax2)
ax2.set_title('Gaussian Naive Bayes')
ax2.set_xlabel('Features')


#3. Decision Tree
DTCM = confusion_matrix(y_test, DT_y_pred)
sns.heatmap(DTCM, annot = True, fmt = 'd', vmin = 0, vmax = 150, cmap = 'viridis', ax = ax3)
ax3.set_title('Decision Tree')
ax3.set_xlabel('Features')


#4. KNN
KNNCM = confusion_matrix(y_test, KNN_y_pred)
sns.heatmap(KNNCM, annot = True, fmt = 'd', vmin = 0 , vmax = 150, cmap = 'YlGnBu', ax = ax4)
ax4.set_title('KNN')
ax4.set_xlabel('Features')


plt.show()


# **Observation:**
# 
# Confusion Matrix table is the table that we used to take a look the numbers of feature predicted correctly by model.<br/>
# In this time, we apply the confusion matrix table into the heat map graph. It's easier to read the result.<br/>
# There's 404 observations in the testing dataset. We can know that **Logistic Regression** is the best performance of predective model.<br/>
# It predict 241 observations corretcly from TP columns & FP columns.<br/>
# 
# 

# # To be Continued...
# 
# commit v24: Model, predict & evaluation
# 
# 1. Import the Classification_report class from the sklearn.metrics to evaluate the performance of model.
