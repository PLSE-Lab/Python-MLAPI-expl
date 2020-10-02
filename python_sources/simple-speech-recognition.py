#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing all the dependencies
import pandas as pd # data frame
import numpy as np # matrix math
from glob import glob # file handling
import librosa # audio manipulation
from sklearn.utils import shuffle # shuffling of data
import os # interation with the OS
from random import sample # random selection
from tqdm import tqdm


# In[3]:


# fixed param
PATH = '../input/train/audio/'


# In[4]:


def load_files(path):
	# write the complete file loading function here, this will return
	# a dataframe having files and labels
	# loading the files
	train_labels = os.listdir(PATH)
	train_labels.remove('_background_noise_')

	labels_to_keep = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']

	train_file_labels = dict()
	for label in train_labels:
		files = os.listdir(PATH + '/' + label)
		for f in files:
			train_file_labels[label + '/' + f] = label

	train = pd.DataFrame.from_dict(train_file_labels, orient='index')
	train = train.reset_index(drop=False)
	train = train.rename(columns={'index': 'file', 0: 'folder'})
	train = train[['folder', 'file']]
	train = train.sort_values('file')
	train = train.reset_index(drop=True)

	def remove_label_from_file(label, fname):
		return path + label + '/' + fname[len(label)+1:]

	train['file'] = train.apply(lambda x: remove_label_from_file(*x), axis=1)
	train['label'] = train['folder'].apply(lambda x: x if x in labels_to_keep else 'unknown')

	labels_to_keep.append('unknown')

	return train, labels_to_keep


# In[16]:


train, labels_to_keep = load_files(PATH)

# making word2id dict
word2id = dict((c,i) for i,c in enumerate(sorted(labels_to_keep)))

# get some files which will be labeled as unknown
unk_files = train.loc[train['label'] == 'unknown']['file'].values
unk_files = sample(list(unk_files), 1000)


# In[17]:


word2id


# In[6]:


unk_files[:10]


# In[7]:


train.sample(5)


# In[13]:


# Writing functions to extract the data, script from kdnuggets: 
# www.kdnuggets.com/2016/09/urban-sound-classification-neural-networks-tensorflow.html
def extract_feature(path):
	X, sample_rate = librosa.load(path)
	stft = np.abs(librosa.stft(X))
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
	return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(files, word2id, unk = False):
    # n: number of classes
    features = np.empty((0,193))
    one_hot = np.zeros(shape = (len(files), word2id[max(word2id)]))
    print(one_hot.shape)
    for i in tqdm(range(len(files))):
        f = files[i]
        mfccs, chroma, mel, contrast,tonnetz = extract_feature(f)
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
        if unk == True:
            l = word2id['unknown']
            one_hot[i][l] = 1.
        else:
            l = word2id[f.split('/')[-2]]
            one_hot[i][l] = 1.
    return np.array(features), one_hot


# In[10]:


files = train.loc[train['label'] != 'unknown']['file'].values
print(len(files))
print(files[:10])


# ## Playing around with the single audio clip
# We now look at a single audio clip and see how it goes.

# In[ ]:


# playing around with the data for now
train_audio_path = '../input/train/audio/'
filename = '/tree/24ed94ab_nohash_0.wav' # --> 'Yes'
sample_rate, audio = wavfile.read(str(train_audio_path) + filename)


# In[ ]:


plt.figure(figsize = (15, 4))
plt.plot(audio)
ipd.Audio(audio, rate=sample_rate)


# In[ ]:


# goto: https://medium.com/@ageitgey/machine-learning-is-fun-part-6-how-to-do-speech-recognition-with-deep-learning-28293c162f7a
# We convert it into chunks of 20ms each i.e. units of 320 
audio_chunks = []
n_chunks = int(audio.shape[0]/320)
for i in range(n_chunks):
    chunk = audio[i*320: (i+1)*320]
    audio_chunks.append(chunk)
audio_chunk = np.array(audio_chunks)


# In[ ]:


# we now convert it to spertogram
# goto: https://www.kaggle.com/davids1992/data-visualization-and-investigation
def log_specgram(audio, sample_rate, window_size=10,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    _, _, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return np.log(spec.T.astype(np.float32) + eps)


# In[ ]:


spectrogram = log_specgram(audio, sample_rate, 10, 0)
spec = spectrogram.T
print(spec.shape)
plt.figure(figsize = (15,4))
plt.imshow(spec, aspect='auto', origin='lower')


# ## Making the data
# Now that we know about the shape of the data, we will finally make the total processed data.

# In[ ]:


# make labels and convert them into one hot encodings
labels = sorted(labels_to_keep)
word2id = dict((c,i) for i,c in enumerate(labels))
label = train['label'].values
label = [word2id[l] for l in label]
print(labels)
def make_one_hot(seq, n):
    # n --> vocab size
    seq_new = np.zeros(shape = (len(seq), n))
    for i,s in enumerate(seq):
        seq_new[i][s] = 1.
    return seq_new
one_hot_l = make_one_hot(label, 12)


# In[ ]:


print(one_hot_l[10:15])


# In[ ]:


one_hot_l[0]


# In[ ]:


# getting all the paths to the files
paths = []
folders = train['folder']
files = train['file']
for i in range(len(files)):
    path = '../input/train/audio/' + str(folders[i]) + '/' + str(files[i])
    paths.append(path)


# In[ ]:


def audio_to_data(path):
    # we take a single path and convert it into data
    sample_rate, audio = wavfile.read(path)
    spectrogram = log_specgram(audio, sample_rate, 10, 0)
    return spectrogram.T

def paths_to_data(paths,labels):
    data = np.zeros(shape = (len(paths), 81, 100))
    indexes = []
    for i in tqdm(range(len(paths))):
        audio = audio_to_data(paths[i])
        if audio.shape != (81,100):
            indexes.append(i)
        else:
            data[i] = audio
    final_labels = [l for i,l in enumerate(labels) if i not in indexes]
    print('Number of instances with inconsistent shape:', len(indexes))
    return data[:len(data)-len(indexes)], final_labels, indexes


# In[ ]:


d,l,indexes = paths_to_data(paths,one_hot_l)


# In[ ]:


labels = np.zeros(shape = [d.shape[0], len(l[0])])
for i,array in enumerate(l):
    for j, element in enumerate(array):
        labels[i][j] = element
print(labels.shape)


# In[ ]:


print(d.shape)
print(labels.shape)


# In[ ]:


d,labels = shuffle(d,labels)


# In[ ]:


print(d[0].shape)
print(labels[0].shape)


# ## Machine learning model
# Using a LSTM network to determine the text

# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


# In[ ]:


model = Sequential()
model.add(LSTM(256, input_shape = (81, 100)))
# model.add(Dense(1028))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(12, activation = 'softmax'))
model.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(d, labels, batch_size = 1024, epochs = 10)


# ## Add further for testing modules
# Add modules for testing and saving the files, will keeo improving the model in the future.

# In[ ]:




