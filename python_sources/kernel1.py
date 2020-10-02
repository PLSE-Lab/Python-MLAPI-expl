#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np  
import pandas as pd 
import numpy as np 
import pandas as pd 
import os
import glob
import pickle
from sklearn.model_selection import train_test_split 
import librosa as lbr
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import os
import librosa.display
print(os.listdir("../input"))


# In[ ]:


INPUT_FOLDER = "../input/"
train_files = glob.glob("../input/train_curated/*.wav")
print(os.listdir(INPUT_FOLDER))


# In[ ]:



TRAIN_CURATED_PATH = INPUT_FOLDER + "train_curated.csv"
TRAIN_NOISY_PATH = INPUT_FOLDER + "train_noisy.csv"
SAMPLE_SUBMISSION_PATH = INPUT_FOLDER + "sample_submission.csv"
TRAIN_CURATED = INPUT_FOLDER + "train_curated/"
TRAIN_NOISY = INPUT_FOLDER + "train_noisy/"
TEST = INPUT_FOLDER + "test/"

train_curated = pd.read_csv(TRAIN_CURATED_PATH)
train_noisy = pd.read_csv(TRAIN_NOISY_PATH)
test = pd.read_csv(SAMPLE_SUBMISSION_PATH)


# In[ ]:





# ### Exploratory Data Analysis

# In[ ]:


print("Number of train examples=", train_curated.shape[0], "  Number of classes=", len(set(train_curated.labels)))
print("Number of test examples=", test.shape[0], "  Number of classes=", len(set(test.columns[1:])))


# Notice that the number of classes in training curated is much larger than in testing file

# In[ ]:


#load both train_curated and noisy
# train_curated['is_curated'] = True
# train_noisy = pd.read_csv('../input/train_noisy.csv')
# train_noisy['is_curated'] = False
#train = pd.concat([train_curated, train_noisy], axis=0)
#del train_noisy


# In[ ]:


#get only the lables that are in the testing file
#train is for one lable per class data, train_curated is the multilabel dataset
train = train_curated[train_curated.labels.isin(test.columns[1:])]
print(len(train))
category_group = train.groupby(['labels']).count()['fname']
category_group.columns = ['counts']


# In[ ]:


plot = category_group.sort_values(ascending=True).plot(
    kind='barh', 
    title="Number of Audio Samples per Category", 
    figsize=(20,20))
plot.set_xlabel("Category")
plot.set_ylabel("Number of Samples");


# In[ ]:


print('Minimum samples per category = ', min(train.labels.value_counts()))
print('Maximum samples per category = ', max(train.labels.value_counts()))


# In[ ]:


# Using wave library
import wave
fname = '../input/train_curated/0164cba5.wav'   # Raindrop
wav = wave.open(fname)
print("Sampling (frame) rate = ", wav.getframerate())
print("Total samples (frames) = ", wav.getnframes())
print("Duration = ", wav.getnframes()/wav.getframerate())


# In[ ]:


def one_hot(labels, src_dict):
    ar = np.zeros([len(labels), len(src_dict)])
    invalid=['77b925c2.wav','f76181c4.wav', '6a1f682a.wav', 'c7db12aa.wav', '7752cc8a.wav','1d44b0bd.wav']
    for i, label in enumerate(labels): 
        if label not in invalid:
            label_list = label.split(',')
            for la in label_list:
                ar[i, src_dict[la]] = 1
    return ar


# In[ ]:


#chacking the multilables
[label.split(',') for i, label in enumerate(train['labels']) if len(label.split(',')) >=2] 


# In[ ]:


#get target names from test 
target_names = test.columns[1:]
target_names.shape


# In[ ]:


num_targets = len(target_names)

src_dict = {target_names[i]:i for i in range(num_targets)}
src_dict_inv = {i:target_names[i] for i in range(num_targets)}


# Analyze the lengths of the audio files in our dataset

# In[ ]:


# train['nframes'] = train['fname'].apply(lambda f: wave.open('../input/train_curated/' + f).getnframes())
# test1 = pd.read_csv(SAMPLE_SUBMISSION_PATH)
# test1['nframes'] = test1['fname'].apply(lambda f: wave.open('../input/test/' + f).getnframes())


# In[ ]:


print('maximum track duration: ', train['nframes'].max()/44100)
print('minimum track duration: ', train['nframes'].min()/44100)


# In[ ]:


#distribution of top 25 categories
import seaborn as sns
idx_sel = category_group.sort_values(ascending=True).index[-25:]
_, ax = plt.subplots(figsize=(20, 4))
sns.violinplot(ax=ax, x="labels", y="nframes", data=train[(train.labels.isin(idx_sel).values)])
plt.xticks(rotation=90)
plt.title('Distribution of audio frames, per label', fontsize=16)
plt.show()


# Analyze the frame length distribution in train and test

# In[ ]:


fig, ax = plt.subplots(2, 1, figsize=(16,8))
train.nframes.hist(bins=100, grid=True, rwidth=0.5, ax=ax[0], color='deeppink')
test.nframes.hist(bins=100, grid=True, rwidth=0.5, ax=ax[1], color='darkslateblue')
ax[0].set_xlim(0, 2700000)
ax[1].set_xlim(0, 2700000)
plt.suptitle('Frame Length Distribution in train and test', ha='center', fontsize='large');


# ### Note:
# Majority of the audio files are short.

# In[ ]:


train.query("nframes > 2500000")


# ### Listen to the audio

# In[ ]:


import IPython.display as ipd  # To play sound in the notebook
fname = '../input/train_curated/77b925c2.wav'   # Abnormal
ipd.Audio(fname)


# In[ ]:


# get all the files with label "Stream" and compare to abnormal
train[train["labels"]=="Stream"]


# In[ ]:


# one hot encoding
y_proc_tmp = one_hot(train_curated['labels'], src_dict)


# ### Spectrogram Visualization

# In[ ]:


song='../input/train_curated/ca5c5f2c.wav'
audio, sample_rate=lbr.load(song,sr=44100)


# In[ ]:


n_fft = int(0.025 * sample_rate) #25ms window length
hop_length =  n_fft//2
N_MELS = 128 #frequency bins
#X = lbr.stft(audio[0], n_fft=n_fft, hop_length=hop_length)
X=lbr.feature.melspectrogram(audio,n_fft=n_fft, hop_length=hop_length,n_mels=N_MELS )
S = lbr.amplitude_to_db(abs(X))
#S=np.log(X)
plt.figure(figsize=(15, 5))
lbr.display.specshow(S, sr=44100, hop_length=hop_length, x_axis='time',cmap='magma')
plt.colorbar(format='%+2.0f dB')


# In[ ]:


signal = audio[0:int(1 * sample_rate)] #5 secs of audio
plt.plot(signal)


# ### Load data and extract spectrogram features

# In[ ]:


WINDOW_SIZE = int(0.025 * sample_rate) #window size 25ms             
WINDOW_STRIDE = WINDOW_SIZE // 2 #overlap 50%
N_MELS = 128  #frequencies

MEL_KWARGS = {
    'n_fft': WINDOW_SIZE,
    'hop_length': WINDOW_STRIDE,
    'n_mels': N_MELS 
}

def load_track(filename, enforce_shape=None):
    new_input, sample_rate = lbr.load(filename, mono=True,sr= 44100)
    features = lbr.feature.melspectrogram(new_input, **MEL_KWARGS).T
    if enforce_shape is not None:
        if features.shape[0] < enforce_shape[0]:
            delta_shape = (enforce_shape[0] - features.shape[0],
                    enforce_shape[1])
            features = np.append(features, np.zeros(delta_shape), axis=0)
        elif features.shape[0] > enforce_shape[0]:
            features = features[: enforce_shape[0], :]

    features[features == 0] = 1e-6
    return (features, float(new_input.shape[0]) / sample_rate)


# ### Building a model using MFCC

# In[ ]:


from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Activation,          Convolution1D, MaxPooling1D, BatchNormalization, Flatten,GlobalAveragePooling1D
import scipy
from keras import losses
from keras import backend as K
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.utils import Sequence
import shutil


# In[ ]:


class Config(object):
    def __init__(self,
                 sampling_rate=44100, audio_duration=2, #audio duration: specify length of the track in sec
                 n_classes=target_names,
                 use_mfcc=True, n_folds=1, learning_rate=0.0001, 
                 max_epochs=30, n_mfcc=64):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.win_len= int(0.025 * sample_rate) #10ms window length

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/self.win_len)*2))
        else:
            self.dim = (self.audio_length)


# In[ ]:


#lwrap implementation for keras
def tf_one_sample_positive_class_precisions(y_true, y_pred) :
    num_samples, num_classes = y_pred.shape
    
    # find true labels
    pos_class_indices = tf.where(y_true > 0) 
    
    # put rank on each element
    retrieved_classes = tf.nn.top_k(y_pred, k=num_classes).indices
    sample_range = tf.zeros(shape=tf.shape(tf.transpose(y_pred)), dtype=tf.int32)
    sample_range = tf.add(sample_range, tf.range(tf.shape(y_pred)[0], delta=1))
    sample_range = tf.transpose(sample_range)
    sample_range = tf.reshape(sample_range, (-1,num_classes*tf.shape(y_pred)[0]))
    retrieved_classes = tf.reshape(retrieved_classes, (-1,num_classes*tf.shape(y_pred)[0]))
    retrieved_class_map = tf.concat((sample_range, retrieved_classes), axis=0)
    retrieved_class_map = tf.transpose(retrieved_class_map)
    retrieved_class_map = tf.reshape(retrieved_class_map, (tf.shape(y_pred)[0], num_classes, 2))
    
    class_range = tf.zeros(shape=tf.shape(y_pred), dtype=tf.int32)
    class_range = tf.add(class_range, tf.range(num_classes, delta=1))
    
    class_rankings = tf.scatter_nd(retrieved_class_map,
                                          class_range,
                                          tf.shape(y_pred))
    
    #pick_up ranks
    num_correct_until_correct = tf.gather_nd(class_rankings, pos_class_indices)

    # add one for division for "presicion_at_hits"
    num_correct_until_correct_one = tf.add(num_correct_until_correct, 1) 
    num_correct_until_correct_one = tf.cast(num_correct_until_correct_one, tf.float32)
    
    # generate tensor [num_sample, predict_rank], 
    # top-N predicted elements have flag, N is the number of positive for each sample.
    sample_label = pos_class_indices[:, 0]   
    sample_label = tf.reshape(sample_label, (-1, 1))
    sample_label = tf.cast(sample_label, tf.int32)
    
    num_correct_until_correct = tf.reshape(num_correct_until_correct, (-1, 1))
    retrieved_class_true_position = tf.concat((sample_label, 
                                               num_correct_until_correct), axis=1)
    retrieved_pos = tf.ones(shape=tf.shape(retrieved_class_true_position)[0], dtype=tf.int32)
    retrieved_class_true = tf.scatter_nd(retrieved_class_true_position, 
                                         retrieved_pos, 
                                         tf.shape(y_pred))
    # cumulate predict_rank
    retrieved_cumulative_hits = tf.cumsum(retrieved_class_true, axis=1)

    # find positive position
    pos_ret_indices = tf.where(retrieved_class_true > 0)

    # find cumulative hits
    correct_rank = tf.gather_nd(retrieved_cumulative_hits, pos_ret_indices)  
    correct_rank = tf.cast(correct_rank, tf.float32)

    # compute presicion
    precision_at_hits = tf.truediv(correct_rank, num_correct_until_correct_one)
    return pos_class_indices, precision_at_hits

def tf_lwlrap(y_true, y_pred):
    num_samples, num_classes = y_pred.shape
    pos_class_indices, precision_at_hits = (tf_one_sample_positive_class_precisions(y_true, y_pred))
    pos_flgs = tf.cast(y_true > 0, tf.int32)
    labels_per_class = tf.reduce_sum(pos_flgs, axis=0)
    weight_per_class = tf.truediv(tf.cast(labels_per_class, tf.float32),
                                  tf.cast(tf.reduce_sum(labels_per_class), tf.float32))
    sum_precisions_by_classes = tf.zeros(shape=(num_classes), dtype=tf.float32)  
    class_label = pos_class_indices[:,1]
    sum_precisions_by_classes = tf.unsorted_segment_sum(precision_at_hits,
                                                        class_label,
                                                       num_classes)
    labels_per_class = tf.cast(labels_per_class, tf.float32)
    labels_per_class = tf.add(labels_per_class, 1e-7)
    per_class_lwlrap = tf.truediv(sum_precisions_by_classes,
                                  tf.cast(labels_per_class, tf.float32))
    out = tf.cast(tf.tensordot(per_class_lwlrap, weight_per_class, axes=1), dtype=tf.float32)
    return out


# In[ ]:


def audio_norm(data):
#     max_data = np.max(data)
#     min_data = np.min(data)
#     data = (data-min_data)/(max_data-min_data+1e-6)
#     return data - 0.5
    data = ( data - np.mean(data) ) / np.std(data)
   # data /= np.max(data)
    return data


# In[ ]:


config = Config(sampling_rate=44100, audio_duration=3, n_folds=1, learning_rate=0.001, use_mfcc=True, n_mfcc=128,max_epochs=30)


# In[ ]:


def build_model(config):
    
    nclass = len(config.n_classes)
    input_length = config.audio_length
    input_shape = (config.dim[1],config.dim[0])
    print(input_shape)
    rate=0.2
    model_input = Input(input_shape, name='input')
    layer = model_input
    layer = Convolution1D(filters= 32, kernel_size= 5 ,activation=tf.nn.leaky_relu,name='convolution_1' , strides=2,padding='same')(layer)
#     layer = BatchNormalization()(layer) #momentum=0.9
    layer=MaxPooling1D(2, strides=2,padding='same')(layer)
    layer = Dropout(rate)(layer)
    
    layer = Convolution1D(filters= 64, kernel_size= 3 ,activation=tf.nn.leaky_relu,name='convolution_2' , strides=2 ,padding='same')(layer)
#     layer = BatchNormalization()(layer)
    layer=MaxPooling1D(2, strides=2,padding='same')(layer)
    layer = Dropout(rate)(layer)
    
    
    layer = Convolution1D(filters= 128, kernel_size= 3 ,activation=tf.nn.leaky_relu,name='convolution_4', strides=1,padding='same' )(layer)
#     layer = BatchNormalization()(layer)
    layer=MaxPooling1D(2, strides=2,padding='same')(layer)
    layer = Dropout(rate)(layer)
    
    layer = Convolution1D(filters= 256, kernel_size= 3,activation=tf.nn.leaky_relu,name='convolution_5',strides=1,padding='same')(layer)
#     layer = BatchNormalization()(layer)
    layer=MaxPooling1D(2, strides=2,padding='same')(layer)
    layer = Dropout(rate)(layer)
    layer = Flatten()(layer)
    
    layer = Dense(256)(layer)
    layer = Dense(nclass)(layer)
    
    output = Activation('softmax', name='Final_output')(layer)
    model = Model(model_input, output)
   # opt = Adam(lr=config.learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf_lwlrap])
    return model

  


# preparing the data

# In[ ]:


def prepare_data(df, config, data_dir):
    WINDOW_SIZE = int(0.025 * sample_rate) #window size 10ms             
    WINDOW_STRIDE = WINDOW_SIZE // 2 #overlap 50%
    N_MELS = 128  #frequencies

    MEL_KWARGS = {
        'n_fft': WINDOW_SIZE,
        'hop_length': WINDOW_STRIDE,
        'n_mels': N_MELS 
    }
    X = np.empty(shape=(df.shape[0], config.dim[1], config.dim[0]))
    input_length = config.audio_length
   # print(X.shape)
    n_fft = int(0.025 * sample_rate) #10ms window length
    hop_length =  n_fft//2
    N_MELS = 128 #frequency bins
    
 
    invalid=['77b925c2.wav','f76181c4.wav', '6a1f682a.wav', 'c7db12aa.wav', '7752cc8a.wav','1d44b0bd.wav']
    for i, fname in enumerate(df.index):
            
            if fname not in invalid:
                
                file_path = data_dir + fname
                data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")

               # print('data_shape: ',data.shape)
                # Random offset / Padding
                if len(data) > input_length:
                    max_offset = len(data) - input_length
                    offset = np.random.randint(max_offset)
                    data = data[offset:(input_length+offset)]
                else:
                    if input_length > len(data):
                        max_offset = input_length - len(data)
                        offset = np.random.randint(max_offset)
                    else:
                        offset = 0
                    #pad with zeros
                    data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
                #print('before spec: ',data.shape)
               # data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc,**MEL_KWARGS).T
                data = librosa.feature.melspectrogram(data,**MEL_KWARGS).T
                data = lbr.amplitude_to_db(abs(data))
                #print('after padding')
               # print(data.shape)
                X[i,] = data
            else:
                print(fname)
    return X
    


# In[ ]:


train_curated.set_index("fname", inplace=True)
test.set_index('fname',inplace=True)


# In[ ]:


get_ipython().run_line_magic('time', '')
X_train = prepare_data(train_curated, config, '../input/train_curated/')
X_test = prepare_data(test, config, '../input/test/')
y = one_hot(train_curated['labels'], src_dict)


# In[ ]:



X = audio_norm(X_train)
X_test = audio_norm(X_test)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


PREDICTION_FOLDER = "predictions_1d_conv"
if not os.path.exists(PREDICTION_FOLDER):
    os.mkdir(PREDICTION_FOLDER)
if os.path.exists('logs/' + PREDICTION_FOLDER):
    shutil.rmtree('logs/' + PREDICTION_FOLDER)
    
K.clear_session()
checkpoint = ModelCheckpoint('best.h5', monitor='val_loss', verbose=1, save_best_only=True)
#early = EarlyStopping(monitor="val_loss", mode="min", patience=7)
tb = TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/fold', write_graph=True)
callbacks_list = [checkpoint, tb]
model = build_model(config)
history = model.fit(X, y, callbacks=callbacks_list,batch_size=128, epochs=50)
#model.load_weights('best.h5')

# # Save train predictions
# predictions = model.predict(X_train, batch_size=64, verbose=1)
# np.save(PREDICTION_FOLDER + "/train_predictions.npy", predictions)

# # Save test predictions
# predictions = model.predict(X_test, batch_size=64, verbose=1)
# np.save(PREDICTION_FOLDER + "/test_predictions.npy", predictions)

# # Make a submission file
# top_3 = np.array(config.n_classes)[np.argsort(-predictions, axis=1)[:, :3]]
# predicted_labels = [' '.join(list(x)) for x in top_3]
# test['label'] = predicted_labels
# test[['label']].to_csv(PREDICTION_FOLDER + "/predictions.csv") 


# In[ ]:


submission= model.predict(X_test, batch_size=64, verbose=1)


# In[ ]:


# Output all random to see a baseline
sample_sub = pd.read_csv('../input/sample_submission.csv')
sample_sub.iloc[:,1:] = submission
sample_sub.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:


working_dir='../working/predictions_1d_conv/'
os.listdir(working_dir)


# In[ ]:


submission=pd.read_csv(working_dir+'sample_submission.csv')
submission.head()


# In[ ]:





# In[ ]:


pred1=pd.read_csv(working_dir+'predictions.csv')
pred1.head()


# In[ ]:


pred=np.load(working_dir+'test_predictions.npy')
pred.shape


# ### Ignore the following code

# In[ ]:


#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(Sequence):
    def __init__(self, config, data_dir, list_IDs, labels=None, 
                 batch_size=64, preprocessing_fn=lambda x: x):
        self.config = config
        self.data_dir = data_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.preprocessing_fn = preprocessing_fn
        self.on_epoch_end()
        self.dim = self.config.dim

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return self.__data_generation(list_IDs_temp)
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        cur_batch_size = len(list_IDs_temp)
        X = np.empty((cur_batch_size, *self.dim))

        input_length = self.config.audio_length
        for i, ID in enumerate(list_IDs_temp):
            file_path = self.data_dir + ID
            
            # Read and Resample the audio
            data, _ = librosa.core.load(file_path, sr=self.config.sampling_rate,
                                        res_type='kaiser_fast')

            # Random offset / Padding
            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length+offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
                else:
                    offset = 0
                data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
                
            # Normalization + Other Preprocessing
            if self.config.use_mfcc:
                data = librosa.feature.mfcc(data, sr=self.config.sampling_rate,
                                                   n_mfcc=self.config.n_mfcc)
                data = np.expand_dims(data, axis=-1)
            else:
                data = self.preprocessing_fn(data)[:, np.newaxis]
            X[i,] = data

        if self.labels is not None:
            y = np.empty(cur_batch_size, dtype=int)
            for i, ID in enumerate(list_IDs_temp):
                y[i] = self.labels[ID]
            return X, to_categorical(y, num_classes=self.config.n_classes)
        else:
            return X


# In[ ]:


working_dir='../working/predictions_1d_conv/'
os.listdir(working_dir)


# In[ ]:


pred1=pd.read_csv(working_dir+'predictions_1.csv')
pred1.head()


# In[ ]:


pred=np.load(working_dir+'test_predictions_0.npy')
pred


# In[ ]:


train_generator = DataGenerator(config, '../input/train_curated/', train_set.index, 
                                    train_set.label_idx, batch_size=64,
                                    preprocessing_fn=audio_norm)
    val_generator = DataGenerator(config, '../input/train_curated/', val_set.index, 
                                  val_set.label_idx, batch_size=64,
                                  preprocessing_fn=audio_norm)
    
    history = model.fit_generator(train_generator, callbacks=callbacks_list, validation_data=val_generator,
                                  epochs=config.max_epochs, use_multiprocessing=True, max_queue_size=20)
    
#     model.load_weights('../working/best_%d.h5'%i)
    
    # Save train predictions
    train_generator = DataGenerator(config, '../input/train_curated/', train.index, batch_size=128,
                                    preprocessing_fn=audio_norm)
    predictions = model.predict_generator(train_generator, use_multiprocessing=True, 
                                          max_queue_size=20, verbose=1)
    np.save(PREDICTION_FOLDER + "/train_predictions_%d.npy"%i, predictions)
    
    # Save test predictions
    test_generator = DataGenerator(config, '../input/test/', test.index, batch_size=128,
                                    preprocessing_fn=audio_norm)
    predictions = model.predict_generator(test_generator, use_multiprocessing=True, 
                                          max_queue_size=20, verbose=1)
    np.save(PREDICTION_FOLDER + "/test_predictions_%d.npy"%i, predictions)
    
    # Make a submission file
    top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test['label'] = predicted_labels
    test[['label']].to_csv(PREDICTION_FOLDER + "/predictions.csv"%)
    


# In[ ]:


pred_list = []
for i in range(config.n_folds):
    pred_list.append(np.load("../working/predictions_1d_conv/test_predictions_%d.npy"%i))
prediction = np.ones_like(pred_list[0])
for pred in pred_list:
    prediction = prediction*pred
prediction = prediction**(1./len(pred_list))
# Make a submission file
top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
predicted_labels = [' '.join(list(x)) for x in top_3]
test = pd.read_csv('../input/sample_submission.csv')
test['label'] = predicted_labels
test[['fname', 'label']].to_csv("1d_conv_ensembled_submission.csv", index=False)


# ### Building a model using MFCC

# Preparing data

# In[ ]:





# In[ ]:


PREDICTION_FOLDER = "predictions_2d_conv"
if not os.path.exists(PREDICTION_FOLDER):
    os.mkdir(PREDICTION_FOLDER)
if os.path.exists('logs/' + PREDICTION_FOLDER):
    shutil.rmtree('logs/' + PREDICTION_FOLDER)

skf = StratifiedKFold(n_splits=config.n_folds)

for i, (train_split, val_split) in enumerate(skf.split(train.index, train.label_idx)):
    K.clear_session()
    X, y, X_val, y_val = X_train[train_split], y_train[train_split], X_train[val_split], y_train[val_split]
    checkpoint = ModelCheckpoint('best_%d.h5'%i, monitor='val_loss', verbose=1, save_best_only=True)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    tb = TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/fold_%i'%i, write_graph=True)
    callbacks_list = [checkpoint, early, tb]
    print("#"*50)
    print("Fold: ", i)
    model = get_2d_dummy_model(config)
    history = model.fit(X, y, validation_data=(X_val, y_val), callbacks=callbacks_list,  batch_size=64, epochs=config.max_epochs)
    model.load_weights('best_%d.h5'%i)

    # Save train predictions
    predictions = model.predict(X_train, batch_size=64, verbose=1)
    np.save(PREDICTION_FOLDER + "/train_predictions_%d.npy"%i, predictions)

    # Save test predictions
    predictions = model.predict(X_test, batch_size=64, verbose=1)
    np.save(PREDICTION_FOLDER + "/test_predictions_%d.npy"%i, predictions)

    # Make a submission file
    top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test['label'] = predicted_labels
    test[['label']].to_csv(PREDICTION_FOLDER + "/predictions_%d.csv"%i)


# In[ ]:


pred_list = []
for i in range(config.n_folds):
    pred_list.append(np.load("../working/predictions_2d_conv/test_predictions_%d.npy"%i))
prediction = np.ones_like(pred_list[0])
for pred in pred_list:
    prediction = prediction*pred
prediction = prediction**(1./len(pred_list))
# Make a submission file
top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
predicted_labels = [' '.join(list(x)) for x in top_3]
test = pd.read_csv('../input/sample_submission.csv')
test['label'] = predicted_labels
test[['fname', 'label']].to_csv("2d_conv_ensembled_submission.csv", index=False)


# In[ ]:


working_dir='../working/predictions_2d_conv/'
os.listdir(working_dir)


# In[ ]:


pred=np.load(working_dir+'test_predictions_1.npy')


# In[ ]:


target_names


# In[ ]:




