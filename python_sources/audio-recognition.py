#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow==1.13.1')


# In[ ]:


import tensorflow as tf
print(tf.__version__)


# **Freesound General-Purpose Audio Tagging Challenge**
# 
# Aim: To build a general-purpose automatic audio tagging system using a dataset of audio files covering a wide range of real-world environments.
# 

# In[ ]:


import tensorflow as tf
print(tf.__version__)
from IPython.display import Image
Image("../input/blockdiagram/block diia.png")


# **Method**

# In[ ]:


Image("../input/methodology/Method.png")


# Generally, ML problems are solved by using data in relevant form that can be fed to train a model, which can further classify/predict outputs for unseen data. In this problem, the input data consists of audio files split into training and testing datasets. The audio clips (.wav) are represented in the form of spectrogram, which along with labels is given to the model for training. Spectrogram is explained in a later section.
# 
# The sequence of events is represented in the block diagram above.

# In[ ]:


# import dependencies

import numpy as np
import pandas as pd
import librosa
import librosa.display
import wave
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras import backend as K
from keras.models import Sequential
from keras.layers import (Conv2D, GlobalAveragePooling2D, BatchNormalization, Flatten,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation, Input, Dense)
import keras.optimizers
import matplotlib.pyplot as plt
from keras.utils import plot_model


# **Data Analysis**
# 
# Freesound Dataset Kaggle 2018 (or FSDKaggle2018 for short) is an audio dataset containing 18,873 audio files annotated with labels from Google's AudioSet Ontology. The audio clips are unequally distributed in 41 categories. 

# In[ ]:


# load data
train_data = pd.read_csv("../input/freesound-audio-tagging/train.csv")
test_data = pd.read_csv("../input/freesound-audio-tagging/sample_submission.csv")
print(train_data.head())
print("Number of training examples=", train_data.shape[0], "  Number of classes=", len(train_data.label.unique()))
print(test_data.head())
print("Number of testing examples=", test_data.shape[0], "  Number of classes=", len(test_data.label.unique()))


# The training data includes 9,473 samples and testing data includes 9400 samples. From the output, it can be seen that the correct labels of testing data are not given.

# **Distribution of Data**

# In[ ]:


category_group = train_data.groupby(['label']).count()
plot = category_group.unstack().plot(kind='bar', stacked=True, title="Number of Audio Samples per Category", figsize=(16,10))
plot.set_xlabel("Category")
plot.set_ylabel("Number of Samples");


# The histogram shows that the number of audio samples per category is not uniform. The minimum number of samples per category is 94 and the maximum is 300.

# **Frame Length**

# Majority of the files are short and not constant. The length of audio files range from 300 ms to 30s.
# As the model takes data in equal sizes, the data needs to be preprocessed. The audio clips that are shorter than the required sample length are padded. Whereas the audio clips that are longer than the required sample length are cut-short.
# 
# Citation: Eduardo Fonseca, Manoj Plakal, Frederic Font, Daniel P. W. Ellis, Xavier Favory, Jordi Pons, Xavier Serra. General-purpose Tagging of Freesound Audio with AudioSet Labels: Task Description, Dataset, and Baseline. In Proceedings of DCASE2018 Workshop, 2018. URL: https://arxiv.org/abs/1807.09902

# **Spectrogram**
# 
# A spectrogram is a visual representation of frequencies of a signal as it varies. An example of spectrogram can be seen below.

# In[ ]:


fname = "../input/freesound-audio-tagging/audio_test/audio_test/8319139c.wav"
samples, sample_rate = librosa.core.load(fname, sr=44100)
s = librosa.feature.melspectrogram(samples,sr=sample_rate)
s = librosa.power_to_db(s)
s = s.astype(np.float32)
plt.figure(figsize=(10, 4))
librosa.display.specshow(s,y_axis='mel', fmax=8000,x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()


# A spectrogram has three dimensions. X-axis represents time, Y-axis represents frequency and the final dimension represents the amplitude of a particular frequency at a particular time is represented by the intensity or color of each point in the image.

# **Data Preprocessing**
# 
# Before the data is fed into the model, it needs to be preprocessed into a desired form. Preprocessing of data includes the following steps:
# 1. Extract samples from audio, adjust the length and extract spectrograms.
# 2. Normalize the data

# In[ ]:


# Loading and processing Data

def get_spectrogram(filename):
    '''
    Input: A dataframe of filepaths of the audio clips
    Returns: A numpy array of spectrograms of the clips in the dataframe.
    
    For every audio clip in the dataset, samples are extracted using Librosa library. The length of the samples are
    adjusted as required.
    Spectrogram for the samples is calculated using the same library Librosa which is then appended to a numpy array
    '''
    import tqdm
    x = []
    duration = 5
    sample_length = 44100 * duration
    for fname in tqdm.tqdm(filename):
        samples, sample_rate = librosa.core.load(fname, sr=44100)
        if len(samples) > sample_length: # long enough
            samples = samples[0:sample_length]
        else: # pad blank
            padding = sample_length - len(samples)
            offset = padding // 2
            if len(samples) == 0:
                samples = np.pad(samples, (offset, sample_length - len(samples) - offset), 'constant')
            else:
                while(len(samples)<sample_length):
                    padding = sample_length - len(samples)
                    samples = np.append(samples, samples[0:padding], axis=0)
        #mfcc = librosa.feature.mfcc(samples,sr=sample_rate)
        s = librosa.feature.melspectrogram(samples,sr=sample_rate)
        s = librosa.power_to_db(s)
        s = s.astype(np.float32)
        x.append(s)
    # Stack them using axis=0.
    x = np.stack(x)
    print(x.shape)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2],1)
    print("done")
    return x


def get_labels(y):
    '''
    Input: A list of Labels
    Returns: A list of number encoded labels and a map of labels to number code
    '''
    le = LabelEncoder()
    le.fit(y)
    y = le.fit_transform(y)
    le_mapping = dict(zip(le.transform(le.classes_),le.classes_))
    return y, le_mapping


def normalize(X):
    '''
    Input: Dataset
    Returns: Z-Score normalization of the data in the dataset.
    '''
    eps = 0.001
    normalized_dataset = []
    for img in X:
        if np.std(img) != 0:
            img = (img - np.mean(img)) / np.std(img)
        else:
            img = (img - np.mean(img)) / eps
        normalized_dataset.append(img)
    return np.array(normalized_dataset)


# In[ ]:


def process_data(train_data,test_data):
    '''
    Input: Dataframes of training and testing data given in the workspace of kaggle
    Returns: Processed training and testing data along with labels and label mapping.
    '''
    filename_train = '../input/freesound-audio-tagging/audio_train/audio_train/' + train_data['fname']
    filename_test = '../input/freesound-audio-tagging/audio_test/audio_test/' + test_data['fname']
    x_train = get_spectrogram(filename_train)
    x_test = get_spectrogram(filename_test)
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    y_train, le_mapping = get_labels(train_data['label'])
    y_test = np.array(test_data['label'])
    return x_train, y_train,x_test, y_test, le_mapping


train_data = pd.read_csv("../input/freesound-audio-tagging/train.csv")
test_data = pd.read_csv("../input/freesound-audio-tagging/sample_submission.csv")

# transform and extract relevant data
x_train,y_train,x_test,y_test, le_mapping = process_data(train_data,test_data)


# **Model**
# 
# Model defined is a 2D Convolution Neural Network. The architecture is defined using Keras.

# In[ ]:


def train(x_train,y_train):
    '''
    Input: Training data
    Returns: A trained model and training metrics
    '''
    
    batch_size = 50
    optimizer = keras.optimizers.SGD(lr=0.01)
    input_shape=(x_train.shape[1], x_train.shape[2],1)
    model = Sequential()
    model.add(Conv2D(64, (3,6),input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(MaxPool2D())
    
    model.add(Conv2D(32, (3,6)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D())
    
    model.add(Conv2D(32, (3,6)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D())
    
    model.add(Flatten())
    model.add(Dense(41,activation='softmax'))
    model.summary()
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer=optimizer,metrics = ['accuracy'])
    history = model.fit(x_train, y_train,batch_size=batch_size,epochs=50,verbose=1,validation_split=0.3)
    
    return model, history

model, history = train(x_train,y_train)
model.save('my_keras_model.h5')
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# **Training Metrics**

# In[ ]:


model.evaluate(x_train,y_train)
print("Output: ",model.outputs)
print("Input: ",model.inputs)
x_axis = np.linspace(1,50,50)
plt.subplot(2, 1, 1)
plt.plot(x_axis, history.history['acc'], history.history['val_acc'])
plt.title('Training metrics')
plt.ylabel('Accuracy')

plt.subplot(2, 1, 2)
plt.plot(x_axis, history.history['loss'], history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()


# **Testing**
# 
# Tested a sample of test data with the model

# In[ ]:


def test(x_test,le_mapping, model,y_test):
    y_pred = model.predict_classes(x_test)
    pred = pd.DataFrame([],columns = ['Predicted','Actual'])
    for i in range(len(y_pred)):
        pred = pred.append({'Predicted':le_mapping[y_pred[i]], 'Actual':y_test[i]}, ignore_index=True)
    print(pred)
        
# test model
x_test  = ['0038a046.wav','007759c4.wav','00ae03f6.wav','00eac343.wav','010a0b3a.wav','01a5a2a3.wav','02107093.wav','02960f07.wav'
          ,'02fb6c5b.wav']
x_test = pd.DataFrame(x_test,columns=['fname'])
y_test = ["Bass_drum",'Saxophone','Chime','Electric_piano','Shatter','bark','Electric_piano','Scissors','Knock']
filename_test = '../input/freesound-audio-tagging/audio_test/audio_test/' + x_test['fname']
x_test = get_spectrogram(filename_test)
x_test = normalize(x_test)
test(x_test,le_mapping,model,y_test)        


# In[ ]:


def export_model_for_mobile(model_name, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out',         model_name + '_graph.pbtxt')

    tf.train.Saver().save(K.get_session(), 'out/' + model_name + '.chkp')

    freeze_graph.freeze_graph('out/' + model_name + '_graph.pbtxt', None,         False, 'out/' + model_name + '.chkp', output_node_name,         "save/restore_all", "save/Const:0",         'out/frozen_' + model_name + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + model_name + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/tensorflow_lite_' + model_name + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())


# In[ ]:


from keras import backend as K
import tensorflow as tf

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):

    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, "model", "my_model.pb", as_text=False)


# In[ ]:


print(le_mapping)

