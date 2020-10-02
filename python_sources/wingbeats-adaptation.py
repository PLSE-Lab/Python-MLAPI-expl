
### One needs to first run https://www.kaggle.com/left13/various-nets-densenet121-0-96-acc-full-set
### DenseNet121 N_FFT 256 - 23 EPOCHS - 0.96 ACC ON 20% TEST

##############################################################################
import os, math
import numpy as np
seed = 2018
np.random.seed(seed)

import librosa
from scipy import signal

from matplotlib import pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.layers import Input
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger

from keras import Model
from keras import backend as K

from keras.utils import np_utils
from keras.preprocessing import image
 
from keras.applications.densenet import DenseNet121

import seaborn as sn
from sklearn.metrics import confusion_matrix
import pandas as pd
import wget
import zipfile

##############################################################################
current_model = DenseNet121

model_name = 'wingbeats_' + current_model.__name__

best_weights_path = model_name + '.h5'
adapted_best_weights_path = 'adapted_' + model_name + '.h5'
scratch_best_weights_path = 'scratch_' + model_name + '.h5'
log_path = model_name + '.log'
monitor = 'val_acc'
batch_size = 32
epochs = 100
es_patience = 7
rlr_patience = 3

SR = 8000
N_FFT = 512
HOP_LEN = N_FFT / 24
input_shape = (N_FFT/2+1, 239, 1)
#input_shape = (N_FFT/2+1, 120, 1)

##############################################################################
target_names_wingbeats = ['Ae. aegypti', 'Ae. albopictus', 'An. gambiae', 'An. arabiensis', 'C. pipiens', 'C. quinquefasciatus']
target_names_adapted = ['aedes_male', 'fuit_flies', 'house_flies', 'new_aedes_female', 'new_stigma_male','new_tarsalis_male', 'quinx_female', 'quinx_male', 'stigma_female', 'tarsalis_female']

def print_confusion(y_test, y_pred):
	cm = confusion_matrix(y_test, y_pred)
	df_cm = pd.DataFrame(cm, columns=target_names_adapted, index=target_names_adapted)
	plt.figure(figsize = (15,10))
	sn.heatmap(df_cm, annot=True, fmt="d")

	print("")

	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	df_cm = pd.DataFrame(cm, columns=target_names_adapted, index=target_names_adapted)
	plt.figure(figsize = (15,10))
	sn.heatmap(df_cm, annot=True)
	plt.show()


def crop_rec(data):

	f, t, Zxx = signal.stft(data, SR, nperseg=256)
	Z = np.sum(np.abs(Zxx), axis=0)
	max_pos = np.argmax(Z)
	mid_x = 1+128*max_pos
	nsamples = 5000
	mid_x = np.max([nsamples/2, mid_x])
	mid_x = np.min([len(data)-nsamples/2, mid_x])
	x = data[-nsamples/2 + mid_x + range(nsamples)]

	# data original signal, x: cropped signal
	return x

###########################################
# Download UCR database from https://www.dropbox.com/s/23dk89s6r88cb0j/increasing%20dataset.zip?dl=0
url = 'https://www.dropbox.com/s/23dk89s6r88cb0j/increasing%20dataset.zip?dl=1'

filename = wget.download(url)

zip_ref = zipfile.ZipFile(filename, 'r')
zip_ref.extractall('')
zip_ref.close()
###########################################

X_names = []
y = []
target_count = []

for i, target in enumerate(target_names_adapted):
    target_count.append(0)
    path = './increasing dataset/' + target + '/'
    for [root, dirs, files] in os.walk(path, topdown = False):
        for filename in files:
            name,ext = os.path.splitext(filename)
            if ext == '.wav':
                name = os.path.join(root, filename)
                y.append(i)
                X_names.append(name)
                target_count[i]+=1
    print (target, '#recs = ', target_count[i])

print ('total #recs = ', len(y))

X_names, y = shuffle(X_names, y, random_state = seed)
X_train, X_test, y_train, y_test = train_test_split(X_names, y, stratify = y, test_size = 0.20, random_state = seed)

print ('train #recs = ', len(X_train))
print ('test #recs = ', len(X_test))
##############################################################################

def random_data_shift(data, u):
    if np.random.random() < u:
        data = np.roll(data, int(round(np.random.uniform(-(len(data)*0.15), (len(data)*0.15)))))
    return data

def train_generator():
    while True:
        for start in range(0, len(X_train), batch_size):
            x_batch = []
            y_batch = []
            
            end = min(start + batch_size, len(X_train))
            train_batch = X_train[start:end]
            labels_batch = y_train[start:end]
            
            for i in range(len(train_batch)):
                data, rate = librosa.load(train_batch[i], sr = SR)
                data = crop_rec(data)

                data = random_data_shift(data, u = .2)

                data = librosa.stft(data, n_fft = N_FFT, hop_length = HOP_LEN)
                data = librosa.amplitude_to_db(data)

                data = np.flipud(data)

                data = np.expand_dims(data, axis = -1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            
            y_batch = np_utils.to_categorical(y_batch, len(target_names_adapted))
            
            yield x_batch, y_batch

def valid_generator():
    while True:
        for start in range(0, len(X_test), batch_size):
            x_batch = []
            y_batch = []
            
            end = min(start + batch_size, len(X_test))
            test_batch = X_test[start:end]
            labels_batch = y_test[start:end]
            
            for i in range(len(test_batch)):
                data, rate = librosa.load(test_batch[i], sr = SR)
                data = crop_rec(data)

                data = librosa.stft(data, n_fft = N_FFT, hop_length = HOP_LEN)
                data = librosa.amplitude_to_db(data)

                data = np.flipud(data)
                data = np.expand_dims(data, axis = -1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            
            y_batch = np_utils.to_categorical(y_batch, len(target_names_adapted))
            
            yield x_batch, y_batch
##############################################################################

##############################################################################
## train model from scratch
img_input = Input(shape = input_shape)

model = current_model(input_tensor = img_input, classes = len(target_names_adapted), weights = None)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])  

callbacks_list = [ModelCheckpoint(monitor = monitor,
                                filepath = scratch_best_weights_path, 
                                save_best_only = True, 
                                save_weights_only = True,
                                verbose = 1), 
                    EarlyStopping(monitor = monitor,
                                patience = es_patience, 
                                verbose = 1),
                    ReduceLROnPlateau(monitor = monitor,
                                factor = 0.1, 
                                patience = rlr_patience, 
                                verbose = 1),
                    CSVLogger(filename = log_path)]


model.fit_generator(train_generator(),
    steps_per_epoch = int(math.ceil(float(len(X_train)) / float(batch_size))),
    validation_data = valid_generator(),
    validation_steps = int(math.ceil(float(len(X_test)) / float(batch_size))),
    epochs = epochs,
    callbacks = callbacks_list,
    shuffle = False)

model.load_weights(scratch_best_weights_path)

loss, acc = model.evaluate_generator(valid_generator(),
        steps = int(math.ceil(float(len(X_test)) / float(batch_size))))

#print('loss:', loss)
print('Test accuracy:', acc)

print ("")
y_pred = np.argmax(model.predict_generator(valid_generator(),
        steps = int(math.ceil(float(len(X_test)) / float(batch_size)))),axis=1)

# Print the confusion matrix
print_confusion(y_test, y_pred)
##############################################################################

## train adapted model
img_input = Input(shape = input_shape)

model = current_model(input_tensor = img_input, classes = len(target_names_wingbeats), weights = None)

# Discard last layers
x_new = model.layers[-3].output

# Load pretrained weights. You can get them after running the code in the following link https://www.kaggle.com/left13/various-nets-densenet121-0-96-acc-full-set
model.load_weights(best_weights_path)

x_new = GlobalAveragePooling2D(name='avg_pool')(x_new)
x_new = Dense(len(target_names_adapted), activation='softmax', name='fc10')(x_new)
    
model = Model(img_input, x_new, name='densenet')

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])  

callbacks_list = [ModelCheckpoint(monitor = monitor,
                                filepath = adapted_best_weights_path, 
                                save_best_only = True, 
                                save_weights_only = True,
                                verbose = 1), 
                    EarlyStopping(monitor = monitor,
                                patience = es_patience, 
                                verbose = 1),
                    ReduceLROnPlateau(monitor = monitor,
                                factor = 0.1, 
                                patience = rlr_patience, 
                                verbose = 1),
                    CSVLogger(filename = log_path)]


model.fit_generator(train_generator(),
    steps_per_epoch = int(math.ceil(float(len(X_train)) / float(batch_size))),
    validation_data = valid_generator(),
    validation_steps = int(math.ceil(float(len(X_test)) / float(batch_size))),
    epochs = epochs,
    callbacks = callbacks_list,
    shuffle = False)

# refined weights
model.load_weights(adapted_best_weights_path)

loss, acc = model.evaluate_generator(valid_generator(),
        steps = int(math.ceil(float(len(X_test)) / float(batch_size))))

#print('loss:', loss)
print('Test accuracy:', acc)

print ("")
y_pred = np.argmax(model.predict_generator(valid_generator(),
        steps = int(math.ceil(float(len(X_test)) / float(batch_size)))),axis=1)

# Print the confusion matrix
print_confusion(y_test, y_pred)

##############################################################################
