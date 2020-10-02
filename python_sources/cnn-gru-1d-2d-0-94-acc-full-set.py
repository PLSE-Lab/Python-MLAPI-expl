### 2D N_FFT 256 - 25 EPOCHS - 0.945 ACC FULL SET // 20% TEST

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
from keras.layers import BatchNormalization
from keras.layers import Conv1D, Conv2D
from keras.layers import Activation
from keras.layers import MaxPooling1D, MaxPooling2D
from keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers import Permute
from keras.layers import Reshape
from keras.layers import CuDNNGRU, CuDNNLSTM
from keras.layers import Dropout
from keras.layers import Dense

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger

from keras import Model
from keras import backend as K

from keras.utils import np_utils
from keras.preprocessing import image
##############################################################################

##############################################################################
model_name = 'cnn_gru'
best_weights_path = model_name + '.h5'
log_path = model_name + '.log'
monitor = 'val_acc'
batch_size = 32
epochs = 100
es_patience = 7
rlr_patience = 3

SR = 8000
#model_mode, input_shape_1D = '1D_psd', (129, 1)
#model_mode, input_shape_1D = '1D', (5000, 1)
#model_mode, N_FFT, HOP_LEN, input_shape_2D = '2D', 128, 128, (65, 40, 1)
model_mode, N_FFT, HOP_LEN, input_shape_2D = '2D', 256, 256 / 6, (129, 120, 1)
##############################################################################

##############################################################################
target_names = ['Ae. aegypti', 'Ae. albopictus', 'An. gambiae', 'An. arabiensis', 'C. pipiens', 'C. quinquefasciatus']

X_names = []
y = []
target_count = []

for i, target in enumerate(target_names):
    target_count.append(0)
    path = './Wingbeats/' + target + '/'
    for [root, dirs, files] in os.walk(path, topdown = False):
        for filename in files:
            name,ext = os.path.splitext(filename)
            if ext == '.wav':
                name = os.path.join(root, filename)
                y.append(i)
                X_names.append(name)
                target_count[i]+=1
                #if target_count[i] > 20000:
                    #break
    print (target, '#recs = ', target_count[i])

print ('total #recs = ', len(y))

X_names, y = shuffle(X_names, y, random_state = seed)
X_train, X_test, y_train, y_test = train_test_split(X_names, y, stratify = y, test_size = 0.20, random_state = seed)

print ('train #recs = ', len(X_train))
print ('test #recs = ', len(X_test))
##############################################################################

##############################################################################
def basic_cnn_gru_1D(cols, channels, num_classes):
    inputs = Input(shape = (cols, channels))

    x = BatchNormalization() (inputs)
    x = Conv1D(16, kernel_size = 3, padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation('elu') (x)
    x = MaxPooling1D((2)) (x)
    x = Conv1D(32, kernel_size = 3, padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation('elu') (x)
    x = MaxPooling1D((2)) (x)
    x = Conv1D(64, kernel_size = 3, padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation('elu') (x)
    x = MaxPooling1D((2)) (x)
    x = Conv1D(128, kernel_size = 3, padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation('elu') (x)
    x = MaxPooling1D((2)) (x)
    x = Conv1D(256, kernel_size = 3, padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation('elu') (x)
    x = MaxPooling1D((2)) (x)

    #x = GlobalAveragePooling1D() (x)

    x = CuDNNGRU(units = 128, return_sequences = True) (x)
    x = CuDNNGRU(units = 128, return_sequences = False) (x)
    #x = CuDNNLSTM(units = 128, return_sequences = True) (x)
    #x = CuDNNLSTM(units = 128, return_sequences = False) (x)

    x = Dropout(0.5) (x)

    x = Dense(num_classes) (x)
    outputs = Activation('softmax') (x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model

def basic_cnn_gru_2D(rows, cols, channels, num_classes):
    inputs = Input(shape = (rows, cols, channels))

    x = BatchNormalization() (inputs)
    x = Conv2D(16, kernel_size = (3, 3), padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation('elu') (x)
    x = MaxPooling2D((2,2)) (x)
    x = Conv2D(32, kernel_size = (3, 3), padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation('elu') (x)
    x = MaxPooling2D((2,2)) (x)
    x = Conv2D(64, kernel_size = (3, 3), padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation('elu') (x)
    x = MaxPooling2D((2,2)) (x)
    x = Conv2D(128, kernel_size = (3, 3), padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation('elu') (x)
    x = MaxPooling2D((2,2)) (x)
    x = Conv2D(256, kernel_size = (3, 3), padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation('elu') (x)
    x = MaxPooling2D((2,2)) (x)
    
    #x = GlobalAveragePooling2D() (x)

    x = Permute((2, 1, 3)) (x)

    if N_FFT == 128:
        x = Reshape((1, 2 * 256)) (x)
    elif N_FFT == 256:
        x = Reshape((3, 4 * 256)) (x)

    x = CuDNNGRU(units = 128, return_sequences = True) (x)
    x = CuDNNGRU(units = 128, return_sequences = False) (x)
    #x = CuDNNLSTM(units = 128, return_sequences = True) (x)
    #x = CuDNNLSTM(units = 128, return_sequences = False) (x)

    x = Dropout(0.5) (x)

    x = Dense(num_classes) (x)
    outputs = Activation('softmax') (x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model
##############################################################################

##############################################################################
def shift(x, wshift, hshift, row_axis = 0, col_axis = 1, channel_axis = 2, fill_mode = 'constant', cval = 0.):
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = hshift * h
    ty = wshift * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    transform_matrix = translation_matrix
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def random_data_shift(data, w_limit = (-0.25, 0.25), h_limit = (-0.0, 0.0), cval = 0., u = 0.5):
    if np.random.random() < u:
        wshift = np.random.uniform(w_limit[0], w_limit[1])
        hshift = np.random.uniform(h_limit[0], h_limit[1])
        data = shift(data, wshift, hshift, cval = cval)
    return data

# def random_data_shift(data, u = 0.5):
#     if np.random.random() < u:
#         data = np.roll(data, int(round(np.random.uniform(-(len(data)), (len(data))))))
#     return data

def train_generator(mode = '2D'):
    while True:
        for start in range(0, len(X_train), batch_size):
            x_batch = []
            y_batch = []
            
            end = min(start + batch_size, len(X_train))
            train_batch = X_train[start:end]
            labels_batch = y_train[start:end]
            
            for i in range(len(train_batch)):
                data, rate = librosa.load(train_batch[i], sr = SR)

                #data = random_data_shift(data, u = 1.0)

                if mode == '1D_psd':
                    XX = np.zeros(129).astype('float32')
                    XX = 10 * np.log10(signal.welch(data, fs = SR, window = 'hanning', nperseg = 256, noverlap = 128 + 64)[1])
                    data = XX

                if mode == '2D':
                    data = librosa.stft(data, n_fft = N_FFT, hop_length = HOP_LEN)
                    data = librosa.amplitude_to_db(data)

                    data = np.flipud(data)

                    data = np.expand_dims(data, axis = -1)
                    data = random_data_shift(data, w_limit = (-0.25, 0.25), h_limit = (-0.0, 0.0), cval = np.min(data), u = 1.0)
                    data = np.squeeze(data, axis = -1)

                data = np.expand_dims(data, axis = -1)

                # data = np.squeeze(data, axis = -1)
                # plt.imshow(data, cmap = 'gray')
                # plt.show()
                # data = np.expand_dims(data, axis = -1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            
            y_batch = np_utils.to_categorical(y_batch, len(target_names))
            
            yield x_batch, y_batch

def valid_generator(mode = '2D'):
    while True:
        for start in range(0, len(X_test), batch_size):
            x_batch = []
            y_batch = []
            
            end = min(start + batch_size, len(X_test))
            test_batch = X_test[start:end]
            labels_batch = y_test[start:end]
            
            for i in range(len(test_batch)):
                data, rate = librosa.load(test_batch[i], sr = SR)

                if mode == '1D_psd':
                    XX = np.zeros(129).astype('float32')
                    XX = 10 * np.log10(signal.welch(data, fs = SR, window = 'hanning', nperseg = 256, noverlap = 128 + 64)[1])
                    data = XX

                if mode == '2D':
                    data = librosa.stft(data, n_fft = N_FFT, hop_length = HOP_LEN)
                    data = librosa.amplitude_to_db(data)

                    data = np.flipud(data)

                data = np.expand_dims(data, axis = -1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            
            y_batch = np_utils.to_categorical(y_batch, len(target_names))
            
            yield x_batch, y_batch
##############################################################################

##############################################################################
if model_mode != '2D':
    model = basic_cnn_gru_1D(input_shape_1D[0], input_shape_1D[1], len(target_names))
else:
    model = basic_cnn_gru_2D(input_shape_2D[0], input_shape_2D[1], input_shape_2D[2], len(target_names))

callbacks_list = [ModelCheckpoint(monitor = monitor,
                                filepath = best_weights_path, 
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

model.fit_generator(train_generator(mode = model_mode),
    steps_per_epoch = int(math.ceil(float(len(X_train)) / float(batch_size))),
    validation_data = valid_generator(mode = model_mode),
    validation_steps = int(math.ceil(float(len(X_test)) / float(batch_size))),
    epochs = epochs,
    callbacks = callbacks_list,
    shuffle = False)

model.load_weights(best_weights_path)

loss, acc = model.evaluate_generator(valid_generator(mode = model_mode),
        steps = int(math.ceil(float(len(X_test)) / float(batch_size))))

print('loss:', loss)
print('Test accuracy:', acc)
##############################################################################