from keras import backend as K
import tensorflow as tf
with K.tf.device('/gpu:0'):
    config = tf.ConfigProto(intra_op_parallelism_threads=4,\
           inter_op_parallelism_threads=4, allow_soft_placement=True,\
           device_count = {'CPU' : 4, 'GPU' : 1})
    session = tf.Session(config=config)
    K.set_session(session)


#console 1-A
#console 3-A
from pandas import read_csv
dataset_consider = read_csv('../input/solar-data-with-lag-24-hours-as-features/Solar_Lag1Day.csv', header=0, infer_datetime_format=True, 
                            parse_dates=['datetime'], index_col=['datetime'])   
###############################################################################   
import numpy as np
subset1 = dataset_consider[:91980]
subset2 = dataset_consider[91980:]


subset1_values = subset1.values
subset2_values = subset2.values

subset1_values =  np.float32(subset1_values)
subset2_values =  np.float32(subset2_values)

# split a multivariate sequence into samples
from numpy import array
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
###############################################################################
n_steps = 28
X_train, y_train = split_sequences(subset1_values, n_steps)
X_test, y_test = split_sequences(subset2_values, n_steps)
###############################################################################
X_train, y_train = X_train[:-1,:,:], y_train[:-1]
X_test, y_test = X_test[:-1,:,:], y_test[:-1]
###############################################################################
trainX = X_train
testX = X_test

from sklearn.preprocessing import MinMaxScaler

scalers = {}
for i in range(X_train.shape[2]):
    scalers[i] = MinMaxScaler()
    X_train[:, :, i] = scalers[i].fit_transform(X_train[:, :, i]) 

for i in range(X_test.shape[2]):
    X_test[:, :, i] = scalers[i].transform(X_test[:, :, i])
###############################################################################
n_features = X_train.shape[2]
###############################################################################
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
path_checkpoint = '3_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=10, verbose=1)

callback_tensorboard = TensorBoard(log_dir='./1_logs/',
                                   histogram_freq=0,
                                   write_graph=False)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.80,
                                       min_lr=1e-6,
                                       patience=2,
                                       verbose=1)

callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]  
###############################################################################
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers import Dense
from keras.layers.merge import concatenate
from keras import optimizers

# head 1
inputs1 = Input(shape=(n_steps,n_features))
conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
conv1_2 = Conv1D(filters=32, kernel_size=5, activation='relu')(conv1)
drop1 = Dropout(0.5)(conv1_2)
flat1 = Flatten()(drop1)
# head 2
inputs2 = Input(shape=(n_steps,n_features))
conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs2)
conv2_2 = Conv1D(filters=32, kernel_size=7, activation='relu')(conv2)
drop2 = Dropout(0.5)(conv2_2)
flat2 = Flatten()(drop2)
# head 3
inputs3 = Input(shape=(n_steps,n_features))
conv3 = Conv1D(filters=64, kernel_size=7, activation='relu')(inputs3)
conv3_2 = Conv1D(filters=32, kernel_size=9, activation='relu')(conv3)
drop3 = Dropout(0.5)(conv3_2)
flat3 = Flatten()(drop3)
# merge
merged = concatenate([flat1, flat2, flat3])
# interpretation
dense1 = Dense(16, activation='relu')(merged)

outputs = Dense(1, activation='relu')(dense1)
model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
adam = optimizers.adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam)
history = model.fit([X_train,X_train,X_train], y_train, epochs=200, batch_size=128, verbose=1, validation_split=0.10, shuffle=False, callbacks=callbacks)
################################################################################

print(history.history.keys())
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()
###############################################################################
try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)
    
predictions = model.predict([X_test,X_test,X_test])
##############################################################################
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt

y_cv = np.expand_dims(y_test, axis=1)

rms = sqrt(mean_squared_error(y_cv, predictions))
mae = mean_absolute_error(predictions, y_cv)
r2 = r2_score(y_cv, predictions) 


outfile = 'predictions.npz'
np.savez_compressed(outfile, a=predictions)

