import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import CuDNNLSTM
from keras.models import model_from_json
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import pickle

X_train = np.load('../input/units150/X_train.npy')
y_train = np.load('../input/units150/y_train.npy')
X_test = np.load('../input/units150/X_test.npy')
y_test = np.load('../input/units150/y_test.npy')

model = load_model('../input/units150/my_model_epoch5.h5')
#results = model.predict(X_test)
#rms = optimizers.RMSprop(lr = 0.0009)
#model.compile(optimizer = rms, loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())

# checkpoint
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
callbacks_list = [checkpoint]

# fit and save history
hist = model.fit(X_train, y_train, epochs = 30, validation_data = (X_test, y_test), batch_size = 64, callbacks = callbacks_list, initial_epoch = 5)
with open('trainHistoryDict', 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)
    
model.save('my_model.h5')



#print(history_loss['loss'])

#partHistory = history_loss['loss'][1:]
#print(partHistory)

#plt.plot(partHistory)
#plt.title('model loss')
#plt.xlabel('epoch')
#plt.ylabel('loss')
#plt.show()