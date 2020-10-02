import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import numpy as np

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

X_train = np.array(train.values[:,1:])
Y_train = np.array(train.values[:,0])
X_test = test.values
Y_test = test.values[:,0]

Y_train_one_hot = np.zeros((Y_train.shape[0], 10))
Y_train_one_hot[[range(len(Y_train)),Y_train]] = 1
Y_test_one_hot = np.zeros((Y_test.shape[0], 10))
Y_test_one_hot[[range(len(Y_test)),Y_test]] = 1

model = Sequential()
model.add(Dense(200, input_dim=784, init='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, init='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, init='normal', activation='softmax'))
early_stopping = EarlyStopping(patience=5)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train_one_hot, batch_size=128, nb_epoch=100, verbose=2, validation_split=0.3, callbacks=[early_stopping])
predictions_one_hot = np.array(model.predict(X_test))

one_to_nine = np.array(range(10))
predictions = np.dot(predictions_one_hot, np.transpose(one_to_nine))
predictions = predictions.astype(int)
f = open('predictions.csv', 'w+')
f.write('ImageId,Label\n')
indices = list(test.index.values)
for i in range(len(predictions)):
    f.write(str(indices[i]+1) + ',' + str(predictions[i]) + str('\n'))
    
f.close()