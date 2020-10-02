import pandas as pd
import numpy as np

# load training data
X = pd.read_csv('../input/X_train.csv')
Y = pd.read_csv('../input/Y_train.csv')
X_test = pd.read_csv('../input/X_test.csv')

# show availible features
X.columns

# preprocessing features
X['PTS_AVG'] = X['PTS'] / X['GP']
X['AST_AVG'] = X['AST'] / X['GP']
X['STL_AVG'] = X['STL'] / X['GP']
X['BLK_AVG'] = X['BLK'] / X['GP']
X['TOV_AVG'] = X['TOV'] / X['GP']
X['REB_AVG'] = X['REB'] / X['GP']
X['HEIGHT'] = X['HEIGHT'].map(lambda height: 12*int(height.split('-')[0]) + int(height.split('-')[1]))


# preprocessing features
X_test['PTS_AVG'] = X_test['PTS'] / X_test['GP']
X_test['AST_AVG'] = X_test['AST'] / X_test['GP']
X_test['STL_AVG'] = X_test['STL'] / X_test['GP']
X_test['BLK_AVG'] = X_test['BLK'] / X_test['GP']
X_test['TOV_AVG'] = X_test['TOV'] / X_test['GP']
X_test['REB_AVG'] = X_test['REB'] / X_test['GP']
X_test['HEIGHT'] = X_test['HEIGHT'].map(lambda height: 12*int(height.split('-')[0]) + int(height.split('-')[1]))


from keras.utils import to_categorical

# select features
selected_features = ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'PTS_AVG', 'AST_AVG', 'STL_AVG',
                     'BLK_AVG', 'TOV_AVG', 'REB_AVG', 'HEIGHT', 'WEIGHT']
X = X[selected_features].values.copy()
Y = Y['POSITION'].values.copy()
X_test = X_test[selected_features].values.copy()

# data normalization
for i in range(X.shape[1]):
    X[:,i] -= np.mean(X[:,i])
    X[:,i] /= np.std(X[:,i])
    X_test[:,i] -= np.mean(X_test[:,i])
    X_test[:,i] /= np.std(X_test[:,i])

# split data into training set and validation set
X_val = X[9000:]
Y_val = Y[9000:]
X_train = X[:9000]
Y_train = Y[:9000]

# one-hot encoding
Y_train = to_categorical(Y_train)
Y_val = to_categorical(Y_val)


# build a linear model 
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# logistic regression:0.75
model.add(Dense(5, input_dim = len(selected_features), activation = 'softmax'))

# deep model
# model.add(Dense(100, input_dim = len(selected_features), activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(25, activation = 'relu'))
# model.add(Dense(5, activation = 'softmax'))

# compile model
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics = ['acc'])

# training
model.fit(X_train, Y_train, epochs=300, batch_size=128, validation_data=(X_val,Y_val))

# predict
predicted = np.argmax(model.predict(X_test), axis = 1)
out_file = 'out.csv'
with open(out_file,'w') as f:
    f.write('id,label\n')
    for i in range(len(predicted)):
        f.write(str(i)+','+str(predicted[i])+'\n')