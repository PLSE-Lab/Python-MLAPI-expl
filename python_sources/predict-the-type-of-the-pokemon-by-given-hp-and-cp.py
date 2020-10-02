from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras import backend as K
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import random
import math

# Dictionary object which can help change string of type to numeric index
type_string_2_int = dict()
type_int_2_string = dict()

# scaler
scaler = MinMaxScaler()

class Net(object):
    def __init__(self):
        input_layer = Input((2,))

        # Define auto-encoder structure
        self.encode_layer = Dense(units=128, activation='elu')(input_layer)
        decode_layer = Dense(units=2, activation='elu')(self.encode_layer)
        self.encode_model = Model(inputs=input_layer, outputs=decode_layer)

        # Define DNN structure
        self.network = Dense(units=128, activation='elu')(self.encode_layer)
        self.network = Dropout(0.5)(self.network)
        self.network = Dense(units=128, activation='elu')(self.network)
        self.network = Dropout(0.5)(self.network)
        self.network = Dense(units=64, activation='elu')(self.network)
        self.network = Dense(units=64, activation='elu')(self.network)
        self.network = Dense(units=64, activation='elu')(self.network)
        self.network = Dense(units=64, activation='elu')(self.network)
        self.output = Dense(units=17, activation='sigmoid')(self.network)
        self.dnn_model = Model(inputs=input_layer, outputs=self.output)        

    def fit(self, x, y, epoch=1000, batch_size=32):
        # Train auto-encoder first
        self.encode_model.compile(
            loss='mse',
            optimizer='adam'
        )
        self.encode_model.fit(x, x, epochs=epoch, batch_size=200)

        # Train classification model
        for i in range(len(self.encode_model.layers)):
            self.dnn_model.layers[i].trainable = False
        self.dnn_model.compile(
            loss='mse',
            optimizer='adam',
            metrics=['accuracy']
        )
        self.dnn_model.fit(x, y, epochs=epoch, batch_size=batch_size)

    def predict(self, x):
        return self.dnn_model.predict(x)

    def close(self):
        K.clear_session()

def load(file_name='../input/pokemonGO.csv'):
    global type_int_2_string
    global type_string_2_int
    global scaler

    # Drop useless feature
    df = pd.read_csv(file_name)
    df = df.drop(['Name', 'Pokemon No.', 'Image URL'], axis=1)
    df.columns = ['type1', 'type2', 'cp', 'hp']

    # Swap column
    column_list = list(df)
    column_list[0], column_list[2] = column_list[2], column_list[0]
    column_list[1], column_list[3] = column_list[3], column_list[1]
    df = df.ix[:, column_list]

    # Build mapping
    _counter = 0
    for i in range(len(df)):
        if df['type1'][i] not in type_string_2_int:
            type_string_2_int[df['type1'][i]] = _counter
            _counter += 1
        if type(df['type2'][i]) == str:
            if df['type2'][i] not in type_string_2_int:
                type_string_2_int[df['type2'][i]] = _counter
                _counter += 1
    type_int_2_string = { type_string_2_int[i]: i for i in type_string_2_int }

    # Change categorical data to numeric index
    for i in range(len(df)):
        df.set_value(i, 'type1', type_string_2_int[df['type1'][i]])
        if type(df['type2'][i]) == str:
            df.set_value(i, 'type2', type_string_2_int[df['type2'][i]])    
    return df

def mergeMultipleTypes(df):
    df = pd.melt(df, id_vars=['cp', 'hp'], var_name='type')
    df = df.dropna()
    df = df.reset_index()
    df = df.drop(['index', 'type'], axis=1)
    return df

def splitData(df, shuffle=True, split_rate=0.005):
    df = generateData(df, 10)
    if shuffle == True:   
        df = shuffleDataFrame(df)
    y = df.get_values().T[2:].T
    x = df.get_values().T[:2].T
    x = scaler.fit_transform(x)
    return train_test_split(x, y, test_size=0.005)

def generateData(df, times=1):
    origin_len = len(df)
    columns_list = df.columns
    print(columns_list)
    for i in range(times):
        for j in range(origin_len):
            random_seed = 1 + (random.random() - 1) / 10
            _new_row = []
            _new_row.append(df['cp'][j] * random_seed)
            _new_row.append(df['hp'][j] * random_seed)
            if len(columns_list) == 4:
                _new_row.append(df['type1'][j])
                _new_row.append(df['type2'][j])
            if len(columns_list) == 3:
                _new_row.append(df['value'][j])
            df = df.append(pd.DataFrame([_new_row], columns=columns_list))
        df = df.reset_index().drop(['index'], axis=1)
    return df

def shuffleDataFrame(data_frame):
    result_table = pd.DataFrame(columns=data_frame.columns)
    rest_table = data_frame
    for i in range(len(data_frame)):
        sample_row = rest_table.sample(1)
        rest_table = rest_table.drop(sample_row.index[0])
        result_table = result_table.append(sample_row)
    return result_table

def matchRate(tag_arr, predict_arr):
    _count = 0
    if len(np.shape(tag_arr)) == 2:
        for i in range(len(tag_arr)):
            if predict_arr[i] == tag_arr[i][0] or \
                predict_arr[i] == tag_arr[i][1]:
                _count += 1
    else:
        for i in range(len(tag_arr)):
            if predict_arr[i] == tag_arr[i]:
                _count += 1
    return _count / len(tag_arr)

def oneHotEncode(arr):
    res = np.zeros([len(arr), np.nanmax(arr) + 1])
    for i in range(len(res)):
        res[i][arr[i][0]] = 1
        if math.isnan(arr[i][1]) == False:
            res[i][arr[i][1]] = 1
    return res

def oneHotDecode(arr):
    return np.argmax(arr, axis=1)

if __name__ == '__main__':
    # -----------------------
    # Random forest model
    # -----------------------

    # Load data
    data_frame = load()
    merge_frame = mergeMultipleTypes(data_frame)
    train_x, test_x, train_y, test_y = splitData(merge_frame)

    # Train
    clf = RandomForestClassifier(verbose=0)
    clf.fit(train_x, list(np.reshape(train_y, [-1])))

    # Test
    print('<< predict >>')
    predict_output = clf.predict(test_x)
    print('match rate: ', matchRate(np.reshape(test_y, [-1]), predict_output))


    # -----------------------
    # DNN model
    # -----------------------

    # Load data
    train_x, test_x, train_y, test_y = splitData(data_frame)

    # Train
    clf = Net()
    clf.fit(train_x, oneHotEncode(train_y))

    # Test
    print('<< predict >>')
    predict_output = oneHotDecode(clf.predict(test_x))
    print('match rate: ', matchRate(test_y, predict_output))
    clf.close()