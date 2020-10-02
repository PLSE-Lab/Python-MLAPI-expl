import math
import time
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import ShuffleSplit, KFold

#keras
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, Dropout, merge
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU, SReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD
from keras.models import Model


seed = 101472016
np.random.seed(seed)


hash_digit = (2 ** 5) * (2 ** 10) #32*1024

#keras
dim = int(2 ** 4)
hidden = int(2 ** 8)

default_batch_size = (2 ** 8)
L2_reg = 10 ** -6


#utilities
##########################################################
step = 2 ** 15 #max=755000

def inversed_scl(x):
    return math.exp(float(x)) * step
    #return step * float(x)

def rescale(x):
    return math.log(float(x) / step)
    #return float(x) / step


def mask_columns(columns = {}, mask = {}):
    for str1 in mask:
        #if columns.count(str1) > 0:
        if str1 in columns:
            columns.remove(str1)
    return columns


def label_encode(data, mask = {}):
    
    start_time = time.time()
    
    columns = mask_columns(data.columns.tolist(), mask)
    #encode
    print('One-hot encoding {} features of {} samples'.format(len(columns), len(data)))
    for c in columns:
        data[c] = LabelEncoder().fit_transform(data[c].astype(str).values)
        
    print('Encoded samples: {} minutes\n'.format(round((time.time() - start_time)/60, 2)))
    return data


##########################################################
def create_submission(pred_id, prediction, score, prefix = '', digit = 5):
    now = dt.datetime.now()
    filename = 'submission_r_' + prefix + '_d' + str(dim) + '_s' + str(round(score, digit)) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Make submission:{}\n'.format(filename))    

    target = 'SalePrice'
    #print(prediction)
    prediction = pd.DataFrame(prediction, columns=[target])
    prediction = prediction.apply(lambda x: round(inversed_scl(x), digit), axis=1)
    
    submission = pd.DataFrame(pred_id, columns=['Id'])
    submission[target] = prediction
    submission.to_csv(filename, index=False, header=True)


def load_data(filename):
    
    print('\nLoad ' + filename)
    
    start_time = time.time()

    data = pd.read_csv(filename,
                    converters={'SalePrice': lambda scl: rescale(scl)},
                    dtype={'Id': np.str})

    print('Load in {} samples: {} minutes'.format(len(data), round((time.time() - start_time)/60, 2)))
    
    return data


###########################################################################
digit = 1

path='../input/'

#read train
datafile = 'train.csv'

train = load_data(path + datafile)


train.drop('Id', axis=1, inplace=True)


#read test
datafile = 'test.csv'


#read test
test = load_data(path + datafile)
test_id = test['Id'].values
test = test.drop('Id', axis=1)


#one-hot
mask = {'Id', 'SalePrice'}
data = pd.concat([train, test])
data = label_encode(data, mask)
#data.drop('id', axis=1, inplace=True)
train = data[:train.shape[0]]
test = data[train.shape[0]:]

#########model    
#remove target and id by remove its column name
mask = {'Id', 'SalePrice'}   
columns = mask_columns(data.columns.tolist(), mask)
print('\nApplied {} features'.format(len(columns)))
for i, j in enumerate(columns, start=1):
        print(i, ': ', j)    

print('\ndesign model structure')

flatten_layers = []
inputs = []
    
for c in columns:
    inputs_c = Input(shape=(1,), dtype='int32')
    inputs.append(inputs_c)
    
    num_c = len(np.unique(data[c].values))
    
    dim = int(math.log2(num_c) * 2)
    
    embed_c = Embedding(
            num_c,
            dim,
            dropout=0.1,
            input_length=1
            )(inputs_c)
        
    flatten_c = Flatten()(embed_c)
    flatten_layers.append(flatten_c)
        
#end embedding layer adding
del data
    
#merge in else
flatten_merged = merge(flatten_layers, mode='concat')
    
#deep layers which stacking from the flatten layer
deep = Dropout(0.25)(flatten_merged)  
#deep = Dense(hidden, activation='relu')(deep)
deep = Dense(hidden, activation='sigmoid')(deep)
deep = Dropout(0.25)(deep)  
deep = Dense(hidden, activation='sigmoid')(deep)
#deep = Dense(hidden, activation='softplus')(deep)
deep = Dropout(0.5)(deep)

#set output
#W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)
outputs = Dense(1, activation='softplus', 
        W_regularizer=l2(L2_reg), activity_regularizer=activity_l2(L2_reg)
    )(deep)

model = Model(input=inputs, output=outputs)
model.compile(
    loss='mean_squared_error', 
    optimizer='adam',
    #optimizer='sgd',
    #optimizer='RMSprop',
    )
             
#print('model complied', model.summary(), '\n')    


##########prepare data
train_y = train['SalePrice'].values
train = train.drop(['SalePrice'], axis=1)


#test
X_t = test[columns].values
X_t = [X_t[:,i] for i in range(X_t.shape[1])]
del test

score = 0
count = 0
X = train.values#np.array
kf = KFold(len(train_y), n_folds=5, shuffle=False)

for kf_tr, kf_va in kf:
    
    y_train, y_valid = train_y[kf_tr], train_y[kf_va]
    
    real_y_valid = []
    for i in y_valid:
        real_y_valid.append(inversed_scl(i))
    
    X_train = X[kf_tr]
    X_valid = X[kf_va]
    X_valid = [X_valid[:,i] for i in range(X_valid.shape[1])]
    
    ss = ShuffleSplit(len(y_train), n_iter=7, test_size=0.2)
    for ind_tr, ind_va in ss:
        
        y1_train = train_y[ind_tr]
        X1_train = X_train[ind_tr]
        X1_train = [X1_train[:,i] for i in range(X1_train.shape[1])]
            
        model.fit(
            X1_train, y1_train,
            batch_size=default_batch_size, 
            nb_epoch=100,
            verbose=2, shuffle=True,
            validation_data=[X_valid, y_valid],
            #callbacks = [model_checkpoint, auc_callback,],
        )
        
        y_preds = model.predict(X_valid, batch_size=default_batch_size)
        
        real_y_preds = []
        for i in y_preds:
            real_y_preds.append(inversed_scl(i))
        
        count += 1
        score = math.sqrt(mean_squared_error(real_y_valid, real_y_preds))
        print('\niter {}: validation = {}\n'.format(count, round(score, digit)))
    #break

        
print('predict test')
outcome = model.predict(X_t, batch_size=default_batch_size)
create_submission(test_id, outcome, score, 'test', digit)



