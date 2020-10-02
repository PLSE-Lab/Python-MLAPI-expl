import math
import itertools
import time
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import ShuffleSplit, KFold

#keras
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, Dropout, Reshape, Merge
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU, SReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD
from keras.models import Model


seed = 101472016
np.random.seed(seed)


hash_digit = (2 ** 5) * (2 ** 10) #32*1024

#keras
dim = int(2 ** 3)
hidden = int(2 ** 8)

default_batch_size = (2 ** 7)
L2_reg = 10 ** -6


#utilities
##########################################################
#step = 2 ** 19 #max=755000
#step = 163000 * 2 ##set training median at 0.5
step = 34900 #min

def inversed_scl(x):
    return  step * math.pow(10.0, float(x))
    #return step * float(x)

def rescale(x):
    return math.log10(float(x) / step)
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


def feature_reduce(data):
    
    start_time = time.time()
    
    data['sMS'] = ''
    data['sLot'] = ''
    data['StAlley'] = ''
    data['sLand'] = ''
    data['sBldg'] = ''
    data['sCondition'] = ''
    data['sOverall'] = ''
    data['sRoof'] = ''
    data['sExter'] = ''
    data['sBsmt'] = ''
    data['sBath'] = ''
    data['Utilities_more'] = ''
    data['sGarage'] = ''
    data['sMisc'] = ''

    data.fillna('na', inplace=True)

    data['YearDiff'] = data['YearRemodAdd'] - data['YearBuilt']
    data.drop('YearBuilt', axis=1, inplace=True)

    for c in data.columns.tolist():
        if str(c).startswith('MS'):
            data['sMS'] += data[c].astype(str) + '_'
            data.drop(c, axis=1, inplace=True)
        
        elif ((str(c).find('Area') > -1) or 
                (str(c).find('SF') > -1) or (str(c).find('Porch') > -1)):
            if data[c].dtypes != str:
                data[c].replace('na', 0, inplace=True)
                data[c] = data[c].apply(lambda scl: int(math.sqrt(float(scl) + 1.0)))
        
        elif str(c).startswith('Lot'):
            if c == 'LotFrontage':
                data[c].replace('na', 0, inplace=True)
                data[c] = data[c].apply(lambda scl: int(math.sqrt(float(scl) + 1.0)))
            else:
                data['sLot'] += data[c].astype(str) + '_'
                data.drop(c, axis=1, inplace=True)

        elif c == 'Street' or c == 'Alley':
            data['StAlley'] += data[c].astype(str) + '_'
            data.drop(c, axis=1, inplace=True)

        elif str(c).startswith('Land'):
            data['sLand'] += data[c].astype(str) + '_'
            data.drop(c, axis=1, inplace=True)
            
        elif str(c).find('Condition') > -1:
            data['sCondition'] += data[c].astype(str) + '_'
            data.drop(c, axis=1, inplace=True)

        elif c == 'BldgType' or c == 'HouseStyle':
            data['sBldg'] += data[c].astype(str) + '_'
            data.drop(c, axis=1, inplace=True)
            
        elif str(c).startswith('Overall'):
            data['sOverall'] += data[c].astype(str) + '_'
            data.drop(c, axis=1, inplace=True)
            
        elif str(c).startswith('Roof'):
            data['sRoof'] += data[c].astype(str) + '_'
            data.drop(c, axis=1, inplace=True)
            
        elif str(c).startswith('Exter'):
            data['sExter'] += data[c].astype(str) + '_'
            data.drop(c, axis=1, inplace=True)
            
        elif (str(c).find('Bath') > -1):#include Bsmt Bath
            data['sBath'] += data[c].astype(str) + '_'
            data.drop(c, axis=1, inplace=True)
            
        elif str(c).startswith('Bsmt') and (str(c).find('SF') == -1):
            data['sBsmt'] += data[c].astype(str) + '_'
            data.drop(c, axis=1, inplace=True)

        elif (str(c).startswith('Heating') or 
                c == 'Utilities' or c == 'CentralAir' or c == 'Electrical'):
            data['Utilities_more'] += data[c].astype(str) + '_'
            data.drop(c, axis=1, inplace=True)
            
        elif (str(c).startswith('GarageT') or str(c).startswith('GarageCo') or
                str(c).startswith('GarageQ') or str(c).startswith('GarageF') or 
                c == 'PavedDrive'):
            data['sGarage'] += data[c].astype(str) + '_'
            data.drop(c, axis=1, inplace=True)
            
        elif str(c).startswith('Misc'):
            data['sMisc'] += data[c].astype(str) + '_'
            data.drop(c, axis=1, inplace=True)

    print('Reduse feature: {} minutes\n'.format(round((time.time() - start_time)/60, 2)))
    return data



##########################################################
def create_submission(pred_id, prediction, score, prefix = '', digit = 5):
    now = dt.datetime.now()
    filename = 'submission_' + prefix + '_d' + str(dim) + '_h' + str(hidden) + '_s' + str(round(score, digit)) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
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

print('param: dim {}, hidden {}, batch size {}'.format(
        dim, hidden, default_batch_size))

path='../input/'


#read train
datafile = 'train.csv'
train = load_data(path + datafile)
train.drop('Id', axis=1, inplace=True)

#read test
datafile = 'test.csv'
test = load_data(path + datafile)
test_id = test['Id'].values
test = test.drop('Id', axis=1)


#one-hot
mask = {'Id', 'SalePrice'}
data = pd.concat([train, test])
data = feature_reduce(data)
data = label_encode(data, mask)
#data.drop('id', axis=1, inplace=True)
train = data[:train.shape[0]]
test = data[train.shape[0]:]

#########model    
#remove target and id by remove its column name
mask = {'Id', 'SalePrice'}   
columns = mask_columns(data.columns.tolist(), mask)
print('\nApplied {} features: {}'.format(len(columns), sorted(columns)))
print('\ndesign model structure')

inputs = []
embed_layers = []
dot_layers = []
    
for c in columns:
    inputs_c = Input(shape=(1,), dtype='int32')
    inputs.append(inputs_c)
    
    num_c = len(np.unique(data[c].values))
    
    vdim = int(math.log10(num_c) * dim)
    
    embed_c = Embedding(
            num_c,
            #dim,
            vdim,
            dropout=0.1,
            input_length=1
            )(inputs_c)
        
    embed_layers.append(embed_c)
        
#end embedding layer adding
del data

#dot layer
comb_dot = [list(x) for x in itertools.combinations(embed_layers, 2)]

for i in comb_dot:
    dot_embed = Merge(mode='dot', dot_axes=1)(i)
    flatten_dot = Flatten()(dot_embed)
    dot_layers.append(flatten_dot)

concat_dot = Merge(mode='concat')(dot_layers)
concat_dot = Dropout(0.25)(concat_dot)
dense_dot = Dense(hidden, activation='sigmoid')(concat_dot)

#normal layer
concat_embed = Merge(mode='concat')(embed_layers)
flatten_concat = Flatten()(concat_embed)
dense_c = Dense(hidden, activation='sigmoid')(flatten_concat)
#dense_c = Dropout(0.25)(dense_c)

#merge dot and others
merged_ds = Merge(mode='concat')([dense_c, dense_dot])
    
#deep layers which stacking from the flatten layer
deep = Dropout(0.25)(merged_ds)  
deep = Dense(hidden, activation='sigmoid')(deep)
deep = Dropout(0.5)(deep) 
#deep = Dense(hidden, activation='softplus')(deep)


#set output
#W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)
outputs = Dense(1, activation='softplus', 
        W_regularizer=l2(L2_reg), activity_regularizer=activity_l2(L2_reg)
    )(deep)

model = Model(input=inputs, output=outputs)
model.compile(
    loss='mean_squared_logarithmic_error', 
    optimizer='adam',
    #optimizer='sgd',
    #optimizer='RMSprop',
    )
             
#print('model complied', model.summary(), '\n') 
ckp_va = EarlyStopping(monitor='val_loss', patience=10)
ckp_tr = EarlyStopping(monitor='loss', patience=10)

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
kf = KFold(len(train_y), n_folds=3, shuffle=True)

for kf_tr, kf_va in kf:
    y_train, y0_valid = train_y[kf_tr], train_y[kf_va]
    
    X_train = X[kf_tr]
    X0_valid = X[kf_va]

    ss = ShuffleSplit(len(y_train), n_iter=3, test_size=0.1)
    for ind_tr, ind_va in ss:
        
        iter_start_ckp = time.time()
        #training
        y1_train = train_y[ind_tr]
        X1_train = X_train[ind_tr]
        X1_train = [X1_train[:,i] for i in range(X1_train.shape[1])]
        
        #valid
        y1_valid = train_y[ind_va]
        y_valid = np.concatenate([y0_valid, y1_valid])
        X1_valid = X_train[ind_va]
        X_valid = np.concatenate([X0_valid, X1_valid])
        X_valid = [X_valid[:,i] for i in range(X_valid.shape[1])]
        
        model.fit(
            X1_train, y1_train,
            batch_size=default_batch_size, 
            nb_epoch=50,
            verbose=2, shuffle=True,
            validation_data=[X_valid, y_valid],
            callbacks=[ckp_tr, ckp_va],
        )
        
        y_preds = model.predict(X_valid, batch_size=default_batch_size)
        
        real_y_valid = []
        for i in y_valid:
            real_y_valid.append(inversed_scl(i))
            
        real_y_preds = []
        for i in y_preds:
            real_y_preds.append(inversed_scl(i))
        
        count += 1
        score = math.sqrt(mean_squared_error(real_y_valid, real_y_preds))
        iter_end_ckp = time.time()
        print('\niter {}: validation = {}, {} s\n'.format(
            count, round(score, digit), round(iter_end_ckp - iter_start_ckp, 2)))
    
    #break

print('predict test')
outcome = model.predict(X_t, batch_size=default_batch_size)
create_submission(test_id, outcome, score, 'test', digit)
