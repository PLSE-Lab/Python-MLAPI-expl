
# collection of scripts from Allstate and beyond, just adding more information
# got 0.585 from gtx 1060 6g in 2h great to see speed of Titan X!
import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import zscore

from sklearn.cross_validation import KFold
#from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.utils.np_utils import to_categorical


data_path = "../input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)

features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]

# count of photos #
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

# count of "features" #
train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)

# count of words present in description column #
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

# convert the created column to datetime object so as to extract more features 
train_df["created"] = pd.to_datetime(train_df["created"])
test_df["created"] = pd.to_datetime(test_df["created"])

# Let us extract some features like year, month, day, hour from date columns #
train_df["created_year"] = train_df["created"].dt.year
test_df["created_year"] = test_df["created"].dt.year
train_df["created_month"] = train_df["created"].dt.month
test_df["created_month"] = test_df["created"].dt.month
train_df["created_day"] = train_df["created"].dt.day
test_df["created_day"] = test_df["created"].dt.day
train_df["created_hour"] = train_df["created"].dt.hour
test_df["created_hour"] = test_df["created"].dt.hour

# adding all these new features to use list #
features_to_use.extend(["num_photos", "num_features", "num_description_words", "created_year", "created_month", "created_day", "listing_id", "created_hour"])

categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)

train_df['features'] = train_df["features"].apply(lambda x: " ".join(x))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(x))
#print(train_df["features"].head())
tfidf = TfidfVectorizer(stop_words='english', max_features=15)
tr_sparse = tfidf.fit_transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])


train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

train_X = train_X.toarray()
test_X = test_X.toarray()
print(train_X.shape)
print(test_X.shape)

# Scale train_X and test_X together
traintest = np.vstack((train_X, test_X))

traintest = preprocessing.StandardScaler().fit_transform(traintest)

train_X = traintest[range(train_X.shape[0])]
test_X = traintest[range(train_X.shape[0], traintest.shape[0])]

## neural net
def nn_model():
    model = Sequential()
    
    model.add(Dense(500, input_dim = train_X.shape[1], init = 'he_normal', activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    model.add(PReLU())
    
    model.add(Dense(50, init = 'he_normal', activation='sigmoid'))
    model.add(BatchNormalization())    
    model.add(Dropout(0.35))
    model.add(PReLU())
	
    model.add(Dense(3, init = 'he_normal', activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')#, metrics=['accuracy'])
    return(model)


target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

train_y = to_categorical(train_y)

do_all = True
## cv-folds
nfolds = 10
if do_all:
	if nfolds>1:
		folds = KFold(int(len(train_y)), n_folds = nfolds, shuffle = True, random_state = 111)
	pred_oob = np.zeros((len(train_y), 3))
	testset = test_X
else:
	folds = KFold(int(len(train_y)*0.8), n_folds = nfolds, shuffle = True, random_state = 111)
	pred_oob = np.zeros((int(len(train_y)*0.8), 3))
	testset = train_X[range(int(len(train_y)*0.8), len(train_y))]
	ytestset = train_y[int(len(train_y)*0.8):(len(train_y))]


## train models
nbags = 5

from time import time
import datetime

pred_test = np.zeros((testset.shape[0], 3))
begintime = time()
count = 0
filepath="weights.best.hdf5"
if nfolds>1:
	for (inTr, inTe) in folds:
	    count += 1
	    
	    xtr = train_X[inTr]
	    ytr = train_y[inTr]
	    xte = train_X[inTe]
	    yte = train_y[inTe]
	    pred = np.zeros((xte.shape[0], 3))
	    for j in range(nbags):
	        print(j)
	        model = nn_model()
	        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
	        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True)
	        
	        model.fit(xtr, ytr, nb_epoch = 1200, batch_size=1000, verbose = 0, validation_data=[xte, yte])
	        # EDIT if to use early_stop and checkpoint do instead
	        # model.fit(xtr, ytr, nb_epoch = 1200, batch_size=1000, verbose = 0, validation_data=[xte, yte], callbacks=[early_stop, checkpoint])
	        # model = nn_model()
	        # model.load_weights(filepath)
	        
	        pred += model.predict_proba(x=xte, verbose=0)
	        
	        pred_test += model.predict_proba(x=testset, verbose=0)
	        
	        print(log_loss(yte,pred/(j+1)))
	        if  not do_all:
	        	print(log_loss(ytestset,pred_test/(j+1+count*nbags)))
	        print(str(datetime.timedelta(seconds=time()-begintime)))
	    pred /= nbags
	    pred_oob[inTe] = pred
	    score = log_loss(yte,pred)
	    print('Fold ', count, '- logloss:', score)
	    if not do_all:
	    	print(log_loss(ytestset, pred_test/(nbags * count)))
else:
    for j in range(nbags):
        print(j)
        model = nn_model()
        model.fit(train_X, train_y, nb_epoch = 1200, batch_size=1000, verbose = 0)
        pred_test += model.predict_proba(x=testset, verbose=0)
        print(str(datetime.timedelta(seconds=time()-begintime)))

if nfolds>1:
	if do_all:
		print('Total - logloss:', log_loss(train_y, pred_oob))
	else:
		print('Total - logloss:', log_loss(train_y[0:int(len(train_y)*0.8)], pred_oob))


if do_all:
	## train predictions
	if nfolds>1:
		out_df = pd.DataFrame(pred_oob)
		out_df.columns = ["high", "medium", "low"]
		out_df["listing_id"] = train_df.listing_id.values
		out_df.to_csv("keras_starter_train.csv", index=False)

	## test predictions
	pred_test /= (nfolds*nbags)
	out_df = pd.DataFrame(pred_test)
	out_df.columns = ["high", "medium", "low"]
	out_df["listing_id"] = test_df.listing_id.values
	out_df.to_csv("keras_starter_test_full.csv", index=False)

