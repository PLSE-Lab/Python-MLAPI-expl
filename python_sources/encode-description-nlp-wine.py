#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import RegexpTokenizer
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc, mean_absolute_error
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from xgboost import XGBRegressor


# In[ ]:


test = pd.read_csv('/kaggle/input/ammi-ghana-bootcamp-kaggle-competition/test.csv')
# Convert the data to a Pandas data frame
data = pd.read_csv('/kaggle/input/ammi-ghana-bootcamp-kaggle-competition/train.csv')

# Shuffle the data
data = data.sample(frac=1)

# Print the first 5 rows
data.head()


# In[ ]:


import category_encoders as ce 
import re

def combine_train_test_set(train, test):
    train['train'] = 1
    test['train'] = 0
    train['test'] = 0
    test['test'] = 1
    return pd.concat([train, test[train.columns.tolist()]])

def add_text_features(df, col):
    df = df.copy()
    # Count number of \n
    df["ant_slash_n"] = df[col].apply(lambda x: count_regexp_occ(r"\n", x))
    # Get length in words and characters
    df[col + "_word_len"] = df[col].apply(lambda x: len(x.split()))
    df[col+"_char_len"] = df[col].apply(lambda x: len(x))
    # Get the new length in words and characters
    df[col+"_word_len"] = df[col].apply(lambda x: len(x.split()))
    df[col+"_char_len"] = df[col].apply(lambda x: len(x))
    # Number of different characters used in a comment
    # Using the f word only will reduce the number of letters required in the comment
    df["clean_chars"] = df[col].apply(lambda x: len(set(x)))
    df["clean_chars_ratio"] = df[col].apply(lambda x: len(set(x))) / df[col].apply(
        lambda x: 1 + min(99, len(x)))
    return df
    
def split_train_test_set(combine):
    test = combine[combine['test'] == 1].drop(columns=['test', 'train'])
    train = combine[combine['train'] == 1].drop(columns=['test', 'train'])
    return train, test

def make_lower_case(text):
    return text.lower()

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

def clean_text_data(data, cols):
    for col in cols:
        data[col].fillna('#na', inplace=True)
        data[col] = data[col].str.replace(r'\d+','')
        data[col] = data.description.apply(func=remove_punctuation)
        data[col] = data.description.apply(func=make_lower_case)
    
def remove_null_rows(data, cols):
    for c in cols:
        data = data[pd.notnull(data[c])]
    return data

def plot_len(data, col, bins=50, title='Token per sentence', ylabel='# samples', xlabel='Len (number of token)'):
    plt.hist(data[col].apply(len).values, bins=50)
    plt.title(title)
    plt.xlabel(ylabel)
    plt.ylabel(xlabel)
    plt.show()
    

def transform_text(data, col, n_components=5):
    #Train tfidf and svd
    tf = TfidfVectorizer(analyzer='word', min_df=10, ngram_range=(1, 2), stop_words='english')
    svd = TruncatedSVD(n_components=n_components)

    #Fit tfidf and svd, and transform training data
    tfidf_matrix = tf.fit_transform(data[col])
    lsa_features = pd.DataFrame(svd.fit_transform(tfidf_matrix))

    #Creat meaningful column names
    collist = map(str, range(0, 5))
    collist = ["latent_" + col + '_' + s for s in collist]
    lsa_features.columns = collist
    lsa_features = lsa_features.set_index(data.index)
    return lsa_features

def map_to_pd_categorical(data, cols, fill_na='#nan'):
    for col in cols:
        data[col] = pd.Categorical(data[col].fillna(fill_na))
    return data

def cat_to_int(data, cols, fill_na='#nan'):
    for col in cols:
        if data[col].dtype.name == 'object':
            data[col] = pd.Categorical(data[col].fillna(fill_na))
        elif data[col].dtype.name == 'category': 
            data[col] = data[col].cat.codes
        else:
            print('Not categorical or object : ' + col + ' ', data[col].dtype)
    return data
        
def rmse_error(predictions, y_test):
    return np.sqrt(mean_squared_error(predictions, y_test))

def save_submissions(submissions, test, name='submissions.csv'):
    if 'id' in test.columns:
        test.set_index('id', inplace=True)
    submissions = pd.DataFrame(submissions, columns=['price'], index=test.index)
    submissions.to_csv(name)
    return submissions

def limit_categorical_by_count(data, cols=[], thres=500, fill=np.nan):
    for c in cols:
        value_counts = data[c].value_counts()
        to_remove = value_counts[value_counts <= thres].index
        if to_remove.size > 0:
            data.replace(to_remove, fill, inplace=True)
        data = data[pd.notnull(data[c])]
    return data


def print_str_info(X, features):
    for feat in features:
        print('Unique {} : {}'.format(feat, X[feat].nunique()))
        print('Data with No {} values : {}'.format(feat, X[feat].isna().sum()))
        print('Max length : ', X[feat].apply(lambda x: len(str(x).split())).max())

def train_model(model, X_train, y_train, X_test, y_test):
    model.fit( X_train, y_train)
    predictions = model.predict(X_test)
    print("RMSE on test data: "+ str(rmse_error(predictions, y_test)))
    return model

def target_encoding(data, cols, fit_data, y=None,y_label=None, fill_na='#nan'):
    # Target with default parameters
    ce_target = ce.TargetEncoder(cols = cols)
    y = fit_data[y_label] if y is None  else y
    print(y.shape, fit_data.shape)
    ce_target.fit(fit_data, y)
    # Must pass the series for y in v1.2.8
    return ce_target.transform(data)
    
def target_encoding_smoothing(data, cols, fit_data, y=None,y_label=None, fill_na='#nan'):
    # Target with smoothing higher
    ce_target = ce.TargetEncoder(cols = cols, smoothing = 10)
    y = fit_data[y_label] if y is None  else y
    print(y.shape, fit_data.shape)
    ce_target.fit(fit_data, y)
    # Must pass the series for y in v1.2.8
    return ce_target.transform(data)

def add_length(data, col, fill_na='#nan'):
    # Target with default parameters
    data[col].fillna(value='#nan', inplace=True)
    data[col+'_len'] = data[col].apply(lambda s: len(s.split()))
    
# Let turn our string to tokens
def text_to_seq(data, col, max_features = 6000):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(data[col])
    list_tokenized = tokenizer.texts_to_sequences(data['title_desc'])
    return pad_sequences(list_tokenized, maxlen=maxlen), tokenizer

def count_regexp_occ(regexp="", text=None):
    """ Simple way to get the number of occurence of a regex"""
    return len(re.findall(regexp, text))


# In[ ]:


data = remove_null_rows(data, ['country', 'price'])


# In[ ]:


combine = combine_train_test_set(data, test)
combine.head()


# In[ ]:


# Do some preprocessing to limit the # of wine varities in the dataset
# Clean it from null values
combine = limit_categorical_by_count(combine, cols=['variety'], fill='nan')
combine.head()


# In[ ]:


def plot_keras():
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
# plot_keras()


# In[ ]:


combine.head()


# Smoothing with target encoding

# In[ ]:


categorical_features = ['country','designation', 'province','region_1','region_2', 'taster_name','taster_twitter_handle', 'variety','winery']
encoded = target_encoding_smoothing(combine,categorical_features, combine[combine['train'] == 1],y_label='price')
add_length(encoded, 'description')
data, test = split_train_test_set(encoded)
data.head()


# In[ ]:


X = data.drop(columns=['title', 'description', 'id', 'price'])
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


#PROCESS TEXT: RAW
print("Text to seq process...")
print("   Fitting tokenizer...")
from keras.preprocessing.text import Tokenizer
encoded.title=encoded.title.astype(str)
encoded.description=encoded.description.astype(str)
clean_text_data(encoded[['title', 'description']], cols=['title', 'description'])
raw_text = np.hstack([encoded.title.str.lower(), encoded.description.str.lower()])
print(raw_text[0])

tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)
print("   Transforming text to seq...")
encoded["seq_title"] = tok_raw.texts_to_sequences(encoded.title.str.lower())
encoded["seq_description"] = tok_raw.texts_to_sequences(encoded.description.str.lower())


# In[ ]:


encoded.head()


# In[ ]:


encoded.seq_title.apply(lambda x: len(x)).hist()


# In[ ]:


encoded.seq_description.apply(lambda x: len(x)).hist()


# In[ ]:


#EMBEDDINGS MAX VALUE
#Base on the histograms, we select the next lengths
MAX_TITLE_SEQ = 13
MAX_DESC_SEQ = 70
MAX_TEXT = np.max(encoded.seq_description.max())+1


# In[ ]:


# #SCALE target variable
# from sklearn.preprocessing import MinMaxScaler
# y_train = np.log(encoded[encoded['train'] == 1].price+1)
# target_scaler = MinMaxScaler(feature_range=(-1, 1))
# y_train = target_scaler.fit_transform(y_train.values.reshape(-1,1))
# pd.DataFrame(y_train).hist()


# In[ ]:


#EXTRACT DEVELOPTMENT TEST
train, test = split_train_test_set(encoded.drop(columns=['description', 'title']))
train.head()


# In[ ]:


#KERAS DATA DEFINITION
from keras.preprocessing.sequence import pad_sequences
X = train.drop(columns=['id', 'price'])
y = train['price']
def get_seq_data(dataset):
    return {
    'title':pad_sequences(dataset.seq_title, maxlen=MAX_TITLE_SEQ),
    'description': pad_sequences(dataset.seq_description, maxlen=MAX_DESC_SEQ)
#     ,'num_vars': np.array(get_num_data(dataset))
}

def get_num_data(data):
    return data[categorical_features+['points','description_len']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, train_size=0.9)

X_train = get_seq_data(X_train)
X_test = get_seq_data(X_test)


# In[ ]:


#KERAS MODEL DEFINITION
from keras.layers import Input, Dropout, Dense,Bidirectional, LSTM, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten, BatchNormalization, GlobalMaxPool1D, Conv1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K

def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

def get_model():
    #params
    dr_r = 0.05
    dr_d = 0.05
    
    #Inputs
    title = Input(shape=[X_train["title"].shape[1]], name="title")
    description = Input(shape=[X_train["description"].shape[1]], name="description")
#     num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    
    #Embeddings layers
    emb_size = 55
    emb_title = Embedding(MAX_TEXT, 32)(title)
    emb_item_desc = Embedding(MAX_TEXT, 60)(description)
    
    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(emb_item_desc)
    lstm_layer = Dropout(0.4)(lstm_layer)
    lstm_layer = LSTM(32)(lstm_layer)
    lstm_layer_2 = Bidirectional(LSTM(32, return_sequences=True))(emb_title)
    lstm_layer_2 = Dropout(0.4)(lstm_layer_2)
    lstm_layer_2 = LSTM(32)(lstm_layer_2)
    
    #main layer
    main_l = concatenate([ lstm_layer , lstm_layer_2 ])
    
    main_l = Dropout(dr_r) (Dense(512,activation='relu') (main_l))
    main_l = Dropout(dr_r) (Dense(256,activation='relu') (main_l))
    main_l = Dropout(dr_d) (Dense(16,activation='relu') (main_l))
    
    #output
    output = Dense(1, activation="linear") (main_l)
    
    #model
    model = Model([description, title], output)
    model.compile(loss="mse", optimizer="adam", metrics=["mae",'acc', root_mean_squared_error])
    
    return model


# In[ ]:


model = get_model()
model.summary()


# In[ ]:


#FITTING THE MODEL
BATCH_SIZE = 512
epochs = 10

model = get_model()
history = model.fit(X_train, y_train, epochs=epochs, batch_size=BATCH_SIZE , validation_data=[X_test, y_test], verbose=1)


# In[ ]:


predictions = model.predict(get_seq_data(X))
print("RMSE on test data: "+ str(rmse_error(predictions, y)))
plot_keras()


# In[ ]:





# In[ ]:


dec_encode = model.predict(get_seq_data(encoded))


# In[ ]:





# In[ ]:


encoded['description_encoded'] = pd.DataFrame(dec_encode, columns=['description_encoded'], index=encoded.index)['description_encoded']


# In[ ]:


encoded.head()


# In[ ]:


import lightgbm as lgb


# In[ ]:





# In[ ]:


encoded_data = encoded.drop(columns=['seq_title','seq_description'])
data, test = split_train_test_set(encoded_data)
data.head()


# In[ ]:


test_set = test.drop(columns=['title', 'description', 'price'])
test_set = test_set.fillna(-1)
test_set.head()


# In[ ]:


X = data.drop(columns=['title', 'description', 'price', 'description_len'])
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)


# In[ ]:


evals_result = {}  # to record eval results for plotting

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'num_leaves': 35,
    'max_depth':7,
    'metric': 'rmse',
    'verbose': 0,
    'bagging_fraction': 0.8
}

gbm_deeplearning = lgb.train(params,
                lgb_train,
                num_boost_round=3000,
                valid_sets=[lgb_train, lgb_test],
                feature_name=X_train.columns.tolist(),
#                 categorical_feature=categorical_features,
                evals_result=evals_result,
                verbose_eval=10)


# In[ ]:


predictions_gbm_deeplearning = gbm_deeplearning.predict(X_test)
predictions_gbm_deep_test = pd.DataFrame(predictions_gbm_deeplearning, columns=['price'], index=y_test.index)
print(rmse_error(predictions_gbm_deeplearning, y_test))


# In[ ]:


predictions_gbm_deeplearning = gbm_deeplearning.predict(test_set.drop(columns=['description_len']))
predictions_gbm_deeplearning = pd.DataFrame(predictions_gbm_deeplearning, columns=['price'], index=test_set.index)
predictions_gbm_deeplearning.to_csv('predictions_gbm_deeplearning_13.csv')


# In[ ]:


X = data.drop(columns=['title', 'description', 'price'])
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:



xgb_deep_learning = XGBRegressor(max_depth=5, n_estimators=1000, min_child_weight=5)
train_model(xgb_deep_learning,  X_train, y_train, X_test, y_test)


# In[ ]:


predictions_xgb_deep_learning = xgb_deep_learning.predict(X_test)
predictions_xgb_deep_test = pd.DataFrame(predictions_xgb_deep_learning, columns=['price'], index=y_test.index)
print(rmse_error(predictions_xgb_deep_learning, y_test))


# In[ ]:


predictions_xgb_deep_learning = xgb_deep_learning.predict(test_set)
predictions_xgb_deep_learning = pd.DataFrame(predictions_xgb_deep_learning, columns=['price'], index=test_set.index)
predictions_xgb_deep_learning.to_csv('predictions_xgb_deep_learning_11.csv')


# In[ ]:


predictions_xgb_deep_learning.to_csv('predictions_xgb_deep_learning_11.csv')


# In[ ]:


avg = (predictions_xgb_deep_learning + predictions_gbm_deeplearning)/2
avg = pd.DataFrame(avg, columns=['price'], index=test_set.index)
avg.to_csv('hybrid_deep_learning_11.csv')


# In[ ]:


avg = (predictions_xgb_deep_test + predictions_gbm_deep_test)/2
print(rmse_error(avg, y_test))

