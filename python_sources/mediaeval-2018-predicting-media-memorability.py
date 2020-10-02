#!/usr/bin/env python
# coding: utf-8

# # Predicting Memorability

# ### Spearman correlation

# In[ ]:


def Get_score(Y_pred,Y_true):
    '''Calculate the Spearmann"s correlation coefficient'''
    Y_pred = np.squeeze(Y_pred)
    Y_true = np.squeeze(Y_true)
    if Y_pred.shape != Y_true.shape:
        print('Input shapes don\'t match!')
    else:
        if len(Y_pred.shape) == 1:
            Res = pd.DataFrame({'Y_true':Y_true,'Y_pred':Y_pred})
            score_mat = Res[['Y_true','Y_pred']].corr(method='spearman',min_periods=1)
            print('The Spearman\'s correlation coefficient is: %.3f' % score_mat.iloc[1][0])
        else:
            for ii in range(Y_pred.shape[1]):
                Get_score(Y_pred[:,ii],Y_true[:,ii])


# In[ ]:


get_ipython().system('pip install pyprind')
get_ipython().system('pip install keras')
get_ipython().system('pip install tensorflow')


# ## Predicting video memorability using captions

# In[ ]:


import pandas as pd
from keras import Sequential
from keras import layers
from keras import regularizers
import numpy as np
from string import punctuation
import pyprind
from collections import Counter
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


# In[ ]:


# for reproducability
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)


# ## 1. Loading the captions and the memorability scores

# In[ ]:


# load labels and captions
def read_caps(fname):
    """Load the captions into a dataframe"""
    vn = []
    cap = []
    df = pd.DataFrame();
    with open(fname) as f:
        for line in f:
            pairs = line.split()
            vn.append(pairs[0])
            cap.append(pairs[1])
        df['video']=vn
        df['caption']=cap
    return df

# load the captions
df_cap=read_caps('../input/data/data/dev-set_video-captions.txt')

# load the ground truth values
labels=pd.read_csv('../input/data/data/dev-set_ground-truth.csv')


# In[ ]:


labels.head()


# ## 2. Cleaning / Pre-Processing

# In[ ]:


from nltk.stem import WordNetLemmatizer
from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re


# ### 2.1 Stripping special characters 

# In[ ]:


df = df_cap.copy()
import re
def strip_character(dataCol):
    r = re.compile(r'[^a-zA-Z]')
    return r.sub(' ', str(dataCol))

df['caption'] = df['caption'].apply(strip_character)


# In[ ]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet')


# ### 2.2 Removing Stopwords

# In[ ]:


stop = stopwords.words('english') 


# In[ ]:


df['caption'] = df['caption'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))
df['caption'].head()


# ### 2.3 Lemmatization

# In[ ]:


df['caption'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in df['caption']]
df['caption'].head()


# ### 2.4 Extracting unique words count

# In[ ]:


counts = Counter()
for i, cap in enumerate(df['caption']):
    counts.update(cap.split())


# In[ ]:


df.caption.values


# ### 2.5 Trying an n-Gram approach- Using TF-IDF

# In[ ]:


vect = TfidfVectorizer(ngram_range = (1,4)).fit(df.caption)
vect_transformed_X_train = vect.transform(df.caption)
len_token = len(vect.get_feature_names())


# In[ ]:


len_token


# ### 2.6 Maping each unique word to an integer (one-hot encoding)

# In[ ]:


# build the word index
len_token = len(counts)
tokenizer = Tokenizer(num_words=len_token)


# In[ ]:


tokenizer.fit_on_texts(list(vect.get_feature_names())) #fit a list of captions to the tokenizer
#the tokenizer vectorizes a text corpus, by turning each text into either a sequence of integers 


# In[ ]:


one_hot_res = tokenizer.texts_to_matrix(list(df.caption.values),mode='binary')
#sequences = tokenizer.texts_to_sequences(list(df.caption.values))


# In[ ]:


len(one_hot_res)


# ## 3.  Predicting video memorability using captions

# In[ ]:


one_hot_res.shape


# ### 3.1 Recurrent Neural Network model

# In[ ]:


Y = labels[['short-term_memorability','long-term_memorability']].values
X = one_hot_res;
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

# add dropout
# add regularizers

model = Sequential()
model.add(layers.Dense(200,activation='relu',kernel_regularizer=None,input_shape=(len_token,)))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(2,activation='sigmoid'))


          
# compile the model 
model.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])

# training the model 
history = model.fit(X_train,Y_train,epochs=20,validation_data=(X_test,Y_test))

# visualizing the model
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()


# In[ ]:


predictions = model.predict(X_test)
print(predictions.shape)


# In[ ]:


Get_score(predictions, Y_test)


# ### Saving the model

# In[ ]:


model.save('CaptionPrdiction_NN.h5')  # creates a HDF5 file 'my_model.h5'


# ### 3.2 Random Forest Model

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf3 = RandomForestRegressor(n_estimators = 10, random_state = 0).fit(X_train, Y_train);


# In[ ]:


y_predict = rf3.predict(X_test)
Get_score(y_predict, Y_test)


# ### 3.3 Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
modelLR = LinearRegression().fit(X_train, Y_train)


# In[ ]:


ynew = modelLR.predict(X_test)
Get_score(y_predict, Y_test)


# ### 3.4 Support Vector Machines - SVR
# 
# #### SVR-Short Term memorablity

# In[ ]:


from sklearn.svm import SVR
Y_short = labels[['short-term_memorability']].values
X = one_hot_res;
X_train, X_test, Y_train_short, Y_test_short = train_test_split(X,Y_short, test_size=0.2, random_state=42)
modelSVR_short = SVR(C=100).fit(X_train,Y_train_short)


# In[ ]:


predictionsSVR_short = modelSVR_short.predict(X_test)
Get_score(predictionsSVR_short,Y_test_short)


# #### SVR-Long Term memorablity

# In[ ]:


Y_long = labels[['long-term_memorability']].values
X_train, X_test,Y_train_long, Y_test_long = train_test_split(X,Y_long, test_size=0.2, random_state=42)


# In[ ]:


Y_test_long.shape
modelSVR_long = SVR(C=100).fit(X_train,Y_train_long)


# In[ ]:


predictionsSVR_long = modelSVR_long.predict(X_test)
Get_score(predictionsSVR_long,Y_test_long)


# ### 3.5 Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


regr = DecisionTreeRegressor(max_depth=10)
regr.fit(X_train, Y_train)
pred_test_dtr = regr.predict(X_test)


# In[ ]:


Get_score(pred_test_dtr, Y_test)


# ## 4.  Predicting video memorability using Video features

# ### 4.1 Using C3D feature

# In[ ]:


def read_C3D(fname):
    """Scan vectors from file"""
    with open(fname) as f:
        for line in f:
            C3D =[float(item) for item in line.split()] # convert to float type, using default separator
    return C3D

def vname2ID(vnames):
    """Parse video digital id from its name
    vnames: a list contains file names"""
    vid = [ os.path.splitext(vn)[0]+'.webm' for vn in vnames]
    return vid


# In[ ]:


C3D_Feat_path = '../input/data/data/dev-set_features/'
# Load video related features first
# it helps with the organization of the video names
vid = labels.video.values

C3D_Features = pd.DataFrame({'video': vid,
                   'C3D': [read_C3D(C3D_Feat_path+'C3D'+'/'+os.path.splitext(item)[0]+'.txt') for item in vid],
                       })


# In[ ]:


C3D_X = np.stack(C3D_Features['C3D'].values)
C3D_Y = labels[['short-term_memorability','long-term_memorability']].values

C3D_X_train, C3D_X_test, C3D_Y_train, C3D_Y_test = train_test_split(C3D_X,C3D_Y, test_size=0.2, random_state=42)


# #### 4.1.1 Recurrent Neural Network

# In[ ]:


C3D_model = Sequential()
C3D_model.add(layers.Dense(200,activation='relu',kernel_regularizer=None,input_shape=(C3D_X.shape[1],)))
C3D_model.add(layers.Dropout(0.1))
C3D_model.add(layers.Dense(2,activation='sigmoid'))
C3D_model.compile(optimizer='rmsprop',loss=['mae'])
history=C3D_model.fit(x=C3D_X_train,y=C3D_Y_train,batch_size=50,epochs=20,validation_split=0.2,shuffle=True,verbose=True)
C3D_Y_pred = C3D_model.predict(C3D_X_test)
Get_score(C3D_Y_pred,C3D_Y_test)


# #### 4.1.2 Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
C3D_clf = RandomForestRegressor()
C3D_clf.fit(C3D_X_train,C3D_Y_train)
pred_test_rfr = C3D_clf.predict(C3D_X_test)
Get_score(pred_test_rfr, C3D_Y_test)


# ### 4.2 Using HMP feature

# In[ ]:


def read_HMP(fname):
    """Scan HMP(Histogram of Motion Patterns) features from file"""
    with open(fname) as f:
        for line in f:
            pairs=line.split()
            HMP_temp = { int(p.split(':')[0]) : float(p.split(':')[1]) for p in pairs}
    # there are 6075 bins, fill zeros
    HMP = np.zeros(6075)
    for idx in HMP_temp.keys():
        HMP[idx-1] = HMP_temp[idx]            
    return HMP


# In[ ]:


HMP_Feat_path = '../input/data/data/dev-set_features/'
# Load video related features first
# it helps with the organization of the video names
vid = labels.video.values
HMP_Features = pd.DataFrame({'video': vid,
                   'HMP': [read_HMP(HMP_Feat_path+'HMP'+'/'+os.path.splitext(item)[0]+'.txt') for item in vid],
                       })


# In[ ]:


HMP_X = np.stack(HMP_Features['HMP'].values)
HMP_Y = labels[['short-term_memorability','long-term_memorability']].values
HMP_X_train, HMP_X_test, HMP_Y_train, HMP_Y_test = train_test_split(HMP_X,HMP_Y, test_size=0.2, random_state=42)


# #### 4.2.1 Recurrent Neural Network

# In[ ]:


HMP_model = Sequential()
HMP_model.add(layers.Dense(200,activation='relu',kernel_regularizer=None,input_shape=(HMP_X.shape[1],)))
HMP_model.add(layers.Dropout(0.1))
HMP_model.add(layers.Dense(2,activation='sigmoid'))
HMP_model.compile(optimizer='rmsprop',loss=['mae'])
history=HMP_model.fit(x=HMP_X_train,y=HMP_Y_train,batch_size=50,epochs=20,validation_split=0.2,shuffle=True,verbose=True)
HMP_Y_pred = HMP_model.predict(HMP_X_test)
Get_score(HMP_Y_pred,HMP_Y_test)


# #### 4.2.2 Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
HMP_clf = RandomForestRegressor()
HMP_clf.fit(HMP_X_train,HMP_Y_train)
pred_test_rfr = HMP_clf.predict(HMP_X_test)
Get_score(pred_test_rfr, HMP_Y_test)


# ## 5. Final ground truth prediction-Test data

# In[ ]:


# load the captions
#cap_path = '/media/win/Users/ecelab-adm/Desktop/DataSet_me18me/me18me-devset/dev-set/dev-set_video-captions.txt'
cap_path = '../input/data/data/test-set_video-captions.txt'
df_test=read_caps(cap_path)

# load the ground truth values
test_ground_truth=pd.read_csv('../input/data/data/test-set_ground-truth.csv')

df_test['caption'] = df_test['caption'].apply(strip_character)

df_test['caption'] = df_test['caption'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))
df_test['caption'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in df_test['caption']]

counts_test = Counter()
for i, cap in enumerate(df_test['caption']):
    counts_test.update(cap.split())


# In[ ]:


vect_test = TfidfVectorizer(ngram_range = (1,4)).fit(df_test.caption)


# In[ ]:


# using the training data's words length for the testing as well to avoid the 
tokenizer_test = Tokenizer(num_words=len_token)
tokenizer_test.fit_on_texts(list(vect_test.get_feature_names()))
one_hot_res_test = tokenizer_test.texts_to_matrix(list(df_test.caption.values),mode='binary')


# In[ ]:


X_testpredict = one_hot_res_test;


# In[ ]:


np.ndim(one_hot_res_test)
one_hot_res_test.shape
testdata = test_ground_truth.copy()


# ### 4.1 Predicting short-term memorability - using SVR

# In[ ]:


testpredict_SVR_short = modelSVR_short.predict(X_testpredict)


# In[ ]:


testpredict_SVR_short.shape
type(testpredict_SVR_short)
testdata['short-term_memorability'] = testpredict_SVR_short


# ### 4.2 Predicting long-term memorability - using Random Forest

# In[ ]:


testpredict_RFR_long = rf3.predict(X_testpredict)


# In[ ]:


testdata['long-term_memorability'] = testpredict_RFR_long[:,1]


# In[ ]:


testdata.tail()


# ## 5. Saving the final submission file

# In[ ]:


testdata.to_csv('finalgroundTruth_test.csv', encoding='utf-8', index=False)

