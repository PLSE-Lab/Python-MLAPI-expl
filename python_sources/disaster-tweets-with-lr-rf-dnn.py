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


import string
import nltk
import re
import xgboost as xgb
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,make_scorer
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

PUNCT_TO_REMOVE = string.punctuation
STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()


# In[ ]:


# import training data
df_train = pd.read_csv('../input/nlp-getting-started/train.csv')
df_test = pd.read_csv('../input/nlp-getting-started/test.csv')


# In[ ]:


df_train.head(2)


# ### TEXT PREPROCESSING

# In[ ]:


def text_preprocessing(text):
    '''
    input: string to be processed
    output: preprocssed string
    '''
    text = text.lower() # make everything lower case
    text = re.compile(r'https?://\S+|www\.\S+').sub(r'', text) #remove url
    text = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE)) #remove punctuation
    text = " ".join([word for word in str(text).split() if word not in STOPWORDS]) #remove stop words
    text = " ".join([stemmer.stem(word) for word in text.split()])
    
    return text


# In[ ]:


text_preprocessing('''#Flashflood causes #landslide in Gilgit #Pakistan Damage to 20 homes
                   farmland roads and bridges #365disasters  http://t.co/911F3IXRH0''')


# In[ ]:


df_train['text_processed'] = df_train['text'].apply(text_preprocessing)
df_test['text_processed'] = df_test['text'].apply(text_preprocessing)


# In[ ]:


df_train.head()


# ### Train Test Split

# In[ ]:


X = df_train['text_processed']
y = df_train['target']


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1,random_state=42)


# In[ ]:


vectorizer=TfidfVectorizer(ngram_range=(1,3),min_df=3,strip_accents='unicode', 
                           use_idf=1,smooth_idf=1, sublinear_tf=1,max_features=None)
vectorizer.fit(list(df_train['text_processed'])+list(df_test['text_processed']))
print('vocab length',len(vectorizer.vocabulary_))


# In[ ]:


X_train_onehot = vectorizer.transform(X_train).todense()
X_val_onehot = vectorizer.transform(X_val).todense()


# ### Baseline Model: Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(max_iter=150,penalty='l2',solver='lbfgs',random_state=0)
lr_clf.fit(X_train_onehot, y_train)
lr_pred = lr_clf.predict(X_val_onehot)

print('accuracy score: ',accuracy_score(lr_pred,y_val))
print(classification_report(y_val, lr_pred))


# In[ ]:


from sklearn.metrics import log_loss
logloss_lr = log_loss(y_val,lr_clf.predict_proba(X_val_onehot))
print('logloss_lr:',logloss_lr)


# ### Naive Bayes

# In[ ]:


from sklearn.naive_bayes import MultinomialNB

mnb_clf = MultinomialNB()
mnb_clf.fit(X_train_onehot, y_train)
mnb_pred = mnb_clf.predict(X_val_onehot)

print('accuracy score: ',accuracy_score(mnb_pred,y_val))
print(classification_report(y_val, mnb_pred))


# In[ ]:


logloss_mnb = log_loss(y_val,mnb_clf.predict_proba(X_val_onehot))
print('logloss_mnb:',logloss_mnb)


# Next, we will try RF, XGBoost and DNN

# ### RandomForest

# In[ ]:


rf_clf = RandomForestClassifier(random_state=0,n_estimators=100,
                                max_depth=None, verbose=0,n_jobs=-1)
rf_clf.fit(X_train_onehot, y_train)
rf_pred = rf_clf.predict(X_val_onehot)

print('accuracy score: ',accuracy_score(rf_pred,y_val))
print(classification_report(y_val, rf_pred))


# In[ ]:


logloss_rf = log_loss(y_val,rf_clf.predict_proba(X_val_onehot))
print('logloss_rf:',logloss_rf)


# ### XGBoost

# In[ ]:


# Fitting a simple xgboost on tf-idf
xgb_clf = xgb.XGBClassifier(n_estimators=100,n_jobs=-1,max_depth=15,
                            min_child_weight=3,objective='binary:logistic'
                            ,colsample_bytree=0.4)
xgb_clf.fit(X_train_onehot, y_train)
xgb_predictions = xgb_clf.predict(X_val_onehot)

print('accuracy score: ',accuracy_score(xgb_predictions,y_val))
print(classification_report(y_val, xgb_predictions))


# In[ ]:


logloss_xgb = log_loss(y_val,xgb_clf.predict_proba(X_val_onehot))
print('logloss_rf:',logloss_xgb)


# ### DNN

# In[ ]:


np.shape(X_train_onehot)[1]


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras import regularizers

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=np.shape(X_train_onehot)[1],
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=10**-8, decay=0.0001, amsgrad=False)
model.compile(optimizer= adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())

# Train the model, iterating on the data in batches of 32 samples
hist = model.fit(X_train_onehot, y_train,validation_data = (X_val_onehot,y_val), epochs=20, batch_size=16)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

history = pd.DataFrame(hist.history)
plt.figure(figsize=(12,12));
plt.plot(history["loss"]);
plt.plot(history["val_loss"]);
plt.title("Loss as function of epoch");
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show();


# In[ ]:


dnn_pred = model.predict_classes(X_val_onehot)


# In[ ]:


print('accuracy score: ',accuracy_score(dnn_pred,y_val))
print(classification_report(y_val, dnn_pred))


# In[ ]:


logloss_dnn = log_loss(y_val,model.predict_proba(X_val_onehot))
print('logloss_dnn:',logloss_dnn)


# ### Averaging

# In[ ]:


lr_predictions_val = lr_clf.predict_proba(X_val_onehot)
mnb_predictions_val = mnb_clf.predict_proba(X_val_onehot)
#rf_predictions_val = rf_clf.predict_proba(X_val_onehot)
xgb_predictions_val = xgb_clf.predict_proba(X_val_onehot)
#dnn_predictions_val = model.predict_proba(X_val_onehot).ravel()


# In[ ]:


#predictions_val = 1/5*lr_predictions_val[:,1]+1/5*gnb_predictions_val[:,1] \
#                +1/5*rf_predictions_val[:,1]+1/5*xgb_predictions_val[:,1]+1/5*dnn_predictions_val
predictions_val = 1/3*lr_predictions_val[:,1]+1/3*mnb_predictions_val[:,1]                 +1/3*xgb_predictions_val[:,1]

predictions_val = np.where(predictions_val>0.5, 1, 0)


# In[ ]:


print('accuracy score: ',accuracy_score(predictions_val,y_val))
print(classification_report(y_val, predictions_val))


# ## Prediction on test set

# In[ ]:


df_test = pd.read_csv('../input/nlp-getting-started/test.csv')


# In[ ]:


df_test.head()


# In[ ]:


df_test['text_processed'] = df_test['text'].apply(text_preprocessing)


# In[ ]:


X_test = df_test['text_processed']
X_test_onehot = vectorizer.transform(X_test).todense()


# In[ ]:


lr_predictions = lr_clf.predict_proba(X_test_onehot)
mnb_predictions = mnb_clf.predict_proba(X_test_onehot)
#rf_predictions = rf_clf.predict_proba(X_test_onehot)
xgb_predictions = xgb_clf.predict_proba(X_test_onehot)
#dnn_predictions = model.predict_proba(X_test_onehot).ravel()


# In[ ]:


#predictions = 1/5*lr_predictions[:,1]+1/5*gnb_predictions[:,1]+1/5*rf_predictions[:,1]+1/5*xgb_predictions[:,1]+1/5*dnn_predictions
predictions = 1/3*lr_predictions[:,1]+1/3*mnb_predictions[:,1]+1/3*xgb_predictions[:,1]


# In[ ]:


predictions = np.where(predictions>0.5, 1, 0)


# In[ ]:


df_submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
df_submission['target'] = predictions
df_submission.to_csv('submission.csv',index=False)


# In[ ]:





# In[ ]:




