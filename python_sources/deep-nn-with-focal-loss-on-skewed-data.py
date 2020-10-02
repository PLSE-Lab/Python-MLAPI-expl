#!/usr/bin/env python
# coding: utf-8

# # <center>Deep NN</center>
# ## <div align=right>Made by:</div>
# **<div align=right>Ihor Markevych</div>**

# ## Scenario
# A financial institution desires to refine its targeting strategy and grow the
# client population leveraging third party credit data.
# 

# --------------------------

# In[ ]:


import pandas as pd
import numpy as np
import pickle

import tensorflow as tf

import keras
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization

from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import joblib

import matplotlib.pyplot as plt

# from imblearn.over_sampling import SMOTE

np.random.seed(123)


# ## Preprocessing

# In[ ]:


def preprocessing(X, scaler_filename='scaler.save', columns_to_use_filename='columns_to_use'):
    
    scaler = joblib.load(scaler_filename)
    
    with open (columns_to_use_filename, 'rb') as fp:
        columns_to_use = pickle.load(fp)    
    
    X = X.loc[:, columns_to_use]
    
    numerical_X = X.loc[:, X.dtypes != 'object']
    numerical_X = pd.DataFrame(scaler.transform(numerical_X), columns=numerical_X.columns)
    numerical_X = numerical_X.fillna(0)
    return numerical_X 
#     categorical_X = X.loc[:, X.dtypes == 'object']
#     categorical_X_encoded = pd.get_dummies(categorical_X, dummy_na=True, prefix=categorical_X.columns)
#     final_X = pd.concat([numerical_X, categorical_X_encoded], axis=1)
    
#     return final_X


# In[ ]:


data = pd.read_csv('/kaggle/input/Data.csv')
X = data.drop(columns='Flag')
y = data.Flag

columns_to_use = list(X.columns[(X.isna().sum() / len(X)) < 0.4])
X = X.loc[:, columns_to_use]
corrs = X.corrwith(y).abs().sort_values(ascending=False)
# columns_to_use = [x for x in columns_to_use if x not in corrs[corrs < 0.01].index.tolist()]
columns_to_use = corrs[corrs > 0.01].index.tolist()
X = X.loc[:, columns_to_use]

with open('columns_to_use', 'wb') as fp:
    pickle.dump(columns_to_use, fp)

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X.loc[:, X.dtypes != 'object'])
joblib.dump(scaler, "scaler.save") 

X_preprocessed = preprocessing(X)

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, stratify=y, test_size=0.1)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1)


# Columns used:

# In[ ]:


print(columns_to_use)


# ## EDA

# In[ ]:


print(f'Number of samples: {len(X)}.')


# In[ ]:


y.value_counts(normalize=True)


# We can see quite strong disbalance in classes. We will need to account that in our model.

# In[ ]:


y.isna().any()


# We can also see that our target does not have any missing values.

# In[ ]:


(X.isna().sum() / X.shape[0]).sort_values()


# In[ ]:


X.isna().sum(axis=1).sort_values() / X.shape[1]


# We see that some columns have a lot of missing values.  
# Also some rows have missing values (up to all of the values).

# Correlations:

# In[ ]:


corrs


# Correlation is going from quite high values of 0.1 to very small rates of 0.06.

# Number of features (before preprocessing):

# In[ ]:


data.shape[1] - 1


# Number of features (after preprocessing):

# In[ ]:


X_preprocessed.shape[1]


# ## Focal loss

# We will use weighted focal loss to account class disbalance problem.  
# https://arxiv.org/pdf/1708.02002.pdf  
# $$Loss(p_t) = \alpha _t* (1 - p_t) ^ \gamma * log(p_t),$$ where $p_t$ is predicted probability of sample to belong to certain class.

# In[ ]:


from tensorflow.keras import backend as K

def focal_loss(gamma=2, alpha=0.89):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)
                 + (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


# ## Hyperparameters

# In[ ]:


OPTIMIZER = keras.optimizers.Adam(learning_rate=0.01, clipvalue=5)
EPOCHS = 50
BATCH_SIZE = 256

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)


# In[ ]:


alpha = y.value_counts(normalize=True)[0] # 0.89
gamma = 0.5


# ## Model

# In[ ]:


import time

def create_model(gamma=gamma, alpha=alpha, seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    else:
        t = int(time.time())
        print(f'Seed: {t}.')
        np.random.seed(t)

    model = Sequential()

    model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(8, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(1, activation='sigmoid', 
                    name='prediction', 
                    bias_initializer=keras.initializers.Constant(np.log(9))))

    model.compile(loss=focal_loss(gamma, alpha), optimizer=OPTIMIZER, metrics=["accuracy"])

    return model


# In[ ]:


def evaluate(model, X_val, y_val):
    y_pred = model.predict_classes(X_val).ravel()
    f1 = f1_score(y_val, y_pred , average="macro")
    fpr, tpr, thresholds = roc_curve(y_val, y_pred)
    auc_model = auc(fpr, tpr)
    conf = pd.DataFrame(confusion_matrix(y_val, y_pred), columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
    prec = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    acc = accuracy_score(y_val, y_pred)
    
    return {'auc': auc_model, 
            'f1':f1, 
            'confusion':conf, 
            'precision': prec,
            'recall': recall,
            'accuracy': acc,
            'roc_cache': (fpr, tpr)}


# In[ ]:


def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS):
            
    history = model.fit(X_train.values, y_train,
                        batch_size=BATCH_SIZE, 
                        epochs=epochs, 
                        validation_split=0.1,
                        use_multiprocessing=True,
                        verbose=0,
                        callbacks=[es]
                       )
    
    return {'model': model,
            'history':history, 
            'eval': evaluate(model, X_val, y_val)}


# ## Validation with Stratified K-Fold

# In[ ]:


from sklearn.model_selection import StratifiedKFold

n_folds = 10

skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

h = []

for i, (train, test) in enumerate(skf.split(X_train, y_train)):
    print("Running Fold", i + 1, "/", n_folds)
    model = None # Clearing the NN.
    model = create_model(seed=1587411287)
    h.append(train_and_evaluate_model(model,
                                      X_train.iloc[train], y_train.iloc[train], 
                                      X_train.iloc[test], y_train.iloc[test]))


# In[ ]:


np.mean([e['eval']['auc'] for e in h])


# ## Final model

# In[ ]:


model = None
model = create_model(seed=1587411287)
model.summary()


# ### Training

# In[ ]:


history = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)


# ### Evaluation

# #### Train performance

# In[ ]:


evaluate(model, X_train, y_train)


# #### Test performance

# In[ ]:


history['eval']


# #### Learning curves

# In[ ]:


plt.title('Loss')
plt.plot(history['history'].history['loss'])
plt.plot(history['history'].history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# #### ROC curve

# In[ ]:


fpr, tpr = history['eval']['roc_cache']
plt.plot(fpr, tpr, color='darkorange', label=f"ROC curve (area = {round(history['eval']['auc'], 3)})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# ## Saving the model

# In[ ]:


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")


# ## Assumptions, limitations, conclusion

# ### Assumptions:
# * Main assumption is that stakeholder will prefer to have more false positives (e.g. sending advertisement to people that most likely won't become a customer) than to have false negatives (missing potential customers).
# * Therefore, ROC-AUC was selected as a single performance evaluation metric (for ease of comparison between different models).

# ### Limitations:
# Main limitation is lack of proper estimate of Bayess Classifier performance. For instance, for tasks like image classification Bayess Classifier error is usually assumed to be zero, as human performance is around 0% error. However, for task like this it's hard to estimate optimal performance, therefore we can't conclude whether our model is too simple (has removable bias), or we are already performing at best possible rate.  
#   
# Another limiting factor is small number of samples with 1 target. It may be possible to significantly increase model performance by having more data for class 1.

# ### Conclusion:
#   
# Focal loss allows to train model on skewed data without data augumentation or downsampling.  
# * $\alpha$ was set to be equal inverse normalized frequency of appearing of this class in data. This can be taken as starting default value that can be tuned later with cross-validation. However, in our case this value appeared to be optimal or close to optimal.  
# * $\gamma$ was tuned using validation set.
#   
# Focal loss gave better performance than using SMOTE to oversample rare class, or than using downsamlping or combinations of above. It also gave better results than simply using weights to account skewed target.  
# Setting bias of the final layer to $log(\frac{1 - \pi}{\pi})$, where $\pi$ is normalized frequency of rare class ensure faster convergence and numerical stability. With SMOTEd and downsampled data local optimas of predicting all 0s were a significant problem.  
# Tweaking $\gamma$ parameter can lead to either more correctly classified 0 classes, or to more correctly classified 1 classes.

# ### What else was tried:
# * SMOTE, 
# * downsampling, 
# * ensemble, 
# * two stage training (SMOTEd data at stage 1 + downsampled data at stage 2).

# ## References:
# 
# 1. https://arxiv.org/pdf/1804.07612.pdf
# 1. https://www.kaggle.com/abazdyrev/keras-nn-focal-loss-experiments
