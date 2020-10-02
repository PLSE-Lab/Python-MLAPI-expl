#!/usr/bin/env python
# coding: utf-8

# This notebook uses Embedding to make more features.
# I am not a pro programmer as you can see but I am still learning. 

# In[ ]:



# importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import random
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score as rauc


# In[ ]:


train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv') 
test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')


# In[ ]:


# get test id for later in submit file
test_id = test.id


# In[ ]:


del train['id']
del test['id']


# In[ ]:


target = train['target']
del train['target']


# In[ ]:


all_data = pd.concat([train, test], axis = 0).copy()


# In[ ]:


plt.figure(figsize=(5,3))
plt.hist(all_data.bin_0);


# 0 has highest incidence we will replace then nans with 0

# In[ ]:


plt.figure(figsize=(5,3))
plt.hist(all_data.bin_1);


# idemn for bin_1

# In[ ]:


plt.figure(figsize=(5,3))
plt.hist(all_data.bin_2);


# also for bin_2

# In[ ]:


# replacing the nans
fill_bin_cols_nan = ['bin_0', 'bin_1', 'bin_2']
all_data[fill_bin_cols_nan] = all_data[fill_bin_cols_nan].fillna(0)


# In[ ]:


# these string types
fill_nom_cols1_nan = ['nom_0','nom_1', 'nom_2', 'nom_3', 'nom_4', 'bin_3', 'bin_4']
all_data[fill_nom_cols1_nan] = all_data[fill_nom_cols1_nan].fillna('NAN')


# In[ ]:


plt.figure(figsize=(5,2))
plt.hist(all_data.ord_0);


# 1 has highes frequency

# In[ ]:


plt.figure(figsize=(5,2))
plt.hist(all_data.day, bins=20);


# day 3 has highest frequency

# In[ ]:


plt.figure(figsize=(5,2))
plt.hist(all_data.month, bins=40);


# Month 8 has the majority

# In[ ]:


all_data['ord_0'] = all_data.ord_0.fillna(1)
all_data['day'] = all_data.day.fillna(3)
all_data['month'] = all_data.month.fillna(8)
# fill all other nans with 'ffffffffff'
columns_to_encode = [ 'ord_1', 'ord_2','ord_3' , 'ord_4', 'ord_5','bin_3', 'bin_4', 
                     'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4',
                    'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
all_data[columns_to_encode] = all_data[columns_to_encode].fillna('fffffffff')


# In[ ]:


all_data.bin_3.unique()


# # Label encoding
# 

# In[ ]:


# label encoding
columns_to_encode = [ 'ord_1', 'ord_2','ord_3' , 'ord_4', 'ord_5','bin_3', 'bin_4', 
                     'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

encoder = LabelEncoder()
for col in columns_to_encode:
    print(col)
    all_data[col][:len(train)] = encoder.fit_transform(all_data[col][:len(train)])
    all_data[col][len(train):] = all_data[col][len(train):].map(lambda s: '<unknown>' if s not in encoder.classes_ else s)
    encoder.classes_ = np.append(encoder.classes_, '<unknown>')
    all_data[col][len(train):] = encoder.transform(all_data[col][len(train):])


# The nominals 5 to 9 have some kind of hexadecimal code. We kan make extra features from each hex digit

# In[ ]:


nom_label_array = np.zeros([len(all_data), len(all_data.nom_5.values[0])])

for step in range(nom_label_array.shape[1]):
    row=0
    for item in all_data.nom_5.values :
        #print(item)
        #print(int(item[0]  ,16))
        nom_label_array[row][step]= int(item[step]  ,16)
        row +=1

for step in range(nom_label_array.shape[1]):        
    col_name = 'nom_5_' + str(step)
    all_data[col_name] = nom_label_array[:,step]


# In[ ]:


# nom_6
for step in range(nom_label_array.shape[1]):
    row=0
    for item in all_data.nom_6.values :
        nom_label_array[row][step]= int(item[step]  ,16)
        row +=1

for step in range(nom_label_array.shape[1]):        
    col_name = 'nom_6_' + str(step)
    all_data[col_name] = nom_label_array[:,step]

# nom_7
for step in range(nom_label_array.shape[1]):
    row=0
    for item in all_data.nom_7.values :
        nom_label_array[row][step]= int(item[step]  ,16)
        row +=1

for step in range(nom_label_array.shape[1]):        
    col_name = 'nom_7_' + str(step)
    all_data[col_name] = nom_label_array[:,step]

# nom_8
for step in range(nom_label_array.shape[1]):
    row=0
    for item in all_data.nom_8.values :
        nom_label_array[row][step]= int(item[step]  ,16)
        row +=1

for step in range(nom_label_array.shape[1]):        
    col_name = 'nom_8_' + str(step)
    all_data[col_name] = nom_label_array[:,step]

# nom_9
for step in range(nom_label_array.shape[1]):
    row=0
    for item in all_data.nom_9.values :
        nom_label_array[row][step]= int(item[step]  ,16)
        row +=1

for step in range(nom_label_array.shape[1]):        
    col_name = 'nom_9_' + str(step)
    all_data[col_name] = nom_label_array[:,step]


# In[ ]:


all_data.head(10).T


# In[ ]:


# drop the nominal features 5 to 9


# In[ ]:


all_data = all_data.drop(['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'], axis='columns')


# In[ ]:


all_data.columns


# In[ ]:


all_data.shape


# Using Keras Embedding to do more label encoding and create more features

# In[ ]:


from numpy.random import seed
seed(1) # for reproducability
import tensorflow
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
all_data_input = np.asarray(all_data)
model_embed = Sequential()
model_embed.add(Embedding(500, 14, ))
model_embed.compile('rmsprop', 'mse')
output_data = model_embed.predict(all_data_input)
output_data.shape


# In[ ]:


embedded_all_data = output_data.reshape(output_data.shape[0], output_data.shape[1]*output_data.shape[2])
print(embedded_all_data.shape)
train = embedded_all_data[:len(train)]
test = embedded_all_data[len(train):]
X_train, X_valid, y_train, y_valid = train_test_split(train, target, test_size=0.2, random_state=1)


# In[ ]:



eval_dataset = Pool(X_valid, y_valid)
model2 = CatBoostClassifier(iterations=1000 ,random_state=2, verbose=0,eval_metric='Accuracy', 
                            learning_rate=0.1, task_type='GPU'  )
                           
history = model2.fit(X_train, y_train, eval_set=eval_dataset,
                     use_best_model=True, verbose=False, plot=True)


# In[ ]:


predictions_proba_for_valid = model2.predict_proba(X_valid)
#print(predictions_proba.shape)
pred_proba = predictions_proba_for_valid[:,1]
#print(pred_proba.shape)
R_AUC_valid = rauc(y_valid, pred_proba)
print('Area under the ROC curve between target and predictions_proba %.5f' % R_AUC_valid)


# In[ ]:


plt.hist(y_valid, bins=4, density=True, label='y_valid', align='left')
plt.hist(model2.predict(X_valid), bins=4, density=True, label='predicted', align='right')
plt.legend()
plt.show()


# It shows that predicted values have not the best balance

# In[ ]:


# training on all train data

model2 = CatBoostClassifier(iterations=800 ,random_state=2, verbose=0,eval_metric='Accuracy', 
                            learning_rate=0.1, task_type='GPU')
                           
history = model2.fit(X_train, y_train, verbose=False, plot=False)


# In[ ]:





# In[ ]:


predictions_proba_test = model2.predict_proba(test)[:,1]


filename = 'submission.csv'
pd.DataFrame({'id': test_id, 
              'target': predictions_proba_test}).to_csv(filename , index=False)


# In[ ]:





# In[ ]:




