#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df1=pd.read_csv('../input/creditcard.csv')
df1.head()


# In[ ]:


df=df1.drop(columns=['Time','Amount'])
df.head()


# In[ ]:


df.describe()


# In[ ]:


df_normal=df[df['Class']==0]
df_normal.shape


# In[ ]:


df_anomaly=df[df['Class']==1]
df_anomaly.shape


# In[ ]:


df_anomaly1,df_anomaly2=train_test_split(df_anomaly,test_size=0.5)
df_anomaly1.shape


# In[ ]:


df_train,df_v=train_test_split(df_normal,test_size=0.005)
df_v.shape


# In[ ]:


df_v1,df_t1=train_test_split(df_v,test_size=0.5)
df_v1.shape


# In[ ]:


df_val=df_v1.append(df_anomaly1).sample(frac=1)
df_val.head()


# In[ ]:


df_test=df_t1.append(df_anomaly2).sample(frac=1)
df_test.head()


# In[ ]:


print(df_train.shape)
print(df_val.shape)
print(df_test.shape)


# In[ ]:


X_train=df_train.iloc[:,:-1].values
X_val=df_val.iloc[:,:-1].values
y_val=df_val.iloc[:,-1].values
X_test=df_test.iloc[:,:-1].values
y_test=df_test.iloc[:,-1].values


# ### scaling

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


sc=MinMaxScaler()


# In[ ]:


sc.fit(X_train)


# In[ ]:


X_train=sc.transform(X_train)
X_val=sc.transform(X_val)
X_test=sc.transform(X_test)


# In[ ]:


print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


# ## Autoencoder

# In[ ]:


inputshape=X_train[0].shape
num_train_sample=len(X_train)
num_val_sample=len(X_val)
batchsize=64


# In[ ]:


from keras.layers import Dense,Dropout,Input
from keras.models import Sequential,Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy


# In[ ]:


input1=Input(shape=inputshape)
x1=Dense(16,activation='relu')(input1)
x2=Dense(8,activation='relu')(x1)
encoded=Dense(4,activation='relu')(x2)


# In[ ]:


d1=Dense(8,activation='relu')(encoded)
d2=Dense(16,activation='relu')(d1)
decoded=Dense(28,activation='sigmoid')(d2)


# In[ ]:


autoencoder=Model(inputs=input1,outputs=decoded)


# In[ ]:


autoencoder.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])


# In[ ]:


history = autoencoder.fit(X_train, X_train,
                    epochs=10,
                    batch_size=batchsize,
                    shuffle=True,
                    validation_data=(X_val, X_val),
                    verbose=1)


# In[ ]:


X_val_pred=autoencoder.predict(X_val)


# In[ ]:


mse = np.mean(np.power(X_val - X_val_pred, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse*1000,'true_class': y_val})
error_df.describe()


# In[ ]:


threshholds=np.linspace(0,100,300)
f1score=[]
for t in threshholds:
    y_hat=error_df.reconstruction_error>t
    f1score.append(f1_score(y_val,y_hat))


# In[ ]:


f1score


# In[ ]:


scores = np.array(f1score)


# In[ ]:


scores.max()


# In[ ]:


scores.argmax()


# In[ ]:


threshholds[scores.argmax()]


# In[ ]:


final_thresh=threshholds[scores.argmax()]


# ### prediction on test dataset

# In[ ]:


X_test_pred=autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)
test_error_df = pd.DataFrame({'reconstruction_error': mse*1000,'true_class': y_test})
test_error_df['y_hat']=test_error_df.reconstruction_error>final_thresh
test_error_df.head()


# In[ ]:


f1_score(y_test,test_error_df.y_hat)


# In[ ]:


precision_score(y_test,test_error_df.y_hat)


# In[ ]:


recall_score(y_test,test_error_df.y_hat)


# In[ ]:


confusion_matrix(y_test,test_error_df.y_hat)


# ## GaussianMixture
# 

# In[ ]:


from sklearn.mixture import GaussianMixture


# In[ ]:


gmm = GaussianMixture(n_components=3, n_init=4, random_state=42)
gmm.fit(X_train)
print(gmm.score(X_val))


# In[ ]:


y_scores=gmm.score_samples(X_val)


# In[ ]:


df_erroe_val = pd.DataFrame({'log_prob': y_scores,'true_class': y_val})


# In[ ]:


df_erroe_val.describe()


# In[ ]:


threshhold2=np.linspace(-400,96,500)
f1score_2=[]
for t in threshhold2:
    y_hat=df_erroe_val.log_prob<t
    f1score_2.append(f1_score(y_val,y_hat))


# In[ ]:


f1score_2=np.array(f1score_2)


# In[ ]:


f1score_2.max()


# In[ ]:


f1score_2.argmax()


# In[ ]:


final_thresh2=threshhold2[f1score_2.argmax()]
final_thresh2


# ### prediction on test data

# In[ ]:


y_scores_test=gmm.score_samples(X_test)


# In[ ]:


y_pred2=y_scores_test<final_thresh2


# In[ ]:


print(f1_score(y_test,y_pred2))
print(precision_score(y_test,y_pred2))
print(recall_score(y_test,y_pred2))


# In[ ]:


confusion_matrix(y_test,y_pred2)


# ## unsupervise anomaly detection

# #### Autoencoder

# In[ ]:


df.head()


# In[ ]:


X1=df.iloc[:,:-1].values
y1=df.iloc[:,-1].values


# In[ ]:


#X_train3,X_test3,y_train3,y_test3=train_test_split(X1,y1,test_size=0.02)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
sc2=MinMaxScaler()
sc2.fit(X1)


# In[ ]:


X_train3=sc2.transform(X1)


# In[ ]:


X_train3.shape


# In[ ]:


inputshape=X_train3[0].shape
num_train_sample=len(X_train3)
batchsize=64


# In[ ]:


from keras.layers import Dense,Dropout,Input
from keras.models import Sequential,Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy


# In[ ]:


input1=Input(shape=inputshape)
x1=Dense(16,activation='relu')(input1)
x2=Dense(8,activation='relu')(x1)
encoded=Dense(2,activation='relu')(x2)


# In[ ]:


d1=Dense(8,activation='relu')(encoded)
d2=Dense(16,activation='relu')(d1)
decoded=Dense(28,activation='sigmoid')(d2)


# In[ ]:


autoencoder=Model(inputs=input1,outputs=decoded)


# In[ ]:


autoencoder.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])


# In[ ]:


history = autoencoder.fit(X_train3, X_train3,
                    epochs=3,
                    batch_size=batchsize,
                    shuffle=True,
                    verbose=1)


# In[ ]:


y1.sum()


# In[ ]:


X_train_pred3=autoencoder.predict(X_train3)


# In[ ]:


mse3 = np.mean(np.power(X_train3 - X_train_pred3, 2), axis=1)
error_df3 = pd.DataFrame({'reconstruction_error': mse3*1000,'true_class': y1})
error_df3.describe()


# In[ ]:


y_pred3=error_df3.reconstruction_error>4.13


# In[ ]:


confusion_matrix(y1,y_pred3)


# In[ ]:


precision_score(y1,y_pred3)


# In[ ]:


recall_score(y1,y_pred3)


# In[ ]:


f1_score(y1,y_pred3)


# #### GaussianMixture

# In[ ]:


from sklearn.mixture import GaussianMixture


# In[ ]:


gmm = GaussianMixture(n_components=3, n_init=4, random_state=42)
gmm.fit(X_train3)
print(gmm.score(X_train3))


# In[ ]:


y_scores=gmm.score_samples(X_train3)


# In[ ]:


df_erroe_val = pd.DataFrame({'log_prob': y_scores,'true_class': y1})
df_erroe_val.describe()


# In[ ]:


y_pred3=df_erroe_val.log_prob<52.84


# In[ ]:


confusion_matrix(y1,y_pred3)


# In[ ]:


precision_score(y1,y_pred3)


# In[ ]:


recall_score(y1,y_pred3)


# In[ ]:


f1_score(y1,y_pred3)


# In[ ]:




