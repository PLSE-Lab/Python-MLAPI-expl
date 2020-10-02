#!/usr/bin/env python
# coding: utf-8

# ![](http://)# **Counter Strike : Global Offensive Analysis & Predictions**

# In[ ]:


from IPython.display import Image # Module to show image in Jupyter-Notebook


# In[ ]:


Image('http://esportscenter.com/wp-content/uploads/2015/11/Csgo-8.jpg') #Path of Image to be displayed, fed into Image() function.


# ## Modules required for Data Analysis.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler, LabelEncoder# to scale the dataset in a desired range.
from sklearn.model_selection import train_test_split # splits data set into Train/Test.
import pandas as pd # to manipulate data, to make it usable.
import numpy as np # module for linear algebra calculations.
import matplotlib.pyplot as plt # to visualize numerical data


# ## Importing Data to Preprocess

# In[ ]:


csgo_data = pd.read_csv('../input/mm_master_demos.csv')


# In[ ]:


work_data = csgo_data[['map','is_bomb_planted','bomb_site','ct_eq_val', 't_eq_val', 'avg_match_rank','winner_side',]]


# In[ ]:


x_data = work_data.iloc[:,0:5]


# In[ ]:


y_data = work_data.iloc[:,6]


# In[ ]:


x_data.head()


# ## Text data is processed into numerical labels to feed into NN.

# In[ ]:


y_data = LabelEncoder().fit_transform(y_data)


# In[ ]:


x_data['map'] = LabelEncoder().fit_transform(x_data['map'])


# In[ ]:


x_data['is_bomb_planted'] = LabelEncoder().fit_transform(x_data['is_bomb_planted'])


# In[ ]:


x_data.fillna('not_planted',axis=1,inplace=True)


# In[ ]:


x_data['bomb_site'] = LabelEncoder().fit_transform(x_data['bomb_site'])


# In[ ]:


x_data[['ct_eq_val','t_eq_val']] = MinMaxScaler(feature_range=(0,2)).fit_transform(x_data[['ct_eq_val','t_eq_val']])


# In[ ]:


x_data = np.array(x_data)


# In[ ]:


trainx,testx,trainy,testy = train_test_split(x_data,y_data,test_size = 0.15,random_state=64)


# ## Modules for Learning Model.

# In[ ]:


# from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import SGDClassifier
# from sklearn import svm


# In[ ]:


from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras import Sequential


# In[ ]:


model = Sequential()


# In[ ]:


model.add(Dense(units=5, input_shape=(5,),activation='relu'))
model.add(Dense(units=36 ,activation='relu'))
model.add(Dense(units=2,activation='softmax'))


# In[ ]:


model.compile(optimizer=Adam(lr=0.003),loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


model.fit(x=trainx,y=trainy,batch_size=100,epochs=1)


# In[ ]:


predictions = model.predict(x=testx,batch_size=100,verbose=2)


# In[ ]:


pred_cls = np.argmax(predictions,axis=1)


# In[ ]:


pred_cls


# In[ ]:


evaluation_keras = model.evaluate(x=testx,y=testy,batch_size=100)


# In[ ]:


print('This model is %.2f Percent Accurate.'%float(evaluation_keras[1]*100))


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


evaluation_sklearn = accuracy_score(y_true=testy,y_pred=pred_cls)


# In[ ]:


print('This model is %.2f Percent Accurate.'%float(evaluation_sklearn*100))


# In[ ]:




