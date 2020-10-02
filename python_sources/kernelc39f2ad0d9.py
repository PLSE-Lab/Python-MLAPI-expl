#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from keras.layers import Dense, Activation, Merge, Reshape
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import RMSprop
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from keras.layers.embeddings import Embedding

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt


# In[14]:


train_data=pd.read_csv('/kaggle/input/train.csv')
test_data=pd.read_csv('/kaggle/input/test.csv')


# In[ ]:


from sklearn.preprocessing import Imputer

imputer=Imputer()
cols_remove_nan=["Age","SibSp","Parch","Fare"]

for col in cols_remove_nan:
    train_data[col]=imputer.fit_transform(train_data[col].values.reshape(-1,1))
    test_data[col]=imputer.fit_transform(test_data[col].values.reshape(-1,1))


# In[ ]:


cols_not_concerned=["Ticket","Cabin","Survived","PassengerId","Name"]
#cols_not_concerned=["Ticket","Fare","Cabin","Survived","PassengerId","Name"]
data=train_data.drop(cols_not_concerned,axis=1)
tcols_not_concerned=["Ticket","Cabin","PassengerId","Name"]
#tcols_not_concerned=["Ticket","Fare","Cabin","PassengerId","Name"]
tdata=test_data.drop(tcols_not_concerned, axis=1)


# In[ ]:


data['Embarked'] = data['Embarked'].fillna('C') 
tdata['Embarked']=tdata['Embarked'].fillna('C')


# In[ ]:


le = LabelEncoder()
le.fit(["male","female"])
data["Sex"]=le.transform(data["Sex"])
tdata["Sex"]=le.transform(tdata["Sex"])

le = LabelEncoder()
le.fit(["Q","C","S"])
data["Embarked"]=le.transform(data["Embarked"])
tdata["Embarked"]=le.transform(tdata["Embarked"])


# In[ ]:


embed_columns=[col for col in data.columns if (train_data[col].dtype=='object')]
non_embed_columns=[col for col in data.columns if(col not in embed_columns)]


# In[ ]:


embed_data = pd.DataFrame()
non_embed_data = pd.DataFrame()
for col in embed_columns:
    embed_data[col]=data[col]
for col in non_embed_columns:
    non_embed_data[col]=data[col] 
    


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
non_embed_data=sc.fit_transform(non_embed_data)


# In[ ]:


labels=train_data['Survived']


# In[ ]:


models=[]

model_sex = Sequential()
model_sex.add(Embedding(2, 2, input_length=1))
model_sex.add(Reshape(target_shape=(2,)))
models.append(model_sex)

model_embark = Sequential()
model_embark.add(Embedding(3, 2, input_length=1))
model_embark.add(Reshape(target_shape=(2,)))
models.append(model_embark)


model_input = Sequential()
model_input.add(Dense(8,input_dim=5))
models.append(model_input)
                
model = Sequential()
model.add(Merge(models, mode='concat'))
model.add(Activation('relu'))
model.add(Dense(6, kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dense(1,activation='sigmoid'))                
                
                


# In[ ]:


from keras.optimizers import RMSprop
from keras.optimizers import SGD
learning_rate=0.1
optimizer=RMSprop(lr=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# In[ ]:


history=model.fit([embed_data['Sex'],embed_data['Embarked'],non_embed_data],labels,epochs=500,batch_size=128,validation_split=0.1)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()


# In[ ]:




