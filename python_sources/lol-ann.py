#!/usr/bin/env python
# coding: utf-8

# > **Importing the libraries**

# In[ ]:




import numpy as np
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import seaborn as sns
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[ ]:


df = pd.read_csv('/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')


# In[ ]:


df.head(5)


# In[ ]:


dataset = df.iloc[:,1:40]
dataset.head(5)


# # EDA ** **

# In[ ]:


corr = dataset.corr('pearson')


# In[ ]:


sns.countplot(data = dataset,x='blueWins')


# In[ ]:


dataset.isnull().sum()


# In[ ]:


blue_df = dataset.iloc[:,0:19]
blue_df.drop(columns = ['blueDeaths'],inplace = True)
blue_df.head(5)


# In[ ]:


corr = blue_df.corr('pearson')
plt.figure(figsize = (10,10))
sns.heatmap(corr,annot = True)


# - Since `blueDeaths` is having negative correlation with the blueWins, we drop the attribute. 
# - Along with the other Red Team attributes

# In[ ]:


corr['blueWins'].sort_values(ascending=False)


# **Let us attempt to tackle it using deep learning**

# In[ ]:


X = blue_df.iloc[:,1:]
y = blue_df.iloc[:,1]


# In[ ]:


scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X),columns=X.columns)
X.head(5)


# In[ ]:


y = blue_df['blueWins']
y = to_categorical(y, 2)
y


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state=0)


# In[ ]:


model = Sequential()
model.add(Dense(units=18,activation='relu',input_dim=len(X.columns)))
model.add(Dense(36,activation = 'relu'))
model.add(Dense(72,activation = 'relu'))
model.add(Dense(units=2,activation='softmax'))
model.summary()


# In[ ]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train,y_train,
                   epochs=50,
                   validation_data=(X_test,y_test))


# In[ ]:


plt.figure(figsize=(8,8))
plt.plot(history.history['val_accuracy'])
#plt.legend(['Training Accuracy','Validation Accuracy'])
plt.title('Accuracy curves')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:




