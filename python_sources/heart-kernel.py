#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/heart-disease-uci/heart.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(x='target',data=df)


# In[ ]:


plt.figure(figsize=(12,5))
sns.heatmap(df.corr(),annot=True)
plt.ylim(0, 14)
plt.xlim(0, None)


# In[ ]:


sns.boxplot(x='target',y='thalach',data=df)


# In[ ]:


plt.figure(figsize=(12,6))
df.corr()['target'][:-1].sort_values().plot(kind='bar')


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df.drop('target',axis=1).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


model = Sequential()

model.add(Dense(18,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')


# In[ ]:


early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=50)


# In[ ]:


model.fit(x=X_train,y=y_train,epochs=300,validation_data=(X_test,y_test),callbacks=[early_stop])


# In[ ]:


losses = pd.DataFrame(model.history.history)


# In[ ]:


losses.head()


# In[ ]:


losses.plot()


# In[ ]:


predictions = model.predict_classes(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,predictions))
print('')
print(classification_report(y_test,predictions))


# In[ ]:




