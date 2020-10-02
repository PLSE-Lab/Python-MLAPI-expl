#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')


# In[ ]:


df.head()


# In[ ]:


df.describe().transpose()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


sns.countplot(df['target_class'])


# If we look at the graph above we can see that it is an imbalanced dataset.So we are going to make a new dataframe with balanced data
# 

# In[ ]:


df['target_class'].value_counts()


# In[ ]:


df1=df[df['target_class']==0].head(1639)


# In[ ]:


df2=df[df['target_class']==1].head(1639)


# In[ ]:


new_df=pd.concat([df1,df2]).sample(frac=1)


# In[ ]:


new_df


# In[ ]:


sns.countplot(new_df['target_class'])


# In[ ]:


sns.scatterplot(x=' Mean of the integrated profile',y=' Standard deviation of the integrated profile',data=new_df)


# In[ ]:


sns.scatterplot(x='target_class',y=' Standard deviation of the integrated profile',data=new_df)


# In[ ]:


sns.scatterplot(x=' Excess kurtosis of the integrated profile',y=' Mean of the DM-SNR curve',data=new_df)


# In[ ]:


sns.scatterplot(x=' Mean of the integrated profile',y=' Mean of the DM-SNR curve',data=new_df)


# In[ ]:


sns.jointplot(new_df[' Skewness of the DM-SNR curve'],new_df[' Excess kurtosis of the DM-SNR curve'])


# In[ ]:


sns.pairplot(new_df)


# In[ ]:


sns.heatmap(new_df.corr(),annot=True)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x=new_df.drop('target_class',axis=1).values
y=new_df['target_class'].values


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler=MinMaxScaler()


# In[ ]:


x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout


# In[ ]:


model=Sequential()
model.add(Dense(30,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)


# In[ ]:


model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),epochs=700,verbose=1,callbacks=[es])


# In[ ]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# In[ ]:


predictions=model.predict_classes(x_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[ ]:




