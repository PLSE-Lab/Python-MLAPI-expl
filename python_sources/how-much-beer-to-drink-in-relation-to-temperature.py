#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import Adam, SGD

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


path = "../input/beer-consumption-sao-paulo/Consumo_cerveja.csv"
dataset = pd.read_csv(path)


# In[ ]:


dataset.head()


# In[ ]:


dataset.columns=["fecha", "media", "min", "max", "precipitacao", "finde", "cerveza"]


# In[ ]:


dataset = dataset.dropna()
dataset.head()


# In[ ]:


media= np.array(dataset.media)


# In[ ]:


print(type(media[1]))


# In[ ]:


media = media.tolist()
print(type(media))


# In[ ]:


media_2 = []
for i in media:
    media_2.append(float(str(i).replace(",", ".")))


# In[ ]:


type(media_2)


# In[ ]:


X=np.array(media_2)
y_true=dataset[['cerveza']].values


# In[ ]:


model = Sequential()
model.add(Dense(1, input_shape=(1,)))


# In[ ]:


model.summary()


# In[ ]:


model.compile(Adam(lr=0.8), 'mean_squared_error')


# In[ ]:


model.fit(X,y_true, epochs=35, batch_size=110)


# In[ ]:


y_pred= model.predict(X)


# In[ ]:


plt.scatter(X,y_true)
plt.plot(X, y_pred, color='red', linewidth=3)


# In[ ]:


w,b=model.get_weights()


# In[ ]:


print("Valor de w:",w)
print("Valor de b:",b)


# In[ ]:


Xnew = np.array([[15.]])
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))


# In[ ]:


X


# In[ ]:


y_pred


# In[ ]:


#dataset = pd.DataFrame({'Temperatura Media (C)':X, 'Consumo de cerveja (litros)':y_pred})


# In[ ]:


X.shape


# In[ ]:


y_pred = np.squeeze(y_pred)


# In[ ]:


y_pred.shape


# In[ ]:


dataset = pd.DataFrame({'Temperatura Media (C)':X, 'Consumo de cerveja (litros)':y_pred})


# In[ ]:


dataset.head()


# In[ ]:


dataset.to_csv('predictions.csv' , index=False)


# In[ ]:


ls


# In[ ]:




