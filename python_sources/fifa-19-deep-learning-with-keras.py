#!/usr/bin/env python
# coding: utf-8

# # 1. Packages

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns

# Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
import keras.backend as K

# Train-Test
from sklearn.model_selection import train_test_split

# Scale
from sklearn.preprocessing import StandardScaler

# Classification Report
from sklearn.metrics import classification_report


# # 2. Data

# In[ ]:


df = pd.read_csv("../input/fifa19/data.csv")

# Remove Missing Values 
na = pd.notnull(df["Position"])
df = df[na]

df.head()


# Position Category

# In[ ]:


forward = ["ST", "LW", "RW", "LF", "RF", "RS","LS", "CF"]
midfielder = ["CM","RCM","LCM", "CDM","RDM","LDM", "CAM", "LAM", "RAM", "RM", "LM"]
defender = ["CB", "RCB", "LCB", "LWB", "RWB", "LB", "RB"]


# In[ ]:


df.loc[df["Position"] == "GK", "Position"] = 0
df.loc[df["Position"].isin(defender), "Position"] = 1
df.loc[df["Position"].isin(midfielder), "Position"] = 2
df.loc[df["Position"].isin(forward), "Position"] = 3


# In[ ]:


df["Position"].value_counts()


# In[ ]:


df["Position"].unique()


# In[ ]:


df = df[["Position", 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes']]
df.head()


# In[ ]:


x = df.drop("Position", axis = 1)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

x = pd.DataFrame(ss.fit_transform(x))

y = df["Position"]

x.head()


# In[ ]:


y.head()


# In[ ]:


from keras.utils.np_utils import to_categorical
y_cat = to_categorical(y)

y_cat[:10]


# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x.values, y_cat,
                                                    test_size=0.2)


# In[ ]:


y.shape


# In[ ]:


x.shape


# # 3. Deep Learning

# In[ ]:


import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, Adadelta, RMSprop


# In[ ]:


K.clear_session()
model = Sequential()
model.add(Dense(60, input_shape = (33,), activation = "relu"))
model.add(Dense(15, activation = "relu"))
model.add(Dense(4, activation = "softmax"))
model.compile(Adam(lr = 0.01), "categorical_crossentropy", metrics = ["accuracy"])
model.summary()


# In[ ]:


60 * 33 + 60


# In[ ]:


model.fit(x_train, y_train, verbose=1, epochs=10)


# In[ ]:


y_pred_class = model.predict_classes(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


y_pred = model.predict(x_test)


# In[ ]:


y_test_class = np.argmax(y_test, axis=1)


# In[ ]:


y_test_class


# In[ ]:


confusion_matrix(y_test_class, y_pred_class)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test_class, y_pred_class))

