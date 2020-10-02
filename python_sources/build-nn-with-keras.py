#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import numpy as np
import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt


# In[ ]:


plt.style.use('seaborn-darkgrid')


# In[ ]:


from sklearn.preprocessing import StandardScaler

def data_preprocessing(df_input):
    # numeric feature standardization
    sc = StandardScaler()    
    column_names = df_input.columns[1:11]
    df = pd.DataFrame(sc.fit_transform(df_input.iloc[:, 1:11]))
    df.columns = column_names
    
    # reverse one-hot encoding
    Wilderness_Area = df_input.iloc[:, 11:15].idxmax(1).str.replace('Wilderness_Area', '')
    df['Wilderness_Area'] = pd.to_numeric(Wilderness_Area)
    
    # reverse one-hot encoding
    Soil_Type = df_input.iloc[:, 15:55].idxmax(1).str.replace('Soil_Type', '')
    df['Soil_Type'] = pd.to_numeric(Soil_Type)
    
    return df.join(df_input.iloc[:, 11:55])


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


train_df.head()


# In[ ]:


X = data_preprocessing(train_df)
y = train_df.iloc[:, -1]


# In[ ]:


X.describe()


# In[ ]:


# oversampling by Soil_Type

for i in range(5):
    threshold = X.Soil_Type.value_counts().median()
    need_over_sample_types = X.Soil_Type.value_counts()[(X.Soil_Type.value_counts() < threshold)].index
    oversampling_rows = X[X['Soil_Type'].isin(need_over_sample_types)].copy()
    
    y = y.append(y[oversampling_rows.index])
    X = X.append(oversampling_rows)

    y.reset_index(drop=True, inplace=True)
    X.reset_index(drop=True, inplace=True)


# In[ ]:


# oversampling by Wilderness_Area

for i in range(5):
    oversampling_rows = X[X['Wilderness_Area']==X.Wilderness_Area.value_counts().idxmin()].copy()
    
    y = y.append(y[oversampling_rows.index])
    X = X.append(oversampling_rows)

    y.reset_index(drop=True, inplace=True)
    X.reset_index(drop=True, inplace=True)


# ### Build Network

# In[ ]:


# prepare data for model
labels = pd.get_dummies(y)
num_labels = len(set(y))
feature_size = X.shape[1]


# In[ ]:


# setting hyperparameter
epochs = 3000
batch_size = 4096
train_ratio = 0.8

reg_rate = 0.001
drop_rate = 0.1

init_nodes = 256


# In[ ]:


from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers

model = tf.keras.Sequential()
model.add(Dense(init_nodes, activation='relu', input_shape=(feature_size,)))

nodes = init_nodes//2
for i in range(5):
    model.add(Dropout(drop_rate))
    model.add(Dense(nodes, activation='relu', kernel_regularizer=regularizers.l2(reg_rate)))
    nodes = nodes//2

model.add(Dense(num_labels, activation='softmax'))

print(model.summary())


# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])


# In[ ]:


# setting early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=50)


# In[ ]:


model_history = model.fit(x=X, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_split=1-train_ratio,
                          verbose=0,
                          callbacks=[early_stopping])


# In[ ]:


# print final acc/loss values
for key, value in model_history.history.items():
    print(key, '=', value[-1])


# In[ ]:


# plot train histories
for key, value in model_history.history.items():
    plt.plot(value, label=key)

plt.ylim(0, 5)

plt.legend()
plt.show()


# In[ ]:


# perform prediction
X_test = data_preprocessing(test_df)
y_out = model.predict_classes(X_test)
predict_class = y_out+1


# ### Write predicted results to .csv file

# In[ ]:


output = pd.DataFrame({'Id': test_df.loc[:, 'Id'], 'class': predict_class})
output.to_csv('./output-keras.csv', index=False)


# In[ ]:




