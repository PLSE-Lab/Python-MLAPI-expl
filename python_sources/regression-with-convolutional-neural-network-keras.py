#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns


# In[ ]:


data = pd.read_csv("/kaggle/input/mercedes-benz-greener-manufacturing/train.csv")


# In[ ]:


dtest = pd.read_csv("/kaggle/input/mercedes-benz-greener-manufacturing/test.csv")


# ### Missing Values
# 
# * No missing values

# In[ ]:


# make a list of the variables that contain missing values
vars_with_na = data.columns[data.isnull().any()].tolist()
vars_with_na_test = dtest.columns[dtest.isnull().any()].tolist()
print (len(vars_with_na), len(vars_with_na_test))


# ### Group the Data [train and test]

# In[ ]:


train_test_data = [data, dtest]


# In[ ]:


# list of numerical variables
for dataset in train_test_data:
    num_vars = [var for var in dataset.columns if dataset[var].dtypes != 'O']

    print('Number of numerical variables: ', len(num_vars))


# ### Check for value greater than 1

# In[ ]:


num_vars = [var for var in data.columns if data[var].dtypes != 'O']
Xdatanum = data[num_vars].drop(["ID", "y"], axis = 1)
for var in Xdatanum.columns:
    if Xdatanum[var].max()>1:
        print ("there are value greater than 1")


# In[ ]:


# list of categorical variables
cat_vars = [var for var in data.columns if data[var].dtypes == 'O']

print('Number of categorical variables: ', len(cat_vars))

# visualise the numerical variables
data[cat_vars].head()


# In[ ]:


import matplotlib.pyplot as plt

for c in data[cat_vars]:
    value_counts = data[c].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title('Categorical feature {} - Cardinality {}'.format(c, len(np.unique(data[c]))))
    plt.xlabel('Feature value')
    plt.ylabel('Occurences')
    plt.bar(range(len(value_counts)), value_counts.values)
    ax.set_xticks(range(len(value_counts)))
    ax.set_xticklabels(value_counts.index, rotation='vertical')
    plt.show()


# ### Looking at individual plot
# 
# * change the label column

# In[ ]:


sns.boxplot(x = data["X0"] , y = "y" , data= data)


# In[ ]:


sns.scatterplot(x = data["X4"] , y = "y" , data= data)


# ### Suspicious data

# In[ ]:


suspiciousData = []

for col in data:
    
    if len(data[col].unique()) == 1:
        suspiciousData.append(col)
data[suspiciousData].describe()


# ### Drop suspicious features

# In[ ]:


for dataset in train_test_data:

    dataset = dataset.drop(suspiciousData, 1, inplace = True)


# ### Type of data

# In[ ]:


dtype_df = data.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()


# ## Target analysis

# In[ ]:


sns.distplot(data["y"])


# In[ ]:


data["y"].describe()


# ## Categorical data

# In[ ]:


categoricalData = data[cat_vars]
categoricalData.info()


# In[ ]:


for var in cat_vars:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x = var, y = "y" , data = data)


# # Count encoder

# In[ ]:


import category_encoders as ce

# Create the encoder itself
Count_enc = ce.CountEncoder(cols=cat_vars)
# Fit the encoder using the categorical features 
Count_enc.fit(data[cat_vars], data["y"])

data = data.join(Count_enc.transform(data[cat_vars]).add_suffix('_count'))
dtest = dtest.join(Count_enc.transform(dtest[cat_vars]).add_suffix('_count'))


# In[ ]:


data = data.drop(data[cat_vars], axis = 1)
dtest = dtest.drop(dtest[cat_vars], axis = 1)


# # Feature Scaling

# In[ ]:


cat_count = ['X0_count', 'X1_count', 'X2_count', 'X3_count',
       'X4_count', 'X5_count', 'X6_count', 'X8_count']


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data[cat_count])
data[cat_count] = scaler.transform(data[cat_count])
dtest[cat_count] = scaler.transform(dtest[cat_count])
print(scaler.data_max_)


# In[ ]:


data.describe()


# ## Outilier

# In[ ]:


#data = data[data["y"] < 220]


# # Define X and y

# In[ ]:


data =  data.drop("ID" , axis = 1)


# In[ ]:


X = data.drop("y" , axis =  1)
y = data["y"]
X.shape


# In[ ]:


X= X.values
y = y.values


# ## Train test split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size = 0.2 )


# # Build and train the CNN model

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K


# In[ ]:


batch_size = 150
epochs =100
# input image dimensions
img_rows, img_cols = 28, 13
#inputshape = X.shape[1]


# In[ ]:


if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# In[ ]:


def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


# In[ ]:


import matplotlib.pyplot as plt

plt.imshow(X_train[0].reshape(28,13))


# In[ ]:


from keras.layers.normalization import BatchNormalization

model = Sequential()
#model.add(Dense(256, activation='relu', input_dim=366))
model.add(Conv2D(64, (3, 3), activation='relu', input_shape = input_shape))
#model.add(Conv2D(128, (3, 3), activation='relu'))
#model.add(Conv2D(64, (3, 3), init='uniform'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))

model.add(Dense(1, activation='linear'))


model.compile(loss='mean_squared_error', # one may use 'mean_absolute_error' as  mean_squared_error
                  optimizer='adam',
                  metrics=[r2_keras] # you can add several if needed
                 )

model.summary()


# In[ ]:


model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)


# In[ ]:


print('Test loss:', score)


# # visualise the predictions and residuals

# In[ ]:


preds = model.predict(X_test)
preds = preds[:,0]
plt.scatter(y_test, preds)


# In[ ]:


residuals = y_test - preds
sns.distplot(residuals)


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test, preds)  


# # testing

# In[ ]:


test_data = dtest.drop("ID", axis=1).copy()
#test_data = dtest[selectedFeatures].copy()

X = test_data.values
X.shape


# In[ ]:


if K.image_data_format() == 'channels_first':
    X = X.reshape(X.shape[0], 1, img_rows, img_cols)
    
    input_shape = (1, img_rows, img_cols)
else:
    X = X.reshape(X.shape[0], img_rows, img_cols, 1)
    
    input_shape = (img_rows, img_cols, 1)

X = X.astype('float32')

print('X shape:', X.shape)

print(X.shape[0], 'test samples')


# In[ ]:


prediction = model.predict(X)


# In[ ]:


prediction = prediction[:,0]


# In[ ]:


dtest.info()


# In[ ]:


submission = pd.DataFrame({
        "ID": dtest["ID"],
        "y": prediction
    })

submission.to_csv('submission_5.csv', index=False)


# In[ ]:




