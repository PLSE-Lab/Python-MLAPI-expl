#!/usr/bin/env python
# coding: utf-8

# # Mobile Price Classification
# 
# ## Objectives
# Help Bob find out the relationship between features of a mobile phone and its price range. Additionally, build a model to help him predict the price range when values of features are given.
# 
# Therefore:
# - Dependent variable: y = phone price range
# - Independent variables: x = 
#     - id: ID
#     - battery_power: Total energy a battery can store in one time measured in mAh
#     - blue: Has bluetooth or not
#     - clock_speed: speed at which microprocessor executes instructions
#     - dual_sim: Has dual sim support or not
#     - fc: Front Camera mega pixels
#     - four_g: Has 4G or not
#     - int_memory: Internal Memory in Gigabytes
#     - m_dep: Mobile Depth in cm
#     - mobile_wt: Weight of mobile phone
#     - n_cores: Number of cores of processor
#     - pc: Primary Camera mega pixels
#     - px_height: Pixel Resolution Height
#     - px_width: Pixel Resolution Width
#     - ram: Random Access Memory in Megabytes
#     - sc_h: Screen Height of mobile in cm
#     - sc_w: Screen Width of mobile in cm
#     - talk_time: longest time that a single battery charge will last when you are
#     - three_g: Has 3G or not
#     - touch_screen: Has touch screen or not
#     - wifi: Has wifi or not
# 
# ## Steps
# 1. Data Preprocessing
# 2. Data Exploration
#     - General view of the phone market
#     - Relationship between phone attributes and price
#     - Attributes of high-priced vs. low-priced phones
# 3. Build deep learning model to predict price range

# ## 1. Data Preprocessing 

# #### 1.1. Load data 

# In[ ]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#load data
test = pd.read_csv("../input/mobile-price-classification/test.csv")
train = pd.read_csv("../input/mobile-price-classification/train.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# #### 1.2. Check for null values 

# In[ ]:


train.info()


# In[ ]:


test.info()


# This dataset is clean with no null values.

# ## 2. Data Exploration

# #### 2.1. General view of phone market

# In[ ]:


numerical = ['battery_power','clock_speed','fc','int_memory','m_dep','mobile_wt', 'n_cores', 'pc', 'px_height',
             'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']
categorical = ['blue','dual_sim','four_g','three_g', 'touch_screen','wifi']


# In[ ]:


print(len(numerical))
print(len(categorical))


# In[ ]:


#categorical attributes
df = pd.melt(train[categorical])
sns.countplot(data=df,x='variable', hue='value')


# Two-third of the phone market has 3g, while other attributes share similiar distribution.

# In[ ]:


#numerical attributes
fig = plt.figure(figsize=(15,20))
for i,col in enumerate(numerical):
    ax=plt.subplot(5,3,i+1) 
    train[col].plot.hist(ax = ax).tick_params(axis = 'x',labelrotation = 360)
    ax.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.1))
plt.show()


# - Numerical attributes having skewed distributions: clock_speed,fc,m_dep, px_height, sc_w. We will apply median as a measure of central tendency.
# - Other numerical attributes having symmetrical distributions, we will apply mean as a measure of central tendency.

# In[ ]:


skewed = ['clock_speed','fc','m_dep', 'px_height', 'sc_w']
no_skewed = ['battery_power','int_memory','mobile_wt','n_cores','pc','px_width','ram','sc_h','talk_time']


# #### 2.2. What attribute has the most effect on phone price? 

# In[ ]:


#correlation between attributes
corr = train.corr()
fig, (ax) = plt.subplots(1,1,sharey = True, figsize = (20,10))
sns.heatmap(corr, cmap = 'Blues')


# We can see these attributes having relationship with each other:
# - Price range vs. ram: high positive correlation
# - fc vs. pc: positive correlation
# - four_g vs. three_g: positive correlation
# - pc_height vs. pc_width: positive correlation

# In[ ]:


#correlation between price and phone attributes
corr.sort_values(by=["price_range"],ascending=False).iloc[0].sort_values(ascending=False)


# Here we can see that Ram has high positive correlation with Price Range. The larger the Ram the higher the price. Other attributes do not affect phone price as much as ram. 

# In[ ]:


train.groupby('price_range').mean()['ram'].plot(kind = 'bar', legend = True).tick_params(axis = 'x', labelrotation = 360)


# #### 2.3. Attributes of high-priced vs. low-priced phones 

# #### 2.3.1 Numerical Variables 

# In[ ]:


#variables with symmetrical distributions
group_no_skewed = train.groupby('price_range')[no_skewed].mean().reset_index()
fig = plt.figure(figsize=(15,20))
for i,col in enumerate(group_no_skewed.iloc[:,1:].columns):
    ax=plt.subplot(5,3,i+1) 
    group_no_skewed.iloc[:,1:][col].plot.bar(ax = ax).tick_params(axis = 'x',labelrotation = 360)
    ax.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.1))
plt.show()


# In[ ]:


#variables with skewed distributions
group_skewed = train.groupby('price_range')[skewed].median().reset_index()
fig = plt.figure(figsize=(15,20))
for i,col in enumerate(group_skewed.iloc[:,1:].columns):
    ax=plt.subplot(5,3,i+1) 
    group_skewed.iloc[:,1:][col].plot.bar(ax = ax).tick_params(axis = 'x',labelrotation = 360)
    ax.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.1))
plt.show()


# High-priced phones seem to have:
#     - Better battery: higher 'battery_power' and higher 'talk_time' 
#     - Better camera: 
#         - higher 'pc': primary camera mega pixels
#         - higher 'px_height': Pixel Resolution Height
#         - higher 'px_width': Pixel Resolution Width
#     - Better memory:
#         - higher 'int_memory': internal memory
#         - higher 'ram': Random Access Memory

# #### 2.3.2. Categorical Variables 

# In[ ]:


#bluetooth, wifi vs. price
sns.catplot('price_range', col='blue',hue = 'wifi',data = train,  kind = 'count', col_wrap=2)


# Findings:
# - Bluetooth and Wifi seem to not have a significant affect to phone price since they have similar distribution in every price range.

# In[ ]:


#3g, 4g vs. price
sns.catplot('price_range', col='three_g',hue = 'four_g',data = train,  kind = 'count', col_wrap=2)


# Findings:
# - Nearly half of the phones have both 3g and 4g.
# - Phones must have 3g in order to have 4g.
# - These attributes seem to not affect the price very much as they have similar distributions in every price range.

# In[ ]:


#dual_sim vs. price
sns.catplot('price_range', col='dual_sim',data = train,  kind = 'count')


# Findings:
# - Whether the phone has dual_sim or not seem to not have a significant affect to phone price since they have similar distribution in every price range.

# In[ ]:


#touch_screen vs. price
sns.catplot('price_range', col='touch_screen',data = train,  kind = 'count')


# Findings:
# - Whether the phone has touch screen or not seem to not have a significant affect to phone price since they have similar distribution in every price range.

# ## 3. Build model to predict price range

# #### 3.1. Preprocessing data

# In[ ]:


#scale numeric variables of training data
from sklearn.preprocessing import MinMaxScaler
scaler_train = MinMaxScaler()
train_num_scaled = scaler_train.fit_transform(train[numerical])
scaler_train.data_max_
scaler_train.data_min_


# In[ ]:


train_num_scaled = pd.DataFrame(train_num_scaled,columns=train[numerical].columns)
train_num_scaled


# In[ ]:


#scale numeric variables of test data
from sklearn.preprocessing import MinMaxScaler
scaler_test = MinMaxScaler()
test_num_scaled = scaler_test.fit_transform(test[numerical])
scaler_test.data_max_
scaler_test.data_min_


# In[ ]:


test_num_scaled = pd.DataFrame(test_num_scaled,columns=test[numerical].columns)


# In[ ]:


test_final = pd.concat([test[categorical],test_num_scaled], axis = 1)
test_final.head()


# #### 3.2. Split data into train, validation & test set

# In[ ]:


#X & Y array
import tensorflow as tf
X = pd.concat([train[categorical],train_num_scaled], axis = 1)
y = tf.keras.utils.to_categorical(train['price_range'], 4)


# In[ ]:


X.head()


# In[ ]:


y


# In[ ]:


#Split the original train data into train and val data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, train['price_range'], test_size=0.33, random_state=101)


# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


# #### 3.3. Build deep learning model to predict price range

# In[ ]:


#import deep learning libraries
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


#build model
model_1 = Sequential()
model_1.add(Dense(25, input_dim=20, activation='relu'))
model_1.add(Dense(25, activation='relu'))
model_1.add(Dense(4, activation='softmax'))
model_1.summary()


# In[ ]:


model_1.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
hist_1 = model_1.fit(X_train, y_train, epochs=20, batch_size=25, 
                   validation_data=(X_val,y_val))


# #### 3.4. Model Evaluation 

# In[ ]:


plt.plot(hist_1.history['loss'])
plt.plot(hist_1.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])


# In[ ]:


score = model_1.evaluate(X_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


#test data prediction
prediction_test = np.argmax(model_1.predict(test_final), axis=1)
pd.DataFrame({'id' : test['id'],'price_range' : prediction_test})

