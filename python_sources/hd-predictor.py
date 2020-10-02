#!/usr/bin/env python
# coding: utf-8

# Thanks to [IntiPic](https://www.kaggle.com/intipic) for explaining the dataset in [this](https://www.kaggle.com/ronitf/heart-disease-uci/discussion/105877) post. To me this looks more like a derivation of the [Statlog Heart dataset](https://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29). Especially since the attributes (features) are an exact match.

# In[ ]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

seed = 51

BATCH_SIZE=32


# In[ ]:


data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
data.sample(5)


# In[ ]:


data.info()


# **Feature engineering:** There is a hypothesis that high cholesterol actually protects elderly patients. So lets create a chol_age feature that reduces the value of chol as we age:

# In[ ]:


data['chol_age'] = data['chol']/data['age']
data.sample(5)


# Let's fix the scaling.

# In[ ]:


from sklearn.preprocessing import RobustScaler

data['age'] = RobustScaler().fit_transform(data['age'].values.reshape(-1, 1))
data['chol_age'] = RobustScaler().fit_transform(data['chol_age'].values.reshape(-1, 1))
data['trestbps'] = RobustScaler().fit_transform(data['trestbps'].values.reshape(-1, 1))
data['chol'] = RobustScaler().fit_transform(data['chol'].values.reshape(-1, 1))
data['thalach'] = RobustScaler().fit_transform(data['thalach'].values.reshape(-1, 1))
data['oldpeak'] = RobustScaler().fit_transform(data['oldpeak'].values.reshape(-1, 1))

data.sample(10)


# Let's make some of the values clearer:

# In[ ]:


data['cp'][data['cp'] == 0] = 'asymptomatic'
data['cp'][data['cp'] == 1] = 'atypical angina'
data['cp'][data['cp'] == 2] = 'non-anginal pain'
data['cp'][data['cp'] == 3] = 'typical angina'

data['restecg'][data['restecg'] == 0] = 'left ventricular hypertrophy'
data['restecg'][data['restecg'] == 1] = 'normal'
data['restecg'][data['restecg'] == 2] = 'ST-T wave abnormality '

data['slope'][data['slope'] == 0] = 'down'
data['slope'][data['slope'] == 1] = 'flat'
data['slope'][data['slope'] == 2] = 'up'


# Check correlation of each attribute to the target

# In[ ]:


corr = data.corr()
corr.sort_values(["target"], ascending = False, inplace = True)
corr.target


# It is possible that our engineered feature, chol_age, is proving the hypothesis by showing a slight correlation with the abscense of heart disease. In the future it might be interesting to replace the chol feature with our new chol_age feature.
# 
# Let's take a closer look at cp, slope, restcg, and thal correlation. Here is what they mean:
# 
# cp: chest pain type
# * -- Value 0: asymptomatic
# * -- Value 1: atypical angina
# * -- Value 2: non-anginal pain
# * -- Value 3: typical angina
# 
# slope: the slope of the peak exercise ST segment
# * 0: downsloping; 
# * 1: flat; 
# * 2: upsloping
# 
# restecg: resting electrocardiographic results
# * -- Value 0: showing probable or definite left ventricular hypertrophy by Estes' criteria
# * -- Value 1: normal
# * -- Value 2: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
# 
# thal: 
# * 1 = fixed defect; 
# * 2 = normal; 
# * 7 = reversable defect
# 
# So, let's one hot encode them

# In[ ]:


from sklearn.preprocessing import OneHotEncoder

OH_cols = ['cp', 'slope', 'restecg','thal']

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_data = pd.DataFrame(OH_encoder.fit_transform(data[OH_cols]))

# One-hot encoding put in generic column names, use feature names instead
OH_cols_data.columns = OH_encoder.get_feature_names(OH_cols)

# # remove the original columns
# for c in OH_cols:
#     cols_to_use.remove(c)
    
# # Add one-hot columns to cols_to_use
# for c in OH_cols_data.columns:
#     cols_to_use.append(c)

# # print(cols_to_use)

# One-hot encoding removed index; put it back
OH_cols_data.index = data.index

# Remove categorical columns (will replace with one-hot encoding)
num_data = data.drop(OH_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_data = pd.concat([num_data, OH_cols_data], axis=1)

data = OH_data


# Now, lets check the correlation again:

# In[ ]:


corr = data.corr()
corr.sort_values(["target"], ascending = False, inplace = True)
corr.target


# As expected chol has very little correlation with the target. **Note** that a more negative number correlates with having the disease (target = 0). A more positive number correlates with not having the disease (target = 1). A small number means there is little correlation with the target.
# 
# Highest correlations are with asymptomatic chest pain, thal_3(probably reversible defect), and exang(angina=yes).

# Create some training and test data to use

# In[ ]:


from sklearn.model_selection import train_test_split

X = data.drop(['target'], axis=1)
y = data['target']

def setup_data(X_in, y_in):
    return train_test_split(X_in, y_in, test_size=0.2, random_state=seed)


# Create model

# In[ ]:


import tensorflow
tensorflow.random.set_seed(seed) 
from tensorflow.keras.layers import Input, Dense, ELU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

input = Input(shape=X.shape[1])

m = Dense(1024)(input)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

#####

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

output = Dense(1, activation='sigmoid')(m)

model = Model(inputs=[input], outputs=[output])

model.summary()


# In[ ]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True)

rlp = ReduceLROnPlateau(monitor='val_loss', patience=9, verbose=1, factor=0.5, cooldown=5, min_lr=1e-10)


# Let's set aside 20% of the data as a test set and use the rest for training.

# In[ ]:


X_remainder, X_test, y_remainder, y_test = setup_data(X,y)


# Train our model

# In[ ]:


X_train, X_validation, y_train, y_validation = setup_data(X_remainder, y_remainder)

history = model.fit(X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=200,
    verbose=2,
    callbacks=[es, rlp],
    validation_data=(X_validation, y_validation),
    shuffle=True
         ).history


# Visualize the training

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 14))

ax1.plot(history['loss'], label='Train loss')
ax1.plot(history['val_loss'], label='Validation loss')
ax1.legend(loc='best')
ax1.set_title('Loss')

ax2.plot(history['acc'], label='Train accuracy')
ax2.plot(history['val_acc'], label='Validation accuracy')
ax2.legend(loc='best')
ax2.set_title('Accuracy')

plt.xlabel('Epochs')
sns.despine()
plt.show()


# For curiosity's sake, let's evaluate the model on the full data set

# In[ ]:


model.evaluate(X, y, batch_size=BATCH_SIZE, verbose=1)


# It appears our model can predict the target accurately. But of course this would be a lot more interesting with a much larger training dataset and a separate test dataset.
# 
# OK. Let's do a final training on the full training set without validation.

# In[ ]:


history = model.fit(X_remainder,
    y_remainder,
    batch_size=BATCH_SIZE,
    epochs=200,
    verbose=2,
    callbacks=[es, rlp],
    shuffle=True
         ).history


# Let's see how it does for the test data we set aside earlier.

# In[ ]:


model.evaluate(X_test, y_test, verbose=0)


# > > So we achieved more than 80% accuracy. Pretty reasonable for such a small dataset.

# In[ ]:


from sklearn.metrics import confusion_matrix

y_prob = model.predict(X_test)
y_pred = np.around(y_prob)
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix


# Sensitivity = true positive rate, Specificity = true negative rate

# In[ ]:


total=sum(sum(confusion_matrix))

sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
print('Sensitivity : ', sensitivity )

specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
print('Specificity : ', specificity)

