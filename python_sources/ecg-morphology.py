#!/usr/bin/env python
# coding: utf-8

# # Heartbeat classification from ECG morphology using Machine learning.
# 
# ## Motivation
# 
# Acording to [Wikipedia](https://en.wikipedia.org/wiki/Heart_arrhythmia) 
# Arrhythmia affects millions of people in the world. In Europe and North America, as of 2014, atrial fibrillation affects about 2% to 3% of the population. Atrial fibrillation and atrial flutter resulted in 112,000 deaths in 2013, up from 29,000 in 1990. Sudden cardiac death is the cause of about half of deaths due to cardiovascular disease and about 15% of all deaths globally. About 80% of sudden cardiac death is the result of ventricular arrhythmias. Arrhythmias may occur at any age but are more common among older people. Arrhythmias are coused by problems with the electrical conduction system of the heart. A number of tests can help with diagnosis including an electrocardiogram (ECG) and Holter monitor. Regarding ECG, the diagnosis is based on the carefully analysis that a specialized doctor perform on the shape and structure of the independent heartbeats. This process is tedious and requires time. 
# ![ecg](./pics/ecg.png)
# 
# In this work, we aim to classify the heart beats extracted from an ECG using machine learning, based only on the lineshape (morphology) of the individual heartbeats. The goal would be to develop a method that automatically detects anomallies and help for the prompt diagnosis of arrythmia.
# 
# ## Data
# 
# The original data comes from the [MIT-BIH Arrythmia database](https://physionet.org/content/mitdb/1.0.0/). Some details of the dataset are briefly summarized below:
# 
# + 48.5 hour excerpts of two-channel ambulatory ECG recordings
# + 48 subjects studied by the BIH Arrhythmia Laboratory between 1975 and 1979.
# + 23 recordings randomly selected from a set of 4000 24-hour ambulatory ECG recordings collected from a mixed at Boston's Beth Israel Hospital.
# + 25 recordings were selected from the same set to include less common but clinically significant arrhythmias.
# + Two or more cardiologists independently annotated each record (approximately 110,000 annotations in all).
# 
# Although the one that its currently being used here is taken from [kaggle](https://www.kaggle.com/alexandrefarb/mitbih-arrhythmia-database-de-chazal-class-labels). In this dataset the single heartbeats from the ECG were extracted using the [Pam-Tompkins algorithm](https://en.wikipedia.org/wiki/Pan-Tompkins_algorithm). Each row of the dataset represents a QRS complex as the one schematically shown below:
# <img src="./pics/qrs.png" width="300">
# These QRS are taken from the MLII lead from the ECG. As observed in the firts figure above, there is also the V1 lead, which is not used in this work.
# For further details on how the data was generated, the interested can read the original paper by [Chazal et al.](https://www.ncbi.nlm.nih.gov/pubmed/15248536)
# 
# The different arrythmia classes are:
# 
# 0. Normal
# 1. Supraventricular ectopic beat
# 2. Ventricular ectopic beat
# 3. Fusion Beat
# 
# This means that we are facing a multi-class classification problem with four classes. 
# 
# ## Strategy.
# 
# To achieve our goal we will go the following way:
# 
# 1. Data standardisation
# 2. Selection of three promising ML algorithms.
# 3. Fine tunning of the best models
# 4. Model comparison
# 5. Build  a Neural Network
# 6. Compare 
# 
# Rather important, in order to evaluate the performance of our models is to choose the appropiate metrics. In this case we will be checking the confusion matrix and the f1 score with macro averaging.
# 
# 
# ## Data Loading and first insights.
# 

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier

import seaborn as sn

hbeat_signals = pd.read_csv("../input/mitbih-arrhythmia-database-de-chazal-class-labels/DS1_signals.csv", header=None)
hbeat_labels = pd.read_csv("../input/mitbih-arrhythmia-database-de-chazal-class-labels//DS1_labels.csv", header=None)

print("+"*50)
print("Signals Info:")
print("+"*50)
print(hbeat_signals.info())
print("+"*50)
print("Labels Info:")
print("+"*50)
print(hbeat_labels.info())
print("+"*50)


# In[ ]:


hbeat_signals.head()


# In[ ]:


hbeat_signals.describe()


# In[ ]:


hbeat_signals = hbeat_signals.sub(0.5, axis=0)
hbeat_signals.describe()


# Let's have a look at how the data looks like for the different types of heartbeats.

# In[ ]:


# Collect data of different hheartbeats in different lists
#class 0
cl_0_idx = hbeat_labels[hbeat_labels[0] == 0].index.values
cl_N = hbeat_signals.iloc[cl_0_idx]
#class 1
cl_1_idx = hbeat_labels[hbeat_labels[0] == 1].index.values
cl_S = hbeat_signals.iloc[cl_1_idx]
#class 2
cl_2_idx = hbeat_labels[hbeat_labels[0] == 2].index.values
cl_V = hbeat_signals.iloc[cl_2_idx]
#class 3
cl_3_idx = hbeat_labels[hbeat_labels[0] == 3].index.values
cl_F = hbeat_signals.iloc[cl_3_idx]

# make plots for the different hbeat classes
plt.subplot(221)
for n in range(3):
    cl_N.iloc[n].plot(title='Class N (0)', figsize=(10,8))
plt.subplot(222)
for n in range(3):
    cl_S.iloc[n].plot(title='Class S (1)')
plt.subplot(223)
for n in range(3):
    cl_V.iloc[n].plot(title='Class V (2)')
plt.subplot(224)
for n in range(3):
    cl_F.iloc[n].plot(title='Class F (3)')


# In[ ]:


#check if missing data
print("Column\tNr of NaN's")
print('+'*50)
for col in hbeat_signals.columns:
    if hbeat_signals[col].isnull().sum() > 0:
        print(col, hbeat_signals[col].isnull().sum()) 


# This means that there are no missing values to fill. We can now proceed to check if there are some correlations on the data.

# In[ ]:


joined_data = hbeat_signals.join(hbeat_labels, rsuffix="_signals", lsuffix="_labels")

#rename columns
joined_data.columns = [i for i in range(180)]+['class']


# In[ ]:


joined_data.head()


# In[ ]:


joined_data.describe()


# In[ ]:


categories_counts = joined_data['class'].value_counts()
print(categories_counts)


# In[ ]:


print("class\t%")
joined_data['class'].value_counts()/len(joined_data)


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)

for train_index, test_index in split.split(joined_data, joined_data['class']):
    strat_train_set = joined_data.loc[train_index]
    strat_test_set = joined_data.loc[test_index]    


# Let's now check if the train data fulfills the stratified conditions after the split was done.

# In[ ]:


print("class\t%")
strat_train_set['class'].value_counts()/len(strat_train_set)


# In[ ]:


print("class\t%")
strat_test_set['class'].value_counts()/len(strat_test_set)


# Nice, we see that the amount of data with classes 0 to 3 in the train set maps to those from the original data.
# 
# We are ready to pick some ML models to start training with our data.
# We will use a brute force approach in the sense that we will try several models at once. For each model, we will do a 5-fold cross validation and depending on its metrics we will choose the best among them. To do that we will write a simple function that takes a list of models, and perfom the cross validation for each and prints its metrics, i.e., confusion matrix, precission, recall and f1 score.

# In[ ]:


train_df = strat_train_set
test_df  = strat_test_set


# In[ ]:


from sklearn.utils import resample

df_0 = train_df[train_df['class']==0]
df_1 = train_df[train_df['class']==1]
df_2 = train_df[train_df['class']==2]
df_3 = train_df[train_df['class']==3]

df_0_downsample = resample(df_0,replace=True,n_samples=10000,random_state=122)
df_1_upsample   = resample(df_1,replace=True,n_samples=10000,random_state=123)
df_2_upsample   = resample(df_2,replace=True,n_samples=10000,random_state=124)
df_3_upsample   = resample(df_3,replace=True,n_samples=10000,random_state=125)

train_df=pd.concat([df_0_downsample,df_1_upsample,df_2_upsample,df_3_upsample])


# In[ ]:


plt.figure(figsize=(10,5))

plt.subplot(2,2,1)
plt.plot(df_0.iloc[0,:180])

plt.subplot(2,2,2)
plt.plot(df_1.iloc[0,:180])

plt.subplot(2,2,3)
plt.plot(df_2.iloc[0,:180])

plt.subplot(2,2,4)
plt.plot(df_3.iloc[0,:180])

plt.show()


# In[ ]:


from keras.utils.np_utils import to_categorical

target_train = train_df['class']
target_test  = test_df['class']
y_train = to_categorical(target_train)
y_test  = to_categorical(target_test)


# In[ ]:


X_train = train_df.iloc[:,:180].values
X_test  = test_df.iloc[:,:180].values

X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
X_test  = X_test.reshape(len(X_test), X_test.shape[1], 1)


# In[ ]:


import keras
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.GaussianNoise(0.01, input_shape=(X_train.shape[1], X_train.shape[2])))

model.add(layers.Conv1D(64, 16, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool1D(pool_size=4, strides=2, padding="same"))

model.add(layers.Conv1D(64, 12, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool1D(pool_size=3, strides=2, padding="same"))

model.add(layers.Conv1D(64, 8, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool1D(pool_size=2, strides=2, padding="same"))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(4, activation='softmax'))

print(model.summary())


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


from keras.callbacks import EarlyStopping
callbacks = [EarlyStopping(monitor='val_loss', patience=8)]

history = model.fit(X_train, y_train, callbacks=callbacks, validation_data=(X_test, y_test), epochs = 20, batch_size = 128)


# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.legend()


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import seaborn as sns

y_pred = model.predict_classes(X_test)

y_test_category = y_test.argmax(axis=-1)

# Creates a confusion matrix
cm = confusion_matrix(y_test_category, y_pred) 

# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index   = ['N', 'S', 'V', 'F', ], 
                     columns = ['N', 'S', 'V', 'F', ])

plt.figure(figsize=(10,10))
sns.heatmap(cm_df, annot=True, fmt="d", linewidths=0.5, cmap='Blues', cbar=False, annot_kws={'size':14}, square=True)
plt.title('Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test_category, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test_category, y_pred, target_names=['N', 'S', 'V', 'F']))

