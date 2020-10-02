#!/usr/bin/env python
# coding: utf-8

# # The Previous Best Score is:
# Version 28 - 78.947%

# # Peek to the Data
# 
# First of all, we need to import the `Train` and `Test` Datasets provided to this competition.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import time
start_time = time.time()
import gc


# In[ ]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")


# We will now check the first few elements and the last few elements of each Dataset.

# In[ ]:


print(train.info())
display(train)


# In[ ]:


print(test.info())
display(test)


# In[ ]:


train.hist()


# In[ ]:


test.hist()


# Seeing this, we can determine that we only have a few samples, roughly 891 rows of train data. Further on, we will process and augment the data thorugh Feature Engineering

# # Feature Engineering
# 
# First, we must make a new function to make sure that the application of feature engineering is reproducible across all DataFrames. The basic Feature engineering will be based on the above Histogram for both `train` and `test` data.

# In[ ]:


#Fill the NA
train.Cabin = train.Cabin.fillna('NaN')
test.Cabin = test.Cabin.fillna('NaN')


# In[ ]:


# Define function to extract titles from passenger names
# Based from Saif Uddin's kernel
import re

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def process(dataset):
    # Create a new feature Title, containing the titles of passenger names
    dataset['Title'] = dataset['Name'].apply(get_title)
    # Group all non-common titles into one single grouping "Rare"
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    return dataset['Title']


# In[ ]:


def feature_engineering(from_df):
    df = pd.DataFrame()
    #PassengerID isn't really relevant for training but we will keep it
    df['PassengerId'] = from_df.PassengerId
    
    #For Name titles, we will be focusing on the name prefixes
    dummy = pd.get_dummies(process(from_df))
    for i in dummy.columns:
        df['Title_'+str(i)] = dummy[i]
        
    #For Pclass, we will be making a one-hot-encoding of the three classes
    dummy = pd.get_dummies(from_df.Pclass.fillna(0))
    for i in dummy.columns:
        df['Pclass_'+str(i)] = dummy[i]
        
    #For the sex, we will be turning 'male' to 0, and 'female' to 1
    df['Sex'] = [0 if x=='male' else 1 for x in from_df.Sex]
    
    #For age, we will group them by age group, in gaps of some years up to 100 yo
    gap = 3
    ages = from_df.Age.fillna(0)
    for i in range(gap, 100, gap):
        df[str(i-gap)+'<Age<'+str(i)] = [int(x<i and x>i-gap) for x in ages]
        
    #We wouldn't touch the number of Family (siblings or parent) But we only need
    #to know if there is a family or none
    df['SibSp'] = [1 if x!=0 else 0 for x in from_df.SibSp.fillna(0)]
    df['Parch'] = [1 if x!=0 else 0 for x in from_df.Parch.fillna(0)]
    
    #For the Fare, similar to the age, we would group them by their aggregate amount
    fares = from_df.Fare.fillna(0)
    amount = 20
    for i in range(amount, 600, amount):
        df[str(i-amount)+'<Fare<'+str(i)] = [int(x<i and x>i-amount) for x in fares]
        
    #We will turn the Cabin into a dummy too using the First letter of their Cabins
    cabins = from_df.Cabin
    for key in set(train.Cabin.tolist()+test.Cabin.tolist()):
        try: #To avoid overwriting the whole Cabin Key, we keep the previous values too
            df['Cabin_'+key[0]] = [int(x[0]==key[0] or y) for x, y in zip(cabins, df['Cabin_'+key[0]])]
        except:
            df['Cabin_'+key[0]] = [int(x[0]==key[0]) for x in cabins]
    #For `Embarked` we will use the same patter to `Cabin`
    for i in set(train.Embarked.tolist()+test.Embarked.tolist()):
        df['Embarked_'+str(i)] = [int(x==i) for x in from_df.Embarked]

    return df


# In[ ]:


#For the Train
X = feature_engineering(train)
Y = train.Survived
print(X.info())
display(X)


# In[ ]:


#For the test
test_X = feature_engineering(test)
print(test_X.info())
display(test_X)


# Now, let's analyze the correlation between features using Seaborn.

# In[ ]:


#We will now remove some unused columns so as to simplify the dataset, and we will use the same columns for the
#prediction
drop_cols = ['PassengerId']
for col in X.columns:
    if X[col].sum() == 0:
        drop_cols.append(col)
train_cols = [col for col in X.columns if col not in drop_cols]
display(train_cols)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(X[train_cols].corr(),annot=False, linewidth=0.2)
fig=plt.gcf()
fig.set_size_inches(20,20)
plt.show()


# # Model
# 
# Now that we are finished with our Feature Engineering, we will now design our model. Since we are only doing a binary prediction, we will be using a basic Keras NN Model to do this. As they say "Simple is Best."

# In[ ]:


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split as tts

def c_model(shape, lr=0.001):
    a = 2000
    
    x = Sequential()
    x.add(Dense(a, activation='relu', input_dim=shape))
    x.add(Dropout(0.25))
    
    x.add(Dense(a//4, activation='relu', kernel_regularizer=l1_l2(0.002, 0.002)))
    x.add(BatchNormalization())
    x.add(Dropout(0.75))
    
    x.add(Dense(a//10, activation='relu', kernel_regularizer=l1_l2(0.002, 0.002)))
    x.add(BatchNormalization())
    x.add(Dropout(0.75))
    
    x.add(Dense(a//10, activation='relu', kernel_regularizer=l1_l2(0.002, 0.002)))
    x.add(BatchNormalization())
    x.add(Dropout(0.75))
    
    x.add(Dense(2, activation='softmax'))
    
    opt = Adam(lr=lr, decay=1e-3, beta_1=0.95, beta_2=0.995, amsgrad=True)
    x.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return x


# As you can see, we did as 2-category softmax insted of a 1-target binary. This makes the probability of both "Dying" and "Surviving" apparent, which should increase the accuracy.

# In[ ]:


import matplotlib.pyplot as plt
def plotter(history, n):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('MODEL ACCURACY #%i' %n)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.ylim(top=1, bottom=0.01)
    plt.savefig('history_accuracy_{}.png'.format(n))
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('MODEL LOSS #%i' %n)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    #plt.ylim(top=2, bottom=0.01)
    plt.savefig('history_loss_{}.png'.format(n))
    plt.show()


# We will do a multi-fold training using the same model on different permutations of the data. The prediction is weighted by the model's prediction accuracy before ensemble. Since we will be using a `Model Checkpoint`, the accuracy should be maintained.

# In[ ]:


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
lrr = ReduceLROnPlateau(monitor = 'val_accuracy',
                         patience = 50,
                         verbose = 1,
                         factor = 0.5,
                         min_lr = 1e-8)

es1= EarlyStopping(monitor='val_loss',
                   mode='min',
                   verbose=1,
                   patience=100,
                   restore_best_weights=True)
es2= EarlyStopping(monitor='val_accuracy',
                   mode='max',
                   verbose=1,
                   patience=750,
                   restore_best_weights=True)



folds = 10
epochs = 5000
batch_size = X.shape[0]

train_history = []
all_predictions = None
all_scores = []

for n in range(1, folds+1):
    mcp = ModelCheckpoint(f'model_weights_{n}.hdf5', monitor='val_accuracy', verbose=0,
                          save_best_only=True, mode='max', period=10)
    
    print(f"Currently training on Fold: {n}")
    
    xt, xv, yt, yv = tts(X[train_cols], Y, test_size=0.2, random_state=1771, shuffle=True, stratify=Y)
    model = c_model(xt.shape[1], 3e-4)
    hist = model.fit(xt, yt, validation_data=(xv, yv),
                     epochs=epochs, batch_size=batch_size,
                     callbacks=[lrr, es1, es2, mcp], verbose=0)
    
    train_history.append(hist)
    plotter(hist, n)
    
    loss, acc = model.evaluate(xv, yv)
    predicted = model.predict(test_X[train_cols])
    
    model.load_weights(f'model_weights_{n}.hdf5')
    loss2, acc2 = model.evaluate(xv, yv)
    
    if acc < acc2 or (acc==acc2 and loss < loss2):
        predicted = model.predict(test_X[train_cols])
        loss, acc = loss2, acc2
        
    all_scores.append([loss, acc])
    
    if acc > .77:
        try:
            all_predictions += predicted*acc
        except:
            all_predictions = predicted*acc


# # Prediction
# 
# Last, we do our prediction and save them for the competition.

# In[ ]:


sub = pd.DataFrame()
sub['PassengerId'] = test.PassengerId
sub['Survived'] = np.argmax(all_predictions, axis=1)
display(sub)
sub.to_csv("submission.csv", index=False)


# # End
# 
# That's all for now. Thank you for reading this kernel.

# In[ ]:


total = time.time() - start_time
h = total//3600
m = (total%3600)//60
s = total%60
print("Total Spent time: %i:%i:%i" %(h, m ,s))

