#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Important Remark
# **This notebook is used for newbie students who wants to start their journey in Machine Learning. It is made to explain how to solve a ML problem by building a basic model and techniques to improve it. When we want to solve any ML problem, we better start with basic and simple models and see how they behave to face the actual problem. This important step helps us to understand more our dataset and decide what to do to improve our solution. **
# 
# **This notebook will be updated from time to time.**

# We start by importing the famous libraries. We can always add a new library if needed (it happens all the time).

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from xgboost.sklearn import XGBClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')


# We import our dataset:

# In[ ]:


train_df = pd.read_csv("/kaggle/input/glass/glass.csv", header = 0)


# # Simple Visualization

# First thing is to visual the first 4 lines of our dataset. We can see the columns that contain information related to the component of the glass and the last column "**Type**" which is the label or the class column.

# In[ ]:


train_df.head()


# Now we visualize the number of appearances for each class:

# In[ ]:


train_df['Type'].value_counts()


# We have 7 classes. But one of them (label=4) is not excisting (this is explained in the dataset description). Which means we will focus on the 6 other classes for our visualizaiton because class 4 will be always 0. First thing that I prefer always to do is visualizing our data in details. That would help us understand and decide more on our dataset.

# In[ ]:


per_class = train_df['Type'].value_counts()
plt.figure(figsize=(17,7))
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(per_class, labels=['Class 1','Class 2','Class 3','Class 4','Class 5','Class 6',],colors=['tab:blue','tab:orange','tab:purple','tab:olive','tab:green','tab:red'],autopct='%1.1f%%')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


# In[ ]:


sb.countplot(train_df['Type'])


# We devide our dataset into features and label. The features contain the first nine columns and the label is only the last column (type):

# In[ ]:


y=train_df.iloc[:,9].values
X=train_df.iloc[:,:9].values


# We set the Y values into categorical. It is a way of encoding the data to prepare it for classification.

# In[ ]:


ycateg=to_categorical(y-1,7)


# # Classification model 1: Random Forest Classifier

# First things first: we prepare our dataset by spliting and scaling it.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, ycateg, test_size=0.2)


# We scale or normalize our data. It is important to avoid the negative effect of different scales in our data.

# In[ ]:


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# We import our Random Forest Classifier already definied in scikit learn library.

# In[ ]:


rfc=RandomForestClassifier(n_estimators=250)
rfc.fit(X_train, y_train)
pred_rfc=rfc.predict(X_test)


# For the evaluation of the model, we use the classification report in scikit learn:

# In[ ]:


print(classification_report(y_test,pred_rfc))


# We calculate the cross validation accuracy. (check here if you don't know cross validation: [Here](https://en.wikipedia.org/wiki/Cross-validation_(statistics))

# In[ ]:


accuracies = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)
score=accuracies.mean()
print("Accuracy CV: %.2f%%" % (score*100))


# For large ML problems, this accuracy is very low of course. But for you now, that's a good start. Don't forget that the dataset is small, and that's your first model implemented. It needs improvement of course, we can tune the hyperparameters. But for now, you can first of all start by modifying a bit the number of estimators in your classifier and see how it affects the accuracy. We will talk more in details about this and improve our model after. Now let's see how to build a Neural Networks model.

# # Classification model 2: Neural Networks with Learning rate modification

# We build a simple neural network for our small dataset.

# **IMPORTANT: ** We are changing the Learning rate per epoch. This is a good tuning way to get better results.

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers import Dense, Convolution1D, MaxPool1D, Flatten, Dropout
from keras.layers import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler


model = Sequential()
model.add(Dense(64, input_shape=(9,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='sigmoid'))


# In[ ]:


LR_START = 0.0001
LR_MAX = 0.00005 * 8
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 4
LR_SUSTAIN_EPOCHS = 6
LR_EXP_DECAY = .8
BATCH_SIZE=2

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = np.random.random_sample() * LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
lr_callback = LearningRateScheduler(lrfn, verbose=True)


# In[ ]:


model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['categorical_accuracy'])
STEPS_PER_EPOCH = 171
VAL_STEP=1
    
es_callback = EarlyStopping(min_delta=0, patience=10, verbose=1, mode='auto', restore_best_weights=True)

history=model.fit(X_train, y_train,epochs=200, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VAL_STEP, callbacks = [lr_callback, es_callback],validation_data=(X_test,y_test))


# In[ ]:


def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])


# In[ ]:


display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)


# In[ ]:


display_training_curves(history.history['categorical_accuracy'], history.history['val_categorical_accuracy'], 'accuracy', 212)


# **This notebook will be modified, completed and updated very soon - 1 May 2020**
