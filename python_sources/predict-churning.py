#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


# In[ ]:


data = pd.read_csv("/kaggle/input/churn-modelling/Churn_Modelling.csv")


# In[ ]:


data.head()


# In[ ]:


# drop unnecessary columns

data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1)


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


# transform vairbales in right format

for col in ['NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited']:
    data[col] = data[col].astype('category')


# In[ ]:


data.info()


# In[ ]:


data.describe().T


# ## Exploration

# ### Salary

# In[ ]:


plt.figure(figsize=(12,5))

sns.distplot(data['EstimatedSalary'],kde = False)


# Uniform distribution, no outliers

# ### Credit Score

# In[ ]:


plt.figure(figsize=(12,5))

sns.distplot(data['CreditScore'],kde = False)


# Slight outliers with a credit score of zero and above 800.
# 
# I cut them out of the data to have a somewhat normal distribution.

# In[ ]:


plt.figure(figsize=(12,5))

sns.distplot(data[(data['CreditScore'] <=840) & (data['CreditScore'] >=450)]['CreditScore'],kde = False)


# In[ ]:


data = data[(data['CreditScore'] <=840) & (data['CreditScore'] >=450)]


# ### Balance

# In[ ]:


plt.figure(figsize=(12,5))

sns.distplot(data['Balance'],kde = False)


# Creating binary variable --> balance = 0 or balance > 0

# In[ ]:


balance = [0 if i == 0 else 1 for i in data['Balance']]


# In[ ]:


pd.Series(balance).value_counts()/len(data)*100


# Somewhat balanced, I leave it like this

# In[ ]:


# add the new binary variable and drop the original 

data['has_balance'] = pd.Series(balance)
data = data.drop('Balance', axis = 1)


# ### Country

# In[ ]:


data['Geography'].value_counts()/len(data)*100


# ### Number of products

# In[ ]:


data['NumOfProducts'].value_counts()


# Create new variable and drop original one

# In[ ]:


data['more_than1product'] = pd.Series([0 if i == 1 else 1 for i in data['NumOfProducts']])
data = data.drop('NumOfProducts', axis=1)


# In[ ]:


data['more_than1product'].value_counts()


# In[ ]:


data.head()


# ### Active Member

# In[ ]:


data['IsActiveMember'].value_counts()


# ### Has Credit Card

# In[ ]:


data['HasCrCard'].value_counts()


# Slightly unbalanced

# ### Tenure

# In[ ]:


data['Tenure'].value_counts()


# ### Gender

# In[ ]:


data['Gender'].value_counts()


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data['has_balance'] = pd.Categorical(data['has_balance'])
data['more_than1product'] = pd.Categorical(data['more_than1product'])


# In[ ]:


sns.heatmap(data.corr(), annot=True)


# In[ ]:


len(data)


# In[ ]:


backup = data.copy()


# In[ ]:


data = backup


# ## Target variable

# very unbalanced target variable --> over or undersampling recommended

# In[ ]:


data['Exited'].value_counts()


# Up or downsampling did not improve the results

# In[ ]:


from sklearn.utils import resample

# # Separate majority and minority classes
# df_majority = data[data.Exited==0]
# df_minority = data[data.Exited==1]
#  
# # Upsample minority class
# df_minority_upsampled = resample(df_minority, 
#                                  replace=True,     # sample with replacement
#                                  n_samples=7600,    # to match majority class
#                                  random_state=404) # reproducible results
#  
# # Combine majority class with upsampled minority class
# df_upsampled = pd.concat([df_majority, df_minority_upsampled])
#  
# # Display new class counts
# df_upsampled.Exited.value_counts()


# In[ ]:


len(data)


# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


data.info()


# ## Neural Network

# In[ ]:


from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Dropout

from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn import metrics

from scipy.stats import zscore


# In[ ]:


def encode_columns(column, data):
    
    data = pd.concat([data,pd.get_dummies(data[column],prefix=column)],axis=1)
    data.drop(column, axis=1, inplace=True)
    
    return data


# In[ ]:


data.columns


# In[ ]:


### ------------- encode categorical columns ----------------

categorical_columns = ['Geography',
                       'Gender',
                       'HasCrCard',
                       'IsActiveMember',
                       'has_balance',
                       'more_than1product']
    
for col in categorical_columns:
    data=encode_columns(col,data)


# In[ ]:


data.info()


# In[ ]:


data['CreditScore'] = zscore(data['CreditScore'])
data['Age'] = zscore(data['Age'])
data['Tenure'] = zscore(data['Tenure'])
data['EstimatedSalary'] = zscore(data['EstimatedSalary'])


# In[ ]:


x = data.drop('Exited', axis=1)
y = data['Exited']


# In[ ]:


x = np.asarray(x)
y = np.asarray(y)


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = Sequential()
model.add(Dense(100, input_dim=x.shape[1], activation='relu', kernel_initializer='random_normal'))
model.add(Dropout(0.5))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(25,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# compile the model
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=25, 
                        verbose=1, mode='min', restore_best_weights=True)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())


# In[ ]:


history = model.fit(X_train, y_train, validation_split=0.2, callbacks=[monitor], verbose=1, epochs=1000)

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print('Accuracy: %f' % (accuracy*100))
print('\n')


# In[ ]:


plt.rcParams["figure.figsize"] = (11,5)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# In[ ]:


from sklearn.metrics import roc_curve, auc


# Plot an ROC. pred - the predictions, y - the expected output.
def plot_roc(pred,y):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7,7))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:


prediction_proba = model.predict(X_test)


# In[ ]:


plot_roc(prediction_proba,y_test)


# ## Other models

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold


# In[ ]:


randf = RandomForestClassifier(class_weight={0: 0.60, 1:0.4}, random_state=22, criterion="entropy")

randf.fit(X_train, y_train)

randf_prediction=randf.predict(X_test)

accuracy_score(y_pred = randf_prediction, y_true= y_test)


# In[ ]:


from sklearn.metrics import  plot_roc_curve
plot_roc_curve(randf, X_test, y_test)
plt.title("ROC for RF")
plt.show()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


dtree = DecisionTreeClassifier(class_weight={0: 0.60, 1:0.4},random_state=22, criterion='entropy')
dtree.fit(X_train,y_train)


# In[ ]:


tree_predictions = dtree.predict(X_test)


# In[ ]:


print(classification_report(y_test,tree_predictions))


# In[ ]:


pipe_randf=make_pipeline(StandardScaler(), randf)


# In[ ]:


pipe_dtree = make_pipeline(StandardScaler(), 
                           dtree)


# In[ ]:


CV_dtree=cross_validate(pipe_dtree,X_train,y_train,scoring=["accuracy","recall","precision"],
                      cv=StratifiedKFold(n_splits=5))


# In[ ]:


print("The mean accuracy in the Cross-Validation is: {:.2f}%".format((np.mean(CV_dtree["test_accuracy"])*100)))

