#!/usr/bin/env python
# coding: utf-8

# In[268]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


# In[225]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[288]:


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


# In[227]:


from sklearn import metrics


# In[356]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import Callback
import keras.backend as k
import tensorflow as tf


# In[229]:


data = pd.read_csv('../input/creditcard.csv')


# In[230]:


data.head()


# In[231]:


data.shape


# In[234]:


data.Class.value_counts()


# In[235]:


492 / 284315


# In[236]:


plt.plot(list(range(1, data.shape[0]+1)), data.Class, 'r+')
plt.show()


# In[237]:


np.max(data.Amount), np.min(data.Amount) 


# In[238]:


colors = {1: 'red', 0: 'green'}
fraud = data[data.Class == 1]
not_fraud = data[data.Class == 0]
fig, axes = plt.subplots(1, 2)
axes[0].scatter(list(range(1, fraud.shape[0]+1)), fraud.Amount, color='red')
axes[1].scatter(list(range(1, not_fraud.shape[0]+1)), not_fraud.Amount, color='green')
plt.show()


# In[239]:


X = data.loc[:, data.columns.tolist()[1:30]]


# In[240]:


X = X.as_matrix()


# In[241]:


Y = data.loc[:, 'Class']


# In[242]:


Y = Y.as_matrix()


# In[357]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[66]:


np.bincount(y_train), np.bincount(y_test) 


# In[260]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# In[335]:


def train_model(model):
    model.fit(X_train, y_train)


# In[326]:


#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def predict_model(model):
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(metrics.auc(fpr, tpr))
    print(metrics.classification_report(y_test, y_pred))
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=['0', '1'],
                          title='Confusion matrix, without normalization')


# ## MLPClassifier

# In[263]:


mlc = MLPClassifier()


# In[264]:


train_model(mlc)


# In[270]:


predict_model(mlc)


# In[271]:


mlc_2 = MLPClassifier(hidden_layer_sizes=200)


# In[272]:


train_model(mlc_2)


# In[273]:


predict_model(mlc_2)


# In[274]:


mlc_3 = MLPClassifier(hidden_layer_sizes=50)


# In[275]:


train_model(mlc_3)


# In[276]:


predict_model(mlc_3)


# ## KNeighborsClassifier

# In[277]:


knn = KNeighborsClassifier()


# In[278]:


train_model(knn)


# In[279]:


predict_model(knn)


# In[280]:


knn_3 = KNeighborsClassifier(n_neighbors=3, n_jobs=3)


# In[281]:


train_model(knn_3)


# In[282]:


predict_model(knn_3)


# In[283]:


knn_2 = KNeighborsClassifier(n_neighbors=2, n_jobs=3)


# In[284]:


train_model(knn_2)


# In[285]:


predict_model(knn_2)


# ## SVC

# In[286]:


svc = SVC()


# In[287]:


train_model(svc)


# In[289]:


predict_model(svc)


# ## DecisionTreeClassifier

# In[290]:


dtc = DecisionTreeClassifier()


# In[291]:


train_model(dtc)


# In[292]:


predict_model(dtc)


# ## RandomForestClassifier

# In[293]:


rfc = RandomForestClassifier()


# In[294]:


train_model(rfc)


# In[295]:


predict_model(rfc)


# In[296]:


rfc_20 = RandomForestClassifier(max_depth=20)


# In[298]:


train_model(rfc_20)


# In[299]:


predict_model(rfc_20)


# In[300]:


rfc_30 = RandomForestClassifier(max_depth=30)


# In[301]:


train_model(rfc_30)


# In[302]:


predict_model(rfc_30)


# ## AdaBoostClassifier

# In[303]:


abc = AdaBoostClassifier()


# In[304]:


train_model(abc)


# In[305]:


predict_model(abc)


# ## GaussianNB

# In[306]:


gnb = GaussianNB()


# In[307]:


train_model(gnb)


# In[308]:


predict_model(gnb)


# ## LogisticRegression

# In[309]:


lr = LogisticRegression()


# In[310]:


train_model(lr)


# In[311]:


predict_model(lr)


# In[312]:


lr_2 = LogisticRegression(C=0.1)


# In[313]:


train_model(lr_2)


# In[314]:


predict_model(lr_2)


# In[315]:


lr_3 = LogisticRegression(solver='newton-cg')


# In[316]:


train_model(lr_3)


# In[317]:


predict_model(lr_3)


# ## NN 

# In[371]:


model = Sequential()
model.add(Dense(256, activation='tanh', input_dim=29))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[372]:


model.fit(X_train, y_train, epochs=5)


# In[373]:


y_pred = model.predict_classes(X_test)


# In[374]:


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
print(metrics.auc(fpr, tpr))
print(metrics.classification_report(y_test, y_pred))
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=['0', '1'],
                          title='Confusion matrix, without normalization')


# ## replicate the smallest class

# In[375]:


fraud = X_train[y_train==1]


# In[376]:


y_fraud = y_train[y_train==1]


# In[377]:


X_train.shape


# In[378]:


for _ in range(5):
    copy_fraud = np.copy(fraud)
    y_fraud_copy = np.copy(y_fraud)
    X_train = np.concatenate((X_train, copy_fraud))
    y_train = np.concatenate((y_train, y_fraud_copy))


# In[379]:


p = np.random.permutation(X_train.shape[0])


# In[380]:


X_train = X_train[p]
y_train = y_train[p]


# ## MLP Classifier

# In[349]:


mlp = MLPClassifier()


# In[350]:


train_model(mlp)


# In[351]:


predict_model(mlp)


# ## DecisionTreeClassifier

# In[352]:


dtc = DecisionTreeClassifier()


# In[353]:


train_model(dtc)


# In[354]:


predict_model(dtc)


# ## NN

# In[381]:


model = Sequential()
model.add(Dense(256, activation='tanh', input_dim=29))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[382]:


model.fit(X_train, y_train, epochs=5)


# In[383]:


y_pred = model.predict_classes(X_test)


# In[384]:


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
print(metrics.auc(fpr, tpr))
print(metrics.classification_report(y_test, y_pred))
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=['0', '1'],
                          title='Confusion matrix, without normalization')


# In[385]:


model.fit(X_train, y_train, epochs=5)


# In[386]:


y_pred = model.predict_classes(X_test)


# In[387]:


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
print(metrics.auc(fpr, tpr))
print(metrics.classification_report(y_test, y_pred))
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=['0', '1'],
                          title='Confusion matrix, without normalization')


# In[388]:


model.fit(X_train, y_train, epochs=5)


# In[389]:


y_pred = model.predict_classes(X_test)


# In[390]:


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
print(metrics.auc(fpr, tpr))
print(metrics.classification_report(y_test, y_pred))
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=['0', '1'],
                          title='Confusion matrix, without normalization')


# In[391]:


model.fit(X_train, y_train, epochs=15)


# In[392]:


y_pred = model.predict_classes(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
print(metrics.auc(fpr, tpr))
print(metrics.classification_report(y_test, y_pred))
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=['0', '1'],
                          title='Confusion matrix, without normalization')


# In[398]:


model.fit(X_train, y_train, epochs=10)


# In[399]:


y_pred = model.predict_classes(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
print(metrics.auc(fpr, tpr))
print(metrics.classification_report(y_test, y_pred))
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=['0', '1'],
                          title='Confusion matrix, without normalization')


# In[400]:


model.fit(X_train, y_train, epochs=20)


# In[401]:


y_pred = model.predict_classes(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
print(metrics.auc(fpr, tpr))
print(metrics.classification_report(y_test, y_pred))
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=['0', '1'],
                          title='Confusion matrix, without normalization')


# ### We should stop before the last 20 epocs
