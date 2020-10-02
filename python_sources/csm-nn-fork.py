#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


#fix random seed for reproducibility
np.random.seed(7)


# In[ ]:


#import dataset
dataframe = pd.read_csv('../input/meningitis_dataset.csv', delimiter=',')


# In[ ]:


dataframe.describe()


# In[ ]:


dataframe.info()


# In[ ]:


#non_categorical
#dataframe.dtypes.sample(50)
dataframe.dtypes.sample(20)


# In[ ]:


non_categorical = dataframe.select_dtypes(include='int64')


# In[ ]:


#rearrange noncategorical colums
#non_categorical = non_categorical[
non_categorical.head()
non_categorical.info()


# In[ ]:


#non_categorical = non_categorical['gender_male', 'gender_female', 'rural_settlement', 'urban_settlement', 'report_year', 'age', 'child_group', 'adult_group', 'cholra', 'diarrhoea', 'measles', 'viral_haemmorrhaphic_fever', 'ebola', 'marburg_virus', 'yellow_fever', 'rubella_mars', 'malaria', 'serotype', 'NmA', 'NmC', 'NmW', 'alive', 'dead', 'unconfirmed', 'confirmed', 'null_serotype', 'meningitis']


# In[ ]:


#split dataset into input(X) and output(Y) values
#X = non_categorical.iloc[:, 0,25]
#Y = non_categorical.iloc[:, 25]
#X = non_categorical.iloc[:, 1,26]
#Y = non_categorical.iloc[:, 26]
#X = non_categorical.ix[:, 1,26]
#Y = non_categorical.ix[:, 26]
#X = non_categorical.iloc[:, 1,26]
#X = non_catergorical['gender_male']
#X = non_categorical['gender_male','gender_female','rural_settlement','urban_settlement','report_year','age','child_group','adult_group','cholera','diarrhoea','measles']
X = non_categorical
Y = non_categorical.iloc[:, 26]


# In[ ]:


#create model
model = Sequential()
#model.add(Dense(12, input_dim=25, activation='relu'))
model.add(Dense(12, input_dim=27, activation='relu'))
#model.add(Dense(25, activation='relu'))
model.add(Dense(27, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


#compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


#fit the model
model.fit(X, Y, epochs=10, batch_size=10)


# In[ ]:


model.summary()


# In[ ]:


from keras.utils.vis_utils import plot_model


plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# In[ ]:


import matplotlib.pyplot as plt
history = model.fit(X,Y, validation_split=0.25, epochs=10, batch_size=16, verbose=1)
#plotting training and validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


# In[ ]:


#plotting training and validation accuracy values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Loss')
plt.ylabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


# In[ ]:


#evaluate the model
scores = model.evaluate(X,Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:


#calculate prediction
predictions = model.predict(X)


# In[ ]:


print(predictions)


# In[ ]:


#round predictions
rounded_predictions = [round(X[0]) for X in predictions]
print(rounded_predictions)


# In[ ]:


#Now try to split dataset into test and train
import sklearn
from sklearn.model_selection import train_test_split


# In[ ]:


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[ ]:


Xtrain.shape


# In[ ]:


Ytrain.shape


# In[ ]:


Xtest.shape


# In[ ]:


Ytest.shape


# In[ ]:


#Now fit the model
#fit themodel
#the model
model.fit(Xtrain, Ytrain, epochs=10, batch_size=10)


# In[ ]:


#evaluate the model
scores = model.evaluate(X,Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:


#calculate prediction
predictions = model.predict(Xtest)
print(predictions)


# In[ ]:


#round predictions
rounded_predictions = [round(Xtest[0]) for Xtest in predictions]
print(rounded_predictions)


# In[ ]:


#evaluate the model
scores = model.evaluate(Xtest,Ytest)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:


from sklearn import metrics 


# In[ ]:


print(metrics.f1_score(Ytrain, Xtrain.iloc[:, 26]))


# In[ ]:


from sklearn.metrics import classification_report
test_size=0.33


# In[ ]:



seed=7


# In[ ]:


report= classification_report(Ytest, rounded_predictions)
print(report)


# In[ ]:


from sklearn import model_selection


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


scoring='roc_auc'
model2 = LogisticRegression()
kfold= model_selection.KFold(n_splits=10)
results = model_selection.cross_val_score(model2, Xtest, Ytest, cv=kfold, scoring=scoring)
print("ROC: %.3f (%.3f)" % (results.mean(), results.std()))


# In[ ]:


scoring='neg_log_loss'
model2 = LogisticRegression()
kfold= model_selection.KFold(n_splits=10)
results = model_selection.cross_val_score(model2, Xtest, Ytest, cv=kfold, scoring=scoring)
print("LogLoss: %.3f (%.3f)" % (results.mean(), results.std()))


# In[ ]:


scoring='accuracy'
model2 = LogisticRegression()
kfold= model_selection.KFold(n_splits=10)
results = model_selection.cross_val_score(model2, Xtest, Ytest, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# In[ ]:


scoring='roc_auc'
model2 = LogisticRegression()
kfold= model_selection.KFold(n_splits=10)
results = model_selection.cross_val_score(model2, Xtest, Ytest, cv=kfold, scoring=scoring)
print("ROC: %.3f (%.3f)" % (results.mean(), results.std()))


# In[ ]:




