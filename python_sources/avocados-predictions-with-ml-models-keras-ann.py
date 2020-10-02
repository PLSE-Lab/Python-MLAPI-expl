#!/usr/bin/env python
# coding: utf-8

# # Avocados - ML models, keras ANN, seaborn plots
# ## Introduction
# I like avocados.
# 
# ## Import libraries and data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/avocado.csv')


# In[ ]:


df.head(10)


# In[ ]:


df.describe()


# In[ ]:


df = df.drop(['Unnamed: 0', 'Date'], axis = 1)
df.info()


# ## Visualizations
# Total volume is higly correlated with small bags and total bags which are correleted to each other too.

# In[ ]:


f, ax = plt.subplots(1, 1, figsize=(10,8))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax)
ax.set_title("Correlation Matrix", fontsize=14)
plt.show()


# Indeed, the smaller bags, the higher number of them  are taken

# In[ ]:


sns.jointplot(x='Small Bags',y='Total Bags',data=df, color='red')


# Surprisingly or not, price doesn't change among the years'

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(10,6))
sns.boxplot(x='year',y='AveragePrice',data=df,color='red')


# Average Price distribution shows that for most cases price of avocado is between 1.1, 1.4.

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(10,6))
price_val = df['AveragePrice'].values
sns.distplot(price_val, color='r')
ax.set_title('Distribution of Average Price', fontsize=14)
ax.set_xlim([min(price_val), max(price_val)])


# ##  Implementing machine learning models
# ### Data prepearing and encoding categorical variables
# 

# In[ ]:


X = df.drop(['AveragePrice'], axis = 1).values
y = df['AveragePrice'].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 8] = labelencoder_X_1.fit_transform(X[:, 9])
labelencoder_X_2 = LabelEncoder()
X[:, 9] = labelencoder_X_2.fit_transform(X[:, 10])
labelencoder_X_3 = LabelEncoder()
X[:, 10] = labelencoder_X_3.fit_transform(X[:, 10])


# ### Standardize the variables

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop(['AveragePrice', 'type', 'year', 'region'],axis=1))
scaled_features = scaler.transform(df.drop(['AveragePrice', 'type', 'year', 'region'],axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[1:9])
df_feat.head()


# ### Train test split, label encoding

# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn import preprocessing
from sklearn import utils
lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(y_train)
y_test = lab_enc.fit_transform(y_test)


# ### ML models implementation

# In[ ]:


from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg = OneVsRestClassifier(logreg, n_jobs=1)
logreg.fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn = OneVsRestClassifier(knn, n_jobs=1)
knn.fit(X_train,y_train)
pred_knn = knn.predict(X_test)

from sklearn.svm import SVC
svc = SVC()
svc = OneVsRestClassifier(svc, n_jobs=1)
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)

from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree = OneVsRestClassifier(decision_tree, n_jobs=1)
decision_tree.fit(X_train, y_train)
pred_tree = decision_tree.predict(X_test)


# ### Confusion matrixes

# In[ ]:


sns.jointplot(x=y_test, y=pred_logreg, color= 'g')
sns.jointplot(x=y_test, y=pred_knn, color= 'g')
sns.jointplot(x=y_test, y=pred_svc, color= 'g')
sns.jointplot(x=y_test, y=pred_tree, color= 'g')

plt.show()


# As we see, KNN method works the best here, logistic regression and decision tree completely do not work. We could tune k value.
# 

# In[ ]:


error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# We choose k = 3.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn = OneVsRestClassifier(knn, n_jobs=1)
knn.fit(X_train,y_train)
pred_knn = knn.predict(X_test)
sns.jointplot(x=y_test, y=pred_knn, color= 'g')
plt.plot()


# ## Keras neural network will appear soon. Stay tuned :)
# And it is. Here i present the best model for that moment. If you have any advices, please let me know :). For now it is not working the best, as SVC model. I tried a model with a few number of units, various batch size, drop out or without and chose the best one. Maybe different number of hidden layers would help.

# In[ ]:


print(X_train.shape[1])
print(X_train.shape[0])
print(len(np.unique(y_train)))
print((len(np.unique(y_train)) + X_train.shape[0]) /2)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from numpy import argmax

BATCH_SIZE = 1000
EPOCHS = 30
VALIDATION_SPLIT = 0.1
file_path="weights_base3.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
callbacks_list = [checkpoint, early]

def get_model():
    model = Sequential()
    model.add(Dense(7428, input_dim=X_train.shape[1]))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7428))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy']
                 )
    model.summary()
    return model
model_nn = get_model()
model_nn.fit(X_train, np_utils.to_categorical(y_train),
                  batch_size=BATCH_SIZE, 
                  epochs=EPOCHS,
                  callbacks=callbacks_list,
                  validation_split=VALIDATION_SPLIT
             )
model_nn.load_weights(file_path)
pred_ann = argmax(model_nn.predict(X_test), axis = 1)


# In[ ]:


sns.jointplot(x=y_test, y=pred_ann, color= 'g')
plt.plot()

