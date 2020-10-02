#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense, Reshape, Dropout, LSTM, RepeatVector, TimeDistributed


# In[ ]:


train = pd.read_csv("../input/janatahack-mobility-analysis/train.csv")
test = pd.read_csv("../input/janatahack-mobility-analysis/test.csv")
sample_submission = pd.read_csv("../input/janatahack-mobility-analysis/sample_submission.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.isnull().sum().sum()


# In[ ]:


train.fillna(train.mean(),inplace=True)
train.fillna(train.mode().iloc[0],inplace=True)

test.fillna(test.mean(),inplace=True)
test.fillna(test.mode().iloc[0],inplace=True)


# In[ ]:


# imp = SimpleImputer(missing_values=np.nan,strategy="most_frequent")
# imp.fit_transform(train)


# In[ ]:


train.isnull().sum().sum()


# In[ ]:


train.shape


# In[ ]:


#train.dropna(inplace=True)


# In[ ]:


for col in ["Trip_Distance","Customer_Since_Months","Life_Style_Index","Var1","Var2","Var3"]:
  plt.figure(figsize=(10,4))
  plt.xlim(train[col].min(), train[col].max())
  sns.boxplot(x=train[col])


# In[ ]:


# train = train[train['Life_Style_Index']<4.8]
train = train[train['Var1']<200]
train = train[train['Var2']<120]
train = train[train['Var3']<2000]


# In[ ]:


train.shape


# In[ ]:


le = LabelEncoder()
train["Type_of_Cab_label"] = le.fit_transform(train.Type_of_Cab)
train = train.drop('Type_of_Cab',axis=1)

le = LabelEncoder()
test["Type_of_Cab_label"] = le.fit_transform(test.Type_of_Cab)
test = test.drop('Type_of_Cab',axis=1)


# In[ ]:


le = LabelEncoder()
train["Confidence_Life_Style_Index_label"] = le.fit_transform(train.Confidence_Life_Style_Index)
train = train.drop('Confidence_Life_Style_Index',axis=1)

le = LabelEncoder()
test["Confidence_Life_Style_Index_label"] = le.fit_transform(test.Confidence_Life_Style_Index)
test = test.drop('Confidence_Life_Style_Index',axis=1)


# In[ ]:


le = LabelEncoder()
train["Destination_Type_label"] = le.fit_transform(train.Destination_Type)
train = train.drop('Destination_Type',axis=1)

le = LabelEncoder()
test["Destination_Type_label"] = le.fit_transform(test.Destination_Type)
test = test.drop('Destination_Type',axis=1)


# In[ ]:


le = LabelEncoder()
train["Gender_label"] = le.fit_transform(train.Gender)
train = train.drop('Gender',axis=1)

le = LabelEncoder()
test["Gender_label"] = le.fit_transform(test.Gender)
test = test.drop('Gender',axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.Surge_Pricing_Type.unique()


# In[ ]:


X = train.copy()
y = X['Surge_Pricing_Type']
X = X.drop(['Surge_Pricing_Type','Trip_ID'],axis=1)


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=18)


# In[ ]:


test = test.drop(['Trip_ID'],axis=1)


# Random Forest

# In[ ]:


model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=10, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=150,
                       n_jobs=None, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)
model.fit(X_train,y_train)


# Logistic Regression

# In[ ]:


# model = LogisticRegression(random_state=42,solver = 'liblinear').fit(X_train, y_train)


# Decision Tree Classifier

# In[ ]:


# model = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
#                        criterion='gini', max_depth=9, max_features='auto',
#                        max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_weight_fraction_leaf=0.0,
#                        random_state=42)
# model.fit(X_train,y_train)


# SVM

# In[ ]:


# model = SVC(kernel='rbf')
# model.fit(X_train,y_train)


# CNN

# In[ ]:


# model = Sequential()
# model.add(Reshape((1,X_train.shape[1],1)))
# model.add(Conv2D(filters = 32, kernel_size = (2,2),padding = 'Same',
#              activation ='relu', input_shape = (1,X_train.shape[1],1)))
# model.add(Flatten())
# model.add(Dense (500, activation='tanh'))
# model.add(Dropout(0.1))
# model.add(Dense (100, activation='tanh'))
# model.add(Dense (1, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam',
#               metrics=['accuracy'])


# In[ ]:


# model.fit(np.array(X_train), np.array(y_train), epochs=5, batch_size=1000, validation_data=(np.array(X_val),np.array(y_val)))


# Predict

# In[ ]:


preds = model.predict(X_val)


# In[ ]:


accuracy_score(y_val, preds)


# In[ ]:


preds = model.predict(test)


# In[ ]:


submission = sample_submission.copy()


# In[ ]:


submission['Surge_Pricing_Type'] = preds


# In[ ]:


submission.to_csv("submission.csv",index=False)

