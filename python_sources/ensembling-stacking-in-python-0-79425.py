#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings("ignore")
get_ipython().system('ls ../input')
get_ipython().system('ls ../input/*')

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


train = pd.read_csv("../input/titanic/train.csv")
y_train = train["Survived"]
train.drop("Survived",axis=1,inplace=True)
test = pd.read_csv("../input/titanic/test.csv")
sample_sub = pd.read_csv("../input/titanic/gender_submission.csv")


# <h3>Feature Preprocessing and Extraction</h3>

# In[ ]:


def search_title(name):
    match = re.search("([A-Za-z]+)\.",name)
    if match: return match.group(1)
    else: return ""
train.Name = train.Name.apply(search_title)
test .Name = test .Name.apply(search_title)


# In[ ]:


train["Name"].unique(), test["Name"].unique()


# In[ ]:


train["Age"].fillna(train["Age"].median(), inplace=True)
test ["Age"].fillna(train["Age"].median(), inplace=True)
train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)
test ["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)
train["Cabin"] = train["Cabin"].apply((lambda x: 0 if type(x) == float else 1))
test ["Cabin"] = test ["Cabin"].apply((lambda x: 0 if type(x) == float else 1))
train["TotalFamily"] = train["SibSp"] + train["Parch"]
test ["TotalFamily"] = test ["SibSp"] + test ["Parch"]
test ["Fare"].fillna(train["Fare"].median(), inplace=True)


# In[ ]:


oneHot = OneHotEncoder()
oneHotName = OneHotEncoder(handle_unknown="ignore")
ordinal = OrdinalEncoder()
train = pd.concat([train,
                   pd.DataFrame(oneHot.fit_transform(train[["Embarked"]]).toarray()),
                   pd.DataFrame(ordinal.fit_transform(train[["Sex"]])), 
                   pd.DataFrame(oneHotName.fit_transform(train[["Name"]]).toarray())], axis=1)
train.drop(columns=["Sex","Embarked","Name","Ticket","PassengerId"],inplace=True)

test  = pd.concat([test,
                   pd.DataFrame(oneHot.transform(test[["Embarked"]]).toarray()),
                   pd.DataFrame(ordinal.transform(test[["Sex"]])), 
                   pd.DataFrame(oneHotName.transform(test[["Name"]]).toarray())], axis=1)
test.drop(columns=["Sex","Embarked","Name","Ticket","PassengerId"],inplace=True)


# In[ ]:


print("Train shape: {}\nTest  shape: {}".format(train.shape, test.shape))


# <h3>Tree-based Models</h3>

# In[ ]:


forest = RandomForestClassifier(min_samples_leaf=3, random_state=42)
boost  = GradientBoostingClassifier(min_samples_leaf=3, random_state=42)
#I did not fine tune the models so there are many space for improvements
forest.fit(train, y_train)
boost .fit(train, y_train)
train_preds = forest.predict(train)
train_preds_boost = boost.predict(train)
print("Train Accuracy\nRandomForest: {}\nGBDT:         {}".format(round(precision_score(y_train,train_preds),3),round(precision_score(y_train,train_preds_boost),3)))


# <h3>Linear Models</h3>

# In[ ]:


test.describe()


# In[ ]:


scaler = StandardScaler()
train_scaled = np.hstack([scaler.fit_transform(train.iloc[:,[1,4]]), train.iloc[:,[0,2,3] + [i for i in range(5,len(train.columns))]].values])
test_scaled  = np.hstack([scaler.transform(test.iloc[:,[1,4]]), test.iloc[:,[0,2,3] + [i for i in range(5,len(test.columns))]].values])
#scaling the numeric features and concatenating them with the categorical ones
#scaling does have a significant performance impact on the result (but I didn't leave out a validation set :< , feel free to fork and play around)


# In[ ]:


svc = SVC(random_state=42)
svc.fit(train_scaled, y_train)
train_preds_svc = svc.predict(train_scaled)
log_reg = LogisticRegression(random_state=True,n_jobs=-1)
log_reg.fit(train_scaled, y_train)
train_preds_log = log_reg.predict(train_scaled)
print("Train Accuracy\nLogisticRegression: {}\nSVC:                {}".format(round(precision_score(y_train,train_preds_log),3),round(precision_score(y_train,train_preds_svc),3)))
#0.801, 0.818


# <h3>ANN</h3>

# In[ ]:


model = None
model = tf.keras.Sequential()
model.add(layers.Input((len(train.columns),)))
model.add(layers.Dense(64,activation=tf.nn.elu))
model.add(layers.Dense(32,activation=tf.nn.elu))
model.add(layers.Dense(1))
model.compile(loss="binary_crossentropy",metrics=["accuracy"],optimizer="adam")
model.summary()


# In[ ]:


es = EarlyStopping(monitor="accuracy",patience=5,restore_best_weights=True)
val = int(len(train) * 0.1)
#I left some validation data out for early stopping
history = model.fit(train_scaled[:-val], y_train[:-val].values, validation_data=(train_scaled[-val:], y_train[-val:].values), callbacks=[es], epochs=40, verbose=0)


# In[ ]:


train_preds_ann = model.predict(train_scaled)
train_preds_ann = [int(round(i[0])) for i in train_preds_ann]
print("Train Accuracy\nANN: {}".format(round((np.array(train_preds_ann) == y_train).sum() / len(train),3)))


# <h3>Submission</h3>

# In[ ]:


X_preds = np.hstack([forest.predict(test).reshape(-1,1), 
                     log_reg.predict(test_scaled).reshape(-1,1), 
                     boost.predict(test).reshape(-1,1), 
                     svc.predict(test_scaled).reshape(-1,1), 
                     np.array([int(round(i[0])) for i in model.predict(test_scaled)]).reshape(-1,1)])


# In[ ]:


#I tried to use xgboost as a second-layer model but it did not go well
#It turns out that random forest is a very good predictor so I decided to use it as the main model and change its prediction only if all the other four disagrees
preds = []
for row in X_preds:
    pred = row[0]
    if np.array([row[1:] != pred]).sum() >= 4:
        pred = 0 if pred == 1 else 1
    preds.append(pred)


# In[ ]:


sample_sub["Survived"] = preds
sample_sub.to_csv("submission.csv",index=False)


# In[ ]:


print("\"Predicted survival rate\": {}".format(round((sample_sub["Survived"] == 1).sum() / len(test),3)))
print("Train survival rate:       {}".format(round(((y_train == 1).sum() / len(train)),3)))
#distribution looks good-ish

