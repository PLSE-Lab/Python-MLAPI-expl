#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import preprocessing
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from subprocess import check_output


# In[ ]:


gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test_df = pd.read_csv("../input/titanic/test.csv")
train_df = pd.read_csv("../input/titanic/train.csv")
test_df['Survived'] = gender_submission['Survived']


# In[ ]:


train_df.info()


# In[ ]:


train_data = train_df.copy()
train_data.drop(['Name','Ticket','PassengerId','Cabin'], axis=1, inplace=True)
train_data['Embarked'].fillna('S', inplace=True)
train_data['Age'].fillna(train_df['Age'].median(skipna=True), inplace=True)
train_data['Sex'] = train_data['Sex'].replace(['male','female'],[0,1])
train_data['TravelAlone']=np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)
train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)


# In[ ]:


train_data.head()


# In[ ]:


sns.barplot('Pclass', 'Survived', data=train_df, color="Green")
plt.show()


# In[ ]:


sns.barplot('Embarked', 'Survived', data=train_df, color="Red")
plt.show()


# In[ ]:


sns.barplot('TravelAlone', 'Survived', data=train_data, color="Blue")
plt.show()


# In[ ]:


sns.barplot('Sex', 'Survived', data=train_df, color="Yellow")
plt.show()


# In[ ]:


train_data.drop('Embarked', axis=1, inplace=True)


# In[ ]:


train_data.head()


# In[ ]:


plt.figure(figsize=(14,6))
ax = sns.kdeplot(train_data["Age"][train_data.Survived == 1], color="lightblue", shade=True)
sns.kdeplot(train_data["Age"][train_data.Survived == 0], color="orange", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for survived and died passengers')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


# In[ ]:


plt.figure(figsize=(14,6))
ax = sns.kdeplot(train_data["Fare"][train_data.Survived == 1], color="lightblue", shade=True)
sns.kdeplot(train_data["Fare"][train_data.Survived == 0], color="orange", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Fare for survived and died passengers')
ax.set(xlabel='Fare')
plt.xlim(-20,200)
plt.show()


# In[ ]:


X = 
y = 
theta = 


# In[ ]:


def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def net_input(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)

def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(theta, x))

def cost_function(self, theta, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x)))
    return total_cost

def gradient(self, theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,   x)) - y)


# In[ ]:


def fit(self, x, y, theta):
    opt_weights = fmin_tnc(func=cost_function, x0=theta,
                  fprime=gradient,args=(x, y.flatten()))
    return opt_weights[0]
parameters = fit(X, y, theta)


# In[ ]:


def predict(self, x):
    theta = parameters[:, np.newaxis]
    return probability(theta, x)
def accuracy(self, x, actual_classes, probab_threshold=0.5):
    predicted_classes = (predict(x) >= 
                         probab_threshold).astype(int)
    predicted_classes = predicted_classes.flatten()
    accuracy = np.mean(predicted_classes == actual_classes)
    return accuracy * 100
accuracy(X, y.flatten())


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


X = train_data['Pclass','Sex','Travel_Alone','Age','Fare']
y = train_data['Survived']
model = LogisticRegression()


# In[ ]:


from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss

# create X (features) and y (response)
X = train_data['Pclass','Sex','Travel_Alone','Age','Fare']
y = train_data['Survived']

# use train/test split with different random_state values
# we can change the random_state values that changes the accuracy scores
# the scores change a lot, this is why testing scores is a high-variance estimate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# check classification scores of logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))


# In[ ]:


plt.scatter(train_data['Age'], train_data['Fare'])


# In[ ]:




