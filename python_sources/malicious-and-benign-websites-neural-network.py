#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import warnings  
warnings.filterwarnings('ignore')


# # Overview
# The goal of this notebook is to see if a simple neural network can be trained to classify a website as malicious or benign. 

# In[ ]:


dataset = pd.read_csv('../input/malicious-and-benign-websites/dataset.csv')
dataset.describe(include='all')


# ## Data Analysis and Preparation
# 
# Let's begin by having a look at the data - cleaning it up where appropriate. 

# In[ ]:


dataset.head()


# The URL column is a unique identifier so we may as well remove that. 

# In[ ]:


dataset.drop('URL', axis =1, inplace=True)


# In[ ]:


# Look for null values 
print(dataset.isnull().sum())


# There are null values each for the DNS_QUERY_TIMES and SERVER columns, so we could easily drop these records / place a dummy value instead without affecting the data too much. The CONTENT_LENGTH column is a bit more concerning, we can't afford to drop that many records (almost half the dataset) and interpolating might distort the data somewhat. Given that there are plenty of other features, I'm choosing to drop the column.

# In[ ]:


dataset.drop('CONTENT_LENGTH', axis =1, inplace=True)
dataset.dropna(inplace=True)
print(dataset.isnull().sum())


# Looking at the data there are several categorical features (WHOIS_COUNTRY, SERVER etc.). For simplicity we'll ignore these features in our network.

# In[ ]:


dataset.drop(['CHARSET', 'SERVER', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE', ], axis =1, inplace=True)


# Let's have a look at the correlations of the remaining features.

# In[ ]:


corr = dataset.corr()
corr.style.background_gradient(cmap='coolwarm')


# Looks like there are some highly correlated features in there. A lot of these make sense - for example longer URLs will probably contain more special characters. Let's remove some of the more highly correlated features. 

# In[ ]:


dataset.drop(['TCP_CONVERSATION_EXCHANGE','URL_LENGTH','APP_BYTES','SOURCE_APP_PACKETS','REMOTE_APP_PACKETS','SOURCE_APP_BYTES','REMOTE_APP_BYTES'], axis = 1, inplace=True)
corr = dataset.corr()
corr.style.background_gradient(cmap='coolwarm')


# That looks a lot better, let's examine the data to see if it's suitable to use in the model.

# In[ ]:


import seaborn as sns
sns.distplot(dataset.loc[dataset['Type'] == 1]['NUMBER_SPECIAL_CHARACTERS'], bins = 50, color='red')
sns.distplot(dataset.loc[dataset['Type'] == 0]['NUMBER_SPECIAL_CHARACTERS'], bins = 50, color='blue')


# The red bars are the malicious websites - there are some definite odd spikes there so this looks promising.

# In[ ]:


sns.distplot(dataset.loc[dataset['Type'] == 1]['DIST_REMOTE_TCP_PORT'], bins = 50, color='red')


# In[ ]:


sns.distplot(dataset.loc[dataset['Type'] == 0]['DIST_REMOTE_TCP_PORT'], bins = 50, color='blue')


# In[ ]:


print(dataset.loc[dataset['Type'] == 0]['DIST_REMOTE_TCP_PORT'].value_counts())


# In[ ]:


print(dataset.loc[dataset['Type'] == 1]['DIST_REMOTE_TCP_PORT'].value_counts())


# Looks like malicious websites generally have don't have a port associated (0 isn't a valid port?) not sure if we can infer anything from this or nor but we'll leave it in for now. It may be worth removing it later and observing the affect on the model.

# In[ ]:


sns.distplot(dataset.loc[dataset['Type'] == 1]['REMOTE_IPS'], bins = 50, color='red')
sns.distplot(dataset.loc[dataset['Type'] == 0]['REMOTE_IPS'], bins = 50, color='blue')


# REMOTE_IPS is described as 'this variable has the total number of IPs connected to the honeypot'. Looks like malicious websites have a slightly lower grouping of remote IPs connected than benign.

# In[ ]:


sns.distplot(dataset.loc[dataset['Type'] == 1]['APP_PACKETS'], bins = 50, color='red')
sns.distplot(dataset.loc[dataset['Type'] == 0]['APP_PACKETS'], bins = 50, color='blue')


# Hard to infer much from this, looks like there might be an outlier. APP_PACKETS is '...the total number of IP packets generated during the communication between the honeypot and the server'. So a large number of these could be a technical error.

# In[ ]:


sns.boxplot(dataset['APP_PACKETS'])


# Given there's a couple of values definitely way out of the normal range we'll remove these. 

# In[ ]:


dataset = dataset[((dataset.APP_PACKETS - dataset.APP_PACKETS.mean()) / dataset.APP_PACKETS.std()).abs() < 3]


# In[ ]:


sns.boxplot(dataset['APP_PACKETS'])


# That looks a lote better. Let's have a look at the histogram again.

# In[ ]:


sns.distplot(dataset.loc[dataset['Type'] == 1]['APP_PACKETS'], bins = 50, color='red')
sns.distplot(dataset.loc[dataset['Type'] == 0]['APP_PACKETS'], bins = 50, color='blue')


# In[ ]:


print(dataset.loc[dataset['Type'] == 1]['DIST_REMOTE_TCP_PORT'].value_counts())


# In[ ]:


print(dataset.loc[dataset['Type'] == 0]['DIST_REMOTE_TCP_PORT'].value_counts())


# Nothing too obvious. Again the spike at 0, these might correlate with the 0 on remote IPs. 

# In[ ]:


sns.distplot(dataset.loc[dataset['Type'] == 1]['DNS_QUERY_TIMES'], bins = 50, color='red')
sns.distplot(dataset.loc[dataset['Type'] == 0]['DNS_QUERY_TIMES'], bins = 50, color='blue')


# In[ ]:


print(dataset['DNS_QUERY_TIMES'].value_counts())


# The graph is slightly misleading - there aren't actually any negative values (thankfully). Again nothing too obvious but we'll leave it our model. 

# ## The Model
# 
# Given our reduced dataset, let's start trying to create a model. We'll start by scaling the data then splitting it into a test and train set. We'll then use a simple logistic regression model to baseline the accuracy then see if we can perform better using a nerual network.

# In[ ]:


# Scale data then split
from sklearn import preprocessing
# Separate into train and test as well as features and predictor
X = dataset.drop('Type',axis=1) #Predictors
y = dataset['Type']
X = preprocessing.scale(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)


# In[ ]:


# Method for evaluating results
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
def calculateScores(y_test, predictions):
    accuracy = 100*accuracy_score(y_test, predictions)
    precision = 100*precision_score(y_test, predictions)
    recall = 100*recall_score(y_test, predictions)
    f1 = 100*f1_score(y_test, predictions)
    print (' Accuracy  %.2f%%' % accuracy)
    print (' Precision %.2f%%'% precision)
    print (' Recall    %.2f%%'% recall)
    print (' F1        %.2f%%'% f1)
    print('Confusion Matrix')
    print(confusion_matrix(y_test,predictions))
    return {'Accuracy':accuracy, 'F1': f1}


# In[ ]:


from sklearn.linear_model import LogisticRegression
reg = LogisticRegression(solver='lbfgs')
model = reg.fit(X_train, y_train)
predictions = model.predict(X_test)
scores = calculateScores(y_test, predictions)


# So our base model has 86% accueracy, but pretty poor precision and recall. From the confusion matrix we can see that while it's pretty good at predicting benign websites, it's poor at predicting the malicious ones, which makes it pretty useless if we wanted to use it in the real world. 
# 
# Let's see if we can improve this with a neural network. We'll use the standard SciKit learn class, starting off by using all the default values, before attempting to optimise it by adjusting its parameters.

# In[ ]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=1)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
scores = calculateScores(y_test, predictions)


# Straight away that's a small improvement on the simple model, it is better at classifying the malicious websites, as seen in the confusion matrix and the F1 score.
# 
# Let's have a play with the parameters to see if this can be improved.

# In[ ]:


def predict( X_train, y_train, **kwargs):
    mlp = MLPClassifier(**kwargs, random_state=1)
    mlp.fit(X_train, y_train)
    return mlp.predict(X_test)


# In[ ]:


def calculateScoresNoOutput(y_test, predictions):
    accuracy = 100*accuracy_score(y_test, predictions)
    precision = 100*precision_score(y_test, predictions)
    recall = 100*recall_score(y_test, predictions)
    f1 = 100*f1_score(y_test, predictions)
    return {'Accuracy':accuracy, 'F1': f1}


# In[ ]:


# Let's try the different solvers
solvers = ['lbfgs', 'sgd', 'adam']
results = []
for solver in solvers:
    result_dict = calculateScoresNoOutput(y_test, predict(X_train, y_train, solver=solver))
    result_dict['Solver'] = solver
    results.append(result_dict)
df = pd.DataFrame(results, columns = ['Solver','Accuracy', 'F1'])
df


# Looks like the lbfgs solver is the best, with highest accuray and F1 scores. We'll use that from now on. 

# In[ ]:


# Generalise attempting different values
def try_different_values(values, column_name, X_train, y_train, **kwargs):
    results = []
    for value in values:
        kwargs[column_name] = value
        result_dict = calculateScoresNoOutput(y_test, predict(X_train, y_train, **kwargs))
        result_dict[column_name] = value
        results.append(result_dict)
    df = pd.DataFrame(results, columns = [column_name,'Accuracy', 'F1'])
    return df


# In[ ]:


activations = ['identity', 'logistic', 'tanh', 'relu']
try_different_values(activations, 'activation', X_train, y_train, solver='lbfgs')


# Another improvement using the logistic activation. Let's try adjusting the regularisation

# In[ ]:


alphas = []
for i in range(5,40):
     alphas.append(1/(2**i))
alpha_df = try_different_values(alphas, 'alpha', X_train, y_train, solver='lbfgs', activation='logistic')


# In[ ]:


alpha_df.set_index('alpha', inplace=True)
alpha_df.plot()


# In[ ]:


print(alpha_df.loc[alpha_df['Accuracy'].idxmax()])
print(alpha_df.loc[alpha_df['F1'].idxmax()])


# The regularisation parameter doesn't seem to change much. Let's save it and try the batch size.

# In[ ]:


# Store alpha
alpha= 4.76837158203125e-07


# In[ ]:


batch_sizes = [2 ** e for e in range(10)]
batch_df = try_different_values(batch_sizes, 'batch_size', X_train, y_train, solver='lbfgs', activation='logistic', alpha=alpha)


# In[ ]:


batch_df.set_index('batch_size', inplace=True)
batch_df.plot()


# Batch size doesn't seem to be affecting the accuracy - let's move on to the hidden layers. 

# In[ ]:


layers = []
for i in range (1,25):
    layers+= [(i)]
layers_df = try_different_values(layers, 'hidden_layer_sizes', X_train, y_train, solver='lbfgs', activation='logistic', alpha=alpha)


# In[ ]:


layers_df.set_index('hidden_layer_sizes', inplace=True)
layers_df.plot()


# In[ ]:


print(layers_df.loc[layers_df['Accuracy'].idxmax()])
print(layers_df.loc[layers_df['F1'].idxmax()])


# There's a definite advatage at 5 hidden layers. What if we add a second layer. 

# In[ ]:


layers = []
for i in range (1,15):
    for j in range(1,15):
        layers+= [(i,j)]
layers_df = try_different_values(layers, 'hidden_layer_sizes', X_train, y_train, solver='lbfgs', activation='logistic', alpha=alpha)


# In[ ]:


layers_df.set_index('hidden_layer_sizes', inplace=True)
layers_df.plot()


# In[ ]:


#layers_df = layers_df.reset_index()
layers_df = layers_df.reset_index()
print(layers_df.iloc[[layers_df['Accuracy'].idxmax()]])
print(layers_df.iloc[[layers_df['F1'].idxmax()]])


# So the best result with another hidden layer isn't better than a single hidden layer of size. We'll try a third just to see.

# In[ ]:


layers = []
for i in range (1,10):
    for j in range(1,10):
        for k in range (1,10):
            layers+= [(i,j,k)]
layers_3_df = try_different_values(layers, 'hidden_layer_sizes', X_train, y_train, solver='lbfgs', activation='logistic', alpha=alpha)


# In[ ]:


layers_3_df.set_index('hidden_layer_sizes', inplace=True)
layers_3_df.plot()


# In[ ]:


layers_3_df = layers_3_df.reset_index()
print(layers_3_df.iloc[[layers_3_df['Accuracy'].idxmax()]])
print(layers_3_df.iloc[[layers_3_df['F1'].idxmax()]])


# ## Conclusion
# Looking at the final results. We've got up to 91% accuracy, with an F1 score of 65%. This is qutie good considering the rather naive neural network implementation and lazy data preparation. Whilst not the most complete implementation it does show that a neural network is a viable model for this data. 
# ### Future Work
# A more considered approad to implementing the neural network could be considered i.e. is the lbfgs solver still the best with a three layer hidden network? The data could be better prepared and the categorical data should be included to see if it adds any value. The other issue is that there is not a large amont of data, specifically malicious websites. If more data could be collected it could help the model, and help stop any problems of overfitting.
# 
# 
