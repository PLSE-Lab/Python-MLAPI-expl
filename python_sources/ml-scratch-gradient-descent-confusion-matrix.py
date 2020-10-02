#!/usr/bin/env python
# coding: utf-8

# > ## Logistic Regression using Gradient Descent algorithm is implemented with metrics from scratch, no pre-build functions/library is used
# > ### Kick starter for Deep Learning learners

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.datasets import load_iris
iris = load_iris()
print("Keys: {}\nData Shape: {}\nType: {} [{}D]\nFeature Names: {}\nDescription: {}".format(iris.keys(), iris.data.shape, 
    type(iris.data), iris.data.ndim,
    iris.feature_names, iris.DESCR))


# In[ ]:


iris.data[:5]


# In[ ]:


iris.target_names


# In[ ]:


import pandas as pd
data = pd.read_csv('../input/Iris.csv')
data.head()


# ### **As same data available from sklearn dataset in the form of nd-array used here instead csv**

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
X = iris.data[:, :2]        #Take the first two features
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
fig = plt.figure()
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=iris['target'],
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()


# In[ ]:


class LogisticRegression:
    def __init__(self, alpha = 0.0001, iteration = 10000, verbose = (True, 1000), activation='sigmoid'):
        self.alpha = alpha
        self.iteration = iteration
        self.verbose = verbose
        self.theta = np.ndarray
        self.cost = np.ndarray
        self.activation = activation
    
    def print_(self, text, skip=True):        
        if(self.verbose[0] and not skip):
            print(text)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def logistic_sigmoid(self, x):
        return np.exp(x) / 1 + np.exp(x)

    def hx(self, theta, X, n):        
        m = X.shape[0]                                              #Nos. of Records/Rows
        h = np.ones((m, 1))                                         #initializing with 1 for all rows
        theta = theta.reshape(1, n+1)                               #[[theta0....thetaN]] 1D->2D
        for i in range(0, m):            
            if self.activation == 'logistic-sigmoid':
                h[i] = self.logistic_sigmoid(float(np.matmul(theta, X[i])))      
            elif self.activation == 'softmax':
                h[i] = np.exp(np.matmul(theta, X[i]))                  
            else:
                h[i] = self.sigmoid(float(np.matmul(theta, X[i])))   #1/(1+e^-(theta.T*X)) (Sigmoid)
        
        if self.activation == 'softmax':
            h = h / sum(h)                                           #SoftMax Implementation
        h = h.reshape(m)                                             #2D->1D/Flatten
        return h

    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()    #Cross Entropy

    def gradient_descent(self, theta, h, X, y, n):
        #w = np.random.randn(2)
        iteration_count = 0
        m = X.shape[0]
        self.cost = np.ones(self.iteration)
        for i in range(0, self.iteration):
        #i=0
        #while True:
            iteration_count = 0 if iteration_count >= self.verbose[1] else iteration_count + 1                
            theta[0] = theta[0] - (self.alpha/m) * sum(h - y)   #* x0 Omitted as = 1
            for j in range(1, n+1):
                theta[j] = theta[j] - (self.alpha/m) * sum((h - y) * X.T[j])
            h = self.hx(theta, X, n)
            self.cost[i] = self.loss(h, y)
            self.print_('Cost/Iteration[{}]: {}'.format(i, self.cost[i]), not(iteration_count >= self.verbose[1]))
            #i +=1
        self.theta = theta.reshape(1, n+1)
    
    def fit(self, X, y, algo='gradient_descent'):
        n = X.shape[1]                              #Nos. of Features [x1....xN]
        m = X.shape[0]                              #no. of Rows/Records
        x0 = np.ones((m, 1))
        x = np.concatenate((x0, X), axis = 1)       #New Vector X: [x0, x1...xN]
        theta = np.zeros(n+1)                       #Initialize theta: [theta0...thetaN]
        fx = self.hx(theta, x, n)                   #Initial h for all rows = 1
        self.print_('Initial h(x):\n{}'.format(fx))
        self.gradient_descent(theta, fx, x, y, n)
        return self.theta

    def predict(self, X, showlog=False):
        n = X.shape[1]
        m = X.shape[0]
        x0 = np.ones((m, 1))
        x = np.concatenate((x0, X), axis = 1)
        y_pred_values = self.hx(self.theta, x, n)        
        self.print_('Predicted y:\n{}'.format(y_pred_values), skip=not(showlog))
        
        predicted_class = np.ndarray(m)
        for i in range(0, m):
            if y_pred_values[i] > 0.5:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 0
        return predicted_class, y_pred_values 

    def cost_minimization_curve(self, msg=''):
        plt.plot(np.arange(self.iteration), self.cost.tolist())
        plt.xlabel('cost')
        plt.ylabel('iteration')
        plt.title('Cost Minimisation. {}'.format(msg))
        plt.show()


# In[ ]:


class Metric:
    def __init__(self, y_hat, y_actual):
        self.y_hat = y_hat
        self.y_actual = y_actual       
        self.tp = self.fp = self.fn = self.tn = 0        
        
    def accuracy(self):
        k = 0
        for i in range(0, self.y_hat.shape[0]):
            if self.y_hat[i] == self.y_actual[i]:
                k += 1
        accuracy = k / self.y_actual.shape[0]
        return accuracy

    def precision_recall_f1(self):
        precision = recall = f1 = np.inf
        for i in range(0, self.y_hat.shape[0]):
            if self.y_hat[i] == self.y_actual[i] == 0:
                self.tp += 1
            elif self.y_hat[i] == 0 and self.y_actual[i] == 1:
                self.fp += 1
            elif self.y_hat[i] == 1 and self.y_actual[i] == 0:
                self.fn += 1
            elif self.y_hat[i] == self.y_actual[i] == 1:
                self.tn += 1
        try:
            precision = self.tp / (self.tp + self.fp)
        except:
            pass
        try:
            recall = self.tp / (self.tp + self.fn)  
        except:
            pass  
        if precision != np.inf or recall != np.inf:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def plot_confusion_matrix(self, normalized=False, class_names=np.ndarray(0)):
        confusion_matrix = np.array([[self.tp, self.fn], [self.fp, self.tn]])
        if(normalized):
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis = 1)[:, np.newaxis]
        print('Confusion Matrix:\n{}'.format(confusion_matrix))
                
        classes = np.ndarray([0, 1])
        if len(class_names) > 0:
            classes = class_names

        fig, ax = plt.subplots()
        im = plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax = ax)
        ax.set(
            xticks=np.arange(confusion_matrix.shape[1]),
            yticks=np.arange(confusion_matrix.shape[0]),
            # ... and label them with the respective list entries
            xticklabels = classes, yticklabels = classes,
            title = 'Confusion Matrix {}'.format('(Normalized)' if normalized else ''),
            ylabel = 'True label',
            xlabel = 'Predicted label')
                
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")       #Rotate the tick labels and set their alignment
        
         #Loop over data dimensions and create text annotations
        fmt = '.2f' if normalized else 'd'
        threshold = confusion_matrix.max() / 2.
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, 
                    format(confusion_matrix[i, j], fmt), ha='center', va='center',
                    color= 'white' if confusion_matrix[i, j] > threshold else 'black'
                    )
        fig.tight_layout()
        plt.show()


# In[ ]:


import sklearn.model_selection as skModel
X_train, X_test, y_train, y_test = skModel.train_test_split(iris['data'], iris['target'], random_state = 42)
print("X_train: ", X_train.shape, "X_test: ", X_test.shape)
#%%
import pandas as pd
def train_test_df(x_train, y_train, x_test, y_test):
    df_train = pd.DataFrame(X_train)
    df_train.columns = iris.feature_names
    df_train['class'] = y_train
    df_test = pd.DataFrame(X_test)
    df_test.columns = iris.feature_names
    df_test['class'] = y_test
    return df_train, df_test

df_train, df_test = train_test_df(X_train, y_train, X_test, y_test)
df_train[df_train['class'] != 2][:5]
df_train.describe()


# In[ ]:


filter_by_column_value = lambda df, colName, colValue: df[df[colName] != colValue]

train_set_01 = filter_by_column_value(df_train, 'class', 2)
train_set_12 = filter_by_column_value(df_train, 'class', 0)
train_set_20 = filter_by_column_value(df_train, 'class', 1)

train_set_01[:5]
train_set_12[:5]
train_set_20[:5]

test_set_01 = filter_by_column_value(df_test, 'class', 2)
test_set_12 = filter_by_column_value(df_test, 'class', 0)
test_set_20 = filter_by_column_value(df_test, 'class', 1)

test_set_01[:5]
test_set_12[:5]
test_set_20[:5]


# In[ ]:


#Visualisation of Training & Test set on basis of Sepal Features
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(8, 5))
ax1.scatter(train_set_01.iloc[:, 0], train_set_01.iloc[:, 1], c=train_set_01.iloc[:,-1:]['class'], cmap=plt.cm.Set1, edgecolor='k')
ax2.scatter(train_set_12.iloc[:, 0], train_set_12.iloc[:, 1], c=train_set_12.iloc[:,-1:]['class'], cmap=plt.cm.Set1, edgecolor='k')
ax3.scatter(train_set_20.iloc[:, 0], train_set_20.iloc[:, 1], c=train_set_20.iloc[:,-1:]['class'], cmap=plt.cm.Set1, edgecolor='k')
ax1.set_xlabel('sepal length')
ax1.set_ylabel('sepal width')
ax2.set_title('Train Set (sepal) Distribution')
plt.show()

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(8, 5))
ax1.scatter(test_set_01.iloc[:, 0], test_set_01.iloc[:, 1], c=test_set_01.iloc[:,-1:]['class'], cmap=plt.cm.Set1, edgecolor='k')
ax2.scatter(test_set_12.iloc[:, 0], test_set_12.iloc[:, 1], c=test_set_12.iloc[:,-1:]['class'], cmap=plt.cm.Set1, edgecolor='k')
ax3.scatter(test_set_20.iloc[:, 0], test_set_20.iloc[:, 1], c=test_set_20.iloc[:,-1:]['class'], cmap=plt.cm.Set1, edgecolor='k')
ax1.set_xlabel('sepal length')
ax1.set_ylabel('sepal width')
ax2.set_title('Test Set (sepal) Distribution')
plt.show()


# In[ ]:


#Visualisation of Training & Test set on basis of Petal Features
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(8, 5))
ax1.scatter(train_set_01.iloc[:, 2], train_set_01.iloc[:, 3], c=train_set_01.iloc[:,-1:]['class'], cmap=plt.cm.Set1, edgecolor='k')
ax2.scatter(train_set_12.iloc[:, 2], train_set_12.iloc[:, 3], c=train_set_12.iloc[:,-1:]['class'], cmap=plt.cm.Set1, edgecolor='k')
ax3.scatter(train_set_20.iloc[:, 2], train_set_20.iloc[:, 3], c=train_set_20.iloc[:,-1:]['class'], cmap=plt.cm.Set1, edgecolor='k')
ax1.set_xlabel('petal length')
ax1.set_ylabel('petal width')
ax2.set_title('Train Set (petal) Distribution')
plt.show()

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(8, 5))
ax1.scatter(test_set_01.iloc[:, 2], test_set_01.iloc[:, 3], c=test_set_01.iloc[:,-1:]['class'], cmap=plt.cm.Set1, edgecolor='k')
ax2.scatter(test_set_12.iloc[:, 2], test_set_12.iloc[:, 3], c=test_set_12.iloc[:,-1:]['class'], cmap=plt.cm.Set1, edgecolor='k')
ax3.scatter(test_set_20.iloc[:, 2], test_set_20.iloc[:, 3], c=test_set_20.iloc[:,-1:]['class'], cmap=plt.cm.Set1, edgecolor='k')
ax1.set_xlabel('petal length')
ax1.set_ylabel('petal width')
ax2.set_title('Test Set (petal) Distribution')
plt.show()

'''
Hence Taking Petal data for regession
'''


# In[ ]:


lm = LogisticRegression(iteration=80000, verbose=(True, 5000))
theta_01 = lm.fit(train_set_01.iloc[:,2:4], train_set_01.iloc[:,-1:]['class'])
lm.cost_minimization_curve()
print('Thetas[01]: {}'.format(lm.theta))


# In[ ]:


y_pred_01, y_pred_01_values = lm.predict(test_set_01.iloc[:, 2:4], showlog=True)
metric_01 = Metric(y_pred_01, test_set_01.iloc[:,-1:].reset_index()['class'])
pre_re_f1_01 = metric_01.precision_recall_f1() 
print('Accuracy: {}, \n(Precision, Recall, F1): {}'.format(metric_01.accuracy(), pre_re_f1_01))
metric_01.plot_confusion_matrix(normalized=False, class_names=iris.target_names)


# In[ ]:


train_set_12.loc[train_set_12['class'] == 2] = 0
lm = LogisticRegression(iteration=20000, verbose=(True, 2000))
theta_12 = lm.fit(train_set_12.iloc[:,2:4], train_set_12.iloc[:,-1:]['class'])
print('Thetas[12]: {}'.format(lm.theta))
lm.cost_minimization_curve()


# In[ ]:


y_pred_12, y_pred_12_values = lm.predict(train_set_12.iloc[:,2:4], showlog=True)
metric_12= Metric(y_pred_12, train_set_12.iloc[:,-1:].reset_index()['class'])
prf1_12 = metric_12.precision_recall_f1() 
print('Accuracy: {}, \n(Precision, Recall, F1): {}'.format(metric_12.accuracy(), prf1_12))
metric_12.plot_confusion_matrix(normalized=False, class_names=iris.target_names[1:3])


# In[ ]:


train_set_20.loc[train_set_20['class'] == 2] = 1
lm = LogisticRegression(iteration=20000, verbose=(True, 5000))
theta_20 = lm.fit(train_set_20.iloc[:,2:4], train_set_20.iloc[:,-1:]['class'])
print('Thetas[20]: {}'.format(lm.theta))
lm.cost_minimization_curve()


# In[ ]:


test_set_20.loc[test_set_20['class'] == 2] = 1
y_pred_20, y_pred_20_values = lm.predict(test_set_20.iloc[:,2:4], showlog=True)
metric_20 = Metric(y_pred_20, test_set_20.iloc[:,-1:].reset_index()['class'])
prf1_20 = metric_20.precision_recall_f1() 
print('Accuracy: {}, \n(Precision, Recall, F1): {}'.format(metric_20.accuracy(), prf1_20))
metric_20.plot_confusion_matrix(normalized=False, class_names=[iris.target_names[i] for i in (0, 2)])

