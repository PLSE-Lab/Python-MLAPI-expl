#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics


# In[ ]:


data = pd.read_csv('../input/train.csv')
data.head()


# In[ ]:


data.dtypes


# In[ ]:


columns = data.columns[1:-1]
X = data[columns]
y = np.ravel(data['target'])


# In[ ]:


distribution = data.groupby('target').size() / data.shape[0] * 100.0
distribution.plot(kind='bar')
plt.show()


# In[ ]:


# show distribution of a single feature among 9 classes
for id in range(9):
    plt.subplot(3, 3, id + 1)
    data[data.target == 'Class_' + str(id + 1)].feat_20.hist()
plt.show()


# In[ ]:


# show relationship between all pairs of feature correlation
fig = plt.figure()
ax = fig.add_subplot(111) # 1 row, 1 col, 1st plot
cax = ax.matshow(X.corr(), interpolation='nearest')
fig.colorbar(cax)
plt.show()


# **Initiate our neural net with 2 hidden layers**
# **93x30x10x9**

# In[ ]:


print(X.shape[1])


# In[ ]:


#alpha is L-2 regularization coefficient
model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (30, 10), random_state = 0, verbose = True)
model.fit(X, y)


# In[ ]:


model.intercepts_  # weights of each neuron 30x10x9


# In[ ]:


print(model.coefs_[0].shape)
print(model.coefs_[1].shape)
print(model.coefs_[2].shape)


# In[ ]:


pred = model.predict(X)
pred


# In[ ]:


print(model.score(X, y))
print(sum(pred == y) / len(y))


# **Predict on test data**

# In[ ]:


test_data = pd.read_csv('../input/test.csv')
test_data.head()


# In[ ]:


Xtest = test_data[test_data.columns[1:]]
print(Xtest.head())
print(Xtest.shape)
print(X.shape)


# In[ ]:


test_prob = model.predict_proba(Xtest)
solution = pd.DataFrame(test_prob, columns=['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])
solution['id'] = test_data['id']
cols = solution.columns.tolist()
cols = cols[-1:] + cols[:-1]
solution = solution[cols]

solution.to_csv('./otto_prediction.tsv', index = False)

