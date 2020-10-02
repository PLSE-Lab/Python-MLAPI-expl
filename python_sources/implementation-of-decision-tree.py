#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification, make_regression, load_digits, load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error


# In[ ]:


RANDOM_STATE = 17


# # Decision Trees

# In[ ]:


# Let's define quality criterion
# entropy and gini criteria are used for classification
def entropy(y):
    p = [len(y[y==k])/len(y) for k in np.unique(y)]
    return -np.dot(p,np.log2(p))

def gini(y):
    p = [len(y[y==k])/len(y) for k in np.unique(y)]
    return 1 - np.dot(p,p)

# Variance and median criteria are used for regression

def variance(y):
    return np.var(y)

def mad_median(y):
    return np.mean(np.abs(y - np.median(y)))


# In[ ]:


criteria_dict = {'entropy':entropy,'gini':gini,
                'variance':variance,'mad_median':mad_median}


# ### The Node implements a node in the DS**

# In[ ]:


class Node():
    
    def __init__(self,feature_idx=0,threshold=0,labels=None,left=None,right=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.labels = labels
        self.left = left
        self.right = right


# In[ ]:


def regression_leaf(y):
    return np.mean(y)

def classification_leaf(y):
    return np.bincount(y).argmax()


# In[ ]:


class DecisionTree(BaseEstimator):
    def __init__(self, max_depth=np.inf, min_samples_split=2,criterion='gini',debug=False):
        params = {'max_depth':max_depth,
                 'min_samples_split':min_samples_split,
                 'criterion':criterion,
                 'debug':debug}
        self.set_params(**params)
        
        if self.debug:
            print('\nDecisionTree params:')
            print("max_depth = {}, min_samples_split = {},criterion = {}\n"                 .format(max_depth,min_samples_split,criterion))
            
    def set_params(self,**params):
        super().set_params(**params)
        
        self._criterion_function = criteria_dict[self.criterion]
        
        if self.criterion in ['variance','mad_median']:
            self._leaf_value = regression_leaf
        else:
            self._leaf_value = classification_leaf
        return self
    
    # Function for splitting the data by two parts
    def _functional(self, X, y, feature_idx, threshold):
        mask = X[:,feature_idx] < threshold
        n_obj = X.shape[0]
        n_left = np.sum(mask)
        n_right = n_obj - n_left
        if n_left > 0 and n_right>0:
            return self._criterion_function(y) - (n_left / n_obj) *                    self._criterion_function(y[mask]) - (n_right/n_obj) *                    self._criterion_function(y[~mask])
        else:
            return 0
        
    def _build_tree(self, X, y,depth=1):
        max_functional = 0
        best_feature_idx = None
        best_threshold = None
        n_samples, n_features = X.shape
        
        if len(np.unique(y))==1:
            return Node(labels=y)
        
        if depth < self.max_depth and n_samples >= self.min_samples_split:
            if self.debug:
                print("depth = {}, n_samples = {}".format(depth,n_samples))
                
            for feature_idx in range(n_features):
                
                threshold_values = np.unique(X[:,feature_idx])
                functional_values = [self._functional(X,y,feature_idx,threshold)
                                    for threshold in threshold_values]
                
                best_threshold_idx = np.nanargmax(functional_values)
                
                if functional_values[best_threshold_idx] > max_functional:
                    max_functional = functional_values[best_threshold_idx]
                    best_threshold = threshold_values[best_threshold_idx]
                    
                    best_feature_idx = feature_idx
                    best_mask = X[:,feature_idx] < best_threshold
                    
        if best_feature_idx is not None:
            if self.debug:
                print("best feature = {}, best threshold = {}"                     .format(best_feature_idx,best_threshold))
                
            return Node(feature_idx = best_feature_idx,threshold=best_threshold,
                       left=self._build_tree(X[best_mask,:],y[best_mask],depth + 1),
                       right=self._build_tree(X[~best_mask,:],y[~best_mask],depth+1))
        
        else:
            return Node(labels=y)
    
    def fit(self,X,y):
        
        if self.criterion in ['gini','entropy']:
            self._n_classes = len(np.unique(y))
            
        self.root = self._build_tree(X,y)
        
        return self
    
    def _predict_object(self,x,node=None):
        
        node = self.root
        
        while node.labels is None:
            if x[node.feature_idx] < node.threshold:
                node = node.left
            else:
                node = node.right
                
        return self._leaf_value(node.labels)
    
    def predict(self,X):
        return np.array([self._predict_object(x) for x in X])
    
    def _predict_proba_object(self,x,node=None):
        node = self.root
        
        while node.labels is None:
            if x[node.feature_idx] < node.threshold:
                node = node.left
            else:
                node = node.right
                
        return [len(node.labels[node.labels == k]) /                len(node.labels) for k in range(self._n_classes)]
    
    def predict_proba(self, X):
        return np.array([self._predict_proba_object(x) for x in X])


# # Classification

# In[ ]:


X, y = make_classification(n_features=2,n_redundant=0,n_samples=400,
                          random_state=RANDOM_STATE)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=RANDOM_STATE)

clf = DecisionTree(max_depth=4, criterion='gini')
clf.fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test)
prob_pred = clf.predict_proba(X_test)
accuracy = accuracy_score(y_test,y_pred)


# In[ ]:


print("Accuracy:",accuracy)


# In[ ]:


if(sum(np.argmax(prob_pred,axis=1) - y_pred)==0):
    print("predict_proba works!")


# In[ ]:


plt.suptitle("Accuracy = {0:.2f}".format(accuracy))
plt.subplot(121)
plt.scatter(X_test[:,0],X_test[:,1],c=y_pred,cmap=plt.cm.coolwarm)
plt.title("Predicted class labels")
plt.axis("equal")
plt.subplot(122)
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap=plt.cm.coolwarm)
plt.title("True class labels")
plt.axis('equal');


# In[ ]:


digits = load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                   test_size=0.2,random_state=RANDOM_STATE)


clf2 = DecisionTree(max_depth=2,criterion='gini',debug=True)
clf2.fit(X_train, y_train)
print(accuracy_score(clf2.predict(X_test), y_test))


# In[ ]:


clf1 = DecisionTree(max_depth=2,criterion='entropy',debug=True)
clf1.fit(X_train,y_train)

print(accuracy_score(clf1.predict(X_test), y_test))


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntree_params = {'max_depth': list(range(3,11)),\n              'criterion':['gini','entropy']}\n\nclf = GridSearchCV(DecisionTree(),tree_params,cv=5,scoring='accuracy',verbose=True, n_jobs=8)\nclf.fit(X_train,y_train)")


# In[ ]:


clf.best_score_, clf.best_params_


# In[ ]:


scores = np.array(clf.cv_results_['mean_test_score'])
scores = scores.reshape(len(tree_params['criterion']),
                       len(tree_params['max_depth']))

for ind,i in enumerate(tree_params['criterion']):
    plt.plot(tree_params['max_depth'],scores[ind], label=str(i))

plt.legend(loc='best')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.show();


# In[ ]:


clf = DecisionTree(max_depth=9,criterion='entropy')
clf.fit(X_train,y_train)
probs = clf.predict_proba(X_test)


# In[ ]:


mean_probs = np.mean(probs,axis=0)
print(mean_probs)


# # Regression

# In[ ]:


X, y = make_regression(n_features=1, n_samples=200,bias=0,noise=5,
                      random_state=RANDOM_STATE)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

reg = DecisionTree(max_depth=6,criterion='mad_median')
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)


# In[ ]:


mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)


# In[ ]:


plt.scatter(X_test[:,0],y_test,color='black')
plt.scatter(X_test[:,0],y_pred,color='green')
plt.title("MSE = {0:.2}".format(mse));


# In[ ]:


boston = load_boston()

X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=RANDOM_STATE)

clf1 = DecisionTree(max_depth=2,criterion='variance',debug=True)
clf1.fit(X_train, y_train)


# In[ ]:


print(mean_squared_error(clf1.predict(X_test),y_test))


# In[ ]:


clf2 = DecisionTree(max_depth=2, criterion='mad_median',debug=True)
clf2.fit(X_train,y_train)

print(mean_squared_error(clf2.predict(X_test),y_test))


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntree_params = {'max_depth': list(range(2,9)),\n              'criterion':['variance','mad_median']}\n\nreg = GridSearchCV(DecisionTree(), tree_params,\n                  cv=5, scoring='neg_mean_squared_error',n_jobs=8)\n\nreg.fit(X_train, y_train)")


# In[ ]:


scores = -np.array(reg.cv_results_['mean_test_score'])
scores = scores.reshape(len(tree_params['criterion']),len(tree_params['max_depth']))

for ind, i in enumerate(tree_params['criterion']):
    plt.plot(tree_params['max_depth'],scores[ind],label=str(i))
    
plt.legend()
plt.xlabel('max_depth')
plt.ylabel('MSE')
plt.show();


# In[ ]:


print("Best params:",reg.best_params_)
print("Best cross vaildation MSE",abs(reg.best_score_))

