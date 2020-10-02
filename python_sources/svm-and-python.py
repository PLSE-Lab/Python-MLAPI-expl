#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# # Applying SVM to the dataset
# 
# ## Notice about current top results on this competition
# 
# As you can learn from Chris Deotte on his Discussion comment [How to score 97%, 98%, 99%, and 100%](https://www.kaggle.com/c/digit-recognizer/discussion/61480#latest-520048), it is not possible to meet 100% of accuracy in MNIST dataset. So, if you are starting on machine learning as I am, always keep splited datasets for training (usually 60% - 80% of the data, depending on if you already have a test dataset), 20% for validation and a test dataset (20% of the original dataset or a separated dataset, as we have here in this competition).
# 
# You should train your model using the train dataset. NEVER use the test or validation datasets to train, because if you do so, you are going to mask your performance, giving to the model more information about what it is going to predict than what you will get with real tests. You should use the validation dataset to test your model performance, using techniques like balanced accuracy. Finally, the test dataset should be used just against your best models. You should avoid to use the test dataset many times, because, again, if you do, you will get better results than you would have in production with real data. It happens because, more you visit the test dataset, more changes you do to the model to influence it, but, many times you can do changes that are good specifically to the test dataset, not to the real world data.
# 
# Said all that, let's split our data into 2 groups: train and cross validation. We do not need a test dataset because we are going to use the test set provided by Competition.

# In[ ]:


import time
from datetime import datetime, timedelta

# let's measure time
class StopWatch:
    def start(self):
        d = datetime.now()
        print("Started: %02d:%02d:%02d.%d" % (d.hour, d.minute, d.second, d.microsecond))
        self.initial_time = time.time()

    def show(self):
        total = time.time() - self.initial_time
        sec = timedelta(seconds=total)
        d = datetime(1,1,1) + sec
        print("Time: %02d:%02d:%02d.%d" % (d.hour, d.minute, d.second, d.microsecond))
        
sw = StopWatch()


# In[ ]:


from sklearn.model_selection import train_test_split
import pandas as pd

# Read data
train = pd.read_csv('../input/train.csv')
y = train['label'].values
X = train[train.columns[1:]].values
X_test = pd.read_csv('../input/test.csv').values

# reduce sets to validate concepts
# from random import sample 
# random_indexes = sample(range(X.shape[0]), int(X.shape[0] * 0.5))
# X = X[random_indexes,:]
# y = y[random_indexes]

# split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=42)
print(X_train.shape)
print(X_val.shape)


# # SVM
# 
# Now, let's create a model from our trainning data

# In[ ]:


from sklearn import svm

sw.start()

clf = svm.SVC(kernel='poly', C=100, gamma='auto', degree=3, coef0=1, decision_function_shape='ovo')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

sw.show()


# And now we can validate the result

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score

print(confusion_matrix(y_val, y_pred))
print(balanced_accuracy_score(y_val, y_pred))


# ## Parameters search
# ****
# Now, let's try to improve our results using grid search to discover the best values to train our model

# In[ ]:


import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, RandomizedSearchCV

sw.start()

# range to test
C_range = np.logspace(2, 7, 30)
gamma_range = np.logspace(-9, -5, 30)

# test strategy
cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)

# run randomized search
param_dist = {"C": C_range, "gamma": gamma_range}
n_iter_search = 50
random_search = RandomizedSearchCV(SVC(), param_distributions=param_dist, n_iter=n_iter_search, cv=cv, n_jobs=7)
random_search.fit(X_train, y_train)

sw.show()


# In[ ]:


# show results
print("best parameters: {}".format(random_search.best_params_))
print("best score:      {})".format(random_search.best_score_))


# Then, let's apply our best params to train the model

# In[ ]:


sw.start()
y_pred = random_search.predict(X_val)
sw.show()


# And now we can validate the result

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score

print(confusion_matrix(y_val, y_pred))
print(balanced_accuracy_score(y_val, y_pred))


# # Conclusion
# 
# Let's use the grid search result to predict the test dataset...

# In[ ]:


sw.start()
y_pred_test = random_search.predict(X_test)
sw.show()


# ...and output the result

# In[ ]:


# output result
dataframe = pd.DataFrame({"ImageId": list(range(1,len(y_pred_test)+1)), "Label": y_pred_test})
dataframe.to_csv('output.csv', index=False, header=True)

