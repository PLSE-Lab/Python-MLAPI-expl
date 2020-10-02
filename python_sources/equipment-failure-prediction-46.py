#!/usr/bin/env python
# coding: utf-8

# # Predicting equipment failures
# 
# ### Table 46
# 
# ### Anton Leontyev, Madi Muse, Sebastian Monroe

# ## Data import and cleaning

# In[ ]:


import pandas as pd
import os 
import numpy as np


# In[ ]:


arr = os.listdir()
print(arr)


# In[ ]:


df_train = pd.read_csv('/kaggle/input/equipfails/equip_failures_training_set.csv')
df_test = pd.read_csv('/kaggle/input/equipfails/equip_failures_test_set.csv')


# Let's inspect the dataset:

# In[ ]:


df_train


# Let's see what columns do we have:

# In[ ]:


list(df_train)


# What types are these columns?

# In[ ]:


df_train.dtypes


# Object means they need to be coerced to the numerical variables; 
# They are listed as object because na are parsed as strings.

# In[ ]:


df_train1 = df_train.apply(pd.to_numeric, errors='coerce')
df_train1.dtypes


# In[ ]:


df_train1.fillna(0) # we can consider no reading as reading with value 0


# Now, let's aggregate the continous sensor recordings. The simplest aggregation is averaging.
# 

# In[ ]:


sensor7 = ['sensor7_histogram_bin0',
 'sensor7_histogram_bin1',
 'sensor7_histogram_bin2',
 'sensor7_histogram_bin3',
 'sensor7_histogram_bin4',
 'sensor7_histogram_bin5',
 'sensor7_histogram_bin6',
 'sensor7_histogram_bin7',
 'sensor7_histogram_bin8',
 'sensor7_histogram_bin9']
sensor24 = ['sensor24_histogram_bin0',
 'sensor24_histogram_bin1',
 'sensor24_histogram_bin2',
 'sensor24_histogram_bin3',
 'sensor24_histogram_bin4',
 'sensor24_histogram_bin5',
 'sensor24_histogram_bin6',
 'sensor24_histogram_bin7',
 'sensor24_histogram_bin8',
 'sensor24_histogram_bin9']
sensor25 = ['sensor25_histogram_bin0',
 'sensor25_histogram_bin1',
 'sensor25_histogram_bin2',
 'sensor25_histogram_bin3',
 'sensor25_histogram_bin4',
 'sensor25_histogram_bin5',
 'sensor25_histogram_bin6',
 'sensor25_histogram_bin7',
 'sensor25_histogram_bin8',
 'sensor25_histogram_bin9']
sensor26 = ['sensor26_histogram_bin0',
 'sensor26_histogram_bin1',
 'sensor26_histogram_bin2',
 'sensor26_histogram_bin3',
 'sensor26_histogram_bin4',
 'sensor26_histogram_bin5',
 'sensor26_histogram_bin6',
 'sensor26_histogram_bin7',
 'sensor26_histogram_bin8',
 'sensor26_histogram_bin9']
sensor64 = ['sensor64_histogram_bin0',
 'sensor64_histogram_bin1',
 'sensor64_histogram_bin2',
 'sensor64_histogram_bin3',
 'sensor64_histogram_bin4',
 'sensor64_histogram_bin5',
 'sensor64_histogram_bin6',
 'sensor64_histogram_bin7',
 'sensor64_histogram_bin8',
 'sensor64_histogram_bin9']
sensor69 = ['sensor69_histogram_bin0',
 'sensor69_histogram_bin1',
 'sensor69_histogram_bin2',
 'sensor69_histogram_bin3',
 'sensor69_histogram_bin4',
 'sensor69_histogram_bin5',
 'sensor69_histogram_bin6',
 'sensor69_histogram_bin7',
 'sensor69_histogram_bin8',
 'sensor69_histogram_bin9']
sensor105 = ['sensor105_histogram_bin0',
 'sensor105_histogram_bin1',
 'sensor105_histogram_bin2',
 'sensor105_histogram_bin3',
 'sensor105_histogram_bin4',
 'sensor105_histogram_bin5',
 'sensor105_histogram_bin6',
 'sensor105_histogram_bin7',
 'sensor105_histogram_bin8',
 'sensor105_histogram_bin9']
df_train1['sensor7_average'] = df_train1[sensor7].mean(axis=1)
df_train1['sensor24_average'] = df_train1[sensor24].mean(axis=1)
df_train1['sensor25_average'] = df_train1[sensor25].mean(axis=1)
df_train1['sensor26_average'] = df_train1[sensor26].mean(axis=1)
df_train1['sensor64_average'] = df_train1[sensor64].mean(axis=1)
df_train1['sensor69_average'] = df_train1[sensor69].mean(axis=1)
df_train1['sensor105_average'] = df_train1[sensor105].mean(axis=1)


# Now that we created the averaged columns, let's simplify the data a little bit and get rid of histogram values. 

# In[ ]:


not_hists = [col for col in df_train1.columns if not  'histogram' in col]
df_train_2 = df_train1[not_hists]


# Are classes balanced in our dataset? Let's count how many instances of "1" do we have in our dataset

# In[ ]:


df_train_2['target'].astype(bool).sum(axis=0)


# ## Transforming the data

# Recall that we have 60000 rows total. Only 1000 instances of class 2 (underground failures, coded 1), while we have 59000 instances of class 1 (surface-related, coded 0). This is clearly a problem, because any model trained for predicting will achieve great accuracy. But this accuracy would actually be moot - the model will effectively be predicting one class.
# 
# A techinque to somewhat combat it can be found in resampling. More accurately, we can "upsample" the underrepresented dataset by performing sampling with replacement. This technique is also known as bootstrapping. We will upsample the dataset to make it even.

# In[ ]:


df_majority = df_train_2[df_train_2.target==0]
df_minority = df_train_2[df_train_2.target==1]
from sklearn.utils import resample
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=59000)    # to match majority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df_upsampled.target.value_counts()


# Now the classes are balanced.

# In[ ]:


df_upsampled = df_upsampled.fillna(0)
y = df_upsampled.target
X = df_upsampled.drop(['target','id'], axis=1)


# Now, let's evaluate how different models fare on accuracy. We are going to use a cross-validation algorithm that splits the data into K parts ("folds") and iteratively treats 1 of these folds as validation and the rest as training. This process is repeated k times and aggregated measures is generated. We will select the model with the best accuracy and the lowest variability in accuracy (i.e., lowest difference in accuracy between folds).

# In[ ]:


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC


# In[ ]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# Let's visualize the results of our model comparison, with y-axis representing accuracy

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure()
fig.suptitle('Comparison of models')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# It is evident from this picture that CART has the smallest variability and the highest accuracy. So, we will use CART as our model of choice.

# In[ ]:


#split dataset into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


model = DecisionTreeClassifier()
fitted_model = model.fit(X_train, y_train)
y_predict = model.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

print("Accuracy:",accuracy_score(y_test, y_predict))
print("F-1 score:", f1_score(y_test, y_predict))


# Prediction accuracy and F-1 is very high, but is our model is not predicting just one class?

# In[ ]:


import collections

print(collections.Counter(y_predict))


# Finally, let's create a confusion matrix. Confusion matrix  tells contrasts prediction of different classes and allows to better evaluate the performance.

# In[ ]:


from sklearn.metrics import confusion_matrix

pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Predicted Surface Failure', 'Predicted Underground failure'],
    index=['True Surface failure', 'True Underground failure']
)


# Save the model to use it later.

# In[ ]:


import pickle
pkl_filename = "pickled_decision_tree.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)


# # Applying the model to the test data

# Finally, let's make our predictions using test data. For that, we need to apply the same transformations to the test data.

# In[ ]:


df_test2 = df_test

df_test2 = df_test2.apply(pd.to_numeric, errors='coerce')
df_test2.dtypes


# In[ ]:


sensor7 = ['sensor7_histogram_bin0',
 'sensor7_histogram_bin1',
 'sensor7_histogram_bin2',
 'sensor7_histogram_bin3',
 'sensor7_histogram_bin4',
 'sensor7_histogram_bin5',
 'sensor7_histogram_bin6',
 'sensor7_histogram_bin7',
 'sensor7_histogram_bin8',
 'sensor7_histogram_bin9']
sensor24 = ['sensor24_histogram_bin0',
 'sensor24_histogram_bin1',
 'sensor24_histogram_bin2',
 'sensor24_histogram_bin3',
 'sensor24_histogram_bin4',
 'sensor24_histogram_bin5',
 'sensor24_histogram_bin6',
 'sensor24_histogram_bin7',
 'sensor24_histogram_bin8',
 'sensor24_histogram_bin9']
sensor25 = ['sensor25_histogram_bin0',
 'sensor25_histogram_bin1',
 'sensor25_histogram_bin2',
 'sensor25_histogram_bin3',
 'sensor25_histogram_bin4',
 'sensor25_histogram_bin5',
 'sensor25_histogram_bin6',
 'sensor25_histogram_bin7',
 'sensor25_histogram_bin8',
 'sensor25_histogram_bin9']
sensor26 = ['sensor26_histogram_bin0',
 'sensor26_histogram_bin1',
 'sensor26_histogram_bin2',
 'sensor26_histogram_bin3',
 'sensor26_histogram_bin4',
 'sensor26_histogram_bin5',
 'sensor26_histogram_bin6',
 'sensor26_histogram_bin7',
 'sensor26_histogram_bin8',
 'sensor26_histogram_bin9']
sensor64 = ['sensor64_histogram_bin0',
 'sensor64_histogram_bin1',
 'sensor64_histogram_bin2',
 'sensor64_histogram_bin3',
 'sensor64_histogram_bin4',
 'sensor64_histogram_bin5',
 'sensor64_histogram_bin6',
 'sensor64_histogram_bin7',
 'sensor64_histogram_bin8',
 'sensor64_histogram_bin9']
sensor69 = ['sensor69_histogram_bin0',
 'sensor69_histogram_bin1',
 'sensor69_histogram_bin2',
 'sensor69_histogram_bin3',
 'sensor69_histogram_bin4',
 'sensor69_histogram_bin5',
 'sensor69_histogram_bin6',
 'sensor69_histogram_bin7',
 'sensor69_histogram_bin8',
 'sensor69_histogram_bin9']
sensor105 = ['sensor105_histogram_bin0',
 'sensor105_histogram_bin1',
 'sensor105_histogram_bin2',
 'sensor105_histogram_bin3',
 'sensor105_histogram_bin4',
 'sensor105_histogram_bin5',
 'sensor105_histogram_bin6',
 'sensor105_histogram_bin7',
 'sensor105_histogram_bin8',
 'sensor105_histogram_bin9']
df_test2['sensor7_average'] = df_test2[sensor7].mean(axis=1)
df_test2['sensor24_average'] = df_test2[sensor24].mean(axis=1)
df_test2['sensor25_average'] = df_test2[sensor25].mean(axis=1)
df_test2['sensor26_average'] = df_test2[sensor26].mean(axis=1)
df_test2['sensor64_average'] = df_test2[sensor64].mean(axis=1)
df_test2['sensor69_average'] = df_test2[sensor69].mean(axis=1)
df_test2['sensor105_average'] = df_test2[sensor105].mean(axis=1)


# In[ ]:


list(df_test2)


# In[ ]:


df_test3 = df_test2.fillna(0)
not_hists2 = [col for col in df_test3.columns if not  'histogram' in col]
df_test4 = df_test3[not_hists2]
df_test4


# In[ ]:


df_test5 = df_test4.drop(['id'], axis=1)
predictions = model.predict(df_test5)


# Let's check how many instances of each class were predicted

# In[ ]:


import collections

collections.Counter(predictions)


# In[ ]:


#sanity check:
15709 + 292


# In[ ]:


predictions


# Save the output in submission-ready format:

# In[ ]:


res_series = pd.Series(predictions)
headers = ['target']
res_series.index += 1 
res_series.index.name = 'id'
res_series.to_csv('submission5a.csv', header = headers)


# In[ ]:




