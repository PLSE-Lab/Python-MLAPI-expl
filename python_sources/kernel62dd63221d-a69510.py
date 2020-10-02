#!/usr/bin/env python
# coding: utf-8

# # ConocoPhillips 
#  Predicting surface and downhole equipment failures in a West Texas conventional field.
#  
#  A rare event classification problem.
# 
# ## Team Name: 
#  Pregler Mann
# ## Members:
#  Seth Pregler
#  
#  Craig Mann

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        


# Any results you write to the current directory are saved as output.


# ## Read in Data
# 
# #### We read in both of the data sets to prepare for the 

# In[ ]:


df_train = pd.read_csv('/kaggle/input/equipfails/equip_failures_training_set.csv')
df_test = pd.read_csv('/kaggle/input/equipfails/equip_failures_test_set.csv')


# In[ ]:


df = df_train.replace('na',np.nan)
dft = df_test.replace('na', np.nan)


# In[ ]:


non_hist = []

labels = df.columns[2:]

for l in labels:
    if 'histogram' in l:
        non_hist.append(l)
    if (df[l].isna().sum()/df.shape[0]*100) > 25:
        df.drop(l, inplace=True, axis=1)
        dft.drop(l, inplace=True, axis=1)
        #print(l)
    #print(l, df[l].isna().sum()/df.shape[0]*100)


# In[ ]:


df.drop(non_hist, inplace=True, axis=1)


# In[ ]:


dft.drop(non_hist, inplace=True, axis=1)


# In[ ]:


df.shape


# ## Sampling Technique
# 
# #### To alleviate class imbalance which results in a serious bias towards the majority class and increases the number of false negatives, Under-Sampling is used, thus imroving precision and recall.
# 

# In[ ]:


# Class count
count_class_0, count_class_1 = df.target.value_counts()

# Divide by class
df_class_0 = df[df['target'] == 0]
df_class_1 = df[df['target'] == 1]


# In[ ]:



df_class_0_under = df_class_0.sample(count_class_1)
df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)

#print('Random under-sampling:')
#print(df_train_under.target.value_counts())

#df_train_under.target.value_counts().plot(kind='bar', title='Count (target)');


# In[ ]:


df_train_under.fillna(0, inplace=True)
#df.fillna(0, inplace=True)
dft.fillna(0, inplace=True)

# Remove 'id' and 'target' columns
labels = df_train_under.columns[2:]
#labels = df.columns[2:]

X = df_train_under[labels]
#X = df[labels]
y = df_train_under['target']
#y = df['target']
Xt = dft[labels]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# ## Model Selection
# 
# #### Random Forest Classification chosen because ensemble model is more robust, a large number of trees operating in a forest will outperform any of the individual constituent models.
# 
# In another kernel, we also tested a GridSearch cross validation with many parameters that took too long to run, so we took the results from that grid search and applied it to our RandomForestClassifier.
# 
# Between criterion['gini','entropy'], n_estimators [300,500,1000,4000], min_samples_split [2,3,4,5,6], min_samples_leaf [2,3,4,5], test_size [0.2,0.3], we ended up with the parameters below.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(criterion='entropy', n_estimators=4000, random_state=12345, min_samples_leaf=2, min_samples_split=5, max_depth=150)

clf.fit(X_train,y_train)


# # Visualize
# 
# ### Take a quick look to check that the predictions are looking okay.

# In[ ]:


clf.predict(Xt)


# # Prediction Probability
# ### We check the prediction probability to see how well it's determining failures... We might adjust the weighting or cutoff value of a classification if it's closer to a coin-flip than a higher confidence.

# In[ ]:


clf.predict_proba(Xt)


# Add predictions to a series to join to the test set dataframe.

# In[ ]:


turnin = clf.predict(Xt)


# Add the series to the test dataframe.

# In[ ]:


dft['target'] = turnin


# Create a final dataframe with the format required for submission.

# In[ ]:


final = dft[['id','target']]


# Output the predicted targets to a csv for submission in the competition.

# In[ ]:


final.to_csv('output.csv', index=False)


# Save the model for production predictions.
# 
# After storing the model in a pickle file, you can bring it back up in its currently trained state and use it in production.
# 
# It might be wise to create a feedback loop or a timed process that would recreate the model every evening with relevant data.

# In[ ]:


import pickle

pickle.dump(clf, open('ensemble.pkl','wb'))


# ### Define a confusion matrix printing function
# 
# Created a confusion matrix printing function with f1 scores because that is the validation metric that is used in this particular type of problem.

# In[ ]:


def display_confusion(conf_mat):
    if len(conf_mat) != 2:
        raise RuntimeError("  Call to display_confustion invalid"+           " Argument is not a 2x2 Matrix.")
        sys.exit()
    TP = int(conf_mat[1][1])
    TN = int(conf_mat[0][0])
    FP = int(conf_mat[0][1])
    FN = int(conf_mat[1][0])
    n_neg  = TN + FP
    n_pos  = FN + TP
    n_pneg = TN + FN
    n_ppos = FP + TP
    n_obs  = n_neg + n_pos
    print("\nModel Metrics")
    print("{:.<27s}{:10d}".format('Observations', n_obs))
    acc = np.nan
    pre = np.nan
    tpr = np.nan
    tnr = np.nan
    f1  = np.nan
    misc = np.nan
    miscc = [np.nan, np.nan]
    if n_obs>0:
        acc = (TP+TN)/n_obs
    print("{:.<27s}{:10.4f}".format('Accuracy', acc))
    if (TP+FP)>0:
        pre = TP/(TP+FP)
    print("{:.<27s}{:10.4f}".format('Precision', pre))
    if (TP+FN)>0:
        tpr = TP/(TP+FN)
    print("{:.<27s}{:10.4f}".format('Recall (Sensitivity)', tpr))
    if (TN+FP)>0:
        tnr = TN/(TN+FP)
    print("{:.<27s}{:10.4f}".format('Recall (Specificity)', tnr))
    if (2*TP+FP+FN)>0:
        f1 = 2*TP/(2*TP + FP + FN)
    print("{:.<27s}{:10.4f}".format('F1-Score', f1))

    if n_obs>0:
        misc = 100*(FN + FP)/n_obs
    print("{:.<27s}{:9.1f}{:s}".format(            'MISC (Misclassification)', misc, '%'))
    if n_neg>0 and n_pos>0:
        miscc = [100*conf_mat[0][1]/n_neg, 100*conf_mat[1][0]/n_pos]
    lrcc  = [0, 1]
    for i in range(2):
        print("{:s}{:.<16.0f}{:>9.1f}{:<1s}".format(              '     class ', lrcc[i], miscc[i], '%'))      

    print("\n\n     Confusion")
    print("       Matrix    ", end="")
    for i in range(2):
        print("{:>7s}{:<3.0f}".format('Class ', lrcc[i]), end="")
    print("")
    for i in range(2):
        print("{:s}{:.<6.0f}".format('Class ', lrcc[i]), end="")
        for j in range(2):
            print("{:>10d}".format(int(conf_mat[i][j])), end="")
        print("")
         


# ### Print Results
# 
# Visualize the results in a printed output to validate the model and see what might need to improve.

# In[ ]:


from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_true=y, y_pred=clf.predict(X))
conf_train = confusion_matrix(y_true=y_train, y_pred=clf.predict(X_train))
conf_test = confusion_matrix(y_true=y_test, y_pred=clf.predict(X_test))
display_confusion(conf_mat)
display_confusion(conf_train)
display_confusion(conf_test)


# ## Final Thoughts
# 
# #### Having temporal data could lead to better predictions via higher sampling of the data leading up to an equipment failure vs randomly sampling from the entire set
# 
# ## Things we could have done but didn't have time for...
# 
# There was potentially an opportunity to figure out the histogram data, however we opted to drop it outright with other columns that were greater than 25% missing values.  We think that if we were able to turn the histogram into a vector of some sort and use that as a variable, that might have made it score better in the holdout set.

# In[ ]:




