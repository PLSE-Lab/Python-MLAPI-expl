#!/usr/bin/env python
# coding: utf-8

# Hi, today I will show you how to classifiy human activities with 3d human joint data.Let's see what happened!

# # DATA DESCRIPTION
# ![approaching_skel.gif](attachment:approaching_skel.gif)
# 
# ### Data Storage Format
# 
# ..Data
# 
# .......SubjectA&B
# 
# ...................Action01
# 
# ............................01(A to B)
# 
# ............................02(B to A)
# 
# ...................Action02
# 
# ...................
# 
# ...................Action08
# 
# .......SubjectA&B
# 
# .
# 
# .
# ### Actions
# 1. close up
# 2. get away from each other
# 3. kick
# 4. push
# 5. shake hands
# 6. hug
# 7. give a notebook
# 8. punch

# - First we have to parse data.I will just use skeleton joint data to classify.Because it give us 3d coordinates of joints.
# 

# In[ ]:


import pandas as pd 
import os
import numpy as np

path = '/kaggle/input/two-person-interaction-kinect-dataset/'

full_data = pd.DataFrame()

for subdir, dirs, files in sorted(os.walk(path)):
    for file in sorted(files):
        if file.endswith('.txt'):
            #print('subdir:{},name:{}'.format(subdir[-6:-4],file))
            data = pd.read_csv(subdir+'/'+file,header=None)
            data['classs'] = subdir[-6:-4]
            full_data = pd.concat([full_data,data],ignore_index=True)

full_data.drop(full_data.columns[[0]],axis=1,inplace=True)
full_data.head()


# - As you see above, we have 15 joint positions for x,y,z and 2 people so;
# **15(joint) x 3(x,y,z) x 2(person) = 90**
# - 91th column is class of action.(8 classes)

# In[ ]:


full_data.describe()


# - Let's check that is the data ready?

# In[ ]:


full_data.dtypes


# -  91. column is the class of the action, but it is in format of object.To use it correctly, I will change it to "categorical" format.

# In[ ]:


full_data['classs'].astype('category')


# - Let's split data as "x" is data and "y" is label.

# In[ ]:


x = full_data.drop(["classs"],axis=1)
y = full_data.classs.values
print('size of x = ',x.shape)
print('size of y = ',y.shape)


# - In this section, I split data into train and test.

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,shuffle=True)


# - I used KNN, SVM, NN, SGD, RF, NB and the voting classifier.Random forest estimator is 100.You can check hyperparamters.

# In[ ]:


from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

knn_clf = KNeighborsClassifier(n_neighbors=3)
print('......')
svm_clf = SVC(random_state=1)
nb_clf = GaussianNB()
nn_clf = MLPClassifier(solver='lbfgs',max_iter=20000)
sgd_clf = SGDClassifier()
rf_clf = RandomForestClassifier(n_estimators=100,random_state=1)

voting_clf = VotingClassifier(
        estimators=[('knn',knn_clf),('svm',svm_clf),('nb',nb_clf),
                    ('NN',nn_clf),('sgd',sgd_clf),('rf',rf_clf)], voting='hard')

from sklearn.metrics import accuracy_score
accuracies = {}
for clf in (knn_clf, svm_clf, nb_clf, nn_clf, sgd_clf, rf_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__,accuracy_score(y_test, y_pred))
    accuracies[clf.__class__.__name__] = accuracy_score(y_test,y_pred)


# - The best result is Random Forest as you see, but I want to make a Cross Validaton to prove the accuracy and look the precision.

# In[ ]:


from sklearn.model_selection import cross_val_score

accuracy_map = cross_val_score(estimator = rf_clf, X = x_train, y =y_train, cv = 8)
print("avg acc: ",np.mean(accuracy_map))
print("acg std: ",np.std(accuracy_map))


# - As you see, average accuracy of 10 fold CV is very high and standart deviation is veri low which means both accuracy score and precision is very good.
# - Now, let's visualize accuracies to show you better and compare.

# In[ ]:


from matplotlib import pyplot as plt

plt.figure(figsize=(14, 8))
plt.suptitle('Accuracies of classifiers')
plt.subplot(121)
plt.bar(*zip(*sorted(accuracies.items())),color = 'g')
plt.xticks(fontsize=7)
plt.subplot(122)
plt.plot(*zip(*sorted(accuracies.items())),marker='P',linestyle='--',color='r')
plt.xticks(fontsize=7)
plt.grid()
plt.show()


# **That's all, please upvote if you like my notebook:)**
