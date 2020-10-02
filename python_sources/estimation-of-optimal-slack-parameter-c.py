#!/usr/bin/env python
# coding: utf-8

# The following code runs a number of simulation using different values of the slack parameter C in a support vector machine. The aim is to identify the value of C, that has the best predictive value. Different values of C ranging from 0.001 to 100 are used. The data set is split in to three sets, in order to un-bias the final estimator of accuracy. Hence, the value of C with the best average performance is used to find the model's predictive power in an out-of-sample data set. All this is done twice: once using a Linear Kernel and once using a Gaussian Kernel ('rbf'). Everything is summarized in a nice plot in the end.
# 
# I have one  conceptual problem.:
# 
# The out-of-sample predictive power of my best Gaussian Kernel is devastating (blue star in the plot). I tried several things, it is not a coding issue. Why is that performance of the best(!) C in the out-of-sample data set so bad? (The performance for the Linear Kernel deteriorates only slightly).
# 
# Any feedback is welcome!
# 
# Thanks, 
# Jan

# In[ ]:


import numpy as np
import math
from sklearn import preprocessing, svm
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
from sklearn.cross_validation import  train_test_split



df =  pd.read_csv('../input/data.csv', header=0)
df[df.columns[1]] = df[df.columns[1]].map( {'B': 0, 'M': 1} ).astype(int)

df = df[list(df.columns[:13])] 
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['diagnosis'], 1))
y = np.array(df['diagnosis'])


# In[ ]:


#Now I split the set the first time and use 
X_train_CV, X_test_CV, y_train_CV, y_test_CV = train_test_split(X, y, test_size=0.2)

reduced_X=X_train_CV
reduced_y=y_train_CV

#List of C-values and other stuff
C_values=[math.pow(10,x/20) for x in range(-60,30)]
opt_dict_lin={}
opt_dict_gauss={}
C_dependent_acc=[]

number_of_simulations=30    #Adjust at will!


# In[ ]:


C_dependent_acc_lin=[]
yerr=[]
for i in range(len(C_values)):

    accuracies = []
    for i1 in range(number_of_simulations):

        X = preprocessing.scale(reduced_X)
        X_train, X_test, y_train, y_test = train_test_split(X, reduced_y, test_size=0.2)

        clf=svm.SVC(C=C_values[i],kernel='linear')
        clf.fit(X_train,y_train)

        predictions = clf.predict(X_test)
        accuracy=metrics.accuracy_score(y_test, predictions)

        #accuracy=clf.score(X_test,y_test)
        accuracies.append(accuracy)

    mean_acc=(sum(accuracies)/len(accuracies))
    standard_error=np.std(accuracies)/sqrt(len(accuracies))
    C_dependent_acc_lin.append(mean_acc)
    yerr.append(standard_error)
    opt_dict_lin[mean_acc]=[C_values[i]]

norms=sorted([n for n in opt_dict_lin])
opt_choice_lin=opt_dict_lin[norms[-1]]


# In[ ]:





# In[ ]:





# In[ ]:


#Out of sample estimation of accuracy
#Note: If I try preprocessing my data using
#X_train_CV = preprocessing.scale(X_train_CV)
#,the performance is screwed up. Why is that?

clf = svm.SVC(C=opt_choice_lin[0], kernel='linear')
clf.fit(X_train_CV, y_train_CV)


predictions = clf.predict(X_test)
accuracy=metrics.accuracy_score(y_test, predictions)



# In[ ]:



#Now, everything is repeated, only using a Gaussian Kernel instead of a Linear one.

C_dependent_acc=[]
yerr1=[]
for i in range(len(C_values)):

    accuracies = []
    for i1 in range(number_of_simulations):


        X = preprocessing.scale(reduced_X)
        X_train, X_test, y_train, y_test = train_test_split(X, reduced_y, test_size=0.2)

        clf=svm.SVC(C=C_values[i],kernel='rbf')
        clf.fit(X_train,y_train)

        predictions = clf.predict(X_test)
        accuracy=metrics.accuracy_score(y_test, predictions)
        accuracies.append(accuracy)

    mean_acc=(sum(accuracies)/len(accuracies))
    standard_error=np.std(accuracies)/sqrt(len(accuracies))
    C_dependent_acc.append(mean_acc)
    yerr1.append(standard_error)
    opt_dict_gauss[mean_acc] = [C_values[i]]

norms = sorted([n for n in opt_dict_gauss])
print(norms)
opt_choice_gauss = opt_dict_gauss[norms[-1]]

#Testing of optimal value at new data
X = preprocessing.scale(X_train_CV)

clf = svm.SVC(C=opt_choice_gauss[0], kernel='rbf')
clf.fit(X, y_train_CV)

predictions = clf.predict(X_test)
accuracy1=metrics.accuracy_score(y_test, predictions)

#Things are plotted using a log scale
ax = plt.subplot(111)
ax.set_xscale("log")
plt.errorbar(C_values,C_dependent_acc_lin,yerr=yerr,color='b',label='Linear Kernel')
plt.errorbar(C_values,C_dependent_acc,yerr=yerr1,color='r',label='Gaussian Kernel')

plt.scatter(opt_choice_gauss[0],accuracy1,marker='*',s=100,color='r',label='Accuracy of best C of Gaussian kernel on new data')
plt.scatter(opt_choice_lin[0],accuracy,marker='*',s=100,color='b',label='Accuracy of best C of Linear kernel on new data')
plt.xlabel("Log of Slack Parameter 'C'")
plt.ylabel("Accuracy")
#plt.legend(loc='lower right')
print('Accuracy of best C (',opt_choice_gauss[0],') at new data with Gaussian kernel:', accuracy1)
print('Accuracy of best C (',opt_choice_lin[0],') at new data with linear kernel:', accuracy)


# In[ ]:


print('Accuracy of best C (',opt_choice_lin[0],') at new data with linear kernel:', accuracy)
print('Accuracy of best C (',opt_choice_gauss[0],') at new data with Gaussian kernel:', accuracy1)


# In[ ]:




