#!/usr/bin/env python
# coding: utf-8

# # A beginner's look at classification problems
# 
# This is my first kernel on Kaggle, so critique is very welcome.
# 
# My intention for this is to cover some of the more basic approaches to classification problems. My intention is also to make the code easy to follow for beginners, without abstracting too much of the work away through library functions. This is why you will see me doing things like manually calculating precision and recall. For this reason, I won't apologise for excessive use of print statements either. ;-)
# 
# My introduction to machine learning has been through Andrew Ng's excellent course on coursera.org. I really cannot recommend this course highly enough, even to those with some previous experience in machine learning. The majority of the work in this notebook covers some of the techniques and nuances that I have learned in this course.
# 
# There are many more advanced learning methods out there which are now very popular on Kaggle, such as various forms of Gradient Boosting, but the choice of ML algorithm is not the only important factor. I want to demonstrate good implementation of cross-validation and how to manage highly skewed data sets.

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

get_ipython().run_line_magic('matplotlib', 'inline')

random.seed(123) #making the results reproducible

filename = '../input/creditcard.csv'
original_data = pd.read_csv(filename)
data = original_data.copy() # create a copy for use leave the original in case we need it later.


# In[ ]:


data.head()


# # 1 - Logistic Regression Model
# ## 1.1 - First Look at the Data
# 
# The data has 28 anonymised features, V1 to V28, which also appear to have been normalised. In addition, there is the time and the transaction amount, giving us 30 features in total.
# 
# For now, I am going to ignore the time feature. I will also normalise the Amount and add this as a new column.

# In[ ]:


amounts = data.loc[:, ['Amount']]
data['Amount'] = StandardScaler().fit_transform(amounts)

data.head()


# Now we can drop the Time column to remove it

# In[ ]:


data = data.drop(columns=['Time'])
data.head()


# In[ ]:


class_dist = pd.value_counts(data['Class'], sort = True)


# In[ ]:


class_dist


# In[ ]:


print('%.2f%% of transactions are fraudulent.' % (class_dist[1]/sum(class_dist)*100) )


# These data are highly skewed, which means that ordinary error metrics will not provide useful insight for our classifier.
# 
# For example, if we had a classifier that simply predicts that classes every transaction as non fraudulent (class 0), then it would be correct 99.83% of the time!
# 
# We need a better method of measuring the algorithm's error. In this case we should use recall and precision to evaluate the effectiveness of the algorithm.
# 
# In addition we need to take care when we separate our data into training, cross validation and test sets. There is a good chance we may have no fraudulent examples in one of these sets, which we want to avoid.
# 
# So first, we split our fraudulent examples (class 1) and our non-fraudulent examples (class 0).

# In[ ]:


class_1 = data[data.Class == 1]
class_0 = data[data.Class == 0]


# In[ ]:


print(class_0.shape)
print(class_1.shape)


# ## 1.2 - Creating training, cross-validation and test data sets
# The data have been split into fraudulent and non-fraudulent data. Now, let's create different data sets with the following split:
# 
# 1. 70% training data
# 2. 15% cross-validation data
# 3. 15% test data
# 
# So for our two classes these are approximately:

# In[ ]:


class_0_split = [round(x * class_0.shape[0]) for x in [0.7, 0.15, 0.15]]
class_1_split = [round(x * class_1.shape[0]) for x in [0.7, 0.15, 0.15]]

print('Non-fraudulent data split %s' % class_0_split)
print('Fraudulent data split %s' % class_1_split)

print('\nCheck for rounding issues as we want to avoid increasing the total with rounding up: ')
print(sum(class_0_split) / class_0.shape[0])
print(sum(class_1_split) / class_1.shape[0])


# This gives us the required split of fraudulent and non-frudulent data.
# 
# Since we have very skewed data, one method that can be used to create a useful classifier is to resample the data and to reduce the presence of the over-represented class in our training data.
# 
# To do this, let's create a training data set with 50:50 split between fraudulent and non-fraudulent training data. So we will only have 344 non-fraudulent examples in our training set instead of the prescribed 199,020 above

# In[ ]:


class_0_train = class_0.sample(n = 344, random_state = 123,  replace = False) #make our random state reproducible
class_1_train = class_1.sample(n = 344, random_state = 234,  replace = False)


# For the cross-validation, and test samples, we will maintain the normal skewness of the original data.
# 
# We have to take care here not to re-sample the trianing data (or the cross-validation data for the test data), so we use **drop** to remove these examples before sampling again.
# 
# We also have too much extra data from undersampling the training data, so we have to reduce the amount in the final test data to keep to ratio of fraudulent and non-fraudulent data.

# In[ ]:


class_0_cv = class_0.drop(class_0_train.index).sample(n = 42647, random_state = 345,  replace = False)
class_0_test = class_0.drop(class_0_train.index).drop(class_0_cv.index).sample(n = 42647, random_state = 456,  replace = False)

class_1_cv = class_1.drop(class_1_train.index).sample(n = 74, random_state = 567, replace = False)
class_1_test = class_1.drop(class_1_train.index).drop(class_1_cv.index)


# Now, let's re-merge the data into three sets.

# In[ ]:


train_set = pd.concat([class_0_train, class_1_train])
cv_set = pd.concat([class_0_cv, class_1_cv])
test_set = pd.concat([class_0_test, class_1_test])


# In[ ]:


#train_set.to_csv('training.csv')
#cv_set.to_csv('cv.csv')
#test_set.to_csv('test.csv')


# These datasets were examined in Excel and no duplicates were found.

# ## 1.3 - Training the data with logistic regression
# 
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression

X_train = train_set.iloc[:, 0:29]
y_train = train_set.iloc[:, 29]

lr = LogisticRegression(penalty= 'l1', C=1) # default regularisation value for now.

lr.fit(X_train, y_train)


# In[ ]:


lr.score(X_train, y_train)


# In[ ]:


X_cv = cv_set.iloc[:, 0:29]
y_cv = cv_set.iloc[:, 29]


# In[ ]:


print(lr.score(X_cv, y_cv))


# This looks good, but remember our cross-validation data is highly skewed!
# 
# We need to look at the recall and precsision to have a true understanding of this data.
# 
# Let's add the predicted classes to our cross-validation data so that we can compare it with the actual class and see where we have false/true positives and negatives.

# In[ ]:


y_cv_predict = lr.predict(X_cv)

cv_set['Predicted'] = y_cv_predict

cv_set.head()


# In[ ]:


class_list = list(cv_set['Class'])
pred_list = list(cv_set['Predicted'])

def calc_prec_recall(class_list, pred_list):
    
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0



    for i in range(cv_set.shape[0]):
        if class_list[i] == 1:
            if pred_list[i] == 1:
                true_pos += 1
            else:
                false_neg += 1
            
        else:
            if pred_list[i] == 0:
                true_neg += 1
            else:
                false_pos += 1
            
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    return precision, recall                                            
                      
cv_prec, cv_recall = calc_prec_recall(class_list, pred_list)                   


# In[ ]:


print('Precision = %.2f%%' % (100 * cv_prec))
print('Recall = %.2f%%' % (100 * cv_recall))


# The precision is bad, but the recall is good in this case. The low precision means that we have too many false positives and the algorithm often predicts a fraudulent transaction where there isn't one! This would be fine if we wanted to be very cautious about frudulent transactions, but customers would get very frustrated having their credit card wrongly declined.
# 
# On the other hand, we have high recall, which means we don't have many false negatives. This means that we don't miss many fraudulent transactions.
# 
# We can change the trade-off by between precision and recall by adjusting the threshold of our logistic regression classifier.
# 
# We may also be underfitting or overfitting to our training data. We simply used the defulat value for C, the inverse of the regularisation strength.
# 
# Since we don't know whether we would prefer to maximise precision or recall, let's also introduce the F1 score as our metric for evaluating our cross-validation performance:
# 
# \begin{align}
# F_1 = 2 \frac{Precision \times Recall}{Precision + Recall}
# \end{align}
# 
# This is also known as the harmonic mean of the precision and recall. Implmenting this:

# In[ ]:


def f1_score(precision, recall):
    return (2 * precision * recall / (precision + recall))
                                     
fscore = f1_score(cv_prec, cv_recall)
print(fscore)


# ## 1.4 - Tuning the regularisation to improve cross-validation score
# Now, let's choose a range of values for our regularisation parameter, C (inverse of regularisation strength) to find the best overall F-Score

# In[ ]:


C_set = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]     # A set of C values to try

def logistic_test(X_train, y_train, X_cv, y_cv, C_in):
    
    lr = LogisticRegression(penalty= 'l1', C=C_in)

    lr.fit(X_train, y_train)
    
    train_score = lr.score(X_train, y_train)
    cv_score = lr.score(X_cv, y_cv)
    
    class_list = y_cv.tolist()
    pred_list = lr.predict(X_cv).tolist()
    
    cv_prec, cv_recall = calc_prec_recall(class_list, pred_list) 

    fscore = f1_score(cv_prec, cv_recall)
    
    return train_score, cv_score, fscore

training_scores = []
cv_scores = []
cv_f1 = []
best_f1_score = 0

for C in C_set:
    print('\n-----------------------------------------------------------------------')
    print('Fitting logistic regression with regularisation parameter %f' % C)
    print('-----------------------------------------------------------------------')
    a, b, c = logistic_test(X_train, y_train, X_cv, y_cv, C)    
    print('Training score = %f' % a)
    print('Cross-validation score = %f' % b)
    print('F1 score = %f' %c)
    training_scores.append(a)
    cv_scores.append(b)
    cv_f1.append(c)
    if c > best_f1_score:
        best_f1_score =c
    


# In[ ]:


plt.plot(C_set, training_scores, '-r', label='Training')
plt.plot(C_set, cv_scores, '-b', label='Cross Validation')
plt.xscale('log')
plt.xlabel('C (inverse regularisation parameter)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# The graph shows a clear under fit to our data when C is low, due to the high degree of reularisation.
# 
# On the other hand, when C is very high, there is less regularisation and the cross validation results worsen slightly. However in most cases, the accuracy for the cross-validation set is better than for the training set. This is likely due to the highly skewed nature of our Cross Validation set compared to the training set.

# In[ ]:


plt.plot(C_set, cv_f1, '-r', label='Cross Validation F1 Score')
plt.xscale('log')
plt.xlabel('C (inverse regularisation parameter)')
plt.ylabel('F1 Score')
plt.legend()
plt.show()


# ## 1.5 - Checking unseen test data
# 
# Based on our previous tuning to find the best C value via cross-validation, we can set C=0.1
# 
# It is important to note that using cross-validation data to tune our parameters is also a form of data fitting. To avoid overfitting to our cross validation set we should compare to an unseen data set. That is why we created a separate set of test data for the final result.
# 
# Let's see how the model performs on the test set.

# In[ ]:


lr = LogisticRegression(penalty= 'l1', C=0.1)
lr.fit(X_train, y_train)
X_test = test_set.iloc[:, 0:29]
y_test = test_set.iloc[:, 29]

class_list_test = list(test_set['Class'])
pred_list_test = lr.predict(X_test)

test_prec, test_recall = calc_prec_recall(class_list_test, pred_list_test) # pretty sure this is wrong - class list needs to be from test data
test_f1 = f1_score(test_prec, test_recall)

print('Accuracy on unseen test data: %.2f%%' % (100 * lr.score(X_test, y_test)))
print('Recall on unseen test data: %.2f%%' % (100*test_recall))
print('Precision on unseen test data: %.2f%%' % (100*test_prec))
print('F1 Score on unseen test data: %.2f%%' % (100 * test_f1))


# ## 1.6 - Conclusion
# Overall the test data results are on par with the cross-validation set with an F1 score of 15.2% overall. The recall is good, but the precision is poor!
# 
# This is not surprising given that the model used was quite a simple one. We could generate a more complex logisitic regression model, by engineering new features and using these. In future I may come back to this in future, but for now I will move on to Support Vector Machines (SVMs).

# # 2 Using Support Vector Machines (SVMs)
# 
# A support vector machine is a useful way to generate a classifier, which fits a hyperplane to separate the data. A hyperplane is a subspace whose dimension is one less than that of its ambient space. So if we have:
# 
# * Data in two dimensions, it will be separated by a line.
# * Data in three dimensions, the data will be separated by a two dimensional plane.
# * Etc.
# 
# The key thing to note is that, at it's heart an SVM is essentially a linear classifier. To generate more complex, non-linear classifiers, we can use various kernel functions. I won't go into the details, but the number of new features generated scales with the number of training examples. This can be problematic with very large training sets. With our relatively small training set size, this particular problem lends itself well to the use of SVMs.
# 
# These are the kernels we will test:
# 
# * Linear (i.e. no kernel, just a basic SVM).
# * Radial Basis Functions (RBF) - This is a form of the Gaussian function.
# 
# Although other kernels exist and we can even write our own kernels, these are the two most popular kernels. The RBF is capable of generating some very complex decision boundaries and should make an interesting comparison to the linear kernel.
# 
# It is very important to carry out feture scaling before using SVM's. Fortunately, our features are already scaled.
# 
# ## 2.1 - Trying the linear classifier
# Scikit-learn has an inbuilt SVM classifier called SVC. This uses the RBF kernel by default. Although this has the option to use a linear kernel, there is also the option to use LinearSVC classifier. As far as I can tell these function similarly, but according to the documentation the LinearSVC is implemented slightly differently. This makes it less restrictive in the choice of penalties and loss functions. It should scale better to large numbers of samples, but that is not a concern as our training set is quite small.
# 

# In[ ]:


from sklearn.svm import SVC, LinearSVC #import both the Linear and SVM classifier

svm_lin = LinearSVC(C=1.0) # C is effectively a regularisation parameter, we will experiement with changing this later

svm_lin.fit(X_train, y_train)

print("Training set linear classifier score: %.2f%%" % (100 * svm_lin.score(X_train, y_train)))
print("Cross-validation set linear classifier score: %.2f%%" % (100 * svm_lin.score(X_cv, y_cv)))


# Similar to the logistic regression, we got a higher score on the cross training data. This is due to highly skewed nature of the cross validation set compared to the training set.
# 
# Let's see what the preicsion, recall and F1 score are like.

# In[ ]:


svm_lin_pred_list = svm_lin.predict(X_cv).tolist()
cv_prec, cv_recall = calc_prec_recall(class_list, svm_lin_pred_list)
fscore = f1_score(cv_prec, cv_recall)


# In[ ]:


print("The cross-validation precision is: %.2f%%" % (100*cv_prec))
print("The cross-validation recall is: %.2f%%" % (100*cv_recall))
print("The cross-validation F1 score is: %.2f%%" % (100*fscore))


# Overall these results are fairly similar to what we got with our first attempt when using the logistic regression. I will leave linear regression here for the sake of keeping this notebook brief. Normally the same process of tuning the hyperparameters (in this case, C) to improve the cross validation score is the next step.
# 
# ## 2.2 - Radial Basis Function Kernel (Gaussian Kernel)
# 
# Let's move on to using the Gaussian kernel and see how this performs before tuning the model.

# In[ ]:


svm_rbf = SVC(C=1.0, kernel='rbf', gamma='auto') # Using the default C for now and allow the algorithm to choose gamma


# In[ ]:


svm_rbf.fit(X_train, y_train)
print("Training set rbf classifier score: %.2f%%" % (100 * svm_rbf.score(X_train, y_train)))
print("Cross-validation set rbf classifier score: %.2f%%" % (100 * svm_rbf.score(X_cv, y_cv)))


# In[ ]:


svm_rbf_pred_list = svm_rbf.predict(X_cv).tolist()
cv_prec, cv_recall = calc_prec_recall(class_list, svm_rbf_pred_list)
fscore = f1_score(cv_prec, cv_recall)
print("The cross-validation precision is: %.2f%%" % (100*cv_prec))
print("The cross-validation recall is: %.2f%%" % (100*cv_recall))
print("The cross-validation F1 score is: %.2f%%" % (100*fscore))


# This is much worse than our first two attempts! The RBF kernel performed worse on the cross-validation set than the logistic regression and the SVM with the linear kernel. This should not be a concern as we only tuned our logistic regression model.
# 
# ## 2.3 - Tuning the Radial Basis Function SVM
# 
# We have two parameters that we can tune to try to improve the model. We can alter C - which acts as an inverse to regularisation and gamma, which increases or decreases the senstivity of our Gaussian function. I won't go into the full details, but it acts like another regularisation parameter. If you are interested to know more about the details, Andrew Ng's Coursera course is my recommended source.

# In[ ]:


#C_set = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]     # A set of C values to try
#gamma_set = [0.001, 0.003, 0.01, 0.03]     # A set of C values to try

C_set = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]     # A set of C values to try
gamma_set = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]     # A set of C values to try

def rbf_test(X_train, y_train, X_cv, y_cv, C_in, gamma_in):
    
    svm_rbf = SVC(C= C_in, kernel='rbf', gamma=gamma_in)

    svm_rbf.fit(X_train, y_train)
    
    train_score = svm_rbf.score(X_train, y_train)
    cv_score = svm_rbf.score(X_cv, y_cv)
    
    class_list = y_cv.tolist()
    pred_list = svm_rbf.predict(X_cv).tolist()
    
    cv_prec, cv_recall = calc_prec_recall(class_list, pred_list) 

    fscore = f1_score(cv_prec, cv_recall)
    
    return train_score, cv_score, fscore


Index = ['gamma = %s' % g for g in gamma_set]
Columns = ['C = %s' % C for C in C_set]

rbf_training_scores_df = pd.DataFrame(index=Index, columns=Columns)
rbf_cv_scores_df = pd.DataFrame(index=Index, columns=Columns)
rbf_cv_f1_df = pd.DataFrame(index=Index, columns=Columns)

rbf_best_f1_score = 0

for C in C_set:
    for gamma in gamma_set:
        print('\n-----------------------------------------------------------------------')
        print('Fitting Gaussian SVM with regularisation parameter %f and gamma, %f' % (C, gamma))
        print('-----------------------------------------------------------------------')
        a, b, c = rbf_test(X_train, y_train, X_cv, y_cv, C, gamma)    
        #print('Training score = %f' % a)
        #print('Cross-validation score = %f' % b)
        #print('F1 score = %f' %c)
        
        rbf_training_scores_df.iloc[gamma_set.index(gamma), C_set.index(C)] = a
        rbf_cv_scores_df.iloc[gamma_set.index(gamma), C_set.index(C)] = b
        rbf_cv_f1_df.iloc[gamma_set.index(gamma), C_set.index(C)] = c
        
        
        training_scores.append(a)
        cv_scores.append(b)
        cv_f1.append(c)
        if c > best_f1_score:
            best_f1_score =c


# That was noticably slower to run than the logistic regression and the SVM with a linear kernel. Let's take a look at some plots to see how this performed.

# In[ ]:


# For some reason when I looked at my data fram the first time, the plots didn't work as expected
# I looked at the data type for each column using the dtype attribute and they were all objects
# I'm not sure of the reason for this, but it could be that I initialised the arrays without any data
# We can convert all the values to float to fix this.


rbf_training_scores_df = rbf_training_scores_df.astype(float)
rbf_cv_scores_df = rbf_cv_scores_df.astype(float)
rbf_cv_f1_df = rbf_cv_f1_df.astype(float)

plt.figure(figsize=(12,6))
sns.heatmap(rbf_training_scores_df, annot=True, linewidths = 0.5, vmin=0.7, vmax=1)
plt.yticks(rotation=0)
plt.title('Training Data Accuracy', pad=10)
plt.show()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(rbf_cv_scores_df, annot=True, linewidths = 0.5, vmin=0.7, vmax=1)
plt.title('Cross-Validation Data Accuracy', pad=10)
plt.yticks(rotation=0)
plt.show()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(rbf_cv_f1_df, annot=True, linewidths = 0.5, vmin=0, vmax=0.7)
plt.yticks(rotation=0)
plt.title('F1 Score on Cross-Validation Data', pad=10)
plt.show()


# Two things immediately spring to mind.
# 
# 1. The best F1 score on the Cross-validation set is much better than the best for both our logistic regression, and our linear kernel SVM.
# 
# 
# 2. Our selected values for gamma were, in general, too high. The automatic value for gamma is normally 1/n_features, so in our case 1/30. The higher values can deceptively look promising, with good training data and cross-validation accuracy, but the F score is much higher for the lower value of gamma. This suggests to me that the model is severly underfitting the data and predicitng every transaction as non-fraudulent. This shows how our highly skewed data set can be deceiving in cross validating the data. The choice to use F1 score has been validated in my opinion.
# 
# Let's try a different range of values, focussing on lower gamma and C

# In[ ]:


C_set = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]     # A set of C values to try
gamma_set = [0.0001, 0.0003, 0.001, 0.003]     # A set of C values to try

Index = ['gamma = %s' % g for g in gamma_set]
Columns = ['C = %s' % C for C in C_set]

rbf_training_scores_df = pd.DataFrame(index=Index, columns=Columns)
rbf_cv_scores_df = pd.DataFrame(index=Index, columns=Columns)
rbf_cv_f1_df = pd.DataFrame(index=Index, columns=Columns)

rbf_best_f1_score = 0

for C in C_set:
    for gamma in gamma_set:
        #print('\n-----------------------------------------------------------------------')
        #print('Fitting Gaussian SVM with regularisation parameter %f and gamma, %f' % (C, gamma))
        #print('-----------------------------------------------------------------------')
        a, b, c = rbf_test(X_train, y_train, X_cv, y_cv, C, gamma)    
        #print('Training score = %f' % a)
        #print('Cross-validation score = %f' % b)
        #print('F1 score = %f' %c)
        
        rbf_training_scores_df.iloc[gamma_set.index(gamma), C_set.index(C)] = a
        rbf_cv_scores_df.iloc[gamma_set.index(gamma), C_set.index(C)] = b
        rbf_cv_f1_df.iloc[gamma_set.index(gamma), C_set.index(C)] = c
        
        
        training_scores.append(a)
        cv_scores.append(b)
        cv_f1.append(c)
        if c > best_f1_score:
            best_f1_score =c


# In[ ]:


rbf_training_scores_df = rbf_training_scores_df.astype(float)
rbf_cv_scores_df = rbf_cv_scores_df.astype(float)
rbf_cv_f1_df = rbf_cv_f1_df.astype(float)

plt.figure(figsize=(12,6))
sns.heatmap(rbf_training_scores_df, annot=True, linewidths = 0.5, vmin=0.7, vmax=1)
plt.yticks(rotation=0)
plt.title('Training Data Accuracy', pad=10)
plt.show()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(rbf_cv_f1_df, annot=True, linewidths = 0.5, vmin=0, vmax=0.8)
plt.yticks(rotation=0)
plt.title('F1 Score on Cross-Validation Data', pad=10)
plt.show()


# These results seem promising. Let's take a look at more values of C from 0.1 to 10 and more values of gamma below 0.0001

# In[ ]:


C_set = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8]     # A set of C values to try
gamma_set = [3e-06, 0.00001, 0.00003, 0.0001, 0.0003]     # A set of C values to try

Index = ['gamma = %s' % g for g in gamma_set]
Columns = ['C = %s' % C for C in C_set]

rbf_training_scores_df = pd.DataFrame(index=Index, columns=Columns)
rbf_cv_scores_df = pd.DataFrame(index=Index, columns=Columns)
rbf_cv_f1_df = pd.DataFrame(index=Index, columns=Columns)

rbf_best_f1_score = 0

for C in C_set:
    for gamma in gamma_set:
        print('\n-----------------------------------------------------------------------')
        print('Fitting Gaussian SVM with regularisation parameter %f and gamma, %f' % (C, gamma))
        print('-----------------------------------------------------------------------')
        a, b, c = rbf_test(X_train, y_train, X_cv, y_cv, C, gamma)    
        #print('Training score = %f' % a)
        #print('Cross-validation score = %f' % b)
        #print('F1 score = %f' %c)
        
        rbf_training_scores_df.iloc[gamma_set.index(gamma), C_set.index(C)] = a
        rbf_cv_scores_df.iloc[gamma_set.index(gamma), C_set.index(C)] = b
        rbf_cv_f1_df.iloc[gamma_set.index(gamma), C_set.index(C)] = c
        
        
        training_scores.append(a)
        cv_scores.append(b)
        cv_f1.append(c)
        if c > best_f1_score:
            best_f1_score =c


# In[ ]:


rbf_training_scores_df = rbf_training_scores_df.astype(float)
rbf_cv_scores_df = rbf_cv_scores_df.astype(float)
rbf_cv_f1_df = rbf_cv_f1_df.astype(float)

plt.figure(figsize=(12,6))
sns.heatmap(rbf_training_scores_df, annot=True, linewidths = 0.5, vmin=0.7, vmax=1)
plt.yticks(rotation=0)
plt.title('Training Data Accuracy', pad=10)
plt.show()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(rbf_cv_f1_df, annot=True, linewidths = 0.5, vmin=0, vmax=0.9)
plt.yticks(rotation=0)
plt.title('F1 Score on Cross-Validation Data', pad=10)
plt.show()


# ## 2.4 - Choosing a final SVM model and testing it on the test 
# 
# Based on the F1 Score for the cross-validation data, there are two potential candidates for the best model (both shown with a F1 score of 0.82 on the figure above).
# 
# Let's see how these two perform on the test data.

# In[ ]:


best_svm_rbf1 = SVC(C= 1.6, kernel='rbf', gamma=3e-05)
best_svm_rbf1.fit(X_train, y_train)
best_pred_list1 = best_svm_rbf1.predict(X_test).tolist()
best_precision1, best_recall1 = calc_prec_recall(y_test.tolist(), best_pred_list1)

best_svm_rbf2 = SVC(C= 0.4, kernel='rbf', gamma=1e-04)
best_svm_rbf2.fit(X_train, y_train)
best_pred_list2 = best_svm_rbf2.predict(X_test).tolist()
best_precision2, best_recall2 = calc_prec_recall(y_test.tolist(), best_pred_list2)

best_fscore1 = f1_score(best_precision1, best_recall1)
best_fscore2 = f1_score(best_precision2, best_recall2)


# In[ ]:


print("Model 1")
print("F1 Score for 1st model on test data: %.2f%%" % (100 * best_fscore1))
print("Precision: %.2f%%" % (100* best_precision1))
print("Recall: %.2f%%" % (100* best_recall1))

print("\nModel 2")
print("F1 Score for 2nd model on test data: %.2f%%" % (100 * best_fscore2))
print("Precision: %.2f%%" % (100* best_precision2))
print("Recall: %.2f%%" % (100* best_recall2))


# In conclusion, these two models performed reasonably well on the test data set.
# 
# The SVM model with a Gaussian (RBF) kernel was able to identify approximately 80% of all of the fraudulent transations as fraudulent (recall). And of all of the transactions that the model selected as fraudulent, approximately 80% of them were indeed fraudulent (preicsion).
# 
# I am pretty happy with this overall result and demonstration of how to avoid overfitting and how to handle skewed data sets in classification problems.
# 
# I am sure it could be improved further with more data or perhaps a more complex model such as a neural network. 
