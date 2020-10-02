#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# It's been created a new folder (train-n-test-data) with duplicates of train and test data because some minor issues with the competition folder
import sklearn 
import os
print(os.listdir("../input/train-n-test-data"))
#The train data
train_data = pd.read_csv("../input/train-n-test-data/train_data.csv")
train_data.drop('Id',axis = 'columns')
#As the data Id collumn just refers to a specific email, we need this information since the email will be classified in terms of the features and the target(ham) columns
train_data.ham


# In[ ]:


# The features to be classified 
test_features = pd.read_csv("../input/train-n-test-data/test_features.csv")
test_features_with_id = test_features
test_features = test_features.drop('Id',axis = 'columns')
test_features


# Now analyzing a little bit the features from the train data ...
# 
# 
# 

# In[ ]:


train_data.shape


# In[ ]:


train_data['ham'].value_counts()


# In[ ]:


train_data


# In[ ]:


#for special characters let's create the excerpt from the dataset containing the Id, ham and char_freq
special_char = train_data[['char_freq_;','char_freq_(', 'char_freq_[','char_freq_!','char_freq_$','char_freq_#','ham','Id']]
special_char
test_special_char = test_features_with_id[['char_freq_;','char_freq_(', 'char_freq_[','char_freq_!','char_freq_$','char_freq_#','Id']]


# Now, Let's separate the excerpt with the 48 word frequency collumns 
# 

# In[ ]:


word_freq = train_data.loc[:,"word_freq_make":"word_freq_conference"]
test_word_freq = test_features_with_id.loc[:,"word_freq_make":"word_freq_conference"]
#Adding the ham and the id collumn
word_freq.loc[:,'ham'] = train_data[['ham']]
word_freq.loc[:,'Id'] = train_data[['Id']]
test_word_freq.loc[:,'Id'] = test_features_with_id[['Id']]
word_freq.shape
test_word_freq


# In[ ]:


word_freq


# For the remaining the columns ...

# In[ ]:


capital_run = train_data[['capital_run_length_average','capital_run_length_longest','capital_run_length_total','ham','Id']]
test_capital_run_with_id = test_features_with_id[['capital_run_length_average','capital_run_length_longest','capital_run_length_total', 'Id']]
test_capital_run = test_capital_run_with_id.drop('Id', axis = 'columns')


# **KNN**

# First for the entire data frame, then for the selected parts
# 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
X_train_data= train_data
X_train_data = X_train_data.drop('ham',axis = 'columns')
X_train_data= X_train_data.drop('Id',axis = 'columns')
Y_train_data= train_data.loc[:,'ham':'ham']
#setting a binary value for ham
Y_train_data = Y_train_data.astype(int)
# to reset for boolean, just try ~ Y_train_data = Y_train_data.astype(bool)
Y_train_data.ham
X_train_data


# In[ ]:


Scores =[]
for i in range (1,40):
    knn = KNeighborsClassifier(n_neighbors = i)
    scores = cross_val_score(knn, X_train_data, Y_train_data.ham, cv=10)
    Scores.append(scores.mean())
Scores


# In[ ]:


#But scores is not a pd.DataFrame, so:
Scores_df = pd.DataFrame()
Scores_df['Knn Scores']= Scores
Scores_df




# So, graphically the efficiency of knn considering one i distance value is:

# In[ ]:


Scores_df.plot()


# So we will predict the train data with knn = 1

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train_data, Y_train_data.ham)
Y_test_Pred = knn.predict(test_features)


# In[ ]:


Y_test_Pred = Y_test_Pred.astype(bool)


# In[ ]:


Predict = pd.DataFrame()
Predict['Id'] = test_features_with_id['Id']
Predict['ham'] = Y_test_Pred
Predict.set_index('Id', inplace=True)
Predict
#Predict.drop('', axis = 'columns')


# In[ ]:


Predict.to_csv("Prediction_Knn.csv")


# **NAIVE BAYES**

# In[ ]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB,BernoulliNB


# In[ ]:


#For Gaussian Naive Bayes, the classification(clf1) is 

clf1 = GaussianNB()
clf1.fit(X_train_data, Y_train_data.ham)
GaussianNB(priors=None, var_smoothing=1e-09)
Y_Pred_G_NB = clf1.predict(test_features)
Y_Pred_G_NB = Y_Pred_G_NB.astype(bool)
Y_Pred_G_NB


# In[ ]:


Pred_G_NB = pd.DataFrame()
Pred_G_NB['Id'] = test_features_with_id['Id']
Pred_G_NB['ham'] = Y_Pred_G_NB
Pred_G_NB.set_index('Id', inplace=True)
Pred_G_NB


# In[ ]:


Pred_G_NB.to_csv("Prediction_GaussianNB.csv")


# In[ ]:


#For the Multinomial Naive Bayes Classifier (clf2)
clf2 = MultinomialNB()
clf2.fit(X_train_data, Y_train_data.ham)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
Y_Pred_M_NB = clf2.predict(test_features)
Y_Pred_M_NB = Y_Pred_M_NB.astype(bool)
Y_Pred_M_NB


# In[ ]:


Pred_M_NB = pd.DataFrame()
Pred_M_NB['Id'] = test_features_with_id['Id']
Pred_M_NB['ham'] = Y_Pred_M_NB
Pred_M_NB.set_index('Id', inplace=True)
Pred_M_NB


# In[ ]:


Pred_M_NB.to_csv("Prediction_MultinomialNB.csv")


# In[ ]:


#For the ComplementNB, the clf3 (now when can make a little more compact)
  
clf3 = ComplementNB()
clf3.fit(X_train_data, Y_train_data.ham)
ComplementNB(alpha=1.0, class_prior=None, fit_prior=True, norm=False)
Y_Pred_C_NB = clf3.predict(test_features)
Y_Pred_C_NB = Y_Pred_C_NB.astype(bool)
Pred_C_NB = pd.DataFrame()
Pred_C_NB['Id'] = test_features_with_id['Id']
Pred_C_NB['ham'] = Y_Pred_C_NB
Pred_C_NB.set_index('Id', inplace=True)
Pred_C_NB.to_csv("Prediction_ComplementNB.csv")
Pred_C_NB


# In[ ]:


#And finally, the bernoulli NB classifier :

clf4 = BernoulliNB()
clf4.fit(X_train_data, Y_train_data.ham)
BernoulliNB(alpha=0.5, binarize=0.0, fit_prior=True)
Y_Pred_B_NB = clf4.predict(test_features)
Y_Pred_B_NB = Y_Pred_B_NB.astype(bool)
Pred_B_NB = pd.DataFrame()
Pred_B_NB['Id'] = test_features_with_id['Id']
Pred_B_NB['ham'] = Y_Pred_B_NB
Pred_B_NB.set_index('Id', inplace=True)
Pred_B_NB.to_csv("Prediction_BernoulliNB.csv")
Pred_B_NB


# **Now, let's summarize the codes used above and apply them to select features (char_freq and word_freq)**

# In[ ]:


def KNN(n,train_data,x_test,doc_name):
    x_train_data = train_data.drop('ham',axis = 'columns')
    x_train_data= x_train_data.drop('Id',axis = 'columns')
    y_train_data= train_data.loc[:,'ham':'ham']
    x_test_with_id = x_test
    x_test = x_test.drop('Id',axis = 'columns')
    #setting a binary value for ham
    y_train_data = y_train_data.astype(int)
    # to reset for boolean, just try ~ Y_train_data = Y_train_data.astype(bool)
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(x_train_data, y_train_data.ham)
    y_test_Pred = knn.predict(x_test)
    y_test_Pred = y_test_Pred.astype(bool)
    Predict = pd.DataFrame()
    Predict['Id'] = x_test_with_id['Id']
    Predict['ham'] = y_test_Pred
    Predict.set_index('Id', inplace=True)
    Predict.to_csv(doc_name)

    


# In[ ]:


#now using the function ...

KNN(1,word_freq,test_word_freq,"KNN_word_freq.csv")


# In[ ]:


#for special characters...
KNN(1,special_char,test_special_char,"KNN_special_char.csv")


# In[ ]:


#Now, summarizing the four types of naive bayes used above
def Gaussian_NB(train_dt,test_feat,name,priors,smooth):
    train_dt = train_dt.drop('Id', axis = 'columns')
    test_feat = test_feat.drop('Id', axis = 'columns')
    y_train_dt = train_dt.ham
    x_train_dt = train_dt.drop('ham', axis ='columns')
    clf = GaussianNB()
    clf.fit(x_train_dt,y_train_dt)
    GaussianNB(priors=priors, var_smoothing=smooth) #None, 1e-09
    Y_Pred_GNB = clf.predict(test_feat)
    Y_Pred_GNB = Y_Pred_GNB.astype(bool)
    Y_Pred_GNB
    Pred_GNB = pd.DataFrame()
    Pred_GNB['Id'] = test_features_with_id['Id']
    Pred_GNB['ham'] = Y_Pred_GNB
    Pred_GNB.set_index('Id', inplace=True)
    print(Pred_GNB)
    Pred_GNB.to_csv(name)
def  Multinomial_NB(train_dt,test_feat,name,smooth,class_prior, fit_prior):
    train_dt = train_dt.drop('Id', axis = 'columns')
    test_feat = test_feat.drop('Id', axis = 'columns')
    y_train_dt = train_dt.ham
    x_train_dt = train_dt.drop('ham', axis ='columns')
    clf = MultinomialNB()
    clf.fit(x_train_dt,y_train_dt)
    MultinomialNB(alpha=smooth, class_prior=class_prior, fit_prior=fit_prior) # Default = (1.0,None, True)
    Y_Pred_MNB = clf.predict(test_feat)
    Y_Pred_MNB = Y_Pred_MNB.astype(bool)
    Y_Pred_MNB
    Pred_MNB = pd.DataFrame()
    Pred_MNB['Id'] = test_features_with_id['Id']
    Pred_MNB['ham'] = Y_Pred_MNB
    Pred_MNB.set_index('Id', inplace=True)
    print(Pred_MNB)
    Pred_MNB.to_csv(name)
def  Complement_NB(train_dt,test_feat,name,smooth,class_prior,fit_prior,norm):
    train_dt = train_dt.drop('Id', axis = 'columns')
    test_feat = test_feat.drop('Id', axis = 'columns')
    y_train_dt = train_dt.ham
    x_train_dt = train_dt.drop('ham', axis ='columns')
    clf = ComplementNB()
    clf.fit(x_train_dt,y_train_dt)
    ComplementNB(alpha=smooth, class_prior=class_prior, fit_prior=fit_prior, norm=norm) #Default = (1.0,None, True,False)
    Y_Pred_CNB = clf.predict(test_feat)
    Y_Pred_CNB = Y_Pred_CNB.astype(bool)
    Y_Pred_CNB
    Pred_CNB = pd.DataFrame()
    Pred_CNB['Id'] = test_features_with_id['Id']
    Pred_CNB['ham'] = Y_Pred_CNB
    Pred_CNB.set_index('Id', inplace=True)
    print(Pred_CNB)
    Pred_CNB.to_csv(name)
def  Bernoulli_NB(train_dt,test_feat,name,smooth,b,fit_prior):
    train_dt = train_dt.drop('Id', axis = 'columns')
    test_feat = test_feat.drop('Id', axis = 'columns')
    y_train_dt = train_dt.ham
    x_train_dt = train_dt.drop('ham', axis ='columns')
    clf = BernoulliNB()
    clf.fit(x_train_dt,y_train_dt)
    BernoulliNB(alpha=smooth, binarize=b, fit_prior=fit_prior) # Default (1.0,0.0,True)
    Y_Pred_BNB = clf.predict(test_feat)
    Y_Pred_BNB = Y_Pred_BNB.astype(bool)
    Y_Pred_BNB
    Pred_BNB = pd.DataFrame()
    Pred_BNB['Id'] = test_features_with_id['Id']
    Pred_BNB['ham'] = Y_Pred_BNB
    Pred_BNB.set_index('Id', inplace=True)
    print(Pred_BNB)
    Pred_BNB.to_csv(name)
    


# So, for the **word frequency base**:

# In[ ]:


word_freq


# In[ ]:


Gaussian_NB(word_freq,test_word_freq,"GaussianNB_word_freq.csv",None, 1e-09)
#0.74871


# In[ ]:


Multinomial_NB(word_freq,test_word_freq,"MultinomialNB_word_freq.csv",1.0,None, True)
#0.83815


# In[ ]:


Complement_NB(word_freq,test_word_freq,"ComplementNB_word_freq.csv",1.0,None, True,False)
#0.82773


# In[ ]:


Bernoulli_NB(word_freq,test_word_freq,"BernoulliNB_word_freq.csv",1.0,0.0,True)
#0.92374


# Now, for the **special characters**:

# In[ ]:


Gaussian_NB(special_char, test_special_char, "GaussianNB_special_char.csv",None, 1e-09)
#0.91857


# In[ ]:


Multinomial_NB(special_char, test_special_char, "MultinomialNB_special_char.csv",1.0,None, True)
#0.93577


# In[ ]:


Complement_NB(special_char, test_special_char, "ComplementNB_special_char.csv",1.0,None, True,False)
#0.89619


# In[ ]:


Bernoulli_NB(special_char, test_special_char, "BernoulliNB_special_char.csv",1.0,0.0,True)
#0.92638


# Lastly, for the capital run portion of the databases:

# In[ ]:


Bernoulli_NB(capital_run, test_capital_run_with_id, "BernoulliNB_capital_run.csv",1.0,0.0,True)


# So, after that, the most efficient (using the test database) classifier is the Bernoulli Naives Bayes that uses the complete train database. Now, playing with the variables:

# In[ ]:


#I thought that changes in the smoothing parameter and the fit_prior would afect the classification but result remains the same
#Still, the code used to test is presented below 
x=0.0
y=str(0)+str(int(x*10))+ "BNB"
name = y + (".csv")
print(name)


# In[ ]:


for i in range(0,5):
    x=i/5
    y=str(0)+str(int(x*10))+ "BNB"
    name = y +(".csv")
    Bernoulli_NB(train_data, test_features_with_id, name,x,0.0,False)


# f3 scores weights the importance of **recall** (the ratio of true positives divided by all the positives within the actual label - TP and FN), so beta = 3 gives the recall 3 times more importancy than **precision** (the ratio of true positives divided by all the labelled positives - TP and FP)

# In[ ]:


from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import roc_curve, auc
train_w = train_data.drop('ham', axis = 'columns')
train_w = train_data.drop('Id', axis = 'columns')
f3 = make_scorer(fbeta_score, beta=3)
train_f3 = cross_val_score(BernoulliNB(), train_w, train_data.ham, cv=10, scoring = f3 )
train_f3.mean()


# Before the ROC curve, I would like to test the capital run potential

# In[ ]:


capital_run_w = capital_run.drop('ham', axis = 'columns')
capital_run_w  = capital_run.drop('Id', axis = 'columns')
f3 = make_scorer(fbeta_score, beta=3)
train_f3 = cross_val_score(BernoulliNB(), capital_run_w, capital_run.ham, cv=10, scoring = f3 )
train_f3.mean()


# The Receiver Characteristic Curve is the ration between True Positives and False Positive Rates, where alfa(or treshold) denotes the ratio of false positives and false negatives. It means that, for alfa greater than 1, we let the misclassification error of giving one label positive when it should be negative less importancy (and thus, we calibrate the classifier to make this alfa result) than the error of giving one label negative when it should be positive

# (1,1) is point for very small alfa and (0,0) is a point for very large alfa and for a very large alfa we get a fewer false positive rate (and larger false negative rates). The ideal ROC curve would be one that disposes of two segmented lines from (0,0) to (0,1) and then from (0,1) to (1,1). But, as we couldn't get one of these, the best possible point would be the one closest to (0,1) where the line parallel to the no discrimination line (x=y) crosses the ROC curve. It will be the point with less chance of false positive and false negative errors (it is also called the  Youden's J statistic).

# In[ ]:


prob = cross_val_predict(BernoulliNB(), train_w, train_data.ham, cv=10, method = 'predict_proba')
fpr, tpr ,thresholds =roc_curve(train_data.ham,prob[:,1]);
lw=2
plt.plot(fpr,tpr, color='red')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()


# In the end, the best classifier found to the spam/ham problem was the one learned from a bernoulli distribution naive bayes (that is, multiple binomial and independent variables). There was an attempt to clear the features to better results, but until now, no evidence showed effiency in this case.

# ***Below is an attempt to weight some of the columns as word_freq_free, char_freq_!, but apparently this attempt backfired in the sense that the accuracy of the model dropped (probably by overfitting)***

# In[ ]:


word_freq


# In[ ]:


x_train_data = train_data.drop('Id', axis = 'columns')
x_train_data = x_train_data.drop('ham', axis = 'columns')
x_train_data.loc[:,'char_freq_!2'] = train_data['char_freq_!']
test_features.loc[:,'char_freq_!2'] = train_data['char_freq_!']
x_train_data.loc[:,'char_freq_!3'] = train_data['char_freq_!']
test_features.loc[:,'char_freq_!3'] = train_data['char_freq_!']
x_train_data.shape
test_features.shape


# In[ ]:


x_train_data


# In[ ]:


train_data.ham.shape


# In[ ]:


clf5 = BernoulliNB()
clf5.fit(x_train_data, train_data.ham)
BernoulliNB(alpha=0.1, binarize=0.0, fit_prior=True)
Y_Pred_B_NB = clf5.predict(test_features)
Y_Pred_B_NB = Y_Pred_B_NB.astype(bool)
Pred_B_NB = pd.DataFrame()
Pred_B_NB['Id'] = test_features_with_id['Id']
Pred_B_NB['ham'] = Y_Pred_B_NB
Pred_B_NB.set_index('Id', inplace=True)
Pred_B_NB.to_csv("Weight_BernoulliNB.csv")
Pred_B_NB 


# -------------------------------------------------------------------------------------------------------------------------------------------------------
