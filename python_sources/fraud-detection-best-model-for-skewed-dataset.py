#!/usr/bin/env python
# coding: utf-8

# # Fraud Prediction with Skewed Dataset
# 1. Here will see how to deal with imbalanced dataset
# 2. What performance metric we should choose
# 3. How to deal with highly skewed dataset
# 4. Perform Precision-Recall trade off
# 5. Chosing best model from all - with better visulaisation of model performance

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')


# In[ ]:


data.head()


# In[ ]:





# In[ ]:


data.describe()


# In[ ]:


data.Class.value_counts(normalize=True)


# In[ ]:


import matplotlib.pyplot as plt
data.hist(figsize= (20,18))
plt.show()


# In[ ]:


X = data.iloc[:,:30]
Y = data.iloc[:,-1]


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size= 0.3 , random_state= 42,stratify= Y,shuffle = True)


# In[ ]:


y_train.value_counts()


# In[ ]:


y_test.value_counts()


# In[ ]:


from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,precision_recall_curve,recall_score,roc_auc_score,plot_precision_recall_curve,confusion_matrix


# In[ ]:


def Model_comparision_report(x_train,x_test,y_train,y_test):
    algorithms = []

    algorithms.append(("Stochastic Gradient Descent ",SGDClassifier()))
    algorithms.append(("Random Forest ", RandomForestClassifier()))
    algorithms.append(("Xtreme Gradient Boost ", XGBClassifier()))
    algorithms.append(("Bernoulli Naive Bayes ", BernoulliNB()))
    algorithms.append(("AdaBoost ", AdaBoostClassifier()))
    algorithms.append(("Extra Tress Classfier ", ExtraTreesClassifier()))
    algorithms.append(("Gradient Boosting Classifier" ,GradientBoostingClassifier()))
    algorithms.append(("Bagging Classifier ", BaggingClassifier()))
    algorithms.append(("Multi-layer Preceptron", MLPClassifier()))

    names = []

    train_f1 = []
    test_f1 = []

    train_recall = []
    test_recall = []

    train_precision = []
    test_precision = []

    train_auc = []
    test_auc = []

    train_acc = []
    test_acc = []


    for name , clf in algorithms:
        clf.fit(x_train,y_train)
        train_f1.append(f1_score(y_train,clf.predict(x_train)))
        test_f1.append(f1_score(y_test,clf.predict(x_test)))
        train_precision.append(precision_score(y_train,clf.predict(x_train)))
        test_precision.append(precision_score(y_test,clf.predict(x_test)))
        train_recall.append(recall_score(y_train,clf.predict(x_train)))
        test_recall.append(recall_score(y_test,clf.predict(x_test)))
        names.append(name)
        train_auc.append(roc_auc_score(y_train,clf.predict(x_train)))
        test_auc.append(roc_auc_score(y_test,clf.predict(x_test)))
        train_acc.append(accuracy_score(y_train,clf.predict(x_train)))
        test_acc.append(accuracy_score(y_test,clf.predict(x_test)))


    clf_comparision = pd.DataFrame({"Algorithms": names, "Train_Precision" : train_precision,
                                  "Test_Precision" : test_precision, "Train_recall" : train_recall,
                                  "Test_recall": test_recall,"Train_F1" : train_f1,"Test_F1": test_f1,
                                   "Train_AUC": train_auc, "Test_AUC": test_auc,
                                   "Train Accuracy": train_acc, "Test_Accuracy":test_acc})
    return clf_comparision


# # ** ****Base Model - without any sampling**

# In[ ]:


first_report = Model_comparision_report(x_train,x_test,y_train,y_test)
first_report


# # Three techniques:
# 1. Oversampling : Generating more minority samples
# 2. Undersampling : Reducing Majority class samples to number of minority samples
# 3. SMOTE : Synthetic data generation

# In[ ]:


oversample = pd.concat([x_train,y_train],axis = 1)
fraud = oversample.loc[oversample["Class"] == 1]
not_fraud = oversample.loc[oversample["Class"] == 0]


# In[ ]:


from sklearn.utils import resample

fraud_samples = resample(fraud,n_samples = len(not_fraud),random_state = 27)


# In[ ]:


fraud_samples.shape,not_fraud.shape


# In[ ]:


new_data = pd.concat([fraud_samples,not_fraud], axis = 0)
new_data.shape


# In[ ]:


x_train = new_data.iloc[:,:30]
y_train = new_data.iloc[:,-1]
x_train.shape,y_train.shape


# In[ ]:


x_test.shape,y_test.shape


# In[ ]:


oversample_report = Model_comparision_report(x_train,x_test,y_train,y_test)
oversample_report


# In[ ]:


# replace = False because we reducing the majority class 
nfraud_samples = resample(not_fraud,replace= False,n_samples = len(fraud),random_state = 27)

nfraud_samples.shape,fraud.shape,not_fraud.shape


# In[ ]:


under_data = pd.concat([nfraud_samples,fraud],axis = 0)
x_train = under_data.iloc[:,:30]
y_train = under_data.iloc[:,30]
x_train.shape,y_train.shape


# In[ ]:


undersample_report = Model_comparision_report(x_train,x_test,y_train,y_test)
undersample_report


# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size= 0.3 , random_state= 42,stratify= Y,shuffle = True)
sm = SMOTE(random_state=27,k_neighbors=5)
x_train , y_train = sm.fit_sample(x_train,y_train)


# In[ ]:



x_train.shape,y_train.shape


# In[ ]:


y_train.value_counts()


# In[ ]:


y_test.value_counts()


# In[ ]:


smote_report = Model_comparision_report(x_train,x_test,y_train,y_test)
smote_report


# In[ ]:


xtree_clf = ExtraTreesClassifier()
xtree_clf.fit(x_train,y_train)
plot_precision_recall_curve(xtree_clf,x_test,y_test)
plt.show()


# In[ ]:


train_prob = xtree_clf.predict_proba(x_test)
p,r,t = precision_recall_curve(y_test,train_prob[:,1])


# In[ ]:


prt = pd.DataFrame({'precision':p[:-1],'recall':r[:-1],'threshold':t},columns=['precision','recall','threshold'])


# In[ ]:


prt.loc[(prt['recall'] > 0.8) & (prt['precision'] > 0.8)].sort_values(by=['recall'],ascending = False)


# In[ ]:


y_pred_threshold = (train_prob[:,1] >=  0.37)


# In[ ]:


precision_score(y_test,y_pred_threshold)


# In[ ]:


recall_score(y_test,y_pred_threshold)


# In[ ]:


accuracy_score(y_test,y_pred_threshold)


# In[ ]:


f1_score(y_test,y_pred_threshold)


# In[ ]:


confusion_matrix(y_test,y_pred_threshold)


# # Conclusion
# * When we have imbalanced dataset - like this highly skewed then Oversampling and SMOTE really helps our model in terms of F1 score
# * We should intially split our dataset and apply sampling techniques only on train set
# * We should always test our model in original test dataset which does not have samples created by us
# * We have seen that RandomForest Classifier/Extra Trees classifier has the best results among the other powerful classifier
# * We see the accuracy is not a good meteric for evaluation for imbalanced dataset
# * F1 score and Precision-recall trade off or curve is better perfoamce metric for highly skewed data
# * For Extra Trees 0.37 is the threshold which gives highest F1 score of 0.8243
# * We can further play around and optimize the RandomForest Classifier/Extra Tree clasifier as both gave same results
# * Sometimes Neural Network model does not perform well as compared to other ensemble models
