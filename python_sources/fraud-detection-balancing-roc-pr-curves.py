#!/usr/bin/env python
# coding: utf-8

# This kernel will compare various data balancing techniques with powerful boosting models. Models used here are Random Forest, XGBoost and LGB. We will look at precision-recall curve and roc curve.
# ###### Key Notes:
# 1. Dataset is highly imbalanced. Thus, we will work on both i.e., imbalanced and balanced dataset.
# 2. We will how precision-recall curve gives us more accurate details of the result as compare to roc curve.

# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, roc_curve, auc, average_precision_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv("../input/creditcard.csv")
df.head()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(2,1, sharex = True, figsize = [12,4])

ax1.hist(df.Time[df.Class == 1], bins = 50)
ax1.set_title("Fraudulent")
ax2.hist(df.Time[df.Class == 0], bins = 50)
ax2.set_title("Non-Fraudulent")

plt.xlabel('Time (in Seconds)')
plt.ylabel('Number of Transactions')
plt.show()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(2,1, sharex = True, figsize = [12,4])

ax1.hist(df.Amount[df.Class == 1], bins = 30)
ax1.set_title("Fraudulent")
ax2.hist(df.Amount[df.Class == 0], bins = 30)
ax2.set_title("Non-Fraudulent")

plt.xlabel('Amount in $')
plt.ylabel('Number of Transactions')
plt.yscale('log')
plt.show()


# Amount variable is skewed. So we will standardize it with mean = 0 and sd = 1.

# In[ ]:


df["Normalized_Amount"] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
#Drop time & amount variable
df = df.drop(['Time', 'Amount'], axis = 1)
df.head()


# In[ ]:


Class = [len(df.loc[df.Class == 1]), len(df.loc[df.Class == 0])]
pd.Series(Class, index = ['Fraudulent', 'Non-fraudulent'], name = 'target')


# In[ ]:


#Percentage of minority(fraudulent) class
print('% of Fraudulent Class = {:.3f}%'.format(len(df[df.Class == 1])*100 / len(df)))


# Accuracy = TP+TN/Total data
# 
# Precison = TP/(TP+FP)
# 
# Recall = TP/(TP+FN)
# 
# Now for our case recall will be a better option because number of normal transacations are very high as compared to the number of fraud cases and sometime a fraud case will be predicted as normal. So, recall will give us a sense of only fraud cases. Now, it is possible to build our model with 100% recall but the downside will be our precision will be worst and that will result into nothing but useless model. Thus our aim is to attain high recall with maintaing the precision.
# 
# Now, this notebook is going to be too big. Thus, we will not repeat any code again and again. Thus, here I've made a function which will calculate all the necessary and sufficient results that we want and will compare the three classifiers using visualization of PR curve and ROC curve.

# In[ ]:


def results(balancing_technique):
    print(balancing_technique)
    fig, (ax1, ax2) = plt.subplots(1,2,figsize = (12,6))
    model_name = ["RF", "XGB", "LGB"]
    RFC = RandomForestClassifier(random_state = 0)
    XGBC = XGBClassifier(random_state = 0)
    LGBC = LGBMClassifier(random_state = 0)

    for clf,i in zip([RFC, XGBC, LGBC], model_name):
        model = clf.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:,1]
        print("#"*25,i,"#"*25)
        print("Training Accuracy = {:.3f}".format(model.score(X_train, y_train)))
        print("Test Accuracy = {:.3f}".format(model.score(X_test, y_test)))
        print("ROC_AUC_score : %.6f" % (roc_auc_score(y_test, y_pred)))
        #Confusion Matrix
        print(confusion_matrix(y_test, y_pred))
        print("-"*15,"CLASSIFICATION REPORT","-"*15)
        print(classification_report(y_test, y_pred))
        
        #precision-recall curve
        precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_prob)
        avg_pre = average_precision_score(y_test, y_pred_prob)
        ax1.plot(precision, recall, label = i+ " average precision = {:0.2f}".format(avg_pre), lw = 3, alpha = 0.7)
        ax1.set_xlabel('Precision', fontsize = 14)
        ax1.set_ylabel('Recall', fontsize = 14)
        ax1.set_title('Precision-Recall Curve', fontsize = 18)
        ax1.legend(loc = 'best')
        #find default threshold
        close_default = np.argmin(np.abs(thresholds_pr - 0.5))
        ax1.plot(precision[close_default], recall[close_default], 'o', markersize = 8)

        #roc-curve
        fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr,tpr)
        ax2.plot(fpr,tpr, label = i+ " area = {:0.2f}".format(roc_auc), lw = 3, alpha = 0.7)
        ax2.plot([0,1], [0,1], 'r', linestyle = "--", lw = 2)
        ax2.set_xlabel("False Positive Rate", fontsize = 14)
        ax2.set_ylabel("True Positive Rate", fontsize = 14)
        ax2.set_title("ROC Curve", fontsize = 18)
        ax2.legend(loc = 'best')
        #find default threshold
        close_default = np.argmin(np.abs(thresholds_roc - 0.5))
        ax2.plot(fpr[close_default], tpr[close_default], 'o', markersize = 8)
        plt.tight_layout()


# In[ ]:


X = df.drop(columns = 'Class')
y = df['Class']
#Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[ ]:


results("Without Balancing")


# ### Understanding the graph
# 1. Average precision calculated here is the area under the precision-recall curve which is also called as average precision.
# 2. The dot in the graph are the default threshold values which is always equal to 0.5 for trees.
# 
# #### Results:
# 1. As expected, xgboost performed well for imbalanced dataset as boosting methods provide higher weights to minority classes. 2. Besides, for highly imbalanced data, even Random forest performed well (which was beyond my expectations).
# 3. Now, comparing results of RF and XGB, both performed almost same. But there is quite difference between ROC AUC between both. One of the reason for this may be due to the less number of thresholds present in RF.
# 4. Surprisingly, LGB performed very bad. This may be due to the default parameters present as LGB is quite sensitive to parameter setting.

# ### NOTE:
# You may notice that area using model function (roc_auc_score) and area from compare_models function (roc_curve) are different. This is because I've used y_pred with roc_aur_score and y_pred_prob with roc_curve. Using y_pred_prob is preffered because it increases the number of thresholds(so that more points are availble to plot the graph accurately) whereas y_pred will only give 3 threshold(0, 1, and any value in between 0 and 1).

# # Data Balancing Techniques

# ### 1. Down-Sampling

# In[ ]:


print("Minority Class =", len(df[df.Class == 1]))


# In[ ]:


train_majority = df[df.Class == 0]
train_minority = df[df.Class == 1]
train_majority_downsampled = resample(train_majority, replace = False, n_samples = 492, random_state = 0)
train_downsampled = pd.concat([train_majority_downsampled, train_minority])


# In[ ]:


X = train_downsampled.drop(columns = 'Class')
y = train_downsampled['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Although now dataset is balanced so boosting models will not give higher weights to the fraulent class, still let's see if it performs well on a balanced dataset.

# In[ ]:


results("Down Sampling")


# 1. All models performed well but the main disadvantage of down sampling is that it will lose a lot of useful information from the dataset as it is removing more than 99% of non-fraudulent cases from the dataset. Thus, this should not be a practical approach here.

# ### 2. Up-Sampling

# In[ ]:


#Note in up sampling, first split the minority class data into train and test set and then up-sample the train data and test it with test data
X = df.drop(columns = 'Class')
y = df['Class']
#First split data into train and test
X_train_us, X_test, y_train_us, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
#Now resample the train data
df_us = pd.concat([X_train_us, y_train_us], axis = 1)
train_majority = df_us[df_us.Class == 0]
train_minority = df_us[df_us.Class == 1]
train_majority.shape, train_minority.shape


# In[ ]:


train_minority_upsampled = resample(train_minority, replace = True, n_samples = 199019, random_state = 0)
print(train_majority.shape, train_minority_upsampled.shape)
train_upsampled = pd.concat([train_minority_upsampled, train_majority])
X_train = train_upsampled.drop(columns = 'Class')
y_train = train_upsampled['Class']


# In[ ]:


results("Up Sampling")


# 1. Surprisingly, threshold of XGB is very earlier. But, PR curves resembles same as other models. This means using up sampling, XGB performs well on recall but precision is decreased. I mean that XGB classified less FN and more FP with more than 50% probability. Thus, this increases the recall of XGB.
# 2. Further looking at PR curves, we can say that all models performes well. As per requirement, one can set threshold and get the required result.
# 
# Now, the down side of this method here is up-sampling will restrict learning of models due to repeatation of fraudulent cases so many times and eventually will overfit also. Also it costs us very expensive computationally.

# ## 3. SMOTE FAMILY

# ### 3.1 SMOTE Regular

# In[ ]:


sm = SMOTE(random_state = 0)
X = df.drop(columns = 'Class')
y = df['Class']
X_train_sm, X_test, y_train_sm, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
X_train, y_train = sm.fit_sample(X_train_sm, y_train_sm)
X_test = np.array(X_test)
y_test = np.array(y_test)


# In[ ]:


results("SMOTE Regular")


# 1. Applying SMOTE, RF performed very well.
# 2. Looking at ROC-AUC curve, boosting performed well. But in real, I will consider RF performing well as compared to boosting methods because RF maintained precision as compared to boosting .

# ### 3.2 BorderLine SMOTE

# In[ ]:


sm = BorderlineSMOTE(random_state = 0)
X = df.drop(columns = 'Class')
y = df['Class']
X_train_sm, X_test, y_train_sm, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
X_train, y_train = sm.fit_sample(X_train_sm, y_train_sm)
X_test = np.array(X_test)
y_test = np.array(y_test)


# In[ ]:


results("Borderline SMOTE")


# 1. Applying Borderline SMOTE, again RF gave high F1-score.
# 2. Surprisingly, LGB performed worst looking at PR curves and best according to ROC. 

# ### 4. ADASYN

# #### Short Explanation:
# ADASYN =  SMOTE + random values between 0 and 1. This makes dataset somewhat more robust.

# In[ ]:


adasyn = ADASYN(random_state = 0)
X = df.drop(columns = 'Class')
y = df['Class']
X_train_as, X_test, y_train_as, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
X_train, y_train = adasyn.fit_sample(X_train_as, y_train_as)
X_test = np.array(X_test)
y_test = np.array(y_test)


# In[ ]:


results("ADASYN")


# 1. All models performed well.

# ##### RESULTS & CONCLUSIONS:
# 1. ROC curve for all three models were almost same. PR curve however differentiated somewhat performance based on precision.
# 2. Random forest performed best when data was balanced with different techniques as it maintained recall and precision with the default threshold and maintained precision as compared to boosting as the threshold was decreasing.
# 3. For imbalanced data XGB performed slightly well as compared to RF.
# 4. For balanced data, boosting made predictions with high recall and compromising precision and RF made predictions based on both recall and precision.
# 5. From PR curves, one can say that models performed well and almost same with SMOTE Regular. User can set threshold to obtain the required result as per the business goal.

# ##### Limitations:
# 1. I tried to tune the models with best parameters that give high F1-score but hyperparameter tuning was not possible due to high computational time. This is because the dataset is very large.
# 2. Due to privacy issue, we don't have original features. This restrics the part of feature engineering.

# ##### FURTHER WORK:
# 1. Results when data is balanced through SVMSMOTE technique.
# 2. Here I have only compared different data balancing techniques with different models with default parameters. Further, we can look at different parameters of both data balancing techniques and models for better performance.
# 3. Hyperparameter tuning can provide us better results but its not computationally affordable.
# 4. One of the limitation of this project was the unavailability of raw credit-card fraud detection dataset i.e, features. If features are available, we can try feature engineering here (polynomials, interactions, etc.).

# ### I will be very pleased if you would suggest something which I can improve in this work. Further, if you liked my work or it was helpful,  a feedback will be much appreciated. Thank you!
