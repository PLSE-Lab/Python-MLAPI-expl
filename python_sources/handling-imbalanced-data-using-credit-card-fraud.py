#!/usr/bin/env python
# coding: utf-8

# ### Importing the libraries to be used

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm


# ### Import the dataset

# In[ ]:


df= pd.read_csv('../input/creditcard-fraud-detection/creditcard.csv')
df.head()


#  All the variables/ features  are standardised apart from amount and time , hence standardsing it

# In[ ]:


from scipy.stats import zscore
df1=df[['Time','Amount']].apply(zscore)
df.drop(['Time','Amount'],axis=1,inplace=True)
df.head()


# In[ ]:


df1.head()
df2=pd.concat((df,df1),axis=1)
df2.head()


# In[ ]:


# the no of rows and columns:
df2.shape


# In[ ]:


# the descriptive statistics
df2.describe().T


# In[ ]:


# the target
df2['Class'].value_counts(normalize=True)


#  Hence this is an imbalanced dataset with majority of the class belonging to class 0. We will try to balance the data by oversampling using SMOTE

# In[ ]:


# Checking for missing Values
df2.isnull().sum()


# Hence no missing values in the data

# In[ ]:


df3=df2.drop('Class',axis=1)


# In[ ]:


# distribution of the features and target
cols= list(df3.columns)
for i in cols:
    sns.boxplot(y=i,x=df2['Class'],data=df3)
    plt.show()


# From the box plot ,looks like most of the features have huge amount of outliers ,hence we will first keep the ouliers and make the model

# Making the first model using stats

# In[ ]:


x=df3
y=df2['Class']
x_const = sm.add_constant(x)


# OVERSAMPLING USING SMOTE TO HANDEL IMBALANCED DATA

# In[ ]:


import imblearn
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=2)
x, y = smote.fit_sample(x, y.ravel())


# In[ ]:


x.shape


# In[ ]:


y.shape


# After oversampling the number of rows have increased from 284807 to 568630.

# In[ ]:



x_const = sm.add_constant(x)
x_const.shape


# In[ ]:


model1=sm.Logit(y,x_const).fit()
model1.summary()


#  The p-value of the model is <0.05 (considering alpha to be 0.05)  and hence the model is significant.
#  The p-value of individual features shows that  the fetures V21 and V27 are insignificant as the p-values are greater than the alpha.

# In[ ]:


y_pred_proba=model1.predict(x_const)


# In[ ]:


def pro(y_pred):
    if y_pred <0.5:
        y_pred=0
    elif y_pred>0.5:
        y_pred=1
    return y_pred


# In[ ]:


y_pred=y_pred_proba.apply(pro)


# In[ ]:


from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
print('accuracy_ score :',accuracy_score(y,y_pred))


# In[ ]:


print('roc_auc_score:',roc_auc_score(y,y_pred_proba))


# In[ ]:


# removing the insignificant features and making a model


# In[ ]:


x_const=x_const.drop(['V21','V27'],axis=1)


# In[ ]:


model2= sm.Logit(y,x_const).fit()
model2.summary()


# In[ ]:


y_pred_proba = model2.predict(x_const)
y_pred=y_pred_proba.apply(pro)


# In[ ]:


print('accuracy_score:', accuracy_score(y_pred,y))


# In[ ]:


print('roc_auc_score : ',roc_auc_score(y,y_pred_proba))


# From the p-value we  can find that  all the remaining variables are significant

# The accuracy score and roc-auc score shows that their is no significant increase in the performance after removing insignificant features as well

# In[ ]:


MACHINE LEARNING MODEL


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)


# In[ ]:


lr= LogisticRegression()
lr.fit(x_train,y_train)


# In[ ]:


y_pred_train=lr.predict(x_train)
y_pred_test=lr.predict(x_test)
y_train_prob = lr.predict_proba(x_train)[:,1]
y_test_proba = lr.predict_proba(x_test)[:,1]


# In[ ]:


from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,confusion_matrix,classification_report


# In[ ]:


print('accuracy score for train :',accuracy_score(y_train,y_pred_train))
print('accuracy score for test :',accuracy_score(y_test,y_pred_test))
print('roc_auc score for train : ',roc_auc_score(y_train,y_train_prob))
print('roc_auc score for test : ',roc_auc_score(y_test,y_test_proba))


# The accuracy and the AUC score for both train and test is good showing no overfitting issues.

# ##### checking multicollinearity using vif
# 

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
cols=list(x.columns)
vif= [variance_inflation_factor(x.values,i) for i in range(len(cols))]
pd.DataFrame(vif,cols)


#  V7 has the highest vif ,hence we can remove it and try modeling

# In[ ]:


x1=x.copy()


# In[ ]:


x1=x1.drop('V7',axis=1)


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(x1,y,test_size=0.3,random_state=True)


# In[ ]:


lr1=LogisticRegression()
lr1.fit(X_train,Y_train)


# In[ ]:


y_pred_train=lr1.predict(X_train)
y_proba_train = lr1.predict_proba(X_train)[:,1]
y_pred_test =lr1.predict(X_test)
y_proba_test = lr1.predict_proba(X_test)[:,1]


# In[ ]:


print('accuracy score for train :',accuracy_score(y_train,y_pred_train))
print('accuracy score for test :',accuracy_score(y_test,y_pred_test))
print('roc_auc score for train : ',roc_auc_score(y_train,y_proba_train))
print('roc_auc score for test : ',roc_auc_score(y_test,y_proba_test))


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
cols=list(x1.columns)
vif= [variance_inflation_factor(x1.values,i) for i in range(len(cols))]
pd.DataFrame(vif,cols)


# As per vif the V17 column exhibit the highest multicollinearity , hence we will drop that and have a check.

# In[ ]:


x1=x1.drop('V17',axis=1)
X_train,X_test,Y_train,Y_test = train_test_split(x1,y,test_size=0.3,random_state=True)
lr1=LogisticRegression()
lr1.fit(X_train,Y_train)


# In[ ]:


y_pred_train=lr1.predict(X_train)
y_proba_train = lr1.predict_proba(X_train)[:,1]
y_pred_test =lr1.predict(X_test)
y_proba_test = lr1.predict_proba(X_test)[:,1]

print('accuracy score for train :',accuracy_score(y_train,y_pred_train))
print('accuracy score for test :',accuracy_score(y_test,y_pred_test))
print('roc_auc score for train : ',roc_auc_score(y_train,y_proba_train))
print('roc_auc score for test : ',roc_auc_score(y_test,y_proba_test))


# Accuracy score has slightly increases while AUC score has slightly decreased after removal of V17.

# In[ ]:


cols=list(x1.columns)
vif= [variance_inflation_factor(x1.values,i) for i in range(len(cols))]
pd.DataFrame(vif,cols)


# The column V12 is exhibiting the highest VIF hence dropping it to check the accuracy of model

# In[ ]:


x1=x1.drop('V12',axis=1)
X_train,X_test,Y_train,Y_test = train_test_split(x1,y,test_size=0.3,random_state=True)
lr1=LogisticRegression()
lr1.fit(X_train,Y_train)


# In[ ]:


y_pred_train=lr1.predict(X_train)
y_proba_train = lr1.predict_proba(X_train)[:,1]
y_pred_test =lr1.predict(X_test)
y_proba_test = lr1.predict_proba(X_test)[:,1]

print('accuracy score for train :',accuracy_score(y_train,y_pred_train))
print('accuracy score for test :',accuracy_score(y_test,y_pred_test))
print('roc_auc score for train : ',roc_auc_score(y_train,y_proba_train))
print('roc_auc score for test : ',roc_auc_score(y_test,y_proba_test))


# The accuracy and the AUC score of the model is decreasing hence VIF is not helping in improving the performance of the model,so lets try Backward elimination to find significant features.

# In[ ]:


cols=list(x1.columns)
vif= [variance_inflation_factor(x1.values,i) for i in range(len(cols))]
pd.DataFrame(vif,cols)


# We still have features exhibiting multicollinearity as per VIF but since removal of feature is decreasing the scores hence we will try other methods of Feature selection.

# #### Backward elimination to check significant features

# In[ ]:


cols = list(x.columns)
pmax = 0
while (len(cols)>1):
   
    X_1 = x[cols]
    X_1 = sm.add_constant(X_1)
    model2 = sm.Logit(y,X_1).fit()
    p = model2.pvalues     
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)


# Two features were eliminated by Backward Elimination Technique. The features eliminated are 'V21' and 'V27'.

# In[ ]:


x2=x[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V22', 'V23', 'V24', 'V25', 'V26', 'V28', 'Time', 'Amount']]
x2_const = sm.add_constant(x2)
model3= sm.Logit(y,x2_const).fit()
model3.summary()


# In[ ]:


y_pred_proba = model2.predict(x2_const)
y_pred=y_pred_proba.apply(pro)


# In[ ]:


print('roc_auc_score : ',roc_auc_score(y,y_pred_proba))
print('accuracy_score:',accuracy_score(y,y_pred))


# Not much improvement from the base model. The accuracy and AUC score are still the same.

# Forward Selection Method for Feature Selection

# In[ ]:


from mlxtend.feature_selection import SequentialFeatureSelector 

lr = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)


# Build step forward feature selection
sfs = SequentialFeatureSelector(lr,k_features = 30,forward=True,
           floating=False, scoring='r2',
           verbose=2,
           cv=5)

sfs = sfs.fit(X_train, y_train)

sfs.k_feature_names_ 


# According to forward selection all the features are important.Hence the accuracy will remain same as the base moel which was built using all the features.

# All the above methods gets almost similar accuracy and Roc-Auc score hence not much parameter tuning or feature elimination is required to improve the performance of the model

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import pandas as pd
creditcard = pd.read_csv("../input/creditcard-fraud-detection/creditcard.csv")

