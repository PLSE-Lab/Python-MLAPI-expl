#!/usr/bin/env python
# coding: utf-8

# <h4>READING DATASET</h4>

# In[ ]:


# importing pandas for reading the datasets
import pandas as pd


# In[ ]:


# reading the training dataset with a ';' delimiter
bdata=pd.read_csv('../input/mlworkshop/bank-full.csv',delimiter=';')


# In[ ]:


# displaying first 10 observations from the dataset
bdata.head(10)


# In[ ]:


# describing the pandas dataframe bdata
bdata.describe()


# In[ ]:


# getting details of no of attributes and observations
print('No of observations :',bdata.shape[0])
print('No of attributes :',bdata.shape[1])
print('No of numerical attributes :',bdata.describe().shape[1])
print('No of categorical attributes :',bdata.shape[1]-bdata.describe().shape[1])


# In[ ]:


# getting list of attributes
bdata.columns.tolist()


# <h4>UNDERSTANDING FEATURES OF DATASET</h4>
# 
# __default__: has credit in default?
# 
# __housing__: has housing loan? 
# 
# __loan__: has personal loan?
# 
# __day__: last contact day of the week
# 
# __month__: last contact month of year 
# 
# __duration__: last contact duration, in seconds 
# 
# __campaign__: number of contacts performed during this campaign and for this client 
# 
# __pdays__: number of days that passed by after the client was last contacted from a previous campaign
# 
# __previous__: number of contacts performed before this campaign and for this client
# 
# __poutcome__: outcome of the previous marketing campaign

# In[ ]:


# importing matplotlib for plotting the graphs
import matplotlib.pyplot as plt


# __Outcome variable__

# In[ ]:


bdata['y'].value_counts().plot(kind='bar')
plt.title('Subscriptions')
plt.xlabel('Term Deposit')
plt.ylabel('No of Subscriptions')
plt.show()


# We observe that the data is highly imbalanced, however we need a balanced data only for training.

# Since the data preprocessing steps are same for both testing and training dataset, we first perform the data preprocessing and then divide the data into training data and testing data.

# <h4>VISUALIZATION</h4>

# __Job vs Subscription__

# In[ ]:


pd.crosstab(bdata.job,bdata.y).plot(kind='bar')
plt.title('Subscriptions based on Job')
plt.xlabel('Job')
plt.ylabel('No of Subscriptions')
plt.show()


# __Marital Status vs Subscription__

# In[ ]:


pd.crosstab(bdata.marital,bdata.y).plot(kind='bar')
plt.title('Subscriptions based on Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('No of Subscriptions')
plt.show()


# __Education vs Subscription__

# In[ ]:


pd.crosstab(bdata.education,bdata.y).plot(kind='bar')
plt.title('Subscriptions based on Education')
plt.xlabel('Education')
plt.ylabel('No of Subscriptions')
plt.show()


# __Housing Credit vs Subscription__

# In[ ]:


pd.crosstab(bdata.housing,bdata.y).plot(kind='bar')
plt.title('Subscriptions based on Housing Credit')
plt.xlabel('Housing Credit')
plt.ylabel('No of Subscriptions')
plt.show()


# __Personal loan vs Subscription__

# In[ ]:


pd.crosstab(bdata.loan,bdata.y).plot(kind='bar')
plt.title('Subscriptions based on Personal Loan')
plt.xlabel('Personal Loan')
plt.ylabel('No of Subscriptions')
plt.show()


# __Outcome of Previous Campaign vs Subscription__

# In[ ]:


pd.crosstab(bdata.poutcome,bdata.y).plot(kind='bar')
plt.title('Subscriptions based on Outcome of Previous Campaign')
plt.xlabel('Outcome of Previous Campaign')
plt.ylabel('No of Subscriptions')
plt.show()


# __Month vs Subscription__

# In[ ]:


pd.crosstab(bdata.month,bdata.y).plot(kind='bar')
plt.title('Monthly Subscriptions')
plt.xlabel('Month')
plt.ylabel('No of Subscriptions')
plt.show()


# <h4>DATA PREPROCESSING</h4>

# In[ ]:


# creating dummy variables for categorical variables

# creating a list of categorical variables to be transformed into dummy variables
category=['job','marital','education','default','housing','loan','contact',
          'month','poutcome']

# creating a backup
bdata_new = bdata

# creating dummy variables and joining it to the training set
for c in category:
    new_column = pd.get_dummies(bdata_new[c], prefix=c)
    bdata_dummy=bdata_new.join(new_column)
    bdata_new=bdata_dummy


# In[ ]:


bdata_new.head(10)


# In[ ]:


# see the dummy setup of one categorical variable
bdata_new[[col for col in bdata_new if col.startswith('education')]].head(10)


# In[ ]:


# drop the initial categorical variable
bdata_final=bdata_new.drop(category,axis=1)


# In[ ]:


bdata_final.head(10)


# In[ ]:


# coding no as '0' and yes as '1'
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(bdata_final['y'])
bdata_final['y'] = labels


# In[ ]:


bdata_final.y.value_counts()


# In[ ]:


bdata_final.head(10)


# In[ ]:


# feature selection to reduce dimensionality
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

# creating dataframe of features
X=bdata_final.drop(['y'],axis=1)
# creating dataframe of output variable
y=bdata_final['y']

# standard scaling
X_norm = MinMaxScaler().fit_transform(X)

rfe_selector = RFE(estimator=LogisticRegression(solver='liblinear',max_iter=100,multi_class='ovr',n_jobs=1), n_features_to_select=30, step=10, verbose=5)
rfe_selector.fit(X_norm, y)
rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')


# In[ ]:


rfe_feature


# <p>features to be eliminated : age, pdays (30 selected features)</p>
# <p>features that may be eliminated : job, marital, education, loan (20 selected features)</p>

# In[ ]:


# dropping age and pdays
bdata_final=bdata_final.drop(['age','pdays'],axis=1)


# In[ ]:


bdata_final.head(10)


# In[ ]:


cat=[col for col in bdata_final if col.startswith('job')]
mar_cat=[col for col in bdata_final if col.startswith('marital')]
edu_cat=[col for col in bdata_final if col.startswith('education')]
loan_cat=[col for col in bdata_final if col.startswith('loan')]
cat.extend(mar_cat)
cat.extend(edu_cat)
cat.extend(loan_cat)


# In[ ]:


cat


# In[ ]:


# creating a dataframe with lesser dimension
bdata_dr=bdata_final.drop(cat,axis=1)


# In[ ]:


bdata_dr.head(10)


# <h4>TRAIN TEST SPLIT</h4>

# In[ ]:


# importing sklearn for train test split
from sklearn.model_selection import train_test_split


# In[ ]:


# creating training set of features
X=bdata_final.drop(['y'],axis=1)
# creating training set of output variable
y=pd.DataFrame(bdata_final['y'])


# In[ ]:


# splitting the dataset into train and test for both input and output variable
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)


# In[ ]:


X_train.head(10)


# In[ ]:


y_train.head(10)


# In[ ]:


X_test.head(10)


# In[ ]:


y_test.head(10)


# <h4>STANDARDIZING TRAINING AND TESTING SET</h4>

# In[ ]:


# importing the Standard Scaler from sklearn
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()


# In[ ]:


X_train


# In[ ]:


X_test


# In[ ]:


y_train


# In[ ]:


y_test


# **BALANCING THE DATASET**

# In[ ]:


# importing imblearn for Synthetic Minority Over Sampling Technique
# NOTE : SMOTE technique needs the dataset to be numpy array

# from imblearn.over_sampling import SMOTE
# sm = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=0)
# X_res, y_res = sm.fit_resample(X_train, y_train)
# import numpy as np
# np.savetxt('xres.txt', X_res, fmt='%f')
# np.savetxt('yres.txt', y_res, fmt='%d')

# SMOTE applied dataset
import numpy as np
X_res = np.loadtxt('../input/smotedata/xres.txt', dtype=float)
y_res = np.loadtxt('../input/smotedata/yres.txt', dtype=int)


# In[ ]:


print('No 0f 0 case :',y_res[y_res==0].shape[0])
print('No of 1 case :',y_res[y_res==1].shape[0])


# <h4>FITTING MODEL</h4>

# __Random Forest Classifier__

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
# Create the model with 100 trees
modelrf = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')


# In[ ]:


# Fit on training data
modelrf.fit(X_res, y_res)


# In[ ]:


# predicting the testing set results
y_pred = modelrf.predict(X_test)
y_pred = (y_pred > 0.50)


# In[ ]:


# importing confusion matrix and roc_auc_score from sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# importing seaborn for plotting the heatmap
import seaborn as sn

cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction
df_cm = pd.DataFrame(cm, index = ('no', 'yes'), columns = ('predicted no',
                                                           'predicted yes'))
plt.figure(figsize = (5,4))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
plt.show()
print("Test Data Accuracy: %0.4f" % roc_auc_score(y_test, y_pred))


# In[ ]:


# importing roc curve and metrics from sklearn
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics

fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc=roc_auc_score(y_test, y_pred)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# **Support Vector Classifier**

# In[ ]:


from sklearn.svm import LinearSVC
modelsv = LinearSVC(max_iter=100,random_state=0)


# In[ ]:


modelsv.fit(X_res, y_res)


# In[ ]:


# predicting the testing set results
y_pred = modelsv.predict(X_test)
y_pred = (y_pred > 0.50)


# In[ ]:


# importing confusion matrix and roc_auc_score from sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# importing seaborn for plotting the heatmap
import seaborn as sn

cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction
df_cm = pd.DataFrame(cm, index = ('no', 'yes'), columns = ('predicted no',
                                                           'predicted yes'))
plt.figure(figsize = (5,4))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
plt.show()
print("Test Data Accuracy: %0.4f" % roc_auc_score(y_test, y_pred))


# In[ ]:


# importing roc curve and metrics from sklearn
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics

fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc=roc_auc_score(y_test, y_pred)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# **K-Nearest Neighbour Classifier**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
modelkn = KNeighborsClassifier(n_neighbors=3)


# In[ ]:


modelkn.fit(X_res, y_res)


# In[ ]:


# predicting the testing set results
y_pred = modelkn.predict(X_test)
y_pred = (y_pred > 0.50)


# In[ ]:


# importing confusion matrix and roc_auc_score from sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# importing seaborn for plotting the heatmap
import seaborn as sn

cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction
df_cm = pd.DataFrame(cm, index = ('no', 'yes'), columns = ('predicted no',
                                                           'predicted yes'))
plt.figure(figsize = (5,4))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
plt.show()
print("Test Data Accuracy: %0.4f" % roc_auc_score(y_test, y_pred))


# In[ ]:


# importing roc curve and metrics from sklearn
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics

fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc=roc_auc_score(y_test, y_pred)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
modellr = LogisticRegression()


# In[ ]:


modellr.fit(X_res, y_res)


# In[ ]:


# predicting the testing set results
y_pred = modellr.predict(X_test)
y_pred = (y_pred > 0.50)


# In[ ]:


# importing confusion matrix and roc_auc_score from sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# importing seaborn for plotting the heatmap
import seaborn as sn

cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction
df_cm = pd.DataFrame(cm, index = ('no', 'yes'), columns = ('predicted no',
                                                           'predicted yes'))
plt.figure(figsize = (5,4))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
plt.show()
print("Test Data Accuracy: %0.4f" % roc_auc_score(y_test, y_pred))


# In[ ]:


# importing roc curve and metrics from sklearn
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics

fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc=roc_auc_score(y_test, y_pred)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# **Naive Bayes Classifier**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
modelnb = GaussianNB()


# In[ ]:


modelnb.fit(X_res, y_res)


# In[ ]:


# predicting the testing set results
y_pred = modelnb.predict(X_test)
y_pred = (y_pred > 0.50)


# In[ ]:


# importing confusion matrix and roc_auc_score from sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# importing seaborn for plotting the heatmap
import seaborn as sn

cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction
df_cm = pd.DataFrame(cm, index = ('no', 'yes'), columns = ('predicted no',
                                                           'predicted yes'))
plt.figure(figsize = (5,4))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
plt.show()
print("Test Data Accuracy: %0.4f" % roc_auc_score(y_test, y_pred))


# In[ ]:


# importing roc curve and metrics from sklearn
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics

fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc=roc_auc_score(y_test, y_pred)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




