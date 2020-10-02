#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 999)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train_data = pd.read_csv("/kaggle/input/bank-marketing-analysis/bank-additional-full.csv",sep=';')


# In[ ]:


train_data.head()


# In[ ]:


train_data.info()


# In[ ]:


train_data.describe()


# In[ ]:


train_data.isnull().sum()


# In[ ]:


train_data.duplicated().sum()


# ### There are some duplicate values in the dataset

# In[ ]:


rcParams['figure.figsize'] = 17,10
sns.countplot(x=train_data['job'])


# In[ ]:


rcParams['figure.figsize'] = 17,10
sns.countplot(x=train_data['job'],hue=train_data['y'],palette="Set2")


# ### This infers that admin and technician are mostly taking the bank deposit as they are more people in number

# In[ ]:


sns.countplot(x=train_data['education'])


# In[ ]:


rcParams['figure.figsize'] = 15,10
sns.countplot(x=train_data['education'],hue=train_data['y'],palette="Set2")


# ### persons who have university degree and high school are getting the bank deposit

# In[ ]:


rcParams['figure.figsize'] = 15,10
sns.countplot(x=train_data['marital'],hue=train_data['y'],palette="Set2")


# ### Married and single people are accepting the bank deposit 

# In[ ]:


rcParams['figure.figsize'] = 15,10
sns.countplot(x=train_data['housing'],hue=train_data['y'],palette="Set2")


# ### It shows those who have hosuing loan are more tend to accept the bank deposit

# In[ ]:


train_data['loan'].value_counts().plot(kind="bar")


# In[ ]:


rcParams['figure.figsize'] = 15,10
sns.countplot(x=train_data['loan'],hue=train_data['y'],palette="Set2")


# ### Personal Loan 
# #### The person who has no personal loan will subscribe the bank deposit and who has already a personal loan does not subscipe to the bank deposit

# In[ ]:


rcParams['figure.figsize'] = 15,10
sns.countplot(x=train_data['contact'],hue=train_data['y'],palette="Set2")


# In[ ]:


rcParams['figure.figsize'] = 15,10
sns.countplot(x=train_data['nr.employed'],hue=train_data['y'],palette="Set2")


# In[ ]:


rcParams['figure.figsize'] = 15,10
sns.countplot(x=train_data['poutcome'],hue=train_data['y'],palette="Set2")


# ### nonexistant people are more exposed for the subscripton of the bank  deposit

# In[ ]:


train_data['month'].value_counts().plot(kind="pie")


# In[ ]:


rcParams['figure.figsize'] = 15,10
sns.countplot(x=train_data['month'],hue=train_data['y'],palette="Set2")


# In[ ]:


train_data["loan"].value_counts()


# In[ ]:


train_data["education"].value_counts().plot(kind="bar")


# In[ ]:


train_data["housing"].value_counts().plot(kind="pie")


# In[ ]:


train_data["contact"].value_counts().plot(kind="bar")


# In[ ]:


train_data['marital'].value_counts().plot(kind="pie")


# In[ ]:


train_data['campaign'].value_counts()


# In[ ]:


train_data['campaign'].value_counts().plot(kind="bar")


# In[ ]:


rcParams['figure.figsize'] = 15,10
sns.countplot(train_data['campaign'],hue=train_data['y'],palette="Set2")


# In[ ]:


rcParams['figure.figsize'] = 15,10
sns.countplot(train_data['pdays'],hue=train_data['y'],palette="Set2")


# In[ ]:


train_data['cons.price.idx'].value_counts().plot(kind="bar")


# ### This infers most of the clients are the not contacted to bank ,it may be due to various reasons

# In[ ]:


train_data.head()


# In[ ]:


new_df = train_data.copy(deep=True)


# In[ ]:


le = preprocessing.LabelEncoder()

# job
le.fit(new_df['job'])
new_df['job'] = le.transform(new_df['job'])

# maritial feature
le.fit(new_df['marital'])
new_df['marital'] = le.transform(new_df['marital'])

# education_feature
le.fit(new_df['education'])
new_df['education'] = le.transform(new_df['education'])

# housing_feature
le.fit(new_df['housing'])
new_df['housing'] = le.transform(new_df['housing'])

# loan_feature
le.fit(new_df['loan'])
new_df['loan'] = le.transform(new_df['loan'])

# contact_feature
le.fit(new_df['contact'])
new_df['contact'] = le.transform(new_df['contact'])

# Month_feature
le.fit(new_df['month'])
new_df['month'] = le.transform(new_df['month'])

# day of week_feature
le.fit(new_df['day_of_week'])
new_df['day_of_week'] = le.transform(new_df['day_of_week'])

# poutcome_feature
le.fit(new_df['poutcome'])
new_df['poutcome'] = le.transform(new_df['poutcome'])

# default_feature
le.fit(new_df['default'])
new_df['default'] = le.transform(new_df['default'])



# Target_feature
le.fit(new_df['y'])
new_df['y'] = le.transform(new_df['y'])


# In[ ]:


correleation_matrix = new_df.corr()


# In[ ]:


rcParams['figure.figsize'] = 25,20
sns.heatmap(correleation_matrix, cbar=True, square= True,fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')


# ### From this co relation matrix we can find emp.var.rate,cons.proce.idx,euribor3m and nr.employed are more correlated to target columns

# In[ ]:


y = new_df['y']
x = new_df.drop(['y'],axis=1)


# In[ ]:


X_train,X_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.10, random_state=42)


# ## Building Models

# In[ ]:


LR = linear_model.LogisticRegression()


# In[ ]:


LR.fit(X_train,y_train)


# In[ ]:


y_pred = LR.predict(X_test)


# In[ ]:


from sklearn import metrics
f1 = metrics.f1_score(y_true=y_test,y_pred=y_pred)
acc = metrics.accuracy_score(y_true=y_test,y_pred=y_pred)
pres = metrics.precision_score(y_true=y_test,y_pred=y_pred)
recall = metrics.recall_score(y_true=y_test,y_pred=y_pred)


# In[ ]:


print("The accuracy of the model Logistic Regression Model",acc)
print("The F1 Score of the model Logistic Regression Model",f1)
print("The Precision of the model Logistic Regression Model",pres)
print("The recall of the model Logistic Regression Model",recall)


# In[ ]:


from sklearn import ensemble
RFC = ensemble.RandomForestClassifier()


# In[ ]:


RFC.fit(X_train,y_train)


# In[ ]:


y_pred_rfc = RFC.predict(X_test)


# In[ ]:


f1_rfc = metrics.f1_score(y_true=y_test,y_pred=y_pred_rfc)
acc_rfc = metrics.accuracy_score(y_true=y_test,y_pred=y_pred_rfc)
pres_rfc = metrics.precision_score(y_true=y_test,y_pred=y_pred_rfc)
recall_rfc = metrics.recall_score(y_true=y_test,y_pred=y_pred_rfc)
cfn_matrix = metrics.plot_confusion_matrix(RFC,X_test,y_test)


# In[ ]:


print("The accuracy of the model RandomForestClassifier Model",acc_rfc)
print("The F1 Score of the model RandomForestClassifier Model",f1_rfc)
print("The Precision of the model RandomForestClassifier Model",pres_rfc)
print("The recall of the model RandomForestClassifier",recall_rfc)


# In[ ]:


ETC = ensemble.ExtraTreesClassifier()


# In[ ]:


ETC.fit(X_train,y_train)


# In[ ]:


y_pred_ETC = ETC.predict(X_test)


# In[ ]:


f1_ETC = metrics.f1_score(y_true=y_test,y_pred=y_pred_ETC)
acc_ETC = metrics.accuracy_score(y_true=y_test,y_pred=y_pred_ETC)
pres_ETC = metrics.precision_score(y_true=y_test,y_pred=y_pred_ETC)
recall_ETC = metrics.recall_score(y_true=y_test,y_pred=y_pred_ETC)


# In[ ]:


print("The accuracy of the modelExtraTreesClassifier Model",acc_ETC)
print("The F1 Score of the model ExtraTreesClassifier Model",f1_ETC)
print("The Precision of the modelExtraTreesClassifier Model",pres_ETC)
print("The recall of the model ExtraTreesClassifier Model",recall_ETC)


# In[ ]:




