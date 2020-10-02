#!/usr/bin/env python
# coding: utf-8

# <h1>Pima Indians Diabetes Database</h1>

# ![](https://i.imgur.com/tXvb3kL.jpg)

# <h2>Structure of this notebook</h2>
# 
# 1. Loading and understanding the data
# 2. Exploratory Data Analysis
# 3. Cleaning the data and feature scaling
# 4. Modeling the data
# 5. Fine tuning and interpreting our result
# 
# Since our dataset is based on the sector of healthcare, we will use precision and recall with AUC as our performance measurement

# The objective of our notebook is to predict if the patient has diabetes or not. Our prediction will be based on certain diagnostic measurement provided in our dataset. A key point to note here is that this data is taken from a much larger dataset, and this dataset is based on the patient from Pima Indian heritage with at least 21 years old females. 
# 
# I have tried to explain the concept behind concept that I have used in our notebook, which might be helpful for beginners as well. Please **UPVOTE** if you find this notebook useful! 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style = 'whitegrid')
import matplotlib.pyplot as plt
import plotly.offline as ply
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode()


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score,recall_score


# In[ ]:


df = pd.read_csv("../input/diabetes.csv")


# In[ ]:


df.head(5)


# In[ ]:


df.info()


# In[ ]:


df.describe().T


# In[ ]:


df.isna().sum()


# In[ ]:


df.isnull().sum()


# Our data does not have any null or nan value. We can continue with our Exploratory Data Analysis

# <h2>Exploratory Data Analysis</h2>

# We can also visualization the types of data we have in our dataset and count of each.

# In[ ]:


sns.countplot(x=df.dtypes ,data=df)
plt.ylabel("number of data type")
plt.xlabel("data types")


# Below is an examination of correlation among the independent variables, and between the dependent and the independent variables. The Correlation Coefficient value can range from -1 to 1. If the correlation between two variables is -1, they are both highly negatively correlated, whereas if the correlation between two variables is +1, they both are highly positively correlated. A correlation coefficient of 0 indicated that there is no correlation between the two variables.

# In[ ]:


corr_df = df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr_df, cmap = 'coolwarm', linecolor = 'white', linewidth =1, annot = True)


# All our variables are positively correlated with different degree of correlation. Glucose has the highest correlation with Outcome and BloodPressure has the lowest

# In[ ]:


corr_df["Outcome"].sort_values(ascending = False)


# As you can see below the number of outcome 0 is almost as twice as much as the number of outcome 1. This is a key point in our dataset and later it will be used when we split our dataset. Here outcome 1 that diabetes was found in our patient whereas in the case of outcome 0 diabetes was absent

# In[ ]:


values = pd.Series(df["Outcome"]).value_counts()
trace = go.Pie(values=values)
ply.iplot([trace])


# Body Mass Index is a calculation which takes into two components, height and weight. Patient between the age of 20-30 have the highest number with BMI over 50. High BMI patients tends to have high chances of Outcome 1. 

# In[ ]:


plt.figure(figsize = (20,10))
sns.scatterplot(x = df['Age'], y = df['BMI'], palette="ch:r=-.2,d=.3_r", hue = df["Outcome"])


# Distribution Plot are great visualization through which we can understand the distribution of each variable in our dataset. As we can see from our distribution plots, Glucose, BloodPressure, SkinThickness, and Insulin all have 0. The minimum of value of such can not be zero. This part will be dealt in our data cleaning process.

# In[ ]:


fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.distplot(df["Age"], ax = ax[0,0])
sns.distplot(df["Pregnancies"], ax = ax[0,1])
sns.distplot(df["Glucose"], ax = ax[1,0])
sns.distplot(df["BMI"], ax = ax[1,1])
sns.distplot(df["BloodPressure"], ax = ax[2,0])
sns.distplot(df["SkinThickness"], ax = ax[2,1])
sns.distplot(df["Insulin"], ax = ax[3,0])
sns.distplot(df["DiabetesPedigreeFunction"], ax = ax[3,1])


# Pair Plot can be used to visualize and understand one variable w.r.t. to another. The diagonal is a histogram which shows distribution of a single variable.
# 
# Here hue is taken as Outcome, which clearly distincts the Outcome 0 and Outcome 1 in our visualization.

# In[ ]:


sns.pairplot(df, hue='Outcome')


# In[ ]:





# <h2>Feature Scaling and Data Cleaning</h2>

# As previously stated, there are a number of variables which have 0 values. The steps I have taken to clean up this problem:
# 1. Replaced the 0 with Nan. The reason behind this is that when the mean of the variable is calculated, the data field with 0 value will not be considered. This will give the true mean of the valid data entry.
# 2. In the next step, we replace the value of NaN with our calculated mean.

# In[ ]:


df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)
df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace = True)
df['Insulin'].fillna(df['Insulin'].mean(), inplace = True)
df['BMI'].fillna(df['BMI'].mean(), inplace = True)


# Distribution Plot after taking the mean.

# In[ ]:


fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.distplot(df["Age"], ax = ax[0,0])
sns.distplot(df["Pregnancies"], ax = ax[0,1])
sns.distplot(df["Glucose"], ax = ax[1,0])
sns.distplot(df["BMI"], ax = ax[1,1])
sns.distplot(df["BloodPressure"], ax = ax[2,0])
sns.distplot(df["SkinThickness"], ax = ax[2,1])
sns.distplot(df["Insulin"], ax = ax[3,0])
sns.distplot(df["DiabetesPedigreeFunction"], ax = ax[3,1])


# In[ ]:





# In[ ]:


ss = StandardScaler()
X = ss.fit_transform(df)


# In[ ]:


X =  pd.DataFrame(ss.fit_transform(df.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])


# In[ ]:


X.head(5)


# In[ ]:


y = df['Outcome']


# <h3>Creating train set and test set</h3>

# Remember at the beginning of this notebook I pointed out that our data consist of almost twice as much outcome 0 as compared to outcome 1. This is the right time to speak about **Stratification**. 
# 
# The general procedure is to randomly split our dataset in predefined proportion. In such scenario, there can be a posibility that our train set or test set may be dominated by Outcome 0 which can give incorrect results. Thus we use the stratification method which divides our dataset homogeneous subgroups called strata, which gives us proportionate set of data for both our train and test set.

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)


# <h3>Confusion Matrix</h3>
# 
# Confusion Matrix is represented by rows and columns. Each row in the confusion matrix is called the actual class, and each column in the confusion matrix is called the predicted column. 
# The first row is called the negative class where in the case Outcome 0 is considered. The second row is called the positive class where Outcome 1 is considered. 
# The negative which are correctly classified are called *true negative*, whereas the negative wrongly classified as 1 are called *false positive*.
# The positive wrongly classified as 0 are called *false negative*, whereas the positive correctly classified as 1 are called *true positive*.

# ![](http://i.imgur.com/uducbAo.jpg)

# Precision - Accuracy of positive prediction
# 
# P = TP/(TP+FP) where TP = True Positive and FP = False Positive

# Recall/Sensitivity - Positive correctly classified as Positive.
# 
# Recall = TP/(TP+FN) where TP = True Positive and FP = False Negative

# <h2>Modeling Our Data</h2>

# <h3>Logistic Regression</h3>

# In[ ]:


log_reg = LogisticRegression(random_state=42)


# In[ ]:


log_reg.fit(X_train, y_train)


# In[ ]:


logp = log_reg.predict(X_test)


# In[ ]:


y_train_pred_log = cross_val_predict(log_reg, X_train, y_train, cv=3)


# In[ ]:


confusion_matrix(y_train,y_train_pred_log)


# In[ ]:


print('Precision Score {}'.format(round(precision_score(y_test,logp),3)))
print('Recall Score {}'.format(round(recall_score(y_test,logp),3)))
print("ROC AUC {}".format(round(roc_auc_score(y_test,logp),3)))


# <h3>Gradient Boosting</h3>

# In[ ]:


gbrt = GradientBoostingClassifier(random_state=42)


# In[ ]:


gbrt.fit(X_train, y_train)


# In[ ]:


gbrtp = gbrt.predict(X_test)


# In[ ]:


y_gbrt = cross_val_predict(gbrt, X_train, y_train, cv=3)


# In[ ]:


confusion_matrix(y_train,y_gbrt)


# In[ ]:


print('Precision Score {}'.format(round(precision_score(y_test,gbrtp),3)))
print('Recall Score {}'.format(round(recall_score(y_test,gbrtp),3)))
print("ROC AUC {}".format(round(roc_auc_score(y_test,gbrtp),3)))


# <h3>Random Classifier</h3>

# In[ ]:


forest_clf = RandomForestClassifier(random_state=42)


# In[ ]:


forest_clf.fit(X_train,y_train)


# In[ ]:


ranp = forest_clf.predict(X_test)


# In[ ]:


y_train_pred_ran = cross_val_predict(forest_clf, X_train, y_train, cv=3)


# In[ ]:


confusion_matrix(y_train,y_train_pred_ran)


# In[ ]:


print('Precision Score {}'.format(round(precision_score(y_test,ranp),3)))
print('Recall Score {}'.format(round(recall_score(y_test,ranp),3)))
print("ROC AUC {}".format(round(roc_auc_score(y_test,ranp),3)))


# <h3>Hyperparameter Optimization</h3>

# In[ ]:


param_grid = [{'n_estimators':np.arange(1,50)}]


# In[ ]:


forest_reg = RandomForestClassifier(random_state=42)


# In[ ]:


grid_search = GridSearchCV(forest_reg,param_grid,cv=5)


# In[ ]:


grid_search.fit(X_train,y_train)


# In[ ]:


grid_search.best_estimator_


# In[ ]:


print("Best Score {}".format(str(grid_search.best_score_)))
print("Best Parameters {}".format(str(grid_search.best_params_)))


# In[ ]:


forest_g = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=46, n_jobs=None,
            oob_score=False, random_state=42, verbose=0, warm_start=False)


# In[ ]:


forest_g.fit(X_train,y_train)


# In[ ]:


rang = forest_g.predict(X_test)


# In[ ]:


y_train_pred_rang = cross_val_predict(forest_g, X_train, y_train, cv=3)


# In[ ]:


confusion_matrix(y_train,y_train_pred_rang)


# In[ ]:


print('Precision Score {}'.format(round(precision_score(y_test,rang),3)))
print('Recall Score {}'.format(round(recall_score(y_test,rang),3)))
print("ROC AUC {}".format(round(roc_auc_score(y_test,rang),3)))


# <h3>Voting classifier</h3>

# <h4>Logistic and Random Forest Classification</h4>

# In[ ]:


voting_clf = VotingClassifier(estimators=[('lr', log_reg),('rf', forest_g)], voting='hard')


# In[ ]:


voting_clf.fit(X_train,y_train)


# In[ ]:


votinglr= voting_clf.predict(X_test)


# In[ ]:


y_train_pred_vt = cross_val_predict(voting_clf, X_train, y_train, cv=3)


# In[ ]:


confusion_matrix(y_train,y_train_pred_vt)


# In[ ]:


print('Precision Score {}'.format(round(precision_score(y_test,votinglr),3)))
print('Recall Score {}'.format(round(recall_score(y_test,votinglr),3)))


# <h4>Logistic and Gradient Boosting</h4>

# In[ ]:


voting_gb = VotingClassifier(estimators=[('lr', log_reg),('gb', gbrt)], voting='hard')
voting_gb.fit(X_train,y_train)
votinggb= voting_clf.predict(X_test)
y_train_pred_vt = cross_val_predict(voting_gb, X_train, y_train, cv=3)


# In[ ]:


confusion_matrix(y_train,y_train_pred_vt)


# <h4>Random Forest and Gradient Boosting</h4>

# In[ ]:


voting_rg = VotingClassifier(estimators=[('rf', forest_g),('gb', gbrt)], voting='hard')
voting_rg.fit(X_train,y_train)
votingrg= voting_rg.predict(X_test)
y_train_pred_vt = cross_val_predict(voting_rg, X_train, y_train, cv=3)


# In[ ]:


confusion_matrix(y_train,y_train_pred_vt)


# <h4>Logistic, Gradient Boosting, and Random Forest</h4>

# In[ ]:


voting_lgr = VotingClassifier(estimators=[('lr', log_reg),('gb', gbrt),('rf',forest_g)], voting='hard')
voting_lgr.fit(X_train,y_train)
votinglgr= voting_lgr.predict(X_test)
y_train_pred_vt = cross_val_predict(voting_lgr, X_train, y_train, cv=3)


# In[ ]:


confusion_matrix(y_train,y_train_pred_vt)


# In[ ]:


for clf in (voting_clf,voting_gb,voting_rg,voting_lgr):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(round(roc_auc_score(y_test,y_pred),3))


# The Voting Classifier with highest score is of the combination of Logistic Regression and Gradient Boosting. 
# 
# Please share your **feedback** and do upvote.
