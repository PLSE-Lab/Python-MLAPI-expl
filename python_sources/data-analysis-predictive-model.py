#!/usr/bin/env python
# coding: utf-8

# # Data Analysis & Predictive Model Building
# ***
# #### By Omar BOUGACHA
# 
# 

# ## Introduction
# ***

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
base_color = sns.color_palette()[0]


# ## Data Wrangling

# ### Data Gathering

# In[ ]:


data_df = pd.read_csv('/kaggle/input/patient-survey-score-with-demographics/pxdata.csv')
data_df.head(2)


# Lets go for the data quality and tidiness assessment.

# ### Data Quality & Tidiness Assessment

# #### Data Completness (Missing values)

# In[ ]:


data_df.isnull().sum()


# As we can see, all the records are given in this dataset. We do not have any missing values.

# #### Data Accuracy 
# 
# Accurate data should present the right values in the right format. Since we cannot cross-reference the given values to other sources of data, we only check the right format of the data. 

# In[ ]:


data_df.dtypes


# As we can see we have some data accuracy issues: 
# * The Date feature should be of type datetime not string. 
# * Perfect, College, White, and English should be of type boolean. 
# * Age and Stay should be categorical

# #### Data Consistency: 
# 
# We check for each feature if the data points present different values for the same meaning.

# In[ ]:


data_df.nunique()


# In[ ]:


data_df['Perfect'].value_counts()


# In[ ]:


data_df['Rate'].value_counts()


# In[ ]:


data_df['Recommend'].value_counts()


# In[ ]:


data_df['Health'].value_counts()


# In[ ]:


data_df['Mental'].value_counts()


# In[ ]:


data_df['College'].value_counts()


# In[ ]:


data_df['White'].value_counts()


# In[ ]:


data_df['English'].value_counts()


# In[ ]:


data_df['Service'].value_counts()


# In[ ]:


data_df['Specialty'].value_counts()


# In[ ]:


data_df['Unit'].value_counts()


# In[ ]:


data_df['Source'].value_counts()


# In[ ]:


data_df['Home'].value_counts()


# In[ ]:


data_df['Age'].unique()


# We can see in the age values we have two categories that mean the same thing 80+ and 80-90. I believe the 80+ is just a mistyping error. Therefore, it should be fixed to 90+

# In[ ]:


data_df['Stay'].unique()


# The stay categories should be defined as: '1', '2-3', '4-7', '8+'

# In[ ]:


data_df['Visit'].unique()


# ### Data Cleaning

# In[ ]:


cleaned_data = data_df.copy()


# * The Date feature should be of type datetime not string. 
# 
# ##### Define: 
# * convert the type from string to datetime
# 
# ##### Code:

# In[ ]:


cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'])


# ##### Test:

# In[ ]:


cleaned_data.dtypes


# * Perfect, College, White, and English should be of type boolean. 
# 
# ##### Define: 
# * convert the type from string to boolean
# 
# ##### Code:

# In[ ]:


cleaned_data['Perfect'] = cleaned_data['Perfect'].apply(lambda x : True if x==1 else False)
cleaned_data['College'] = cleaned_data['College'].apply(lambda x : True if x=='Y' else False)
cleaned_data['White'] = cleaned_data['White'].apply(lambda x : True if x=='Y' else False)
cleaned_data['English'] = cleaned_data['English'].apply(lambda x : True if x=='Y' else False)


# ##### Test

# In[ ]:


cleaned_data.dtypes


# * the Stay categories: 
# 
# ##### define: 
# * Change the stay categories to: '1', '2-3', '4-7', '8+'
# 
# ##### Code:

# In[ ]:


cleaned_data['Stay'] = cleaned_data['Stay'].replace({'2+':'2-3', '4+': '4-7'})


# ##### Test:

# In[ ]:


cleaned_data['Stay'].unique()


# * 80+ and 80-90 category. 
# 
# ##### Define: 
# * Change the 80+ category to 90+
# 
# ##### Code: 
# 

# In[ ]:


cleaned_data['Age'] = cleaned_data["Age"].apply(lambda x: '90+' if x=='80+' else x)


# ##### Test

# In[ ]:


cleaned_data['Age'].unique()


# * Age should be categorical
# 
# ##### Define:
# * Change Age to categorical type
# 
# ##### Code:

# In[ ]:


from pandas.api.types import CategoricalDtype
age_cat = CategoricalDtype(['18-34', '35-49', '50-64', '65-79', '80-90', '90+'], ordered=True)


# In[ ]:


cleaned_data['Age'] = cleaned_data['Age'].astype(age_cat)


# ##### Test

# In[ ]:


cleaned_data['Age'].dtype


# * Stay should be categorical
# 
# ##### Define:
# * Change Stay type from string to categorical:
# 
# ##### Code:

# In[ ]:


stay_cat = CategoricalDtype(['1', '2-3', '4-7', '8+'], ordered=True)
cleaned_data['Stay'] = cleaned_data['Stay'].astype(stay_cat)


# ##### Test:

# In[ ]:


cleaned_data.dtypes


# ## Exploratory Data Analysis

# In the EDA process, we continue working using the cleaned_data table to analyze the different relationships between the variables. The EDA process has 3 main components: 
# * Univariate Analysis
# * Bivariate Analysis 
# * Multivariate Analysis
# 
# However, in this analysis, we focuse only on univariate and bivariate analysis.

# ### Univariate Analysis

# In[ ]:


cleaned_data['Composite'].hist(bins=100)
plt.xlabel('Composite')
plt.ylabel('Count')
plt.show()


# We can see that the composite feature is highly skewed to the left.

# In[ ]:


fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
sns.countplot(x='Perfect', data=cleaned_data, ax=ax1, color=base_color)
sns.countplot(x='College', data=cleaned_data, ax=ax2, color=base_color)
sns.countplot(x='White', data=cleaned_data, ax=ax3, color=base_color)
sns.countplot(x='English', data=cleaned_data, ax=ax4, color=base_color)
plt.tight_layout()
plt.show()


# In[ ]:


fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
sns.countplot(x='Rate', data=cleaned_data, ax=ax1, color=base_color)
sns.countplot(x='Recommend', data=cleaned_data, ax=ax2, color=base_color)
plt.tight_layout()
plt.show()


# In[ ]:


fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
sns.countplot(x='Health', data=cleaned_data, ax=ax1, color=base_color)
sns.countplot(x='Mental', data=cleaned_data, ax=ax2, color=base_color)
plt.tight_layout()
plt.show()


# In[ ]:


fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
sns.countplot(x='Age', data=cleaned_data, ax=ax1, color=base_color)
sns.countplot(x='Sex', data=cleaned_data, ax=ax2, color=base_color)
sns.countplot(x='Home', data=cleaned_data, ax=ax3, color=base_color)
plt.tight_layout()
plt.show()


# In[ ]:


fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
sns.countplot(x='Source', data=cleaned_data, ax=ax1, color=base_color)
sns.countplot(x='Service', data=cleaned_data, ax=ax2, color=base_color)
plt.tight_layout()
plt.show()


# In[ ]:


fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
sns.countplot(x='Unit', data=cleaned_data, ax=ax1, color=base_color)
sns.countplot(x='Specialty', data=cleaned_data, ax=ax2, color=base_color)
plt.tight_layout()
plt.show()


# In[ ]:


fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
sns.countplot(x='Stay', data=cleaned_data, ax=ax1, color=base_color)
sns.countplot(x='Visit', data=cleaned_data, ax=ax2, color=base_color)
plt.tight_layout()
plt.show()


# In[ ]:


cleaned_data['Date'].hist(bins=100)
plt.xlabel('Date')
plt.ylabel('Count')
plt.show()


# This feature of the date allows us to derive other features and to study them. We can most likely derive the month feature and the day of week. 

# #### Feature Engineering:
# 
# ##### Define: 
# * From Date extract the Month Name
# * From Date extract the Day-of-Week
# 
# ##### Code: 

# In[ ]:


cleaned_data['Month'] = cleaned_data['Date'].apply(lambda x : x.strftime("%B"))


# In[ ]:


cleaned_data['Day_of_Week'] = cleaned_data['Date'].apply(lambda x: x.strftime("%A"))


# Lets change them to categorical type:

# In[ ]:


m_cat = CategoricalDtype(['January', 'February', 'March', 'April', 'May', 'June', 'July', 
                          'August', 'September', 'October', 'November', 'December'], ordered=True)
d_w_cat = CategoricalDtype(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 
                            'Saturday', 'Sunday'], ordered=True)


# In[ ]:


cleaned_data['Month'] = cleaned_data['Month'].astype(m_cat)
cleaned_data['Day_of_Week'] = cleaned_data['Day_of_Week'].astype(d_w_cat)


# ##### Test: 

# In[ ]:


cleaned_data['Month'].unique()


# In[ ]:


cleaned_data['Day_of_Week'].unique()


# Lets now plot their bar plot.

# In[ ]:


fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ch1 = sns.countplot(x='Month', data=cleaned_data, ax=ax1, color=base_color)
ch1.set_xticklabels(labels = ch1.get_xticklabels(), rotation=45)
ch2 = sns.countplot(x='Day_of_Week', data=cleaned_data, ax=ax2, color=base_color)
ch2.set_xticklabels(labels = ch2.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.show()


# ### Bivariate Analysis

# In[ ]:


d_m = cleaned_data[cleaned_data['Sex']=='M']
sns.distplot(d_m['Composite'],kde=False, label='Male')
d_f = cleaned_data[cleaned_data['Sex']=='F']
sns.distplot(d_f['Composite'],kde=False, label='Female')
plt.legend(prop={'size': 12})
plt.title('Distrubtion of Composite per Gender')
plt.xlabel('Composite')
plt.ylabel('Count')
plt.show()


# In[ ]:


fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
sns.countplot(x='Perfect', data=cleaned_data, ax=ax1, hue='Sex')
sns.countplot(x='College', data=cleaned_data, ax=ax2, hue='Sex')
sns.countplot(x='White', data=cleaned_data, ax=ax3, hue='Sex')
sns.countplot(x='English', data=cleaned_data, ax=ax4, hue='Sex')
plt.tight_layout()
plt.show()


# In[ ]:


fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
sns.countplot(x='Rate', data=cleaned_data, ax=ax1, hue='Sex')
sns.countplot(x='Recommend', data=cleaned_data, ax=ax2, hue='Sex')
plt.tight_layout()
plt.show()


# In[ ]:


d_m = cleaned_data[cleaned_data['Visit']==0]
sns.distplot(d_m['Composite'],kde=False, label='No Visit')
d_f = cleaned_data[cleaned_data['Visit']==1]
sns.distplot(d_f['Composite'],kde=False, label='Visit')
plt.legend(prop={'size': 12})
plt.title('Distrubtion of Composite per Visit')
plt.xlabel('Composite')
plt.ylabel('Count')
plt.show()


# In[ ]:


fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
sns.countplot(x='Perfect', data=cleaned_data, ax=ax1, hue='Visit')
sns.countplot(x='College', data=cleaned_data, ax=ax2, hue='Visit')
sns.countplot(x='White', data=cleaned_data, ax=ax3, hue='Visit')
sns.countplot(x='English', data=cleaned_data, ax=ax4, hue='Visit')
plt.tight_layout()
plt.show()


# In[ ]:


fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
sns.countplot(x='Rate', data=cleaned_data, ax=ax1, hue='Visit')
sns.countplot(x='Recommend', data=cleaned_data, ax=ax2, hue='Visit')
plt.tight_layout()
plt.show()


# In[ ]:


fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
sns.countplot(x='Age', data=cleaned_data, ax=ax1, hue='Visit')
sns.countplot(x='Sex', data=cleaned_data, ax=ax2, hue='Visit')
sns.countplot(x='Home', data=cleaned_data, ax=ax3, hue='Visit')
plt.tight_layout()
plt.show()


# In[ ]:


sns.catplot(x='Month', data=cleaned_data, kind='count', hue='Visit', aspect=2)
plt.show()


# In[ ]:


sns.catplot(x='Day_of_Week', kind='count', data=cleaned_data, hue='Visit', aspect=1.2)
plt.show()


# In[ ]:


sns.catplot(x='Day_of_Week', data=cleaned_data, y='Visit', kind='bar', aspect=1.2, ci=None, color=base_color)
plt.ylabel('Probability of Visit')
plt.show()


# In[ ]:


sns.catplot(x='Age', data=cleaned_data, y='Visit', kind='bar', aspect=1.2, ci=None, color=base_color)
plt.ylabel('Probability of Visit')
plt.show()


# In[ ]:


sns.catplot(x='Sex', data=cleaned_data, y='Visit', kind='bar', aspect=0.8, ci=None, color=base_color)
plt.ylabel('Probability of Visit')
plt.show()


# We can see that the gender has an influence on the Visit status. Lets verify if this observation is statistically significant or not. 
# 
# The null hypothesis in this case is that the probablity (ratio) of visit if the gender is Female is the same as the Male: 
# $$H_0: p_{F} = p_{M}$$ 
# 
# The $H_1$ is that the gender has an influence. 
# $$H_1: p_{F} \neq p_{M}$$

# In[ ]:


import statsmodels.api as sm
female = cleaned_data[cleaned_data['Sex']=='F']
male = cleaned_data[cleaned_data['Sex']=='M']
counts = np.array([female['Visit'].sum(), 
                   male['Visit'].sum()])
nobs = np.array([female.shape[0], male.shape[0]])
zstat, pval = sm.stats.proportions_ztest(counts, nobs, alternative='smaller')
zstat, pval


# We can see that the p-value is so small (lower than 0.05). Hence, we succeeded in rejecting the null hypothesis. Therefore, **the gender has an influence of the visit outcome**.

# In[ ]:


sns.catplot(x='Mental', data=cleaned_data, y='Visit', kind='bar', aspect=1.2, ci=None, color=base_color)
plt.ylabel('Probability of Visit')
plt.show()


# In[ ]:


sns.catplot(x='College', data=cleaned_data, y='Visit', kind='bar', aspect=0.8, ci=None, color=base_color)
plt.ylabel('Probability of Visit')
plt.show()


# Lets do the statistical test for this feature also. 
# 
# The null hypothesis in this case is that the probablity (ratio) of visit if the college is True is the same as the False: 
# $$H_0: p_{F} = p_{T}$$ 
# 
# The $H_1$ is that the college has an influence. 
# $$H_1: p_{F} \neq p_{T}$$

# In[ ]:


collage = cleaned_data[cleaned_data['College']]
no_collage = cleaned_data[cleaned_data['College']==False]
counts = np.array([collage['Visit'].sum(), 
                   no_collage['Visit'].sum()])
nobs = np.array([collage.shape[0], no_collage.shape[0]])
zstat, pval = sm.stats.proportions_ztest(counts, nobs, alternative='smaller')
zstat, pval


# The obtained p-value is lower than 0.05. Then, we can safely reject the null hypothesis. Therefore, **the college status has an influence over the visit outcome**.

# In[ ]:


sns.catplot(x='White', data=cleaned_data, y='Visit', kind='bar', aspect=0.8, ci=None, color=base_color)
plt.ylabel('Probability of Visit')
plt.show()


# Lets do the statistical test for this feature also. 
# 
# The null hypothesis in this case is that the probablity (ratio) of visit if the white is True is the same as the False: 
# $$H_0: p_{F} = p_{T}$$ 
# 
# The $H_1$ is that the white has an influence. 
# $$H_1: p_{F} \neq p_{T}$$

# In[ ]:


co = cleaned_data[cleaned_data['White']]
no = cleaned_data[cleaned_data['White']==False]
counts = np.array([co['Visit'].sum(), 
                   no['Visit'].sum()])
nobs = np.array([co.shape[0], no.shape[0]])
zstat, pval = sm.stats.proportions_ztest(counts, nobs, alternative='smaller')
zstat, pval


# We can see that the p-value for this feature is equal to 0.82. Then, $p-value > 0.05$. So we fail to reject the null hypothesis. Hence, **the white or no white does not influence the visit outcome**.

# In[ ]:


cleaned_data.corr()


# In[ ]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
sns.heatmap(cleaned_data.corr(), ax=ax);


# We can see from the correlation heatmap and the matrix that we have two highly correlated couple of features: 
# * the Rate and Recommend
# * the White and English

# ## Modeling:
# *********

# In this project, we suppose that we want to build a model that is able to predict the visit outcome. Hence, we have a binary classification problem. The features, in this work, come in different types (categorical, integer, boolean, and float). Therefore, we should start by preparing the data for the model building. 

# ### Data Preparation:

# In[ ]:


y = cleaned_data['Visit']
X = cleaned_data[[x for x in cleaned_data.columns if x not in ['Survey', 'Visit', 'Date']]]


# * Transforming the boolean columns into binary (0,1)

# In[ ]:


X['White'] = X['White'].astype('int32')
X['College'] = X['College'].astype('int32')
X['English'] = X['English'].astype('int32')
X['Perfect'] = X['Perfect'].astype('int32')


# * Transforming Sex to binary (0 for F, 1 for M)

# In[ ]:


X['Sex'] = X['Sex'].replace({'F':0, 'M':1})


# * Transfroming Source to binary (0 for T and 1 for D)

# In[ ]:


X['Source'] = X['Source'].replace({'D':1,'T':0})


# * Get dummies for features: Service, Home, Specialty, and Unit. 
# * Define baseline for each feature. 
#     * Feature Service baseline is: O
#     * Feature Home baseline is: Y
#     * Feature Specialty baseline is: 1
#     * Feature Unit baseline is: 3

# In[ ]:


X = pd.get_dummies(X, columns=['Service', 'Home', 'Specialty', 'Unit'])


# In[ ]:


X.columns


# In[ ]:


X.drop(['Service_O', 'Home_Y', 'Specialty_1', 'Unit_3'], axis=1, inplace=True)


# * Convert Month, Day_of_Week, Age, and Stay caterogies to numeric

# In[ ]:


X['Month'] = X['Month'].replace({'January':0, 'February':1, 'March':2, 'April':3, 'May':4, 'June':5, 'July':6, 
                                 'August':7, 'September':8, 'October':9, 'November':10, 'December':11})
X['Day_of_Week'] = X['Day_of_Week'].replace({'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 
                                             'Saturday':5, 'Sunday':6})
X['Age'] = X['Age'].replace({'18-34':0, '35-49':1, '50-64':2, '65-79':3, '80-90':4, '90+':5})
X['Stay'] = X['Stay'].replace({'1':0, '2-3':1, '4-7':2, '8+':3})


# In[ ]:


X.dtypes


# ### Methodology:
# 
# To assess the performance of the models, we divide the dataset into two parts train and test. The train part is used to fit the models we are going to build and then we use the model to predict the outcome of the test part. Then, the built models should be compared to select the best model. In this application, we use the accuracy, precision, and f1-score to compare the models. We, also, plot the ROC and compute the area under curve of each model. 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.7)


# In[ ]:


Performances=[]


# ### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)


# In[ ]:


model={'Model': 'DecisionTree'}
model['Accuracy'] = accuracy_score(y_test, dtc.predict(X_test))
model['Precision'] = precision_score(y_test, dtc.predict(X_test))
model['F1-Score'] = f1_score(y_test, dtc.predict(X_test))
Performances.append(model)


# In[ ]:


confusion_matrix(y_test, dtc.predict(X_test))


# In[ ]:


print(classification_report(y_test, dtc.predict(X_test)))


# In[ ]:


dtc_features = pd.DataFrame()
dtc_features['Feature'] = X_train.columns.tolist()
dtc_features['Importance'] = dtc.feature_importances_
sns.catplot(y='Feature', x='Importance', data=dtc_features, kind='bar', height=15, color=base_color)
plt.show()


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=100, random_state=0)
rfc.fit(X_train, y_train)


# In[ ]:


model={'Model': 'RandomForest'}
model['Accuracy'] = accuracy_score(y_test, rfc.predict(X_test))
model['Precision'] = precision_score(y_test, rfc.predict(X_test))
model['F1-Score'] = f1_score(y_test, rfc.predict(X_test))
Performances.append(model)


# In[ ]:


confusion_matrix(y_test, rfc.predict(X_test))


# In[ ]:


print(classification_report(y_test, rfc.predict(X_test)))


# In[ ]:


rfc_features = pd.DataFrame()
rfc_features['Feature'] = X_train.columns.tolist()
rfc_features['Importance'] = rfc.feature_importances_
sns.catplot(y='Feature', x='Importance', data=rfc_features, kind='bar', height=15, color=base_color)
plt.show()


# ### XGBoost

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


xgb = XGBClassifier(n_estimators=100,random_state=0)
xgb.fit(X_train, y_train)


# In[ ]:


model={'Model': 'XGBoost'}
model['Accuracy'] = accuracy_score(y_test, xgb.predict(X_test))
model['Precision'] = precision_score(y_test, xgb.predict(X_test))
model['F1-Score'] = f1_score(y_test, xgb.predict(X_test))
Performances.append(model)


# In[ ]:


confusion_matrix(y_test, xgb.predict(X_test))


# In[ ]:


print(classification_report(y_test, xgb.predict(X_test)))


# In[ ]:


xgb_features = pd.DataFrame()
xgb_features['Feature'] = X_train.columns.tolist()
xgb_features['Importance'] = xgb.feature_importances_
sns.catplot(y='Feature', x='Importance', data=xgb_features, kind='bar', height=15, color=base_color)
plt.show()


# ### CatBoost

# In[ ]:


from catboost import CatBoostClassifier


# In[ ]:


cb = CatBoostClassifier(n_estimators=2500, random_state=0)
cb.fit(X_train, y_train)


# In[ ]:


model={'Model': 'CatBoost'}
model['Accuracy'] = accuracy_score(y_test, cb.predict(X_test))
model['Precision'] = precision_score(y_test, cb.predict(X_test))
model['F1-Score'] = f1_score(y_test, cb.predict(X_test))
Performances.append(model)


# In[ ]:


confusion_matrix(y_test, cb.predict(X_test))


# In[ ]:


print(classification_report(y_test, cb.predict(X_test)))


# In[ ]:


cb_features = pd.DataFrame()
cb_features['Feature'] = X_train.columns.tolist()
cb_features['Importance'] = cb.feature_importances_
sns.catplot(y='Feature', x='Importance', data=cb_features, kind='bar', height=15, color=base_color)
plt.show()


# ### Models Comparaison:

# In[ ]:


pd.DataFrame(Performances)


# From the obtained table we can see that the CatBoost is the best method in terms of accuracy and precision. However, in terms of F1-Score of the class Visit=1 it has the second best value after the Decision Tree. If we compare the macro average of the F1-Score on the two classes we can see that the CatBoost (0.64) is better that the Decision Tree (0.61). This information is obtained from the classification reports. 
# 
# Lets see if the same results are obtained from the ROC Curves.

# ### ROC Curves: 

# In[ ]:


fpr_RFC, tpr_RFC, _ = roc_curve(y_test, rfc.predict_proba(X_test)[:,1])
roc_auc_RFC = auc(fpr_RFC, tpr_RFC)
fpr_dtc, tpr_dtc, _ = roc_curve(y_test, dtc.predict_proba(X_test)[:,1])
roc_auc_dtc = auc(fpr_dtc, tpr_dtc)
fpr_XGB, tpr_XGB, _ = roc_curve(y_test, xgb.predict_proba(X_test)[:,1])
roc_auc_XGB = auc(fpr_XGB, tpr_XGB)
fpr_CAT, tpr_CAT, _ = roc_curve(y_test, cb.predict_proba(X_test)[:,1])
roc_auc_CAT = auc(fpr_CAT, tpr_CAT)


# In[ ]:


plt.figure(figsize=(8,8))
lw = 2
plt.plot(fpr_dtc, tpr_dtc, 
         lw=lw, label='ROC curve Decision Tree (area = %0.2f)' % roc_auc_dtc)
plt.plot(fpr_RFC, tpr_RFC, 
         lw=lw, label='ROC curve Random Forest (area = %0.2f)' % roc_auc_RFC)
plt.plot(fpr_XGB, tpr_XGB, 
         lw=lw, label='ROC curve XGBoost (area = %0.2f)' % roc_auc_XGB)
plt.plot(fpr_CAT, tpr_CAT, 
         lw=lw, label='ROC curve CATBoost (area = %0.2f)' % roc_auc_CAT)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Using All Features')
plt.legend(loc="lower right")
plt.show()


# Using the ROC curve to select the best model is quite easy. We search for the model that goes rapidly to the top and the one with the higher curve and the highest area under curve. According to these criteria, we can conclude that the CatBoost model is the best.

# From the table of models' performances and the ROC curves, we can conclude that the **CatBoost** model is the most suitable model for this application.
