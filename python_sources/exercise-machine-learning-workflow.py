#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import random


# <h2>Understanding the Dataset

# In[ ]:


df = pd.read_csv('../input/data.csv')
df.head(3)


# The dataset contains the following features:
# 1. age: (in years)
# 2. sex: (male, female)
# 3. cp: chest pain type
# 4. trestbps: resting blood pressure (in mm Hg on admission to the hospital)
# 5. chol: serum cholestoral in mg/dl
# 6. fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 7. restecg: resting electrocardiographic results
# 8. thalach: maximum heart rate achieved
# 9. exang: exercise induced angina (1 = yes; 0 = no)
# 10. oldpeak: ST depression induced by exercise relative to rest
# 11. slope: the slope of the peak exercise ST segment
# 12. ca: number of major vessels (0-3) colored by flourosopy
# 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# 14. target: 1 or 0

# In[ ]:


df.dtypes


# In[ ]:


df.shape


# In[ ]:


df.describe()


# **The features described in the above data set are**
# 
# 1. Count = Number of non-empty rows in a feature
# 2. Mean  = Mean value of a feature
# 3. Std   = Standard Deviation Value of a feature
# 4. Min   = Minimum value of a feature
# 5. 25%, 50%, and 75% = Percentile/quartile of each feature
# 6. Max   = Maximum value of a feature

# In[ ]:


df.columns


# <h1>Data Visualization

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# <h2>Plot Features

# In[ ]:


sns.distplot(df['thalach'], kde = False, bins=30, color='blue')


# In[ ]:


sns.distplot(df['chol'], kde=False,color='red')
plt.show()


# In[ ]:


plt.figure(figsize=(20,14))
sns.countplot(x='age',data = df, hue = 'target',palette='GnBu')
plt.show()


# <h2>Scatter Plots

# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='trestbps',y='thalach',data=df, hue='target')
plt.show()


# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='target',y='thalach',data=df, hue='sex')
plt.show()


# <h2>Swarm Plot

# In[ ]:


plt.figure(figsize=(8,6))
sns.swarmplot(x='target',y='thalach',data=df, hue='sex')
plt.show()


# <h2>Box Plot

# In[ ]:


plt.figure(figsize=(8,6))
sns.boxplot(x='target',y='thalach',data=df, hue='sex')
plt.show()


# <h2>Violin Plot

# In[ ]:


plt.figure(figsize=(8,6))
sns.violinplot(x='sex',y='thalach',data=df, hue='target', split=True)
plt.show()


# <h1>Data Preprocessing

# <h2>Handling Missing Values

# In[ ]:


df.isnull().sum()


# <h3>Drop NULL Value Columns

# In[ ]:


df.dropna()


# <h3>Fill NULL Values

# In[ ]:


df.fillna(method='bfill')


# <h3>Some Other Methods of Filling Null Values</h3>
# 
# **df.fillna(0)** - fill with "0" <br>
# **df.fillna(method='bfill')** - fill with before value <br>
# **df.fillna({'age': 10, 'sex': 1, 'cp': 1, 'trestbps': 130})** - fill with given values
# 

# <h2>Outlier Removal

# <h3>Using Z-Score Approach

# In[ ]:


from scipy import stats 


# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='trestbps',y='thalach',data=df, hue='target')
plt.show()


# In[ ]:


df2 = df[(np.abs(stats.zscore(df['thalach'])) < 3)]
df2 = df2[(np.abs(stats.zscore(df2['trestbps'])) < 3)]


# In[ ]:


df2.shape[0]


# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='trestbps',y='thalach',data=df2, hue='target')
plt.show()


# <h3>Using Quantile Approach

# In[ ]:


q = df['thalach'].quantile(0.7)
df3 = df[df['thalach'] < q]

q = df3['trestbps'].quantile(0.99)
df3 = df3[df3['trestbps'] < q]


# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='trestbps',y='thalach',data=df3, hue='target')
plt.show()


# In[ ]:


df3.shape[0]


# In[ ]:


df = df3


# <h3>Encoding Categorical Variables

# In[ ]:


from sklearn import preprocessing


# In[ ]:


df.head(3)


# In[ ]:


df.describe()


# In[ ]:


label_encoder = preprocessing.LabelEncoder()


# In[ ]:


df['sex'] = label_encoder.fit_transform(df['sex'])
df.head(3)


# <h3>Data Normalization

# In[ ]:


x = df[['chol', 'thalach']].values
min_max_scaler = preprocessing.MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(x)


# In[ ]:


df.head(5)


# In[ ]:


pd.DataFrame(df_scaled).head(5)


# <h2>Feature Engineering

# <h3>Correlation Analysis

# In[ ]:


plt.figure(figsize=(18,18))
plt.rcParams["axes.labelsize"] = 20
sns.set(font_scale=1.4)
sns.heatmap(df.corr(), annot = True ,linewidths=.1)
plt.show()


# In[ ]:


def find_correlation(data, threshold=0.9):
    corr_mat = data.corr()
    corr_mat.loc[:, :] = np.tril(corr_mat, k=-1)
    already_in = set()
    result = []
    for col in corr_mat:
        perfect_corr = corr_mat[col][abs(corr_mat[col])> threshold].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat


# In[ ]:


columns_to_drop = find_correlation(df.drop(columns=['target']) , 0.7)
df4 = df.drop(columns=columns_to_drop)


# In[ ]:


df4


# In[ ]:


corr = df.corr()
linear_features=abs(corr).target.drop('target').sort_values(ascending=False)[:5].keys()


# In[ ]:


abs(corr).target.drop('target').sort_values(ascending=False)[:5].plot(kind='barh')


# <h3>Using Random Forest Classifier to Identify Important Features

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
y = df.copy(deep=True)['target']
model = rf.fit(df.drop('target', axis=1),y)


# In[ ]:


importance = rf.feature_importances_
feat_importances_act = pd.Series(importance, index=df.drop('target', axis=1).columns)
feat_importances = feat_importances_act.nlargest(20)
feat_importances.plot(kind='barh')


# <h3>Convert Categorical Variables into Dummy Variables

# In[ ]:


df.dtypes


# In[ ]:


df['sex'] = df['sex'].astype('object')
df['cp'] = df['cp'].astype('object')
df['fbs'] = df['fbs'].astype('object')
df['restecg'] = df['restecg'].astype('object')
df['exang'] = df['exang'].astype('object')
df['slope'] = df['slope'].astype('object')
df['thal'] = df['thal'].astype('object')
df.dtypes


# In[ ]:


df.head(10)


# In[ ]:


df_1 = pd.get_dummies(df, drop_first=True)
df_1.head()


# In[ ]:


df


# <h1>Applying ML Models

# <h2>Split Train and Test sets

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('target',1), df['target'], test_size = .2, random_state=10)


# In[ ]:


X_train.head()


# <h2>Random Forest Classifier

# In[ ]:


from sklearn.metrics import accuracy_score
model = RandomForestClassifier(max_depth=20)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
# cla_pred.append(accuracy_score(y_test,predictions))
print(accuracy_score(y_test,predictions))


# In[ ]:


importance = model.feature_importances_
feat_importances_act = pd.Series(importance, index=X_train.columns)
feat_importances = feat_importances_act.nlargest(20)
feat_importances.plot(kind='barh')


# <h3> Other Models

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
#from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

classifiers=[['Logistic Regression :',LogisticRegression()],
       ['Decision Tree Classification :',DecisionTreeClassifier()],
       ['Gradient Boosting Classification :', GradientBoostingClassifier()],
       ['Ada Boosting Classification :',AdaBoostClassifier()],
       ['Extra Tree Classification :', ExtraTreesClassifier()],
       ['K-Neighbors Classification :',KNeighborsClassifier()],
       ['Support Vector Classification :',SVC()],
       ['Gaussian Naive Bayes :',GaussianNB()]]
cla_pred=[]
for name,model in classifiers:
    model=model
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    cla_pred.append(accuracy_score(y_test,predictions))
    print(name,accuracy_score(y_test,predictions))


# <h2>Hyper Parameter Tuning

# <h3>Using Grid Search

# In[ ]:


random.seed(100)
rfmodel = RandomForestClassifier()


# In[ ]:


#Hyperparameter tuning for Logistic Regression
random.seed(100)
from sklearn.model_selection import GridSearchCV
n_estimators = [10, 20, 50, 100]
max_depth = [5,10,15,20]
hyperparameters = dict(max_depth=max_depth, n_estimators=n_estimators)
h_rfmodel = GridSearchCV(rfmodel, hyperparameters, cv=5, verbose=0)
best_logmodel=h_rfmodel.fit(df.drop('target', 1), df['target'])
print('Best Estimators:', best_logmodel.best_estimator_.get_params()['n_estimators'])
print('Best Max Depth:', best_logmodel.best_estimator_.get_params()['max_depth'])


# In[ ]:


random.seed(100)
rfmodel = RandomForestClassifier(n_estimators=100, max_depth=10)
rfmodel.fit(X_train, y_train)
predictions = rfmodel.predict(X_test)
print(accuracy_score(y_test,predictions))


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))


# <h2> Model Evaluation

# In[ ]:


from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(y_test,predictions)


# In[ ]:


plt.figure(figsize=(24,12))

plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.show()


# ![350px-Precisionrecall.svg.png](attachment:350px-Precisionrecall.svg.png)

# In[ ]:


from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,rfmodel.predict_proba(X_test)[:,1])


# In[ ]:


import scikitplot as skplt
import matplotlib.pyplot as plt

skplt.metrics.plot_roc_curve(y_test,rfmodel.predict_proba(X_test), figsize = (20,20))
plt.figure(figsize=(40,18))
plt.show()

