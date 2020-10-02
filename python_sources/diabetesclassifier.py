#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pyforest')


# In[ ]:


from pyforest import *
from mlxtend.plotting import plot_decision_regions
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[ ]:


diabetes = pd.read_csv('../input/diabetes.csv')
diabetes.head()


# In[ ]:


diabetes.describe()


# In[ ]:


diabetes.isnull().sum()


# In[ ]:


diabetes.info()


# In[ ]:


diabetes.describe().T


# In[ ]:


diabetes1 = diabetes.copy(deep=True)
diabetes1[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes1[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
diabetes1.isnull().sum()


# In[ ]:


p = diabetes.hist(figsize=(20, 20))


# In[ ]:


diabetes1['Glucose'].fillna(diabetes1['Glucose'].mean(), inplace=True)
diabetes1['BloodPressure'].fillna(diabetes1['BloodPressure'].mean(), inplace=True)
diabetes1['SkinThickness'].fillna(diabetes1['SkinThickness'].median(), inplace=True)
diabetes1['Insulin'].fillna(diabetes1['Insulin'].median(), inplace=True)
diabetes1['BMI'].fillna(diabetes1['BMI'].median(), inplace=True)


# In[ ]:


p = diabetes1.hist(figsize=(20, 20))


# In[ ]:


diabetes.shape


# In[ ]:


diabetes.dtypes


# In[ ]:


plt.figure(figsize=(5,5))
sns.set(font_scale=2)
sns.countplot(y=diabetes.dtypes.map(str))
plt.xlabel("count of each data type")
plt.ylabel("data types")
plt.show()


# In[ ]:


get_ipython().system('pip install missingno')


# In[ ]:


import missingno as msno
p = msno.bar(diabetes)


# In[ ]:


color_wheel = {1: '#0392cf', 2: '#7bc043'}
colors = diabetes['Outcome']
print(diabetes.Outcome.value_counts())
p=diabetes.Outcome.value_counts().plot(kind='bar')


# In[ ]:


from pandas.plotting import scatter_matrix
p = scatter_matrix(diabetes, figsize=(25,25))


# In[ ]:


p = sns.pairplot(diabetes1, hue='Outcome')


# In[ ]:


plt.figure(figsize=(15, 15))
p=sns.heatmap(diabetes.corr(), annot=True, cmap='copper')


# In[ ]:


sns.catplot(x='Outcome', y='Insulin', data=diabetes1, kind='box')
plt.show()


# In[ ]:


import plotly.offline as ply
values = pd.Series(diabetes1["Outcome"]).value_counts()
trace = go.Pie(values=values)
ply.iplot([trace])


# In[ ]:


plt.figure(figsize=(15, 15))
p=sns.heatmap(diabetes1.corr(), annot=True, cmap='Blues')


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(diabetes1.drop(['Outcome'], axis=1),),columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])


# In[ ]:


X.head()


# In[ ]:


y = diabetes1.Outcome
y.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=42, stratify=y)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

test_scores = []
train_scores = []

for i in range(1, 15):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train, y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))


# In[ ]:


max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v==max_train_score]
print('Max train score {} % and k={} '.format(max_train_score*100, list(map(lambda x: x+1, train_scores_ind))))


# In[ ]:


max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v==max_test_score]
print('Max test score {} % and k={} '.format(max_test_score*100, list(map(lambda x: x+1, test_scores_ind))))


# In[ ]:


plt.figure(figsize=(12, 5))
p=sns.lineplot(range(1, 15), train_scores, marker='*', label='Train Score')
p=sns.lineplot(range(1, 15), test_scores, marker='o', label='Test Score')


# In[ ]:


knn = KNeighborsClassifier(11)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)


# In[ ]:


value = 20000
width = 20000

plot_decision_regions(X.values, y.values, clf=knn, legend=2, filler_feature_values={2:value, 3:value, 4: value, 5: value, 6: value, 7: value}, filler_feature_ranges={2: width, 3: width, 4: width, 5: width, 6: width, 7: width}, X_highlight=X_test.values)
plt.title('KNN with Diabetes Data')
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[ ]:


y_pred = knn.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p=sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix', y=1.1)
plt.ylabel('Actual Label')
plt.xlabel('Predicted')


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_pred, y_test))


# In[ ]:


from sklearn.metrics import roc_curve
y_pred_proba = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# In[ ]:


plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=11) ROC curve')
plt.show()


# In[ ]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_proba)


# In[ ]:


from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors':np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid,cv=5)
knn_cv.fit(X,y)

print('Best Score:' + str(knn_cv.best_score_))
print('Best Parameters:' + str(knn_cv.best_params_))


# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
lrp = lr.predict(X_test)
print('Accuracy:', metrics.accuracy_score(lrp, y_test))
print(classification_report(lrp, y_test))
print(confusion_matrix(lrp, y_test))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbcp = gbc.predict(X_test)
print('Accuracy:', metrics.accuracy_score(gbcp, y_test))
print(classification_report(gbcp, y_test))
print(confusion_matrix(gbcp, y_test))


# In[ ]:


import xgboost as xgb
xgbc = xgb.XGBClassifier()
xgbc.fit(X_train, y_train)
xgbcp = xgbc.predict(X_test)
print('Accuracy:', metrics.accuracy_score(xgbcp, y_test))
print(classification_report(xgbcp, y_test))
print(confusion_matrix(xgbcp, y_test))


# In[ ]:




