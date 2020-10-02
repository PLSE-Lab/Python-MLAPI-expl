#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import sklearn 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# In[ ]:


dataset= pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")


# 
# # Dataset Basic Information
# 
# 
# 

# In[ ]:


dataset.head()


# In[ ]:


dataset.describe()


# In[ ]:


dataset.info()


# In[ ]:


df= dataset.copy()
dataset.dropna(inplace=True)
dataset.info()


# In[ ]:


dataset.describe()


# bedrooms , bathrroms -> discrete numerical data

# In[ ]:


s1 = dataset.groupby("type").size()
s1.plot(kind='pie',autopct='%1.1f%%',figsize=(10,10))


# In[ ]:


s1 = dataset.groupby("rating").size()
s1.plot(kind='pie',autopct='%1.1f%%',figsize=(10,10))


# In[ ]:


dataset["rating"].value_counts()
#IMBALANCED DATASET


# # Checking Distribution of Each Variable

# In[ ]:


sns.FacetGrid(dataset,hue="rating", height=7)    .map(sns.distplot, "feature1")    .add_legend();
plt.show();


# Almost Everything is Normally Distributed

# In[ ]:


sns.FacetGrid(dataset, height=7)    .map(sns.distplot, "feature2")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(dataset, height=7)    .map(sns.distplot, "feature3")    .add_legend();
plt.show();


# kind of skewed data, we take log to correct it

# In[ ]:


sns.FacetGrid(dataset, height=7)    .map(sns.distplot, "feature4")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(dataset, height=7)    .map(sns.distplot, "feature5")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(dataset, height=7)    .map(sns.distplot, "feature6")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(dataset, height=7)    .map(sns.distplot, "feature7")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(dataset, height=7)    .map(sns.distplot, "feature8")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(dataset,hue="rating", height=7)    .map(sns.distplot, "feature9")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(dataset, height=7)    .map(sns.distplot, "feature10")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(dataset, height=7)    .map(sns.distplot, "feature11")    .add_legend();
plt.show();


# RIGHT SKEWED = feature3,feature5,feature6,feature9 (median < mean => Right Skewed)
# 
# TO NORMALIZE = feature8
# 
# BELL NORMALIZED = feature11, feature10, feature7,feature4,feature2, feature1

# ## LOG NORMALIZATION
# 
# 

# In[ ]:


import math
skewed_features=['feature6','feature5','feature9','feature3']
dataset[skewed_features]= np.log(dataset[skewed_features])


# In[ ]:


sns.FacetGrid(dataset, height=7)    .map(sns.distplot, "feature9")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(dataset, height=7)    .map(sns.distplot, "feature3")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(dataset, height=7)    .map(sns.distplot, "feature6")    .add_legend();
plt.show();


# feature6 distribution has remained same as previous.
# We can after removing outliers do bell-shaped normalization maybe
# 
# 

# In[ ]:


sns.FacetGrid(dataset, height=7)    .map(sns.distplot, "feature5")    .add_legend();
plt.show();


# In[ ]:


dataset.describe()


# # Checking/ Removing Outliers

# In[ ]:


sns.boxplot(x='rating',y='feature1', data=dataset)
plt.show()


# In[ ]:


sns.boxplot(x='rating',y='feature2', data=dataset)
plt.show()


# In[ ]:


sns.boxplot(x='rating',y='feature3', data=dataset)
plt.show()


# In[ ]:


sns.boxplot(x='rating',y='feature4', data=dataset)
plt.show()


# In[ ]:


sns.boxplot(x='rating',y='feature5', data=dataset)
plt.show()


# In[ ]:


sns.boxplot(x='rating',y='feature6', data=dataset)
plt.show()


# In[ ]:


sns.boxplot(x='rating',y='feature7', data=dataset)
plt.show()


# In[ ]:


sns.boxplot(x='rating',y='feature8', data=dataset)
plt.show()


# In[ ]:


sns.boxplot(x='rating',y='feature9', data=dataset)
plt.show()


# In[ ]:


sns.boxplot(x='rating',y='feature10', data=dataset)
plt.show()


# In[ ]:


sns.boxplot(x='rating',y='feature11', data=dataset)
plt.show()


# Visualizing Outliers with BoxPlot
# feature9 , feature 6 has no/veryless outliers 
# I shall remove outliers , by using knowledge of IQR , outlier is considered to be at point 1.5*IQR a/c to convention used by me

# In[ ]:


data= dataset.copy()


# In[ ]:


features=['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11']
for f in features:
  q25, q75 = np.percentile(data[f], 25), np.percentile(data[f], 75)
  iqr = q75 - q25
  cut_off = iqr * 1.5
  lower, upper = q25 - cut_off, q75 + cut_off
  print(f)
  print(lower,upper)
  outliers = [x for x in data[f] if x < lower or x > upper]
  print(outliers)
  data=data.loc[ (data[f]>= lower) & (data[f] <= upper)]


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


sns.boxplot(x='rating',y='feature1', data=data)
plt.show()


# In[ ]:


sns.boxplot(x='rating',y='feature10', data=data)
plt.show()


# This still has outlier, so we need to remove outliers by calculating iqr for every feature , class wise.

# In[ ]:


data2= dataset.copy()


# In[ ]:


data2.info()


# In[ ]:


data2.loc[data2['rating']==4]['feature1']


# In[ ]:


features=['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11']
for f in features:
  for i in range(7):
    dataTemp=data2.loc[data2['rating']==i]
    q25, q75 = np.percentile(dataTemp[f], 25), np.percentile(dataTemp[f], 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    print(f)
    print(lower,upper)
    outliers = [x for x in dataTemp[f] if x < lower or x > upper]
    print(outliers)
    data2.drop( data2[(data2['rating']==i) & ((data2[f]<lower) | (data2[f]>upper))].index,inplace=True)
    #data2.info()
    #data2.loc[data2['rating']==i][f]=dataTemp.loc[ (dataTemp[f]>= lower) & (dataTemp[f] <= upper)]


# In[ ]:


data2.info()


# In[ ]:


sns.boxplot(x='rating',y='feature1', data=data2)
plt.show()


# In[ ]:


sns.boxplot(x='rating',y='feature10', data=data2)
plt.show()


# In[ ]:


data2.loc[(data2['feature1']>=20) & (data2['rating']==4)]['feature1']


# In[ ]:


sns.boxplot(x='rating',y='feature11', data=data2)
plt.show()


# # Normalize Data, i.e. bring everything in range of 0-1

# In[ ]:


newdataset= data.copy()


# In[ ]:


newdataset.info()


# In[ ]:


newdataset= newdataset.drop_duplicates()
newdataset.info()


# In[ ]:


from sklearn import preprocessing


# In[ ]:


newdataset['type'].unique()


# In[ ]:


newdataset['type']= newdataset['type'].map({'new':1,'old':0})


# In[ ]:


newdataset['type'].unique()


# In[ ]:


newdataset['type'].dtypes


# In[ ]:


f, ax = plt.subplots(figsize=(20, 20))
corr = newdataset.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10,as_cmap=True),
            square=True, ax=ax, annot=True)


# In[ ]:


df['type'] = df['type'].map({'new': 1, 'old': 0})
df = df.apply(lambda x: x.fillna(x.mean()),axis=0)


# In[ ]:


y_ = df['rating']
X_ = df.drop('rating', axis = 1)


# In[ ]:


y= newdataset['rating']
X= newdataset.drop(['rating'],axis=1)
X.info()


# In[ ]:


y.describe()


# In[ ]:


from sklearn.preprocessing import StandardScaler
X = pd.DataFrame(StandardScaler().fit_transform(X),columns=X.columns)


# In[ ]:


from sklearn.preprocessing import StandardScaler
X_new = StandardScaler().fit_transform(X_)


# In[ ]:


X.describe()


# In[ ]:


#Standardisation
#PCA-LDA -> NoT DOING NOW
#CO-RELATION MATRIX
#MODEL


# # Splitting Data

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test, y_train,y_test=train_test_split(X,y,test_size=0.33, random_state=62)


# # Balancing Data
# 
# 

# In[ ]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(kind='regular',k_neighbors=1, sampling_strategy='minority')
x_sm2,y_sm2= smote.fit_resample(x_train,y_train)


# In[ ]:


from imblearn.over_sampling import ADASYN
adasyn= ADASYN(n_neighbors=1, sampling_strategy='minority')
x_sm2,y_sm2= adasyn.fit_resample(x_train,y_train)


# In[ ]:


from collections import Counter
print(sorted(Counter(y_sm2).items()))


# In[ ]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
x_sm,y_sm = ros.fit_resample(x_train, y_train)


# In[ ]:


from collections import Counter
print(sorted(Counter(y_sm).items()))


# In[ ]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[ ]:


from imblearn.over_sampling import ADASYN
adasyn= ADASYN(n_neighbors=1, sampling_strategy='minority',random_state=42)
X1, y1 = adasyn.fit_resample(X_new, y_)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X_new, y_, test_size=0.33, random_state=42)


# In[ ]:


X_Train, X_Test, Y_Train, Y_Test = train_test_split(X1, y1, test_size=0.33, random_state=42)


# In[ ]:


dataset["rating"].value_counts()


# # Linear Regression

# In[ ]:


linear_reg= linear_model.LinearRegression()
linear_reg.fit(x_sm,y_sm)
y_pred = linear_reg.predict(x_test)


# In[ ]:


y_pred=np.round(y_pred)
y_pred[y_pred>6]=6
y_pred[y_pred<0]=0


# In[ ]:


np.unique(y_pred)


# In[ ]:


np.unique(y_test)


# In[ ]:


x_test.shape
x_sm.shape


# In[ ]:


y_pred = linear_reg.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
rmse=sqrt(mse)
print("MSE : %.6f" 
      %mse)
print("RMSE : %.6f" 
      %rmse)


# # Ridge Regression

# In[ ]:


ridge_reg= linear_model.Ridge(alpha=0.5)
ridge_reg.fit(x_sm,y_sm)
y_pred = ridge_reg.predict(x_test)
y_pred=np.round(y_pred)
y_pred[y_pred>6]=6
mse=mean_squared_error(y_test,y_pred)
print(ridge_reg.alpha)
rmse=sqrt(mse)
print("MSE : %.6f" 
      %mse)
print("RMSE : %.6f" 
      %rmse)


# # Ridge Regression with Parameter Tuning
# 

# In[ ]:


ridge_regCV= linear_model.RidgeCV(alphas=[3.351,3.352,3.352,3.352,3.352,3.352,3.352,3.352,3.352,3.35])
ridge_regCV.fit(x_sm,y_sm)
print(ridge_regCV.alpha_)
y_pred = ridge_regCV.predict(x_test)
y_pred=np.round(y_pred)
y_pred[y_pred>6]=6
mse=mean_squared_error(y_test,y_pred)
rmse=sqrt(mse)
print("MSE : %.6f" 
      %mse)
print("RMSE : %.6f" 
      %rmse)


# # Lasso Regression

# In[ ]:


lasso_reg= linear_model.LassoCV(alphas=np.logspace(-5,0,40),cv=20)
lasso_reg.fit(x_sm,y_sm)
print(lasso_reg.alpha_)
y_pred = lasso_reg.predict(x_test)
y_pred=np.round(y_pred)
y_pred[y_pred>6]=6
mse=mean_squared_error(y_test,y_pred)
rmse=sqrt(mse)
print("MSE : %.6f" 
      %mse)
print("RMSE : %.6f" 
      %rmse)


# # Bayesian Regression
# 

# In[ ]:


bayes_reg= linear_model.BayesianRidge()
bayes_reg.fit(x_sm,y_sm)
y_pred = bayes_reg.predict(x_test)
y_pred=np.round(y_pred)
y_pred[y_pred>6]=6
mse=mean_squared_error(y_test,y_pred)
rmse=sqrt(mse)
print("MSE : %.6f" 
      %mse)
print("RMSE : %.6f" 
      %rmse)


# # Logistic Regression with hyperparameter tuning
# 
# 
# 

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.classification import accuracy_score, log_loss

alpha = [10 ** x for x in range(-6, 3)]
cv_log_error_array = []
for i in alpha:
    print("for alpha =", i)
    clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='log', random_state=42)
    clf.fit(x_sm, y_sm)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(x_sm, y_sm)
    y_pred= sig_clf.predict_proba(x_test)
    cv_log_error_array.append(log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))
    # to avoid rounding error while multiplying probabilites we use log-probability estimates
    print("Log Loss :",log_loss(y_test,y_pred)) 

fig, ax = plt.subplots()
ax.plot(alpha, cv_log_error_array,c='g')
for i, txt in enumerate(np.round(cv_log_error_array,3)):
    ax.annotate((alpha[i],str(txt)), (alpha[i],cv_log_error_array[i]))
plt.grid()
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error measure")
plt.show()


best_alpha = np.argmin(cv_log_error_array)
clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
clf.fit(x_sm, y_sm)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(x_sm, y_sm)

predict_y = sig_clf.predict_proba(x_sm)
print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_sm, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(x_test)
print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))


# In[ ]:


y_pred = sig_clf.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
rmse=sqrt(mse)
print("MSE : %.6f" 
      %mse)
print("RMSE : %.6f" 
      %rmse)


# # Random Forest 

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


alpha = [100,200,500,1000]
max_depth = [5, 10,15,20,25,30,40]
n= len(max_depth)
cv_accuracy_array = []
for i in alpha:
    for j in max_depth:
        print("for n_estimators =", i,"and max depth = ", j)
        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42, n_jobs=-1)
        clf.fit(x_sm, y_sm)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(x_sm, y_sm)
        sig_clf_probs = sig_clf.predict(x_test)
        cv_accuracy_array.append(accuracy_score(y_test, sig_clf_probs))
        print("Accuracy:",accuracy_score(y_test, sig_clf_probs)) 

best_alpha = np.argmax(cv_accuracy_array)
clfRF = RandomForestClassifier(n_estimators=alpha[int(best_alpha/n)], criterion='gini', max_depth=max_depth[int(best_alpha%n)], random_state=42, n_jobs=-1)
clfRF.fit(x_sm,y_sm)
sig_clfRF = CalibratedClassifierCV(clf, method="sigmoid")
sig_clfRF.fit(x_sm,y_sm)

predict_y = sig_clfRF.predict(x_sm)
print('For values of best estimator = ', alpha[int(best_alpha/n)]," best depth = ", max_depth[int(best_alpha%n)], "The train accuracy is:",accuracy_score(y_sm, predict_y))
predict_y = sig_clfRF.predict(x_test)
print('For values of best estimator = ', alpha[int(best_alpha/n)]," best depth = ", max_depth[int(best_alpha%n)], "The test accuracy is:",accuracy_score(y_test, predict_y))


# In[ ]:


predict_y = sig_clfRF.predict(x_test)
np.unique(predict_y)


# In[ ]:


mse=mean_squared_error(y_test,predict_y)
rmse=sqrt(mse)
print("MSE : %.6f" 
      %mse)
print("RMSE : %.6f" 
      %rmse)


# # RF with SMOTE class balancing 

# In[ ]:


alpha = [100,200,500,1000]
max_depth = [5, 10,15,20,25,30,40]
n= len(max_depth)
cv_accuracy_array = []
for i in alpha:
    for j in max_depth:
        print("for n_estimators =", i,"and max depth = ", j)
        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42, n_jobs=-1)
        clf.fit(x_sm2, y_sm2)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(x_sm2, y_sm2)
        sig_clf_probs = sig_clf.predict(x_test)
        cv_accuracy_array.append(accuracy_score(y_test, sig_clf_probs))
        print("Accuracy:",accuracy_score(y_test, sig_clf_probs)) 

best_alpha = np.argmax(cv_accuracy_array)
clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/n)], criterion='gini', max_depth=max_depth[int(best_alpha%n)], random_state=42, n_jobs=-1)
clf.fit(x_sm2,y_sm2)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(x_sm2,y_sm2)

predict_y = sig_clf.predict(x_sm2)
print('For values of best estimator = ', alpha[int(best_alpha/n)]," best depth = ", max_depth[int(best_alpha%n)], "The train accuracy is:",accuracy_score(y_sm2, predict_y))
predict_y = sig_clf.predict(x_test)
print('For values of best estimator = ', alpha[int(best_alpha/n)]," best depth = ", max_depth[int(best_alpha%n)], "The test accuracy is:",accuracy_score(y_test, predict_y))


# In[ ]:


mse=mean_squared_error(y_test,predict_y)
rmse=sqrt(mse)
print("MSE : %.6f" 
      %mse)
print("RMSE : %.6f" 
      %rmse)


# 
# # RF with ADASYN class balancing  - Chosen Model 1
# 
# 

# In[ ]:


param_grid = { 
    'n_estimators': [150, 200, 500, 1000, 1500],
    'max_depth' : [5,8,15, 25],
    'criterion' :['gini']
}
rfc = RandomForestClassifier(random_state=42)
rf_cv = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5, verbose = 1, n_jobs = -1)
rf_cv.fit(X_Train, Y_Train)


# In[ ]:


rf_cv.best_params_


# In[ ]:


rf_mod = RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 1500, max_depth=25, criterion='gini')
rf_mod.fit(X_Train, Y_Train)
pred = rf_mod.predict(X_Test)
rms = math.sqrt(mean_squared_error(Y_Test, pred))


# In[ ]:


rms


# # Support Vector Machines

# In[ ]:


alpha = [10 ** x for x in range(-6, 3)]
cv_accuracy = []
for i in alpha:
    print("for alpha =", i)
    clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='hinge', random_state=42)
    clf.fit(x_sm, y_sm)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(x_sm, y_sm)
    y_pred= sig_clf.predict(x_test)
    cv_accuracy.append(accuracy_score(y_test,y_pred))
    print("Accuracy:",accuracy_score(y_test,y_pred)) 

# fig, ax = plt.subplots()
# ax.plot(alpha, cv_log_error_array,c='g')
# for i, txt in enumerate(np.round(cv_log_error_array,3)):
#     ax.annotate((alpha[i],str(txt)), (alpha[i],cv_log_error_array[i]))
# plt.grid()
# plt.title("Cross Validation Error for each alpha")
# plt.xlabel("Alpha i's")
# plt.ylabel("Error measure")
# plt.show()


best_alpha = np.argmax(cv_accuracy)
clfSVM = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42)
clfSVM.fit(x_sm, y_sm)
sig_clfSVM = CalibratedClassifierCV(clf, method="sigmoid")
sig_clfSVM.fit(x_sm, y_sm)

predict_y = sig_clfSVM.predict(x_sm)
print('For values of best alpha = ', alpha[best_alpha], "The train Accuracyis:",accuracy_score(y_sm, predict_y))
predict_y = sig_clfSVM.predict(x_test)
print('For values of best alpha = ', alpha[best_alpha], "The test Accuracy is:",accuracy_score(y_test,predict_y))


# # Gradient Boosted Decision Tree

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# parameters = {
#     "loss":["deviance"],
#     "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
#     "min_samples_split": np.linspace(0.1, 0.5, 12),
#     "min_samples_leaf": np.linspace(0.1, 0.5, 12),
#     "max_depth":[3,5,8],
#     "max_features":["log2","sqrt"],
#     "criterion": ["friedman_mse",  "mae"],
#     "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
#     "n_estimators":[10]
#     }
# 
# clfGBDT = GridSearchCV(GradientBoostingClassifier(), parameters, cv=1, n_jobs=-1,verbose=1)
# clfGBDT.fit(x_sm, y_sm)
# 
# predict_y = clfGBDT.predict(x_sm)
# print("The train Accuracyis:",accuracy_score(y_sm, predict_y))
# predict_y = clfGBDT.predict(x_test)
# print("The test Accuracy is:",accuracy_score(y_test,predict_y))
# 

# # STACKING - Chosen Model 2

# In[ ]:


from mlxtend.classifier import StackingCVClassifier

rf = RandomForestClassifier()
et = ExtraTreesClassifier()
svc = SVC()
lr = LogisticRegression()

sclf = StackingClassifier(classifiers=[rf,et, svc], 
                          meta_classifier=lr)


# In[ ]:


params = {'randomforestclassifier__n_estimators': [500, 1000],
           'svc__C' : [1.0, 10.0],
          'meta_classifier__C': [10.0, 20.0]
          }

grid = GridSearchCV(estimator=sclf, 
                    param_grid=params, 
                    cv=5,
                    verbose = 5)


# In[ ]:


grid.fit(X_train, Y_train)


# In[ ]:


cv_keys = ('mean_test_score', 'std_test_score', 'params')

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_[cv_keys[0]][r],
             grid.cv_results_[cv_keys[1]][r] / 2.0,
             grid.cv_results_[cv_keys[2]][r]))

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)


# In[ ]:


params = {'svc__C': [1.0],
          'randomforestclassifier__n_estimators': [500],
          'meta_classifier__C': [10.0]}


# In[ ]:


grid_final = GridSearchCV(estimator=sclf, 
                    param_grid=params, 
                    cv=5,
                    verbose = 1)


# In[ ]:


grid_final.fit(X_train, Y_train)


# In[ ]:


pred = grid_final.predict(X_test)


# In[ ]:


rms = sqrt(mean_squared_error(Y_test, pred))
rms


# # APPLICATION ON TEST DATA

# In[ ]:


test_data= pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")


# In[ ]:


id1= test_data['id']
type(id1)


# In[ ]:


test_data.info()


# In[ ]:


# skewed_features=['feature6','feature5','feature9','feature3']
# test_data[skewed_features]= np.log(dataset[skewed_features])


# In[ ]:


test_data['type']= test_data['type'].map({'new':1,'old':0})


# In[ ]:


test_data=test_data.apply(lambda x: x.fillna(x.mean()),axis=0)
test_data.info()


# In[ ]:


#test_data=test_data.drop(['feature1','feature2','feature4','feature8','feature9','feature11'],axis=1)
test_data.info()


# In[ ]:


test_data = pd.DataFrame(StandardScaler().fit_transform(test_data),columns=test_data.columns)


# In[ ]:


test_data.shape


# ## Using Model 1

# In[ ]:


predict_y = rf_mod.predict(test_data)


# In[ ]:


np.unique(predict_y)


# In[ ]:


predict_y.shape


# In[ ]:


sol_rf= pd.DataFrame({'id':id1,'rating':predict_y})


# In[ ]:


sol_rf.info()


# In[ ]:


sol_rf.to_csv('rf_v1.csv',index=False)


# ## Using Model 2

# In[ ]:


predict_y2 = grid_final.predict(test_data)
sol_st= pd.DataFrame({'id':id1,'rating':predict_y2})
sol_rf.to_csv('resultsStacking.csv',index=False)

