#!/usr/bin/env python
# coding: utf-8

# #### Final Models are at the end of the notebook. The rest of the notebook is a step by step process of arriving at those models.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# # Analysing dataset

# In[ ]:


training_df=pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')
test_df=pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')


# In[ ]:


training_df.head(10)


# In[ ]:


training_df=training_df.drop('id',axis = 1 )


# In[ ]:


training_df_dtype_nunique = pd.concat([training_df.dtypes, training_df.nunique()],axis=1)
training_df_dtype_nunique.columns = ["dtype","unique"]
training_df_dtype_nunique


# In[ ]:


test_df_dtype_nunique = pd.concat([test_df.dtypes, test_df.nunique()],axis=1)
test_df_dtype_nunique.columns = ["dtype","unique"]
test_df_dtype_nunique


# In[ ]:


training_df.info()
test_df.info()


# In[ ]:


training_df.describe()


# In[ ]:


test_df.describe()


# In[ ]:


training_df.replace("?", np.nan, inplace=True)
test_df.replace("?", np.nan, inplace=True)


# In[ ]:


training_df.describe()


# In[ ]:


training_df.isnull().sum()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


#Numerical features
features=['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature11']


# In[ ]:


#Fills NaN with mean of features
training_df.update(training_df[features].fillna(training_df[features].mean()))
test_df.update(test_df[features].fillna(test_df[features].mean()))


# In[ ]:


training_df.info()


# In[ ]:


training_df.isnull().any().any()


# In[ ]:


test_df.isnull().any().any()


# ## Data Analysis

# In[ ]:


sns.distplot(
    training_df['feature1'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Feature1', ylabel='Count');


# In[ ]:


sns.distplot(
    training_df['feature2'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Feature2', ylabel='Count');


# In[ ]:


training_df[features].hist(bins=15, figsize=(15, 6));


# In[ ]:


sns.countplot(training_df['type']);


# In[ ]:


sns.boxplot(x=training_df['type'], y=training_df['rating'], data=training_df)


# In[ ]:


sns.regplot(x=training_df['feature1'], y=training_df['rating'], data=training_df)


# In[ ]:


sns.regplot(y=training_df['feature2'], x=training_df['rating'], data=training_df)


# In[ ]:


plt.figure(figsize=(16,16))
sns.heatmap(training_df.corr(),annot=True)


# In[ ]:


X_analysis=training_df[features].copy()
y=training_df['rating'].copy()


# ## Feature Importance

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X_analysis,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_analysis.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(12,'Score'))  #print best features


# In[ ]:


model = ExtraTreesClassifier(n_estimators=10)
model.fit(X_analysis,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X_analysis.columns)
feat_importances.nlargest(12).plot(kind='barh')


# In[ ]:


#Splitting data
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X_analysis, y,test_size=0.33, random_state=2)


# In[ ]:


#Scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

scaler = RobustScaler()
X_train_n = scaler.fit_transform(X_train[features])


# In[ ]:


X_train = np.concatenate([X_train_n],axis=1)


# In[ ]:


X_val_n = scaler.fit_transform(X_val[features]) 


# In[ ]:


X_val = np.concatenate([X_val_n],axis=1)


# # Trying out different models

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn_classifier =KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

y_pred_lr = knn_classifier.predict(X_val)

rmse_knn = sqrt(mean_squared_error(y_pred_lr,y_val))

print("Root Mean Squared Error of Linear Regression: {}".format(rmse_knn))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dt_classifier =DecisionTreeClassifier().fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

y_pred_lr = dt_classifier.predict(X_val)

rmse_dt = sqrt(mean_squared_error(y_pred_lr,y_val))

print("Root Mean Squared Error of Linear Regression: {}".format(rmse_dt))


# In[ ]:


from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier    
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB  
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix as cm
from math import sqrt


# In[ ]:


classifiers = [
    ExtraTreeClassifier(),  #1
    DecisionTreeClassifier(max_depth=5), #2
    MLPClassifier(alpha=0.5, max_iter=1000), #3
    RadiusNeighborsClassifier(7),#4
    KNeighborsClassifier(3), #5
    SGDClassifier(), #6
    RidgeClassifierCV(), #7
    RidgeClassifier(), #8
    PassiveAggressiveClassifier(),#9
    AdaBoostClassifier(),#10
    GradientBoostingClassifier(),#11
    BaggingClassifier(),#12
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),#13
    BernoulliNB(),#14
    CalibratedClassifierCV(),#15
    GaussianNB(),#16
    LabelPropagation(),#17
    LabelSpreading(),#18
    LinearDiscriminantAnalysis(),#19
    LinearSVC(),#20
    LogisticRegression(),#21
    LogisticRegressionCV(),#22
    NearestCentroid(),#23
    Perceptron(),#24
    QuadraticDiscriminantAnalysis(),#25
    SVC(kernel="linear", C=0.025),#26
    SVC(gamma=2, C=1)#27
    ]


# In[ ]:


#Iterating over classifiers
i=1
for clf in classifiers:
    classifier=clf.fit(X_train, y_train)
    y_pred = classifier.predict(X_val)
    print(cm(y_val, y_pred))
    rmse = sqrt(mean_squared_error(y_pred,y_val))

    print("Root Mean Squared Error of" , i, ": {}".format(rmse))
    i+=1


# In[ ]:


# X_test = test_df[features+['type']].copy()
X_analysis_test=test_df[features].copy()
X_test_n = scaler.fit_transform(X_analysis_test[features]) 
X_analysis_test = np.concatenate([X_test_n],axis=1)


# In[ ]:


#Iterating over different parameters for BaggingClassifier
max_n_ests=80
for j in [500,2000,8000,99999]:
    clf_stump=ExtraTreeClassifier(max_features=None,max_leaf_nodes=j)
    print(j)
    sum=0
    for i in np.arange(1,max_n_ests):
        baglfy=BaggingClassifier(base_estimator=clf_stump,n_estimators=i)#max_samples=1.0
        baglfy=baglfy.fit(X_train,y_train)
        bag_tr_err=baglfy.predict(X_val)
        rmse = sqrt(mean_squared_error(bag_tr_err,y_val))
        print("Root Mean Squared Error of" , i, ": {}".format(rmse))
        sum+=rmse
    print(j, " ", sum/80)


# # Final Model 1: BaggingClassifier

# In[ ]:


#Parameters max_leaf_nodes and n_estimators decided by looping over them to find optimal values.
clf_stump=ExtraTreeClassifier(max_features=None,max_leaf_nodes=8000)
baglfy=BaggingClassifier(base_estimator=clf_stump,n_estimators=80)#max_samples=1.0
baglfy=baglfy.fit(X_train,y_train)
bag_tr_err=baglfy.predict(X_val)
rmse = sqrt(mean_squared_error(bag_tr_err,y_val))

print("Root Mean Squared Error of" , 80, ": {}".format(rmse))


# In[ ]:


y_pred_test=baglfy.predict(X_analysis_test)
y_pred_test


# In[ ]:


df=pd.DataFrame(columns=['id', 'rating'])
df['id']=test_df['id']
df['rating']=y_pred_test


# In[ ]:


# df.to_csv('predictions.csv', index=False)


# # Final Model 2: AdaBoostClassifier

# In[ ]:


#Parameters max_leaf_nodes and n_estimators decided by looping over them to find optimal values.
clf_stump=ExtraTreeClassifier(max_features=None,max_leaf_nodes=500)
baglfy=AdaBoostClassifier(base_estimator=clf_stump,n_estimators=80)#max_samples=1.0
baglfy=baglfy.fit(X_train,y_train)
bag_tr_err=baglfy.predict(X_val)
rmse = sqrt(mean_squared_error(bag_tr_err,y_val))

print("Root Mean Squared Error of" , 80, ": {}".format(rmse))


# In[ ]:


y_pred_test=baglfy.predict(X_analysis_test)
y_pred_test


# In[ ]:


df=pd.DataFrame(columns=['id', 'rating'])
df['id']=test_df['id']
df['rating']=y_pred_test


# In[ ]:


# df.to_csv('predictions.csv', index=False)

