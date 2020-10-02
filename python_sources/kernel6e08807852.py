#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 99% of scores of any metric is 100% perfect. The remainimg 1% is 0.999... or 0.95....

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from xgboost import XGBClassifier
import xgboost
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# In[ ]:


df = pd.read_csv('../input/mushrooms.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


# Let us check the data for missing values
s =set(df.apply(lambda x: sum(x.isnull())))


# In[ ]:


s


# In[ ]:


# calling categorical columns
cat_cols = [x for x in df.dtypes.index if df.dtypes[x]=='object']


# In[ ]:


len(cat_cols)


# In[ ]:



# TO get the feture_importance by XGBClassifier()
#name = 'xgboost'

labelEncoder = preprocessing.LabelEncoder()
df.dtypes

for col in df.columns:
    df[col] = labelEncoder.fit_transform(df[col])


# In[ ]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.26) 
y_train = train['class']
X_train = train[[x for x in train.columns if 'class' not in x]]
y_test = test['class']
X_test= test[[x for x in test.columns if 'class' not in x]]
xgb =  XGBClassifier()
xgb.fit(X_train, y_train)
ax = xgboost.plot_importance(xgb, color='magenta') 


# In[ ]:


df.apply(lambda x : len(x.unique()))


# In[ ]:


# Let us do some frequecy plots

def cplt(col_name):
    sns.countplot(x = df[col_name])


# In[ ]:


cplt('habitat')


# In[ ]:


cplt('gill-color') 


# In[ ]:


cplt('class')


# In[ ]:


# Separate the class column from the dat and call the rest X.
X=df.drop(['class'],axis=1) 
y = df['class']


# In[ ]:


z = df['class'].value_counts()


# In[ ]:


z


# In[ ]:


# Ploting the class again with bar chart.
z.plot(kind='bar', colors=['red', 'blue'])


# In[ ]:


# To see the correlation among the columns
X.corr()


# In[ ]:


c = X.corr()
plt.figure(figsize=(16,8))
sns.heatmap(c, annot=True)


# In[ ]:


# Change all the categorical columns to numerical columns by Label Encoder and get dummies
from sklearn.preprocessing import LabelEncoder
Encoder_X = LabelEncoder()
for col in X.columns:
    X[col] = Encoder_X.fit_transform(X[col])
    Encoder_y=LabelEncoder()
y = Encoder_y.fit_transform(y)
X=pd.get_dummies(X, columns=X.columns, drop_first=True) 


# In[ ]:


X.head()


# In[ ]:


X.isnull().sum()


# In[ ]:


X.corr()


# In[ ]:


c = X.corr()
plt.figure(figsize=(16,8))
sns.heatmap(c, annot=True)


# In[ ]:



#Get the feature_importance for dummy features.
#from numpy import loadtxt
import matplotlib.pyplot as plt
from xgboost import plot_tree
from xgboost import XGBClassifier
import xgboost
model = XGBClassifier()
model.fit(X_train, y_train)
ax = xgboost.plot_importance(model, color='green')    
X.shape


# In[ ]:


# plot single tree
plot_tree(model)
plt.show()
#To get outliers in two different ways
X.columns


# In[ ]:


def find_outliers_tukey(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3-q1 
    floor = q1 - 1.5*iqr
    ceiling = q3 + 1.5*iqr
    outlier_indices = list(x.index[(x < floor)|(x > ceiling)])
    outlier_values = list(x[outlier_indices])
    return outlier_indices, outlier_values


# In[ ]:


tukey_indices, tukey_values = find_outliers_tukey(X['cap-shape_1'])
print(np.sort(tukey_values))

# No outliers


# In[ ]:


# Train Test Split
np.random.seed(1234)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled =  sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.26,random_state=0)  


# In[ ]:


from sklearn.metrics import accuracy_score,roc_curve,confusion_matrix,roc_auc_score
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred )


# In[ ]:


confusion_matrix(y_test,y_pred)
# The model is 100% perfect. Let us see the ROC curve.


# In[ ]:



from sklearn.metrics import roc_curve 
fpr, tpr, threshold = roc_curve(y_test, y_pred)
auc_roc = roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


from sklearn.metrics import roc_curve, auc, confusion_matrix,roc_auc_score
fpt, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpt, tpr)
roc_auc


# In[ ]:


plt.figure(figsize=(10,10))
plt.title('ROC Curve')
plt.plot(fpr,tpr, color ='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[ ]:


# The ROC curve is also 100% perfect.
# In the next part I am going to do 8 different classifiers model together.


# In[ ]:


np.random.seed(1234)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled =  sc.fit_transform(X)
#split train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.26,random_state=0)
from sklearn.metrics import roc_curve, auc,confusion_matrix,roc_auc_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
from sklearn.ensemble import ExtraTreesClassifier
exc = ExtraTreesClassifier()
import xgboost as xgb
boost= xgb.XGBClassifier(n_estimators=200, learning_rate=0.01)
from sklearn.naive_bayes import GaussianNB
model_naive = GaussianNB()
from sklearn.svm import SVC
svm_model= SVC(gamma='scale')
from sklearn.neighbors import KNeighborsClassifier as KNN
kn = KNN()
models = [lg, dt, rfc, exc, boost, model_naive, svm_model, kn]
modnames = ['LogisticRegression', 'DecisionTreeClassifier','RandomForestClassifier',
            'ExtraTreesClassifier', 'XGBClassifier', 'GaussianNB', 'SVC', 'KNeighborsClassifier']


# In[ ]:


for i, model in enumerate(models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    confusion_matrix(y_test,y_pred)
    print('The accuracy of ' + modnames[i] + ' is ' + str(accuracy_score(y_test,y_pred)))
    print('The confution_matrix ' + modnames[i] + ' is ' + str(confusion_matrix(y_test,y_pred)))


# In[ ]:


# Since X_test is a numpy array, we need to convert it to a DataFrame to define the index.
test_X = pd.DataFrame(X_test)


# In[ ]:


output1 = pd.DataFrame({'index':test_X.index,'actual':y_test, 'pred_lg':lg.predict(X_test),'pred_dt':dt.predict(X_test), 'pred_rfc':
                       rfc.predict(X_test), 'pred_exc':exc.predict(X_test), 'pred_boost':boost.predict(X_test), 'pred_model_naive':
                       model_naive.predict(X_test),'pred_svm_model':svm_model.predict(X_test),'pred_kn':kn.predict(X_test),
                      })


# In[ ]:


output1.to_csv('submission1.csv', index=False)


# In[ ]:


df1 = pd.read_csv('submission1.csv')


# In[ ]:


df1.head()


# In[ ]:


# We see that from 8 classifiers 6 of them scored 1 and one close to 1 and the other almost 95%. The confution_matrix
#for all got the same as before.


# In[ ]:


for i, model in enumerate(models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    auc_roc = classification_report(y_test,y_pred)
    print('The auc_roc report ' + modnames[i] + ' is ' + str(auc_roc))


# In[ ]:


# Also these are all perfect for all models.


# In[ ]:


# Two other classifiers
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc2 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
        max_depth=1, random_state=0)

models = [gbc, gbc2]
modnames = ['GradientBoostingClassifier', 'GradientBoostingClassifier']



# In[ ]:


for i, model in enumerate(models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    confusion_matrix(y_test,y_pred)
    print('The accuracy of ' + modnames[i] + ' is ' + str(accuracy_score(y_test,y_pred)))
    print('The confution_matrix ' + modnames[i] + ' is ' + str(confusion_matrix(y_test,y_pred)))


# In[ ]:


for i, model in enumerate(models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    auc_roc = classification_report(y_test,y_pred)
    print('The auc_roc report ' + modnames[i] + ' is ' + str(auc_roc))


# In[ ]:


#**********************************************
# 10 cross validation
from sklearn.neighbors import KNeighborsClassifier as KNN
knn = KNN()
from sklearn.model_selection import cross_val_score
cvs_knn = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
cvs_knn.mean()


# In[ ]:


# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
from sklearn.neighbors import KNeighborsClassifier
knn2 = KNeighborsClassifier(n_neighbors=5)
cvs_knn2 = cross_val_score(knn2, X_train, y_train, cv=10, scoring='accuracy')
cvs_knn2.mean()


# In[ ]:


# use average accuracy as an estimate of out-of-sample accuracy
k_range = list(range(1, 10))
k_scores = []
from sklearn.neighbors import KNeighborsClassifier 
for k in k_range:
    knn3 = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn3, X_train, y_train, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())


# In[ ]:


k_scores     
cvs_knn3 = pd.Series(k_scores)
cvs_knn3.mean()


# In[ ]:


from sklearn.naive_bayes import GaussianNB
model_naive = GaussianNB()
cvs_GNB = cross_val_score(model_naive, X, y, cv=10, scoring='accuracy')
cvs_GNB.mean()


# In[ ]:


output3 = pd.DataFrame({'index':np.arange(1), 'cvs_knn':cvs_knn.mean(), 'cvs_knn2':cvs_knn2.mean(),
                       'cvs_knn3':cvs_knn3.mean(), 'cvs_GNB':cvs_GNB.mean()})


# In[ ]:


output3.to_csv('submission3.csv', index=False)


# In[ ]:


df3 = pd.read_csv('submission3.csv')


# In[ ]:


df3


# In[ ]:


#Exhaustive search over specified parameter values for an estimator.
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
clf = SVC(gamma='auto')
param_grid = {
 'C': [1, 10, 100], 'kernel': ['linear','rbf'],
 'C': [1, 20], 'gamma': [1,0.1,0.01], 'kernel': ['rbf']}


# In[ ]:


model_svm_GRID = GridSearchCV(clf, param_grid, scoring='accuracy',cv=5)
model_svm_GRID.fit(X_train, y_train)
y_pred_GRID= model_svm_GRID.predict(X_test)
from sklearn.metrics import roc_curve, auc,confusion_matrix,roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
roc_auc


# In[ ]:


print(roc_auc_score(y_pred,y_test))
cm = confusion_matrix(y_test,y_pred)
cm


# In[ ]:


plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[ ]:


"""
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)"""


# In[ ]:


#Exhaustive search over specified parameter values for an estimator.
#GridSearcgh using AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()
param_grid = {'n_estimators': [10,25,40 ]}
from sklearn.model_selection import GridSearchCV
model_ada_GRID =  GridSearchCV(ada, param_grid)                              
model_ada_GRID.fit(X_train, y_train)
y_pred_ada_GRID = model_ada_GRID.predict(X_test)
model_ada_GRID.score(X_test, y_pred) 


# In[ ]:


#GridSearch using DecisionTreeClassifier and AdaBoostClassifier tp prevent overfitting of dt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2] }


# In[ ]:



dt2 = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced",max_depth = None)
ada = AdaBoostClassifier(base_estimator = dt2)
# run grid search
GRID_ABC = GridSearchCV(ada, param_grid=param_grid, scoring = 'roc_auc',cv=5)
GRID_ABC.fit(X_train, y_train)
y_pred_GRID_ABC = GRID_ABC.predict(X_test)
GRID_ABC.score(X_test, y_pred)


# In[ ]:


from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import AdaBoostClassifier
ada3 = AdaBoostClassifier(algorithm='SAMME.R', random_state=0)
params = {'n_estimators':[50,100,200],'learning_rate':[1.0, 3.0, 5.0]}
scorer = make_scorer(accuracy_score)
grid_ada3 = GridSearchCV(ada3, params, scoring=scorer)
grid_fit = grid_ada3.fit(X_train,y_train)
best_ada3 = grid_fit.best_estimator_
ada3.fit(X_train, y_train)
y_pred_ada3 = ada3.predict(X_test)
ada3.score(X_test, y_pred)


# In[ ]:


output2 = pd.DataFrame({'index':test_X.index , 'actual':y_test , 'pred_kn':kn.predict(X_test) ,
                        'pred_gbc':gbc.predict(X_test) , 'pred_gbc2':gbc2.predict(X_test), 'y_pred_GRID':
                        model_svm_GRID.predict(X_test),
                       'y_pred_ada_GRID':model_ada_GRID.predict(X_test),'y_pred_GRID_ABC':GRID_ABC.predict(X_test),
                        'y_pred_ada3':ada3.predict(X_test)})


# In[ ]:


output2.to_csv('submission2.csv', index=False)


# In[ ]:


df2 = pd.read_csv('submission2.csv')


# In[ ]:


df2.head()


# In[ ]:


# Using PCA let us see how much data are in 5 first components
from sklearn.decomposition import PCA
pca = PCA(n_components=5,svd_solver='full' )
pca.fit_transform(X)
N = X.values
x = pca.fit_transform(N)
print(pca.explained_variance_ratio_)
pca.explained_variance_ratio_.sum()


# In[ ]:


# So we see that allmost the first components have 48% of whole data


# In[ ]:


# Let us put this data in 10 clusters
plt.figure(figsize = (5,5))
plt.scatter(x[:,0],x[:,1],c='red')
plt.scatter(x[:,0],x[:,2],c='blue')
plt.scatter(x[:,0],x[:,3],c='purple')
plt.scatter(x[:,0],x[:,4],c='magenta')
plt.scatter(x[:,1],x[:,2],c='black')
plt.scatter(x[:,1],x[:,3],c='brown')
plt.scatter(x[:,1],x[:,4],c='yellow')
plt.scatter(x[:,2],x[:,3],c='green')
plt.scatter(x[:,2],x[:,4],c='cyan')
plt.scatter(x[:,3],x[:,4],c='olive')
plt.xlabel("n-components is 5", fontsize=15)
plt.title("0.47658221797685124 PORTION OF DATA", fontsize=12)
plt.show()


# In[ ]:


#How about 10 components
from sklearn.decomposition import PCA
pca = PCA(n_components=10, svd_solver='arpack')
pca.fit_transform(X)
covariance = pca.get_covariance()
covariance
explained_variance = pca.explained_variance_
len(explained_variance)


# In[ ]:


print(pca.explained_variance_ratio_)


# In[ ]:


pca.explained_variance_ratio_.sum()


# In[ ]:


# Let us use Label encoder to get exact data and exact number of components.
from sklearn.preprocessing import LabelEncoder
Encoder_X = LabelEncoder()
for col in X.columns:
    X[col] = Encoder_X.fit_transform(X[col])


# In[ ]:


X=pd.get_dummies(X, columns=X.columns, drop_first=True)
from sklearn.decomposition import PCA


# In[ ]:


pca = PCA()
pca.fit_transform(X)
covariance = pca.get_covariance()


# In[ ]:


covariance


# In[ ]:


explained_variance = pca.explained_variance_
explained_variance


# In[ ]:


with plt.style.context('bmh'):
    plt.figure(figsize=(6, 4))
    plt.bar(range(95), explained_variance, alpha=0.5, align='center',
            label='individual explained variance', color='blue')
    plt.ylabel('Explained variance ratio',fontsize=10)
    plt.xlabel('Principal components',fontsize=10 )
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()


# In[ ]:


# It seems that with 60 compents we can recover the almost all the data. Let us try it in action.
from sklearn.decomposition import PCA
pca = PCA(n_components=60, svd_solver='arpack')
pca.fit_transform(X)
covariance = pca.get_covariance()
covariance


# In[ ]:


explained_variance = pca.explained_variance_
len(explained_variance)


# In[ ]:


print(pca.explained_variance_ratio_)
pca.explained_variance_ratio_.sum()


# In[ ]:


# It is allmost 100%. However we can figure out the exact number of components by other means.


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=95, svd_solver='full')
pca.fit_transform(X)
explained_variance
pca.explained_variance_ratio_
pca.explained_variance_ratio_.cumsum()
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95)+1
d


# In[ ]:


# We see that 38 components is covering the 95% od the data.


# In[ ]:


# Let us take only the first 4 principal components and visualise it using K-means clustering with k=4.
N = X.values
pca = PCA(n_components=4)
x = pca.fit_transform(N)
print(pca.explained_variance_ratio_)
pca.explained_variance_ratio_.sum()
plt.figure(figsize = (5,5))
plt.scatter(x[:,0],x[:,1], x[:,2],x[:,3])
plt.show()


# In[ ]:


# We may see the relation of the dimension of data with its percentage values in a plot.To do that, we need
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=895)
pca = PCA()
pca.fit(X_train)
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)


# In[ ]:


pca.explained_variance_ratio_.sum()


# In[ ]:


d = np.argmax(cumsum >= 0.95)+1
d


# In[ ]:


plt.xlabel('DIMENSIONS', fontsize=15)
plt.ylabel('Explained Variance',fontsize=15)
plt.title('Elbow Curve for 95% of Data',fontsize=17)
plt.plot(cumsum)


# In[ ]:


# The Elbow plot shows that the 95% of the data are recovered by the first 38 principal components.


# In[ ]:


#Final step to get the submission files.

