#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.simplefilter("ignore")


# # Explority Data Analysis

# In[ ]:


data=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
data.dropna(axis=0).any()
X=set(data.columns)
X.remove('Outcome')
X=data[X]
y=data['Outcome']
data.head()


# In[ ]:


print(data.info())
print(data.describe())


# We can learn some statistical informations using info and describe methods

# In[ ]:


age_data=pd.DataFrame(data.groupby(['Age'],as_index=False)['Outcome'].count())
interval={}
temp_sum=0
for i in range(len(age_data)):
    temp_sum+=int(age_data.iloc[i,1])
    if age_data.iloc[i,0]==35:
        interval.update({"20-35":temp_sum})
        temp_sum=0
    elif age_data.iloc[i,0]==50:
        interval.update({"35-50":temp_sum})
        temp_sum=0
    elif age_data.iloc[i,0]==81:
        interval.update({"+50":temp_sum})
plt.figure(figsize=(12,8))
plt.bar(interval.keys(),interval.values(),color=['#cc6699','#339933','#006666'])

plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Counts of Age Interval")
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(12,7))    
sns.heatmap(data.corr(), annot=True,ax=ax)


# Correlation helps us to understand relation between 2 column. If correlation coefficient close to 1, it means when first feature increase, second one is going to increase too. If coefficient close to -1 means, when first one increase second feature is going to decrease (negative relation) But we can not say only check correlation coefficient then they 2 feature affect together directly. Data scientist should check they are related or not.

# In[ ]:


sns.pairplot(data, hue="Outcome") 


# * With pairplot function we see the relation on each 2 feature pairs. It gives us an idea that can we seperate data with a linear seperator or should we use soft margin or should we use one of tree based algorithm. Except diagonal,figure above demonstrate, linear seperator will not be able to seperate well.Even so we will apply these algorithms.****

# # Data Preprocessing

# ## Feature Scaling

#  Scaling one of the most important part about preprocessing.Not only this database but also others,different features might have different scales. Assume that we have a dataset about vehicles. Production year is important. Speed is important also. Let us say production year is 2010 , speed is 280 km / h price is 100.000$.
#  Second car one has year is 2020, speed is 280 kmh price is 220000 \$.As the third option Year is 2010 speed , is 290 km / h and price is 120.000 \$. Are the difference of year and speed same? Changing of 10 unit affects our cost different.Because year scale has just 4 digit and scale of price has on average 6 or more.In This and similar situations we should apply feature scaling. (Normalization & Standardization mostly)

# ## Standardization
# ![Standardization](https://i.hizliresim.com/1zOWgC.png)
# ## Normalization
# ![Normalization](http://i.hizliresim.com/DG8T4r.jpg)
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_scaled=scaler.fit_transform(X)
data_scaled=pd.concat([pd.DataFrame(x_scaled,columns=data.iloc[:,:-1].columns),y],axis=1)


# Generally we use standardization (sklearn standardscaler) instead of normalization (min-max scaling) Because standardization function recreates number around dataset. Normalization gives around -1,1. So out of exceptional datasets I suggest standardization. 

# ## Dimension Reduction

# In dataset, we might get rid of unnecessary features. Actually besides being unnecessary they also might decrease our accurracy.So we need to remove them. One of the most famous dimension reduction algorithm is Principal Component Analysis (PCA). We are going to use this algorithm today.

# ## How PCA Works
# ![How PCA Works](https://i.hizliresim.com/CqpxsW.png)

# Basically PCA tries to represent 2 dimensional feature as one feature.If data has more than 2 features,pca applies this algorithm same dimensional pairs .Some points could not represents but we need to  both save time and hardware resources.

# In[ ]:


from sklearn.decomposition import PCA
pca=PCA().fit(x_scaled)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title("How many variable represents our model with acceptable error PCA")
plt.xlabel("Number of variable")
plt.ylabel("Variance")
plt.grid()


# In[ ]:


pca=PCA(n_components=5)
x_reduced=pca.fit_transform(data_scaled.iloc[:,:-1])
data_reduced=pd.concat([pd.DataFrame(x_reduced),y],axis=1)
data_reduced.head()


# In[ ]:


before_x=[]
after_x=[]
data4iqr=data.copy()
for i in range(len(data4iqr.columns)-1):
    col=data4iqr.iloc[:,i:i+1]

    Q1=col.quantile(0.25)
    Q3=col.quantile(0.75)
    IQR=Q3-Q1
    lower=Q1-1.5*IQR
    upper=Q3+1.5*IQR
    new_col=col[~((col<lower)|(col>upper)).any(axis=1)]
    ex_col=col[((col<lower)|(col>upper)).any(axis=1)]
    before_x.append(col)
    data4iqr.drop(index=ex_col.index,axis=0,inplace=True)
    after_x.append(data4iqr.iloc[:,i:i+1])
data4iqr.reset_index(inplace=True)
print("IQR METHOD",len(data)-len(data4iqr)," Row Effected")
####IQR Visualization####
f, axes = plt.subplots(2,5, figsize=(22, 7))
j=0
for i in range(5):
    if data4iqr.columns[i+1]=="Outcome":
        continue
    sns.boxplot(before_x[i],ax=axes[0,j]).set_title(data4iqr.columns[i+1]+" Before IQR")
    sns.boxplot(after_x[i],ax=axes[1,j]).set_title(data4iqr.columns[i+1]+" After IQR")
    j+=1
plt.show()


# IQR method helps us removing outliers. It checks each column,finds standard deviation and remove which are out of (mean + 2 std) and (mean - 2std) 

# ![Gaussian Distrubution](https://i.hizliresim.com/G8HuaU.jpg)

# Iqr select %95.4 (if feature distr normal) You should not keep outliers whether if they are real. Most machine learning algorithms are not suitable for kind of outliers.

# ## Local Outlier Factor

# In[ ]:


from sklearn.neighbors import LocalOutlierFactor
clf=LocalOutlierFactor(n_neighbors=20,contamination=0.1)
outlier_pred=clf.fit_predict(data_reduced)
x_score=clf.negative_outlier_factor_
x_score=np.abs(x_score)
xscr_mean=x_score.mean()
xscr_std=np.std(x_score)
lower=xscr_mean-(1*xscr_std)
upper=xscr_mean+(1*xscr_std)


# Lof is one of the most effective outlier detection algorithm. It is an unsupervised method which finds an optimum cluster. We chose number of neighbors 20.

# In[ ]:


inliers=data[~((x_score>upper)| (x_score<lower))]
print(len(inliers))
lof_data=inliers.copy()


# Now we have 694 rows. Lof and Iqr are althernative methods, I select lof because generally iqr using if data has one independent variable

# In[ ]:


from sklearn import metrics
from sklearn.model_selection import GridSearchCV,cross_val_score


# In[ ]:


columns=set(lof_data.columns)
columns.remove('Outcome')
x_reduced=lof_data[columns]
y=lof_data['Outcome']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_reduced,y,test_size=0.33,random_state=58)


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression as log_rec
logistic_model=log_rec(C=0.0001,solver='newton-cg')
logistic_model.fit(x_train,y_train)
logistic_pred=logistic_model.predict(x_test)
print("Logistic Regression Accuracy Score Before Tuning %.5f"% metrics.accuracy_score(logistic_pred,y_test))
print("Logistic Regression F1 Score Before Tuning %.5f"%metrics.f1_score(logistic_pred,y_test))


# Logistic Regression is one of the simplest binary classification algorithm. We are going to tune its parameters.

# ## Logistic Regression Hyperparameter Tuning

# In[ ]:


logistic_params={'C':[0.001,0.01,0.1,1,10,100,1000],'solver':[ "liblinear", "sag", "saga","lbfgs"]}
grid=GridSearchCV(log_rec(),logistic_params,scoring='accuracy',cv=3)
grid.fit(x_train,y_train)
grid.best_params_


# In[ ]:


logistic_model=log_rec(C=grid.best_params_['C'],solver=grid.best_params_['solver'])
logistic_model.fit(x_train,y_train)
logistic_pred=logistic_model.predict(x_test)
print("Logistic Regression Accuracy Score After Tuning %.5f"%metrics.accuracy_score(logistic_pred,y_test))
print("Logistic Regression F1 Score After Tuning %.5f"%metrics.f1_score(logistic_pred,y_test))


# # K- Nearest Neighbors Algorithm (KNN)

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train,y_train)
knn_pred=knn_model.predict(x_test)
print("KNN Accuracy Score Before Tuning %.5f"% metrics.accuracy_score(knn_pred,y_test))
print("KNN F1 Score Before Tuning %.5f"% metrics.f1_score(knn_pred,y_test))


# ## K Nearest Neighbors Hyperparameter Tuning

# In[ ]:


knn_params={'n_neighbors':np.arange(3,90,2)}
grid=GridSearchCV(KNeighborsClassifier(),knn_params,scoring='accuracy',cv=3)
grid.fit(x_train,y_train)
grid.best_params_


# In[ ]:


knn_model=KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'])
knn_model.fit(x_train,y_train)
knn_pred=knn_model.predict(x_test)
print("KNN Accuracy Score After Tuning %.5f"% metrics.accuracy_score(knn_pred,y_test))
print("KNN F1 Score After Tuning %.5f" % metrics.f1_score(knn_pred,y_test))
#print(knn_model.predict_proba(x_test[3:5]))


# # Decision Tree
# 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier()
dt_model.fit(x_train,y_train)
dt_pred=dt_model.predict(x_test)
print("Decision Tree Accuracy Score Before Tuning %.5f"% metrics.accuracy_score(dt_pred,y_test))
print("Decision Tree F1 Score Before Tuning %.5f"% metrics.f1_score(dt_pred,y_test))


# ## Decision Tree Hyperparameter Tuning

# In[ ]:


dt_params={'criterion':['gini','entropy'],'max_depth':(2,4,6,8,10,12,16,18,20)}
grid=GridSearchCV(DecisionTreeClassifier(),dt_params,scoring='accuracy')
grid.fit(x_train,y_train)
grid.best_params_


# In[ ]:


dt_model=DecisionTreeClassifier(criterion=grid.best_params_['criterion'],max_depth=4)
dt_model.fit(x_train,y_train)
dt_pred=dt_model.predict(x_test)
print("Decision Tree Accuracy Score After Tuning %.5f"% metrics.accuracy_score(dt_pred,y_test))
print("Decision Tree F1 Score After Tuning %.5f"% metrics.f1_score(dt_pred,y_test))


# ## Visualization Of Decision Tree

# In[ ]:



from sklearn import tree
plt.figure(figsize=(60,40),dpi=400)
tree.plot_tree(dt_model,filled=True,rounded=True,feature_names=X.columns,
            class_names=['Diabetes','No Diabetes'])
plt.show()
#plt.savefig("tree_visual.png")


# # Support Vector Machine(s)

# In[ ]:


from sklearn.svm import SVC
svc_model=SVC()
svc_model.fit(x_train,y_train)
svc_pred=svc_model.predict(x_test)
print("SVM Accuracy Score Before Tuning %.5f"% metrics.accuracy_score(svc_pred,y_test))
print("SVM F1 Score Before Tuning %.5f"%metrics.f1_score(svc_pred,y_test))


# ## SVM HyperParameter Tuning

# In[ ]:


svc_params=({'kernel':['rbf'],'C':[0.001,0.1,1,10,100],'gamma':['auto','scale']})
grid=GridSearchCV(SVC(),param_grid=svc_params,scoring="accuracy",cv=3)
grid.fit(x_train,y_train)
grid.best_params_


# In[ ]:


svc_model=SVC(C=grid.best_params_['C'],kernel=grid.best_params_['kernel'],gamma=grid.best_params_['gamma'])
svc_model.fit(x_train,y_train)
svc_pred=svc_model.predict(x_test)
print("SVM Accuracy Score After Tuning %.5f"% metrics.accuracy_score(svc_pred,y_test))
print("SVM  F1 Score After Tuning %.5f"%metrics.f1_score(svc_pred,y_test))


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier()
rf_model.fit(x_train,y_train)
rf_pred=rf_model.predict(x_test)
print("RandomForest Accuracy Score Before Tuning %.5f"% metrics.accuracy_score(rf_pred,y_test))
print("RandomForest F1 Score Before Tuning %.5f"%metrics.f1_score(rf_pred,y_test))


# ## Random Forest Tuning

# In[ ]:


rf_params={'n_estimators':range(10,110,10),'criterion':['gini','entropy']}
grid=GridSearchCV(RandomForestClassifier(),rf_params,cv=3,scoring='accuracy')
grid.fit(x_train,y_train)
grid.best_params_


# In[ ]:


rf_model=RandomForestClassifier(n_estimators=grid.best_params_['n_estimators'],criterion=grid.best_params_['criterion'],max_depth=4)
rf_model.fit(x_train,y_train)
rf_pred=rf_model.predict(x_test)
print("RandomForest Accuracy Score After Tuning %.5f"% metrics.accuracy_score(rf_pred,y_test))
print("RandomForest F1 Score After Tuning %.5f"%metrics.f1_score(rf_pred,y_test))


# # XGBoost

# In[ ]:


from xgboost import XGBClassifier
xg_model=XGBClassifier()
xg_model.fit(x_train,y_train)
xg_pred=xg_model.predict(x_test)
print("XGBoost Accuracy Score Before Tuning %.5f"% metrics.accuracy_score(xg_pred,y_test))
print("XGBoost F1 Score Before Tuning %.5f"%metrics.f1_score(xg_pred,y_test))


# # Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb_model=GaussianNB()
nb_model.fit(x_train,y_train)
nb_pred=nb_model.predict(x_test)
print("NaiveBayes Accuracy Score  %.5f"% metrics.accuracy_score(nb_pred,y_test))
print("NaiveBayes F1 Score  %.5f"%metrics.f1_score(nb_pred,y_test))


# # K-means 

# Kmeans is a unsupervised algorithm. We will not give train labels and wait model to outputs.

# In[ ]:


from sklearn.cluster import KMeans
km_model=KMeans(n_clusters=2)
km_model.fit(x_train)
km_pred=km_model.predict(x_test)
if metrics.accuracy_score(km_pred,y_test)<0.5:
    zeros=np.where(km_pred==0)
    ones=np.where(km_pred==1)
    km_pred[zeros]=1
    km_pred[ones]=0
metrics.accuracy_score(km_pred,y_test)


# In[ ]:


km_params={'algorithm':["auto", "full", "elkan"],'max_iter':[100,200,300,400,500,600],'init':['k-means++','random']}
grid=GridSearchCV(KMeans(n_clusters=2,random_state=12),km_params,scoring='accuracy',cv=3)
grid.fit(x_train,y_train)
grid.best_params_


# In[ ]:


km_model=KMeans(n_clusters=2,init=grid.best_params_['init'],algorithm=grid.best_params_['algorithm'],max_iter=grid.best_params_['max_iter'])
km_model.fit(x_train)
km_pred=km_model.predict(x_test)
if metrics.accuracy_score(km_pred,y_test)<0.5:
    zeros=np.where(km_pred==0)
    ones=np.where(km_pred==1)
    km_pred[zeros]=1
    km_pred[ones]=0
metrics.accuracy_score(km_pred,y_test)


# # Artificial Neural Network with Backpropagation

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# In[ ]:


model = Sequential()
model.add(Dense(60, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=50)


# In[ ]:


eval_score=model.evaluate(x_train, y_train)
print("Loss:",eval_score[0],"Accuracy:",eval_score[1])


# # Majority Voting

# With this approach, we choose independent and succesfull models and each one has a vote. Hard voting means select which model choose more. Soft voting based on averaging (Mostly using for regression).

# In[ ]:


from sklearn.ensemble import VotingClassifier 


# In[ ]:


est=[]
est.append(('svm',svc_model))
est.append(('rf',rf_model))
est.append(('lr',logistic_model))
est.append(('nb',nb_model))
est.append(('xg',xg_model))
vcls=VotingClassifier(estimators=est,voting='hard')
vcls.fit(x_train,y_train)


# In[ ]:


voting_pred=vcls.predict(x_test)
print("Voting Accuracy Score  %.5f"% metrics.accuracy_score(voting_pred,y_test))
print("Voting F1 Score  %.5f"%metrics.f1_score(voting_pred,y_test))

