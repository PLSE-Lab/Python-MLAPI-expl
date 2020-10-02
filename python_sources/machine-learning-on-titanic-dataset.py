#!/usr/bin/env python
# coding: utf-8

# ## Titanic Dataset Analysis
# 
# > Here the standard titanic dataset has been analysed and reuslts have been portrayed using visualisations along with statistical inferences. Also different models have been created using different algorithms and their accuracy has been printed. 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Assessing</a></li>
# <li><a href="#wrangling">Cleaning</a></li>
# <li><a href="#wrangling">Feature Engineering</a></li>    
# <li><a href="#eda">Exploratory Analysis</a></li>
# <li><a href="#eda">Regression Analysis</a></li>
# <li><a href="#eda">Predictive Analysis</a></li>    
# <li><a href="#conclusions">Conclusions</a></li>    
# </ul>

# ## Introduction
# 
# ### Describing the meaning of the features given the both train & test datasets.
# <h4>Variable Definition Key.</h4>
#  
# > - Survival
# 
# > - 0= No
# > - 1= Yes
# 
# > - pclass (Ticket class)
# 
# > - 1=1st
# > - 2=2nd
# > - 3=3rd
#  
# - sex
# <br>
# 
# - age
# 
# 
# - sibsp (# of siblings / spouses aboard the Titanic)
# <br>
# - parch (# of parents / children aboard the Titanic)
# <br>
# - tickets
# <br>
# - fare
# <br>
# - cabin
# - embarked Port of Embarkation.
#  - C = Cherbourg,
#  - Q = Queenstown,
#  - S = Southampton
# - pclass: A proxy for socio-economic status (SES)
# <br>
# <h4>Passenger Class.</h4>
# 
# > - 1st = Upper
# > - 2nd = Middle
# > - 3rd = Lower
# 

# In[ ]:


# Importing relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Loading the data set into a pandas dataframe
gender=pd.read_csv('../input/gender_submission.csv')
test=pd.read_csv('../input/test.csv')
train=pd.read_csv('../input/train.csv')


# ## Assessing the data

# In[ ]:


gender.info()


# In[ ]:


train.info()


# In[ ]:


train.describe(include='all')


# In[ ]:


# Assessing the first few columns.
train.head()


# ### Dataset Issues
# 
# > PassengerId column is of int type.
# 
# > Cabin information missing for most of the passengers.
# 
# > For the age column also dataset is missing which can prove to be a deciding factor in predicting survival.
# 
# > Two passengers having no information of Embarked ports.
# 

# ## Cleaning the Dataset

# ### Define
# 
# > PasseingerId column is of int type rather it should be of str type.

# #### Code

# In[ ]:


train['PassengerId']=train['PassengerId'].astype(str)


# #### Test

# In[ ]:


train.info()


# In[ ]:


train=train.drop(['Cabin'],axis=1)


# In[ ]:


df1=train[['Survived','Pclass','Age','SibSp','Parch','Fare']]


# In[ ]:


df1.head(5)


# In[ ]:


from fancyimpute import KNN


# In[ ]:


def knn(t,i):
    z=t
    z.loc[0,i]=np.NaN
    z=pd.DataFrame(KNN(k=3).fit_transform(z),columns=z.columns)
    return(z.loc[0,i])


# In[ ]:


def mean(t,i):
    z=t
    z.loc[0,i]=np.NaN
    z=z.loc[:,i].fillna(z.loc[:,i].mean())
    return(z[0])


# In[ ]:


def median(t,i):
    z=t
    z.loc[0,i]=np.NaN
    z=z.loc[:,i].fillna(z.loc[:,i].median())
    return(z[0])


# In[ ]:


# Function for imputing the missing values.
# Here we have first stored a non null value of a particular column stored it in a separate variable and replaced it in the dataframe with nan.
# Then we have imputed the missing value using the mean and median and depending upon which method imputes the value closes to the actual value is used for imputing the missing values in the dataset.
def impute(t):
    for i in miss_val:
            if(sum(t.loc[:,i].isnull())!=0):
                p=mean(t,i)
                q=median(t,i)
                r=knn(t,i)
                if(abs(p-t.loc[0,i]) < abs(q-t.loc[0,i]) and abs(p-t.loc[0,i]) < abs(r-t.loc[0,i])):
                    t.loc[:,i]=t.loc[:,i].fillna(t.loc[:,i].mean())
                elif(abs(q-t.loc[0,i]) < abs(p-t.loc[0,i]) and abs(q-t.loc[0,i]) < abs(r-t.loc[0,i])):
                    t.loc[:,i]=t.loc[:,i].fillna(t.loc[:,i].median())
                else:
                    t=pd.DataFrame(KNN(k=3).fit_transform(t),columns=t.columns)
            else:
                continue
    return(t)      


# In[ ]:


miss_val=['Age']
df=impute(df1)


# In[ ]:


df.info()


# In[ ]:


train.loc[:,['Survived','Pclass','Age','SibSp','Parch','Fare']]=df


# In[ ]:


train.info()


# In[ ]:


train['Embarked'].value_counts()


# In[ ]:


k=train


# In[ ]:


train[train['Embarked'].isnull()]


# In[ ]:


train.loc[:,'Embarked']=train.loc[:,'Embarked'].fillna('S')


# In[ ]:


train.info()


# #### The task can be completed either using loops or functions. Functions have been used here to reduce the execution time.

# ### Cleaning of testing dataset

# In[ ]:


test.info()


# In[ ]:


test=test.drop(['Cabin'],axis=1)
test['PassengerId']=test['PassengerId'].astype(str)


# In[ ]:


test.info()


# In[ ]:


miss_val=['Age','Fare']


# In[ ]:


df1=test[['Pclass','Age','SibSp','Parch','Fare']]


# In[ ]:


df=impute(df1)


# In[ ]:


df.info()


# In[ ]:


test.loc[:,['Pclass','Age','SibSp','Parch','Fare']]=df


# In[ ]:


test.info()


# In[ ]:


#train=k
train.info()


# In[ ]:


train['Survived']=train['Survived'].astype(int)
train['Pclass']=train['Pclass'].astype(int)
train['SibSp']=train['SibSp'].astype(int)
train['Parch']=train['Parch'].astype(int)
test['Pclass']=test['Pclass'].astype(int)
test['SibSp']=test['SibSp'].astype(int)
test['Parch']=test['Parch'].astype(int)


# In[ ]:


train=train.drop(['Ticket'],axis=1)
test=test.drop(['Ticket'],axis=1)


# ## Exploring the dataset

# In[ ]:


bins=np.arange(0,train.Age.max()+10,10)
plt.hist(train['Age'],rwidth=0.6,bins=bins)
plt.title('Distribution of the dataset according to Age.')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show();


# > From the above histogram it is clear that majority of the dataset contains people of age between 20 to 30, followed by those between 30 and 40. 

# In[ ]:


bins=np.arange(0,train.Fare.max()+10,50)
plt.hist(train['Fare'],rwidth=0.6,bins=bins)
plt.title('Distribution of the dataset according to Fare.')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show();


# In[ ]:


bins=np.arange(0,train.SibSp.max()+1,1)
plt.hist(train['SibSp'],rwidth=0.6,bins=bins)
plt.title('Distribution of the dataset according to SibSp.')
plt.xlabel('SibSp')
plt.ylabel('Count')
plt.show();


# In[ ]:


bins=np.arange(0,train.Parch.max()+1,1)
plt.hist(train['Parch'],rwidth=0.6,bins=bins)
plt.title('Distribution of the dataset according to Parch.')
plt.xlabel('Parch')
plt.ylabel('Count')
plt.show();


# In[ ]:


sb.countplot(data=train,x='Survived',hue='Sex');


# > The bar chart above compares the passengers survived or not on gender basis.
# 
# > From above it is clear that among those who survived females are high in number as compared to males and among who did not survive males are high in number.

# In[ ]:


sb.countplot(data=train,x='Survived',hue='Pclass');


# > The bar chart above compares the passengers survived or not on socio economic status.
# 
# > The above chart shows that first class passengers were preferred over other passsengers at the time of rescue,which is quite evident from above graph as the number of third class passengers are high among those who did not survive.
# 

# In[ ]:


plt.figure(figsize=[12,7])
sb.violinplot(data=train,x='Pclass',y='Fare')
plt.show();


# In[ ]:


# Analysing each continuous variable for presence of outliers using boxplot.
plt.figure(figsize=[10,5])
plt.boxplot([train['Age'],train['Fare'],train['SibSp'],train['Parch']])
plt.xlabel(['1. Age', '2. Fare', '3. SibSp', '4. Parch'])
plt.title("BoxPlot of the continuous Variables")
plt.ylabel('Values');


# In[ ]:


train.info()


# In[ ]:


df1=train[['Age','SibSp','Parch','Fare','Survived']]


# In[ ]:


df1.head(3)


# In[ ]:


from scipy import stats
cnames=['Age','SibSp','Parch','Fare']
for i in cnames:
    f, p = stats.f_oneway(df1[i], df1['Survived'])
    print("P value for variable "+str(i)+" is "+str(p))


# In[ ]:


f, ax = plt.subplots(figsize=(20, 8))
corr = df1.corr()
sb.heatmap(corr, mask=np.zeros_like(corr,dtype=np.bool),cmap=sb.diverging_palette(220, 10, as_cmap=True),annot=True,ax=ax,);


# In[ ]:


df1['logage']=np.log(df1['Age'])


# In[ ]:


df1['sqrtfare']=np.sqrt(df1['Fare'])


# In[ ]:


df1['sqrtage']=np.sqrt(df1['Age'])


# In[ ]:


df1.head(2)


# In[ ]:


from scipy import stats
cnames=['Age','SibSp','Parch','Fare','sqrtfare','logage','sqrtage']
for i in cnames:
    f, p = stats.f_oneway(df1[i], df1['Survived'])
    print("P value for variable "+str(i)+" is "+str(p))


# In[ ]:


from scipy.stats import chi2_contingency


# In[ ]:


train['Pclass']=train['Pclass'].astype(str)
cat_names=['Embarked','Sex','Pclass']
for i in cat_names:
    print(i)
    chi2,p,dof,ex=chi2_contingency(pd.crosstab(train['Survived'],train[i]))
    print(p)


# In[ ]:


train['log_age']=np.log(train['Age'])
train['sqrt_age']=np.sqrt(train['Age'])
train['sqrt_fare']=np.sqrt(train['Fare'])
test['log_age']=np.log(test['Age'])
test['sqrt_age']=np.sqrt(test['Age'])
test['sqrt_fare']=np.sqrt(test['Fare'])


# ## Preprocessing the for performing logistics regression 

# In[ ]:


train.info()


# In[ ]:


k=train['Embarked'].value_counts()
k.plot(kind='pie',figsize=(20,10),legend=True)
plt.legend(loc=0,bbox_to_anchor=(1.5,0.5));


# In[ ]:


k=train['Sex'].value_counts()
k.plot(kind='pie',figsize=(20,10),legend=True)
plt.legend(loc=0,bbox_to_anchor=(1.5,0.5));


# In[ ]:


k=train['Pclass'].value_counts()
k.plot(kind='pie',figsize=(20,10),legend=True)
plt.legend(loc=0,bbox_to_anchor=(1.5,0.5));


# In[ ]:


train.head()


# In[ ]:


df=train[['Survived','Pclass','Sex','Embarked','Age','SibSp','Parch','Fare','log_age','sqrt_age','sqrt_fare']]
df=pd.get_dummies(df)


# In[ ]:


df.info()


# In[ ]:


df=df.drop(['Embarked_Q','Sex_female','Pclass_2'],axis=1)


# In[ ]:


df.columns


# In[ ]:


bins=np.arange(0,train.sqrt_fare.max()+1,1)
plt.hist(train['sqrt_fare'],rwidth=0.6,bins=bins)
plt.title('Distribution of the dataset according to Age.')
plt.xlabel('sqrt_fare')
plt.ylabel('Count')
plt.show();


# In[ ]:


bins=np.arange(0,train.sqrt_age.max()+1,1)
plt.hist(train['sqrt_age'],rwidth=0.6,bins=bins)
plt.title('Distribution of the dataset according to Age.')
plt.xlabel('sqrt_age')
plt.ylabel('Count')
plt.show();


# In[ ]:


X=df[['Age', 'SibSp', 'sqrt_fare', 'Pclass_1', 'Pclass_3', 'Sex_male','Embarked_C', 'Embarked_S']]
y=df['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


# ## Logistic Regression using different algorithms and their accuracy on the test set.

# ## Linear Model

# In[ ]:


from sklearn.linear_model import LogisticRegression
clf_LR = LogisticRegression(random_state = 123,tol=1e-3,C=1,solver='lbfgs')
clf_LR.fit(X_train, y_train)

# Predicting the Test set results
y_pred_LR = clf_LR.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_LR)
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred_LR))


# In[ ]:


cm


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred_LR)


# ### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


clf_DT=DecisionTreeClassifier(criterion='entropy',random_state=123,)
clf_DT.fit(X_train,y_train)


# In[ ]:


y_pred_DT = clf_DT.predict(X_test)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred_DT))


# In[ ]:


cm = confusion_matrix(y_test, y_pred_DT)
cm


# In[ ]:


r2_score(y_test, y_pred_DT)


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


clf_RF=RandomForestClassifier(n_estimators=5000,criterion='gini')
clf_RF.fit(X_train,y_train)


# In[ ]:


y_pred_RF = clf_RF.predict(X_test)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred_RF))


# In[ ]:


cm = confusion_matrix(y_test, y_pred_RF)
cm


# In[ ]:


r2_score(y_test, y_pred_RF)


# ### K Nearest Neighbors(KNN)

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


clf_knn=KNeighborsClassifier(n_neighbors=3,algorithm='ball_tree')
clf_knn.fit(X_train,y_train)


# In[ ]:


y_pred_knn = clf_knn.predict(X_test)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred_knn))


# In[ ]:


cm = confusion_matrix(y_test, y_pred_knn)
cm


# In[ ]:


r2_score(y_test,y_pred_knn)


# ### SVM Kernel

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


clf_svm=SVC(kernel='rbf',random_state=123,C=100)
clf_svm.fit(X_train,y_train)


# In[ ]:


y_pred_svc = clf_svm.predict(X_test)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred_svc))


# In[ ]:


cm = confusion_matrix(y_test, y_pred_svc)
cm


# In[ ]:


r2_score(y_test,y_pred_svc)


# ### Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


clf_NB=GaussianNB()
clf_NB.fit(X_train,y_train)


# In[ ]:


y_pred_NB = clf_NB.predict(X_test)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred_NB))


# In[ ]:


cm = confusion_matrix(y_test, y_pred_NB)
cm


# In[ ]:


r2_score(y_test,y_pred_NB)


# In[ ]:


from xgboost import XGBClassifier
clf_XGB=XGBClassifier()


# In[ ]:


clf_XGB.fit(X_train,y_train)


# In[ ]:


y_pred_XGB = clf_XGB.predict(X_test)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred_XGB))


# In[ ]:


cm = confusion_matrix(y_test, y_pred_XGB)
cm


# In[ ]:


r2_score(y_test,y_pred_XGB)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


# Creating a dictionary of parameters for tuning Decision Tree
z={'max_depth':[2,3,5,7,10],'min_samples_leaf':[2,5,7,10,15],'min_samples_split':[2,5,7,10],'max_features':
   ['auto','sqrt','log2'],'criterion':['entropy','gini']}


# In[ ]:


classifier = RandomizedSearchCV(clf_DT,param_distributions=z, random_state=1)


# In[ ]:


best_model = classifier.fit(X_train, y_train)


# In[ ]:


best_model.best_params_


# In[ ]:


clf_DT=DecisionTreeClassifier(criterion='gini',min_samples_split=2,min_samples_leaf=5,max_features='log2',
                              max_depth=3,random_state=123)
clf_DT.fit(X_train,y_train)


# In[ ]:


y_pred_DT = clf_DT.predict(X_test)


# In[ ]:


print(metrics.accuracy_score(y_test,y_pred_DT))


# In[ ]:


cm = confusion_matrix(y_test, y_pred_DT)
cm


# In[ ]:


r2_score(y_test,y_pred_DT)


# In[ ]:


# Creating a dictionary of parameters for tuning Random Forest
z={'n_estimators':[1000,5000,10000,50000],'max_depth':[2,3,5,7,10,15],'min_samples_leaf':[2,3,5,7,10,15],'min_samples_split':[2,3,5,7,10,15],'max_features':
   ['auto','sqrt','log2'],'oob_score':[True],'n_jobs':[-1],'criterion':['entropy','gini']}


# In[ ]:


classifier = RandomizedSearchCV(clf_RF,param_distributions=z, random_state=1)


# In[ ]:


best_model = classifier.fit(X_train, y_train)


# In[ ]:


best_model.best_params_


# In[ ]:


clf_RF=RandomForestClassifier(n_estimators=1000,max_depth=15,min_samples_leaf=2,min_samples_split=10,n_jobs=-1,
                              oob_score=True,max_features='log2',criterion='gini')
clf_RF.fit(X_train,y_train)


# In[ ]:


y_pred_RF = clf_RF.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred_RF))
cm = confusion_matrix(y_test, y_pred_RF)
r2_score(y_test,y_pred_RF)


# In[ ]:


# Creating a dictionary of parameters for tuning Support Vector Machines
z={'C':[1,10,50,70,100,1000],'gamma':[0.01,0.1,0.05,0.5,1],'kernel':['linear','rbf','poly'],
   'tol':[0.001,0.05,0.005,0.01]}


# In[ ]:


classifier = RandomizedSearchCV(clf_svm,param_distributions=z, random_state=123)


# In[ ]:


best_model = classifier.fit(X_train, y_train)


# In[ ]:


best_model.best_params_


# In[ ]:


clf_svm=SVC(n_estimators=1000,max_depth=15,min_samples_leaf=2,min_samples_split=10,n_jobs=-1,
                              oob_score=True,max_features='log2',criterion='gini')
clf_svm.fit(X_train,y_train)


# In[ ]:


y_pred_svc = clf_svm.predict(X_test)


# In[ ]:


print(metrics.accuracy_score(y_test,y_pred_svc))
cm = confusion_matrix(y_test, y_pred_svc)
r2_score(y_test,y_pred_svc)


# In[ ]:


# Creating a dictionary of parameters for tuning XGBoost Classfier
z={'n_estimators':[100,500,1000,2500,5000],'learning_rate':[0.001,0.005,0.01,0.05,0.1,0.5]}


# In[ ]:


classifier = RandomizedSearchCV(clf_XGB,param_distributions=z, random_state=123)


# In[ ]:


best_model = classifier.fit(X_train, y_train)


# In[ ]:


best_model.best_params_


# In[ ]:


clf_XGB=XGBClassifier(n_estimators=1000,learning_rate=)
clf_XGB.fit(X_train,y_train)


# In[ ]:


y_pred_XGB = clf_XGB.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred_XGB))
cm = confusion_matrix(y_test, y_pred_XGB)
r2_score(y_test,y_pred_XGB)


# In[ ]:





# In[ ]:


test['Pclass']=test['Pclass'].astype(str)


# In[ ]:


test.info()


# In[ ]:


test.columns


# In[ ]:


df=test[['Age', 'SibSp', 'sqrt_fare', 'Pclass', 'Sex','Embarked']]
df=pd.get_dummies(df,drop_first=True)


# In[ ]:


df.columns


# In[ ]:


X_test=df[['Age', 'SibSp', 'sqrt_fare', 'Pclass_2', 'Pclass_3', 'Sex_male','Embarked_Q', 'Embarked_S']]


# In[ ]:


predictions=clf_NB.predict(X_test)


# In[ ]:


Y=gender['Survived']


# In[ ]:


print(metrics.accuracy_score(Y,predictions))


# In[ ]:


r2_score(Y,predictions)


# In[ ]:


submiss=gender


# In[ ]:


submiss['Survived']=predictions


# In[ ]:


submiss.info()


# In[ ]:


submiss.to_csv('mycsvfile.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


predictions=y_pred.to_csv()


# ## Conclusion

# > The histogram for age distribution shows that majority of the dataset contains people of age between 20 to 30, followed by those between 30 and 40. 
# 
# > First class passengers were preferred over other passsengers at the time of rescue,as the number of third class passengers are high among those who did not survive.
# 
# > The violinplot shows that distribution of fares of passengers depending upon their socio-economic status (Pclass).From the plot it is clear that passengers belonging to first class have the highest fares along with presence of outliers beyond 500. Those belonging to second class have their fares between 0 to 100.And those belonging to third class majority of their fares are below 50 and the distribution of fare is unimodal. 
# Thus it can be concluded that fares for passengers of First class are higher than that of other class also they belong to the upper class of society. 
# 
# > In the above regression analysis firstly regression has been performed on three features namely gender,age and passenger class where the accuracy of training and testing set were 79.6% and 73.9% respectively.
# 
# > Here different models have been built using different algorithms.On comparing the accuracies of different models on test set we can see that naive bayes model has the maximum accuracy of about 99.5%.
