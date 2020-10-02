#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv("../input/StudentsPerformance.csv")


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.columns


# In[ ]:


data.mean()


# In[ ]:


data.median()


# In[ ]:


data.var()


# In[ ]:


data.sum()


# In[ ]:


passmark = 40


# ## Check Null Data

# In[ ]:


#data.isnull().sum()
data.isnull().any()


# ### Explore Math Score

# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x ="math score", data= data,palette="muted")


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x ="reading score", data= data,palette="muted")


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x ="writing score", data= data,palette="muted")


# In[ ]:


sns.countplot(data['gender'])


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x ="race/ethnicity", data= data,palette="muted")


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x ="parental level of education", data= data,palette="muted")


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x = 'math score' ,hue ="parental level of education", data= data,palette="muted")


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x ="lunch", data= data,palette="muted")


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x ="test preparation course", data= data,palette="muted")


# ## Pass And fail in Reading, writing and Math Score 

# In[ ]:


data['Math_Pass'] = np.where(data['math score'] < passmark,'Fail', 'Pass')


# In[ ]:


data['Reading_Pass'] = np.where(data['reading score'] < passmark,'Fail', 'Pass')


# In[ ]:


data['Writing_Pass'] = np.where(data['writing score'] < passmark,'Fail', 'Pass')


# In[ ]:


del data['writing_pass']
del data['reading_Pass']
data.head()


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(data['Math_Pass'])


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(data['Reading_Pass'])


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(data['Writing_Pass'])


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x = 'parental level of education' , hue = 'Math_Pass', data=data)


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x = 'parental level of education' , hue = 'Reading_Pass', data=data)


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x = 'parental level of education' , hue = 'Writing_Pass', data=data)


# In[ ]:


plt.figure(figsize=(10,6))
sns.stripplot(x = 'parental level of education' , hue = 'Math_Pass', data=data) #swarmplot


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x = 'race/ethnicity' , hue = 'Writing_Pass', data=data)


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x = 'race/ethnicity' , hue = 'Reading_Pass', data=data)


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x = 'race/ethnicity' , hue = 'Math_Pass', data=data)


# In[ ]:


x = data['reading score'] > 40


# In[ ]:


sns.countplot(x)


# In[ ]:


print(data['Math_Pass'].value_counts())
print('_' * 30)
print(data['Reading_Pass'].value_counts())
print('_' * 30)
print(data['Writing_Pass'].value_counts())


# In[ ]:


plt.figure(figsize=(15,10))
x = data['math score']
sns.distplot(x)


# In[ ]:


plt.figure(figsize=(15,10))
x = data['reading score']
sns.distplot(x)


# In[ ]:


plt.figure(figsize=(15,10))
x = data['writing score']
sns.distplot(x)


# In[ ]:


plt.figure(figsize=(15,10))
sns.regplot(x='math score',y ='reading score', data=data, color='red')


# In[ ]:


plt.figure(figsize=(25,10))
sns.jointplot(x='Total_Marks',y ='reading score', data=data, color='red',kind='reg')


# In[ ]:


plt.figure(figsize=(25,10))
sns.boxplot(data['math score'],data['reading score'], data=data)
sns.stripplot(data['math score'],data['reading score'], data=data,  jitter=True, edgecolor="gray")


# In[ ]:


sns.heatmap(df.corr(),annot = True,linewidths = 0.5,cmap='cubehelix_r');


# In[ ]:


plt.figure(figsize=(15,10))
sns.regplot(x='math score',y ='writing score', data=data, color='green')


# In[ ]:


data['OverAll_Pass'] = data.apply(lambda x : 'Fail' if x['Math_Pass'] == 'Fail' or 
                                    x['Reading_Pass'] == 'Fail' or x['Writing_Pass'] == 'Fail' else 'Pass', axis =1)

data.OverAll_Pass.value_counts()


# In[ ]:


plt.figure(figsize=(18,10))
sns.countplot(x='parental level of education', data = data, hue='OverAll_Pass', palette='bright')


# In[ ]:


data['Total_Marks'] = data['math score'] + data['reading score'] + data['writing score']


# In[ ]:


data['Percentage'] = data['Total_Marks'] / 3


# In[ ]:


data.head()


# In[ ]:



plt.figure(figsize=(18,6))
sns.distplot(data['Percentage'])


# ## Assign The Grades
# 
# ### Grading
# 
# ### #Above 90 = A+ Grade
# 
# ### Above 80 = A Grade
# 
# ### 70 to 80 = B Grade
# 
# ### 60 to 70 = C Grade
# 
# ### 50 to 60 = D Grade
# 
# ### 40 to 50 = E Grade
# 
# ### below 40 = F Grade ( means Fail )

# In[ ]:


def getGrade(Percentage, Overall_Pass):
    if(Overall_Pass == 'Fail'):
        return 'Fail'
    if(Percentage >= 90):
        return "A+"
    if(Percentage >= 80):
        return "A"
    if(Percentage >= 70):
        return "B"
    if(Percentage >= 60):
        return "C"
    if(Percentage >= 50):
        return "D"
    if(Percentage >= 40):
        return "E"
    else:
        return 'Fail'
    
data['Grade'] = data.apply(lambda x:  getGrade(x['Percentage'], x['OverAll_Pass']), axis=1)
    


# In[ ]:


data.Grade.value_counts()


# In[ ]:


plt.figure(figsize=(18,7))
sns.countplot(x='Grade', data=data,order=['A+','A','B','C','D','E','Fail'],  palette="muted")


# In[ ]:


data.head()


# In[ ]:


plt.figure(figsize=(18,7))
sns.countplot(x='parental level of education', data = data, hue='Grade', palette='bright')


# # Prediction

# In[ ]:



# loading pakages for model. 
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer
from sklearn.kernel_ridge import KernelRidge

from sklearn import linear_model, model_selection, ensemble, preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet,SGDRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor


# In[ ]:


y = data['Total_Marks']


# In[ ]:


X = data[['math score','reading score','writing score']]


# In[ ]:


y.head()


# In[ ]:


X.head()


# In[ ]:


# X_train, y_train, X_test, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# In[ ]:


from sklearn.metrics import accuracy_score,classification_report,mean_squared_error, mean_absolute_error


# In[ ]:


# print("X Train: " ,X_train.shape)
# print("X Test: " ,X_test.shape)
# print("Y Train: " ,y_train.shape)
# print("Y Test: " ,y_test.shape)


# In[ ]:


# model_Lasso= make_pipeline(RobustScaler(), Lasso(alpha =0.000327, random_state=18))

# model_ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.00052, l1_ratio=0.70654, random_state=18))

# model_KRR = make_pipeline(RobustScaler(), KernelRidge(alpha=0.029963, kernel='polynomial', degree=1.103746, coef0=5.442672))
# model_GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
#                                    max_depth=4, max_features='sqrt',
#                                    min_samples_leaf=15, min_samples_split=10, 
#                                    loss='huber', random_state =18)

# model_KRR = make_pipeline(RobustScaler(), KernelRidge(alpha=0.029963, kernel='polynomial', degree=1.103746, coef0=5.442672))
# forest_reg = RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=None,
#            max_features=60, max_leaf_nodes=None, min_impurity_decrease=0.0,
#            min_impurity_split=None, min_samples_leaf=1,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            n_estimators=70, n_jobs=1, oob_score=False, random_state=42,
#            verbose=0, warm_start=False)


# In[ ]:


# Gender distribution
plt.figure(figsize=(8,7))
plt.title('Gender Distribution')
sns.countplot(data['gender'], palette='rainbow')


# In[ ]:


# race/ethinicity distribution
plt.figure(figsize=(8,7))
plt.title('Race/Ethinicity distribution')
sns.countplot(data['race/ethnicity'], palette='rainbow', hue=data['gender'])


# In[ ]:


#parental level of education
plt.figure(figsize=(12,6))
plt.title('Parental Level of Education')
sns.countplot(data['parental level of education'], palette='rainbow', hue=data['gender'])


# In[ ]:


# test prep course
plt.figure(figsize=(8,7))
plt.title('Test Preparation course')
sns.countplot(data['test preparation course'], palette='rainbow', hue=data['gender'])


# In[ ]:


# students with highest score in math
math_df = data[data['math score']==data['math score'].max()]
math_df


# In[ ]:


plt.figure(figsize=(8,7))
plt.title('Grades of students with gender distribution')
sns.countplot(data['Grade'], hue=data['gender'], palette='Set1')


# In[ ]:


plt.figure(figsize=(8,7))
plt.title('Grades of students with gender distribution')
sns.countplot(data['Grade'], hue=data['race/ethnicity'], palette='Set1')


# In[ ]:


plt.figure(figsize=(15,7))
plt.title('Grades of students with parental lvl of education distribution')
sns.countplot(data['Grade'], hue=data['parental level of education'], palette='Set1')


# In[ ]:


plt.figure(figsize=(15,7))
plt.title('Grades of students with parental lvl of education distribution')
sns.countplot(data['Grade'], hue=data['test preparation course'], palette='Set1')


# # Rename

# In[ ]:


df=data.rename(columns={'parental level of education':'parental_level_of_education',
                      'test preparation course':'test_preparation_course',
                     'math score':'math_score','reading score':'reading_score','writing score':'writing_score'})
df.head()


# In[ ]:


df.parental_level_of_education.unique()


# In[ ]:


df.lunch.unique()


# In[ ]:


df.test_preparation_course.unique()


# In[ ]:


df=df.replace(['group A','group B','group C','group D','group E'],[0,1,2,3,4])
df=df.replace(["bachelor's degree", 'some college', "master's degree","associate's degree", 'high school', 'some high school'],
             [0,1,2,3,4,5])
df=df.replace(['standard', 'free/reduced'],[0,1])
df=df.replace(['none', 'completed'],[0,1])
df=df.replace(['male','female'],[0,1])
df.head()


# In[ ]:



df['Percentage']=df['Percentage'].astype(int)
df.head()


# In[ ]:


x=sns.PairGrid(df,palette='coolwarm')
x=x.map_diag(plt.hist)
x=x.map_offdiag(plt.scatter,color='red',edgecolor='black')


# In[ ]:


plt.figure(figsize=(15,8))
sns.regplot('Total_Marks','math_score', data=df ,marker="d")


# In[ ]:


plt.figure(figsize=(15,8))
sns.regplot('Total_Marks','reading_score', data=df, color='red',marker="+" )


# In[ ]:


plt.figure(figsize=(15,8))
sns.regplot('Total_Marks','writing_score', data=df, color='green',marker="^" )


# In[ ]:



plt.figure(figsize = (15,15))
sns.heatmap(df.corr(),annot = True,linewidths = 0.5,cmap='cubehelix_r');
plt.savefig('Correlation Heatmap.png')


# In[ ]:





# In[ ]:





# In[ ]:


x=df[['gender','race/ethnicity','parental_level_of_education','lunch','test_preparation_course','math_score','reading_score','writing_score']]
y=df['Percentage']


# In[ ]:


x.shape,y.shape


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=7)


# In[ ]:


print("X Train: " ,x_train.shape)
print("X Test: " ,x_test.shape)
print("Y Train: " ,y_train.shape)
print("Y Test: " ,y_test.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model=LinearRegression()
model.fit(x_train,y_train)
#prediction=model.predict(x_test)
#mean_absolute_error(y_test, prediction)


# In[ ]:


prediction=model.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score,classification_report,mean_squared_error, mean_absolute_error
mean_absolute_error(y_test, prediction)


# In[ ]:


mean_squared_error(y_test, prediction)


# In[ ]:


plt.figure(figsize=(11,6))
plt.scatter(y_test,prediction,edgecolors='black',c='red',vmin=30,vmax=70)
#x.set_yticklabels([30,35,40,45,50])


# In[ ]:


plt.figure(figsize=(15,6))
sns.regplot(y_test,prediction, color='green' )


# In[ ]:


model.coef_


# In[ ]:


model.intercept_


# In[ ]:


from sklearn import metrics
mean_sq=metrics.mean_squared_error(y_test,prediction)
RMSE=np.sqrt(mean_sq)
RMSE


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test,prediction)


# In[ ]:


model1= KernelRidge(alpha=0.029963, kernel='polynomial', degree=1.103746, coef0=5.442672)
model1.fit(x_train,y_train)
prediction1=model1.predict(x_test)
print(mean_absolute_error(y_test, prediction1))
print(r2_score(y_test,prediction1))
print(np.sqrt(metrics.mean_squared_error(y_test,prediction1)))


# In[ ]:


plt.figure(figsize=(15,6))
sns.regplot(y_test,prediction1, color='green' )


# In[ ]:


import xgboost as xgb
model1= xgb.XGBRegressor(n_jobs=-1, n_estimators=849, learning_rate=0.015876, 
                           max_depth=58, colsample_bytree=0.599653, colsample_bylevel=0.287441, subsample=0.154134, seed=18)
model1.fit(x_train,y_train)
prediction2=model1.predict(x_test)
print(mean_absolute_error(y_test, prediction2))
print(r2_score(y_test,prediction2))
print(np.sqrt(metrics.mean_squared_error(y_test,prediction2)))


# In[ ]:


plt.figure(figsize=(15,6))
sns.regplot(y_test,prediction2, color='green' )


# In[ ]:


import lightgbm as lgb
model1= lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model1.fit(x_train,y_train)
prediction3=model1.predict(x_test)
print(mean_absolute_error(y_test, prediction3))
print(r2_score(y_test,prediction3))
print(np.sqrt(metrics.mean_squared_error(y_test,prediction3)))


# In[ ]:


plt.figure(figsize=(15,6))
sns.regplot(y_test,prediction3, color='green' )


# In[ ]:


model1= Lasso(alpha =0.000327, random_state=18)
model1.fit(x_train,y_train)
prediction4=model1.predict(x_test)
print(mean_absolute_error(y_test, prediction4))
print(r2_score(y_test,prediction4))
print(np.sqrt(metrics.mean_squared_error(y_test,prediction4)))


# In[ ]:


plt.figure(figsize=(15,6))
sns.regplot(y_test,prediction4, color='green' )


# In[ ]:


model1= ElasticNet(alpha=0.00052, l1_ratio=0.70654, random_state=18)
model1.fit(x_train,y_train)
prediction5=model1.predict(x_test)
print(mean_absolute_error(y_test, prediction5))
print(r2_score(y_test,prediction5))
print(np.sqrt(metrics.mean_squared_error(y_test,prediction5)))


# In[ ]:


plt.figure(figsize=(15,6))
sns.regplot(y_test,prediction5, color='green' )


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
model1= KNeighborsRegressor(n_neighbors=18)
model1.fit(x_train,y_train)
prediction6=model1.predict(x_test)
print(mean_absolute_error(y_test, prediction6))
print(r2_score(y_test,prediction6))
print(np.sqrt(metrics.mean_squared_error(y_test,prediction6)))


# In[ ]:


plt.figure(figsize=(15,6))
sns.regplot(y_test,prediction6, color='green' )


# In[ ]:


model1= SGDRegressor(alpha=0.00052, l1_ratio=0.70654, random_state=18)
model1.fit(x_train,y_train)
prediction7=model1.predict(x_test)
print(mean_absolute_error(y_test, prediction7))
print(r2_score(y_test,prediction7))
print(np.sqrt(metrics.mean_squared_error(y_test,prediction7)))


# In[ ]:


plt.figure(figsize=(15,6))
sns.regplot(y_test,prediction7, color='green' )


# In[ ]:


model1= Ridge(alpha=0.00052, random_state=24)
model1.fit(x_train,y_train)
prediction8=model1.predict(x_test)
p81 = mean_absolute_error(y_test, prediction8)
p82 = r2_score(y_test,prediction8)
p83 = np.sqrt(metrics.mean_squared_error(y_test,prediction8))
print(p81)
print(p82)
print(p83)


# In[ ]:


plt.figure(figsize=(15,6))
sns.regplot(y_test,prediction8, color='green' )


# In[ ]:



# def score(model):
#     score = cross_val_score(model, X_train, y_train, cv=5, scoring=rmse_score).mean()
#     return score


# In[ ]:


# scores['Liner Reg'] = lg
# scores['Random_forest'] = rnd

