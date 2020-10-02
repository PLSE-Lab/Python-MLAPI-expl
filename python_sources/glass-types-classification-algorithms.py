#!/usr/bin/env python
# coding: utf-8

# # Data Information
# 
# This is a Glass Identification Data Set from UCI. It contains 10 attributes including id. The response is glass type(discrete 7 values)
# 
# Attribute information: -
# 1.	Id number: 1 to 214 (removed from CSV file)
# 2.	RI: refractive index
# 3.	Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
# 4.	Mg: Magnesium
# 5.	Al: Aluminum
# 6.	Si: Silicon
# 7.	K: Potassium
# 8.	Ca: Calcium
# 9.	Ba: Barium
# 10.	Fe: Iron
# 11.	Type of glass: (class attribute) -- 1 building_windows_float_processed -- 2 building_windows_non_float_processed -- 3 vehicle_windows_float_processed -- 4 vehicle_windows_non_float_processed (none in this database) -- 5 containers -- 6 tableware -- 7 headlamps
# 
# In this note book, I have made small attempt to understand and use various classification algorithms to predict the best algorithm for this data.
# 
# Notebook by: [Akshay Sb](https://www.linkedin.com/in/akshaybidarkundi/)
# 
# ""Please do upvote, if you find this as useful.""

# ## Importing Libraries

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# ## Reading the data

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df=pd.read_csv('../input/glass.csv')
df.head()


# In[ ]:


df.info()


# The data has 214 observations and 9 features which can be used to predict 10th feature.

# ## Exploratory Data Analysis

# ### Let us check for any null values

# In[ ]:


df.isnull().sum()


# We can see that there are no null values in the data set

# In[ ]:


## Let us check for five point summary of our data
df.describe()


# ### Let us check for any outliers using uni variate analysis

# In[ ]:


features=df.columns[:-1]
cols=list(features)


# In[ ]:


for i in cols:
    skewness=df[i].skew()
    print('Skewness for ',i,'= ',skewness)


# ##### From the skewness data we can see that none of the features are normally distributed. Let us check with box plots to find the outliers. 'Ba' and 'K' have very high skewness in data

# In[ ]:


plt.figure(figsize=(10,10))
df.boxplot()


# 1. From box plot it is evident that there are outliers in the data.
# 2. Mean of 'Si' is way more than mean of other parameters. It is not surprising, since 'Si' is major content in glass

# #### Let us find the outliers.

# In[ ]:


Feat=[]
U_C_L=[]
L_C_L=[]
for i in cols:
    q_25=np.percentile(df[i],25)
    q_75=np.percentile(df[i],75)
    IQR=q_75-q_25
    const=1.5*IQR
    UCL=round((q_75+const),4)
    LCL=round((q_25+const),4)
    Feat.append(i)
    U_C_L.append(UCL)
    L_C_L.append(LCL)


# In[ ]:


limits=pd.DataFrame({'Features':Feat,'Upper Limit':U_C_L,'Lower Limit':L_C_L})
limits


# Here we can see that Upper limit and lower limit are very close to each other. If we remove the outliers there will be huge loss in information. Hence we will remove those rows which have more than two outliers

# In[ ]:


def outliers(df):
    outlier_indices=[]
    
    # iterate over features(columns)
    for col in df.columns.tolist():
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        
        # Interquartile rrange (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > 2 )
    print(multiple_outliers)
    return multiple_outliers
print('The dataset contains %d observations with more than 2 outliers' %(len(outlier_indices)))  


# In[ ]:


outlier_indices = outliers(df[cols])
outlier_indices


# Now, it is better to remove these rows, instead of removing entries based on LCL and UCL. 

# ##### Let us try to see whether there is any correlation between the features

# In[ ]:


plt.figure(figsize=(8,8))
cor_mat=df[cols].corr()
cor_mat
sns.heatmap(cor_mat,annot=True)


# There is good correlation between 'Ca' and 'RI'.
# There is moderate correlation between ('Si' and 'RI'),('Ba' and 'Mg') and ('Al' and 'Mg')

# In[ ]:


plt.figure(figsize=(50,50))
print(df.groupby(['Type'])['RI'].mean())


# #### We can observe that, all the types of glass have similar refractive index.

# ### Let us look at the content of Na,Mg,Al,Si,K,Ca,Ba and Fe in different types of glass

# In[ ]:


(df.groupby(['Type'])['Na'].mean()).plot(kind='bar')
plt.xlabel('Type of glass')
plt.ylabel('"Na" content')
plt.title('Sodium Content in various types of glass')


# ##### Sodium (Na) content varies from 12.5 to 14. It is highest in type-6 glass which is tableware glass

# In[ ]:


(df.groupby(['Type'])['Mg'].mean()).plot(kind='bar')
plt.xlabel('Type of glass')
plt.ylabel('"Mg" content')
plt.title('Magnesium Content in various types of glass')


# ##### Magnesium (Mg) content varies from around 0.5 to 3.5. It is highest in float processed building windows and vehicle windows (Type 1 and Type3) glasses

# In[ ]:


(df.groupby(['Type'])['Al'].mean()).plot(kind='bar')
plt.xlabel('Type of glass')
plt.ylabel('"Al" content')
plt.title('Aluminum Content in various types of glass')


# ##### 1. Aluminum (Al) content varies from 1.15 to around 2.20. It is highest in glass used in headlamps (Type7).
# ##### 2. Glass used in containers also have almost same Aluminum content as that of headlamps. 

# In[ ]:


(df.groupby(['Type'])['Si'].mean()).plot(kind='bar')
plt.xlabel('Type of glass')
plt.ylabel('"Si" content')
plt.title('Silicon Content in various types of glass')


# ##### Every type of glass have almost same SIlicon content

# In[ ]:


(df.groupby(['Type'])['K'].mean()).plot(kind='bar')
plt.xlabel('Type of glass')
plt.ylabel('"K" content')
plt.title('Potassium Content in various types of glass')


# #####  Float and non-float processed building glasses, float-processed window glasses and head-lamp glasses (Type 1,2,3 and 7) have almost same quantity of potassium.
# #####  Glasses used in containers (Type 5) have very high content of potassium compared to other types
# ##### Table ware glasses have no potassium content in them.

# In[ ]:


(df.groupby(['Type'])['Ca'].mean()).plot(kind='bar')
plt.xlabel('Type of glass')
plt.ylabel('"Ca" content')
plt.title('Calcium Content in various types of glass')


# ##### All types of glasses have calcium content inthe range 9 to 10. Calcium content is highest in glass used in containers (Type 5)

# In[ ]:


(df.groupby(['Type'])['Ba'].mean()).plot(kind='bar')
plt.xlabel('Type of glass')
plt.ylabel('"Ba" content')
plt.title('Barium Content in various types of glass')


# ##### Glass used in float processed building and vehicle window glasses and  tableware glasses (Type 1,3,6) almost have no Barium content in them.
# ##### Glass used in non-float processed building window walls and container glass have Barium content in the range 0.05 - 0.2
# ##### Glass used in headlamps have highest Barium content.

# In[ ]:


(df.groupby(['Type'])['Fe'].mean()).plot(kind='bar')
plt.xlabel('Type of glass')
plt.ylabel('"Fe" content')
plt.title('Iron Content in various types of glass')


# ##### Glass used in tablewares have no iron content in them
# ##### Glass used in headlamps have iron content of around 0.01.
# ##### Glass used in non-float processed building window glasses have highest content of iron compared other types of glasses.

# In[ ]:


df1=df.drop(outlier_indices).reset_index(drop=True)


# In[ ]:


df1.shape


# #### Lets plot distribution plots after removing outlier columns

# In[ ]:


for i in cols:
    skewness=df1[i].skew()
    print('Skewness for ',i,'= ',skewness)


# Still there is no appreciable change in skewness. Let us go for transformations
# Here we cannot use Log transformation, since it keeps on reducing the value and there are some features with negetive skew and it will becone further negetive.
# We can try for box-cox transformation

# In[ ]:


y=df1['Type']
x=df1.drop('Type',axis=1)
from sklearn.model_selection import train_test_split


# In[ ]:


from scipy import stats as st


# In[ ]:


for i in x.columns:
    x[i],lambda_val=st.boxcox(x[i]+1.0)


# In[ ]:


for i in x.columns:
    skewness=x[i].skew()
    print('Skewness for ',i,'= ',skewness)


# Results look better than previous after transformation as we can see that skew values have come down. Lets split the data for training and testing

# In[ ]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
xs=ss.fit_transform(x)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)


# In[ ]:


x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)


# In[ ]:


x_test.shape,y_test.shape


# # Model Building

# ### Let us define a function to evaluate a model, which can be used to evaluate the model of all algorithms

# In[ ]:


def model_eval(algo,xtrain,ytrain,xtest,ytest):
    algo.fit(xtrain,ytrain)
    ytrain_pred=algo.predict(xtrain)

    from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve,classification_report

    print('Confusion matrix for train:','\n',confusion_matrix(ytrain,ytrain_pred))

    print('Overall accuracy of train dataset:',accuracy_score(ytrain,ytrain_pred))
    
    print('Classification matrix for train data','\n',classification_report(ytrain,ytrain_pred))

    ytest_pred=algo.predict(xtest)

    print('Test data accuracy:',accuracy_score(ytest,ytest_pred))

    print('Confusion matrix for test data','\n',confusion_matrix(ytest,ytest_pred))
    
    print('Classification matrix for train data','\n',classification_report(ytest,ytest_pred))


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
model_eval(rfc,x_train,y_train,x_test,y_test)


# #### Here we can see that  odel is over-fitting, let us try hyper-parameter tuning for the random forest

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
rfc=RandomForestClassifier(random_state=3)
params={'n_estimators':sp_randint(50,200),'max_features':sp_randint(1,24),'max_depth':sp_randint(2,10),
       'min_samples_split':sp_randint(2,20),'min_samples_leaf':sp_randint(1,20),'criterion':['gini','entropy']}
rs=RandomizedSearchCV(rfc,param_distributions=params,n_iter=500,cv=3,scoring='accuracy',random_state=3,
                      return_train_score=True)
rs.fit(xs,y)


# #### Let us see what are the best parameters and we will build model as per best parameters

# In[ ]:


rfc_best_parameters=rs.best_params_
print(rfc_best_parameters)


# In[ ]:


rfc1=RandomForestClassifier(**rfc_best_parameters)
model_eval(rfc1,x_train,y_train,x_test,y_test)


# #### Even after hyperparameter tuning, there is no improvement in performance of model. Let us try building other models

# # KNN Algorithm

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
model_eval(knn,x_train,y_train,x_test,y_test)


# #### There is around 10% difference between train and test data. Let us try to hyper-tune the parameters.

# In[ ]:


knn_rs=KNeighborsClassifier()

params={'n_neighbors':sp_randint(1,30),'p':sp_randint(1,6)}

rs1=RandomizedSearchCV(knn_rs,param_distributions=params,cv=3,return_train_score=True,random_state=3,n_iter=500)

rs1.fit(xs,y)


# In[ ]:


knn_best_parameters=rs1.best_params_
print(knn_best_parameters)


# In[ ]:


knn1=KNeighborsClassifier(**knn_best_parameters)
model_eval(knn1,x_train,y_train,x_test,y_test)


# #### From train and test accuracy, we can observe that the accuracy scores are closer to each other. Hence we can see that KNeighbours performs better in this case. Let us try with other models before final conclusion

# ### Let us use Boosting method (Lightbgm)

# In[ ]:


import lightgbm as lgb
lgbm=lgb.LGBMClassifier()
model_eval(lgbm,x_train,y_train,x_test,y_test)


# ##### Model seems to over-fit. Let us tune the hyper-parameters to check whether we can resolve over-fitting issues

# In[ ]:


from scipy.stats import uniform as sp_uniform
params={'n_estimator':sp_randint(50,200),'max_depth':sp_randint(2,15),'learning_rate':sp_uniform(0.001,0.5),
       'num_leaves':sp_randint(20,50)}
lgbm_rs=lgb.LGBMClassifier()
rs_lgbm=RandomizedSearchCV(lgbm_rs,param_distributions=params,cv=3,random_state=3,n_iter=500,n_jobs=-1)
rs_lgbm.fit(xs,y)


# In[ ]:


lgbm_best_parameters=rs_lgbm.best_params_
print(lgbm_best_parameters)


# In[ ]:


lgbm_1=lgb.LGBMClassifier(**lgbm_best_parameters)
model_eval(lgbm_1,x_train,y_train,x_test,y_test)


# #### Even after hyperparameter tuning, there is considerable amount of over-fitting

# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='liblinear')
model_eval(lr,x_train,y_train,x_test,y_test)


# #### There is acceptable difference between train and test accuracy.

# # Stacking

# ## Hard-Voting

# In[ ]:


from sklearn.ensemble import VotingClassifier
lr=LogisticRegression(solver='liblinear')
rfc1=RandomForestClassifier(**rfc_best_parameters)
knn1=KNeighborsClassifier(**knn_best_parameters)
lgbm_1=lgb.LGBMClassifier(**lgbm_best_parameters)


# In[ ]:


clf=VotingClassifier(estimators=[('lr',lr),('knn',knn1),('rfc',rfc1),('lgbm',lgbm_1)],voting='hard')
model_eval(clf,x_train,y_train,x_test,y_test)


# ### There is slight difference in the accuracy. We will check with soft voting
# 
# #### Soft Voting with equal weightages

# In[ ]:


clf_sv=VotingClassifier(estimators=[('lr',lr),('knn',knn1),('rfc',rfc1),('lgbm',lgbm_1)],voting='soft')
model_eval(clf_sv,x_train,y_train,x_test,y_test)


# #### Let us try soft voting by giving weightages

# In[ ]:


clf_sv1=VotingClassifier(estimators=[('lr',lr),('knn',knn1),('rfc',rfc1),('lgbm',lgbm_1)],voting='soft',weights=[5,5,1,1])
model_eval(clf_sv1,x_train,y_train,x_test,y_test)


# ## Here we can observe,model is performing better comparing to equal weightages.

# # Since KNN and Logistic Regression models are performing better. Lets try and build stacking model with only KNN and Logistic Regression

# In[ ]:


clf_sv2=VotingClassifier(estimators=[('lr',lr),('knn',knn1)],voting='soft',weights=[5,4])
model_eval(clf_sv2,x_train,y_train,x_test,y_test)


# #### From above results, it can be seen that the model with combinations of Logistic Regression, KNN, Random Forest and Light gbm performs better compared to other models.
