#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Loading the dataframe
df = pd.read_csv('/kaggle/input/adult-census-income/adult.csv')


# # Dataset Description
# 
# ### Categorical Attributes :
# 
# **1. workclass**: (categorical) Private, Self-emp-not-inc(Unincorporated self employment), Self-emp-inc(Incorporated self employment:), Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
#     - Individual work category
# **2. education**: (categorical) Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
#     - Individual's highest education degree
# **3. marital-status**: (categorical) Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
#     - Individual marital status
# **4. occupation**: (categorical) Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
#      - Individual's occupation
# **5. relationship**: (categorical) Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
#     - Individual's relation in a family
# **6. race**: (categorical) White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
#     - Race of Individual
# **7. sex**: (categorical) Female, Male.
# 
# **8. native-country**: (categorical) United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
#     - Individual's native country
# 
# ### Continuous Attributes :
# 
# **1. age**: continuous.
#     - Age of an individual
# **2. education-num**: number of education year, continuous.
#     - Individual's year of receiving education
# **3. fnlwgt**: final weight, continuous.
#     - The weights on the CPS files are controlled to independent estimates of the civilian noninstitutional population of the US. These are prepared monthly for us by Population Division here at the Census Bureau.
#     
#     - The term estimate refers to population totals derived from CPS by creating "weighted tallies" of any specified socio-economic characteristics of the population. People with similar demographic characteristics should have similar weights. 
#     
# **4. capital-gain**: continuous.
# 
# **5. capital-loss**: continuous.
# 
# **6. hours-per-week**: continuous.
#     - Individual's working hour per week

# In[ ]:


#Viewing the dataset
df.head()


# In[ ]:


#Information about dataset
df.info()


# In[ ]:


#Seperating continuous and categorical columns
cont_col = []
cat_col  = []

for i in df.columns :
    if df[i].dtypes == 'O':
        cat_col.append(i)
    else :
        cont_col.append(i)


# In[ ]:


print('The categorical columns are :\n',cat_col)
print()
print('The continuous columns are  :\n',cont_col)


# **CHECKING FOR MISSING VALUES**

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(df.isnull(),cbar=False)
plt.xticks(rotation=60, fontsize=10)
plt.show()


# In[ ]:


df.isnull().sum()


# There are no missing NaN values in the dataset

# **FEATURE ENGINEERING**

# In[ ]:





# In[ ]:


#Gives Frequency of elements in all categorical columns 

for i in cat_col :
    print('\t\t\t\t',i)
    print(df[i].value_counts())
    plt.title('Countplot')
    plt.ylabel('counts')
    plt.xlabel(i)
    sns.barplot(df[i].value_counts().index[:5] , df[i].value_counts().values[:5])
    plt.tight_layout()
    plt.show()
    print('\n\n')


# Observation :
#    -  ? in all columns need to be imputed
#    - Education column can be combined
#    - No changes needed for Relationship , Race and Sex columns
#    - Native country columns needs to be binned together since there are very less values for some countries

# In[ ]:


#Cleaning Marital Status column
df['marital.status'].replace({'Married-civ-spouse' : 'Married' ,
                              'Divorced' : 'Separated' , 
                              'Married-AF-spouse' : 'Married' , 
                              'Married-spouse-absent':'Separated'},inplace = True)

sns.countplot(df['marital.status'])


# In[ ]:


#Cleaning education column

df['education'].replace({'HS-grad':'HighSchool' , 
                         'Some-college':'College' , 
                         'Bachelors' : 'Bachelor degree' , 
                         'Masters' : 'Masters Degree' ,
                         'Assoc-voc' : 'College' , 
                         'Assoc-acdm':'College' , 
                         'Prof-school' : 'Masters Degree' , 
                         'Doctorate' : 'PhD' , 
                         '11th' : 'Dropout' , 
                         '10th' : 'Dropout' ,
                         '7th-8th' : 'Dropout' ,
                         '9th' : 'Dropout' , 
                         '12th' : 'Dropout' ,
                         '5th-6th': 'Dropout' ,
                         '1st-4th': 'Dropout' ,
                         'Preschool':'Dropout'} , inplace = True)

plt.figure(figsize=(10,4))
sns.countplot(df['education'])


# In[ ]:


#The number of columns with both occupation and workclass unknown

df[(df['occupation'] == "?") & (df['workclass'] == "?")]['age'].count()

#So a new cateogry called unknown is created for both occupation and workclass columns


# In[ ]:


#Imputing WorkClass
sns.countplot(df[df['workclass'] == '?']['workclass'] , hue = df['income'])


# In[ ]:


#Since there are around 1800 unknown workclass we will create a new seperate unknown category
df['workclass' ].replace({'?' : 'Unknown'} , inplace = True)
df['occupation' ].replace({'?' : 'Unknown'} , inplace = True)


# **FOR CONTINUOUS VARIABLES**

# In[ ]:


cont_col


# In[ ]:


df.describe()


# In[ ]:


#Box plot and distplot for all continuous variables
for i in ['age','fnlwgt','education.num','hours.per.week']:
    print('\t\t\t\t',i)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title('Boxplot')
    df[i].plot(kind= 'box')
    plt.subplot(1,2,2)
    plt.title('Distribution plot')
    sns.distplot(df[i],hist = False )
    plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr() , annot =True , cmap = 'summer' )


# There is no correlation with any of the columns

# In[ ]:


#Creating copy of the dataframe df_n
df_n = df.copy()


# In[ ]:


#The following code does the one hot encoding only when the frequency of each category is greater than 50. 
#After encoding then the first column is dropped inorder to avoid the curse of dimensionality.
#And it deletes the original column after encoding

print('The Encoding is applied for: ')
for col in cat_col[:-1]:
    freqs=df_n[col].value_counts()
    k=freqs.index[freqs>50][:-1]
    for cat in k:
        name=col+'_'+cat
        df_n[name]=(df_n[col]==cat).astype(int)
    del df_n[col]
    print(col)


# In[ ]:


#EDA is done for our datasets
df_n.head()


# In[ ]:


#Checking our Target column
df_n.groupby(by = 'income').count()


# In[ ]:


#Mapping income greater than 50k as 1
df_n['income'] = df_n['income'].map({'<=50K' : 0 , '>50K' : 1})


# In[ ]:


#Splitting the target variable

X = df_n.drop(columns='income')
y = df_n['income']


# In[ ]:


#Scaling the dataset
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X[X.columns] = ss.fit_transform(X[X.columns])


# In[ ]:


X.head()


# **APPLYING PCA**

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA()
df_pca = pca.fit_transform(X)


# In[ ]:


#Checking for cumulative explained varience ratio to choose the best n components for PCA

pd.DataFrame(np.cumsum(pca.explained_variance_ratio_)*100 , index = range(1, X.shape[1]+1) , columns=['Cum. Var']).T


# In[ ]:


#We will choose 54 PC dimensions since 99% varience is explained in 54 pc dimensions
pca = PCA(n_components=54)
df_pca1 = pca.fit_transform(X)
df_pca1 = pd.DataFrame(df_pca1 , columns=['PC '+str(i) for i in range(1,55)])


# In[ ]:


display(df_pca1.head())
display(df_pca1.shape)


# **TEST TRAIN SPLIT**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_pca1, y, test_size=0.3, random_state=3 )
print('X_train shape',X_train.shape)
print('y_train shape',y_train.shape)
print('X_test  shape',X_test.shape)
print('y_test  shape',y_test.shape)


# **HYPERPARAMETER TUNING (Finding the best paramters using RandomizedSearchCV for Random Forest) :**

# In[ ]:


X1 = pd.concat([df_pca1, y],axis=1)
X1 = X1.sample(frac=0.1, replace=True, random_state=3) 
Xt = X1.drop(columns = 'income')
yt = X1['income']

#we are taking 10% of data with replacement as a sample 
#This is done here to reduce the run time , Since it takes more than 3 hours for a single randomized search to be done.
# This increases the risk of overfitting (the hyperparameters) on that specific test set result. So randomized search needs to be run on full dataset
print(Xt.shape)
print(yt.shape)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

rfc = RandomForestClassifier(random_state=3)
params = { 'n_estimators' : sp_randint(50 , 200) , 
           'max_features' : sp_randint(1 , 54) ,
           'max_depth' : sp_randint(2,15) , 
           'min_samples_split' : sp_randint(2,30) ,
           'min_samples_leaf' : sp_randint(1,30) ,
           'criterion' : ['gini' , 'entropy']
    
}

rsearch_rfc = RandomizedSearchCV(rfc , param_distributions= params , n_iter= 200 , cv = 3 , scoring='roc_auc' , random_state= 3 , return_train_score=True , n_jobs=-1)

rsearch_rfc.fit(Xt,yt)


# In[ ]:


#The best parameters are,
rsearch_rfc.best_params_


# **HYPERPARAMETER TUNING (Finding the best paramters using RandomizedSearchCV for KNN) :**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV , GridSearchCV
from scipy.stats import randint as sp_randint

knn = KNeighborsClassifier()

params = {
    'n_neighbors' : sp_randint(1 , 20) ,
    'p' : sp_randint(1 , 5) ,
}

rsearch_knn = RandomizedSearchCV(knn , param_distributions = params , cv = 3 , random_state= 3  , n_jobs = -1 , return_train_score=True)

rsearch_knn.fit(Xt , yt)


# In[ ]:


#The best parameters are,
rsearch_knn.best_params_


# **HYPERPARAMETER TUNING (Finding the best paramters using RandomizedSearchCV for Decision Tree Classifier) :**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(random_state=3)
params = { 'max_features' : sp_randint(1 , 54) ,
           'max_depth' : sp_randint(2,15) , 
           'min_samples_leaf' : sp_randint(1,30) ,
           'criterion' : ['gini' , 'entropy']
    
}

rsearch_dtc = RandomizedSearchCV(rfc , param_distributions= params , n_iter= 200 , cv = 3 , scoring='roc_auc' , random_state= 3 , return_train_score=True , n_jobs=-1)

rsearch_dtc.fit(Xt,yt)


# In[ ]:


#The best parameters are,
rsearch_dtc.best_params_


# In[ ]:


#Calling remaining classification algorithms
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(fit_intercept=True , solver='liblinear' , random_state=3)

from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier(**rsearch_dtc.best_params_ , random_state=3)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(**rsearch_rfc.best_params_ , random_state=3)

from sklearn.naive_bayes import GaussianNB
gb = GaussianNB()

from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(**rsearch_knn.best_params_ )
knn = KNeighborsClassifier()

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(random_state=3)

import lightgbm as lgb
lgbm = lgb.LGBMClassifier(random_state=3)


# In[ ]:


#Executing various classification algorithms , Test train overall accuracy score , Test train AUC score and ROC AUC curve

from sklearn.metrics import confusion_matrix, classification_report , accuracy_score , roc_auc_score ,roc_curve

algo_arr = [lr ,dt, rfc , gb ,knn , ada ,lgbm]

l={0:'LogisticRegression' ,1:'Decision Tree', 2:'Random Forest' , 3:'Gaussian NB' ,4:'KNN' , 5:'AdaBoost' , 6:'Lightbgm'}
metric=[]
    
for i in range(len(algo_arr)):
    algo_arr[i].fit(X_train , y_train)

    y_train_pred=algo_arr[i].predict(X_train)
    y_train_prob=algo_arr[i].predict_proba(X_train)[:,1]

    print('\t','\t','\t','\t','\t','\t','\t',l[i])
    print('Confusion Matrix - Train' , '\n' , confusion_matrix(y_train,y_train_pred))

    print('Overall Accuracy - Train ', accuracy_score(y_train,y_train_pred))
    metric.append(accuracy_score(y_train,y_train_pred))

    print('AUC - Train: ', roc_auc_score(y_train,y_train_prob))
    metric.append(roc_auc_score(y_train,y_train_prob))

    print('\n')

    y_test_pred = algo_arr[i].predict(X_test)
    y_test_prob=algo_arr[i].predict_proba(X_test)[:,1]

    print('Confusion Matrix - Test ', '\n' , confusion_matrix(y_test,y_test_pred))

    print('Overall Accuracy - Test ', accuracy_score(y_test,y_test_pred))
    metric.append(accuracy_score(y_test,y_test_pred))

    print('AUC - Test: ', roc_auc_score(y_test,y_test_prob))
    metric.append(roc_auc_score(y_test,y_test_prob))
    
    #ROC AUC Curve
    fpr , tpr , threshold = roc_curve(y_test , y_test_prob)
    plt.plot(fpr , tpr)
    plt.plot(fpr , fpr , 'r-')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()


# In[ ]:


#Creating accuracies as dataframe for easy interpretation
al=['LogisticRegression','Decision Tree','Random Forest', 'Gaussian NB','KNN','AdaBoost','Lightbgm']

col=['Train_Accuracy_score', 'Test_Accuracy_score','Train_AUC_score','Test_AUC_score']

me=np.array(metric).reshape(7,4)

c_t=pd.DataFrame(np.round(me,2),columns= col , index=al)


# In[ ]:


#Comparing various algorithms scores
c_t


# In[ ]:




