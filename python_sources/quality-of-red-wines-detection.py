#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv("../input/winequality-red.csv")


# In[ ]:


data.describe()


# ### Bivariate Analysis

# In[ ]:


# Checking all Attributes against the QUALITY attribute.


# In[ ]:


sns.boxplot('quality', 'alcohol', data = data)


# In[ ]:


## Observations from Bivariate analysis using BOXPLOT

1. fixed acidity has outliers at 5 and 6
2. volatile acidity has outliers at 5 and 6 and 7
3. citric acid looks OK
4. residual sugar has outliers at 5 and 6 and 7
5. chlorides has outliers at 4 and 5 and 6
6. free sulfur dioxide has outliers at 5 and 6
7. total sulfur dioxide has outliers at 6 and 7
8. density has outliers at 5 and 6
9. pH has outliers at 6 and 7
10. sulphates has outliers at 5 and 6
11. alcohol has outliers at 5 

Overall the Quality 5, 6 and 7 are mostly having Outliers


# In[ ]:


#count of the target variable
sns.countplot(x='quality', data=data)


# In[ ]:


Observations:
    We can see that Quality 5, 6 and 7 are Contributing more..
    and Outliers are in these 3 Categories..


# In[ ]:


corr = data.corr()
sns.heatmap(data=corr,annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# In[ ]:


density and fixed acidity are correlated @ 0.67
citric acid and fixed acidity are correlated @ 0.67
total sulfur dioxide and free sulfur dioxide are correlated @ 0.67


# In[ ]:


# "total sulfur dioxide" Analysis Before outliers deletion (before LOG)

#IQR = 1.5 * (np.percentile(data['total sulfur dioxide'], 75) - np.percentile(data['total sulfur dioxide'], 25))
#print("IQR of ApplicantIncome is: ", IQR)
#print("\n DataDescriptions of total sulfur dioxide BEFORE LOG: \n",data['total sulfur dioxide'].describe())
#LA_more_IQR = data[data['total sulfur dioxide']>IQR]
#print("Records above the IQR: ", LA_more_IQR.shape)
#print("There are ",
#      (LA_more_IQR['total sulfur dioxide'].count()/data['total sulfur dioxide'].count())*100 , "% of total sulfur dioxide are Outliers")


# In[ ]:


# Removing OutLiners of total sulfur dioxide using LOG

#out_log_LA = np.log(data['total sulfur dioxide'])
#print("\n DataDescriptions of total sulfur dioxide AFTER LOG: \n",out_log_LA.describe())

#data['total sulfur dioxide'] = out_log_LA
#data.boxplot(column = ('total sulfur dioxide'))


# ## Preprocessing

# ## Approach 1 for feature Engneering, using this getting the 83% Model Accuaracy 

# In[ ]:


# 3 and 4 = LOW      #1
# 5 and 6 = Average  #2
# 7 and 8 = High     #3  
Review = []
for i in data['quality']:
    if i >= 3 and i<=4:
        Review.append('1')
    if i >= 5 and i<=6:
        Review.append('2')
    if i >= 7 and i<=8:
        Review.append('3')

data['Review'] = Review


# In[ ]:





# ## Approach 2 for feature Engneering, using this getting the 98% Model Accuaracy

# In[ ]:


#next we shall create a new column called Review. This column will contain the values of 1,2, and 3. 
#1 - Bad
#2 - Average
#3 - Excellent
#This will be split in the following way. 
#1,2,3 --> Bad
#4,5,6,7 --> Average
#8,9,10 --> Excellent
#Create an empty list called Reviews
Review = []
for i in data['quality']:
    if i >= 1 and i <= 3:
        Review.append('1')
    elif i >= 4 and i <= 7:
        Review.append('2')
    elif i >= 8 and i <= 10:
        Review.append('3')
data['Review'] = Review


# In[ ]:


print(data.columns)
print(data['Review'].value_counts())
#print(data.iloc[:,1:len(data.columns)-2])


# ## Prepare Data Model

# In[ ]:


# Start Modeling
# Split Training and Testing data (within training data file only)
# Hold out method validation

from sklearn.model_selection import train_test_split

#X_train = train.loc[:,('LoanAmount','Family_Income','Loan_Amount_Term', 'Gender_Married',  'Education_SelfEmployed', 'Credit_History')]
#Y_train = train.iloc[:,len(train.columns)-1:]  # this represnts the (output) Label LOAN_STATUS

X_train = data.iloc[:,1:len(data.columns)-2] # this represents the input Features
Y_train = data.loc[:,'Review']


# Scaling features (only feature NOT observation)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Scaling - brings to range (0 to 1)
ScaleFn = MinMaxScaler()
X_Scale = ScaleFn.fit_transform(X_train)
# Standardise - brings to Zero mean, Unit variance
ScaleFn = StandardScaler()
X_Strd = ScaleFn.fit_transform(X_train)


# ### Proceed to perform PCA

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA()
x_pca = pca.fit_transform(X_Strd)


# In[ ]:


#plot the graph to find the principal components
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()


# In[ ]:


#AS per the graph, we can see that 8 principal components attribute for 90% of variation in the data. 
#we shall pick the first 8 components for our prediction.
pca_new = PCA(n_components=8)
x_new = pca_new.fit_transform(X_Strd)
print(x_new)


# In[ ]:


test_size = .30
seedNo = 11


X_train,X_test,Y_train,Y_test = train_test_split(x_new,Y_train,test_size = test_size, random_state = seedNo)

print("train X", X_train.shape)
print("train Y", Y_train.shape)
print("test X", X_test.shape)
print("test Y", Y_test.shape)

# Choose algorithm and train
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

mymodel = []
mymodel.append(('LogReg', LogisticRegression()))
mymodel.append(('KNN', KNeighborsClassifier()))
mymodel.append(('DeciTree', DecisionTreeClassifier()))
mymodel.append(('RandForest', RandomForestClassifier()))
mymodel.append(('SVM', SVC()))
mymodel.append(('XGBoost', XGBClassifier()))



All_model_result = []
All_model_name = []
for algoname, algorithm in mymodel:    
    kfoldFn = KFold(n_splits = 11, random_state = seedNo)
    Eval_result = cross_val_score(algorithm, X_train, Y_train, cv = kfoldFn, scoring = 'accuracy')
    
    All_model_result.append(Eval_result)
    All_model_name.append(algoname)
    print("Modelname and Model accuracy:", algoname, 100*Eval_result.mean(),"%")


# In[ ]:





# In[ ]:




