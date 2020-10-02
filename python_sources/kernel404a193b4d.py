#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ***HERE WE ARE SIMPLY IMPORTING THE DATASET***

# In[ ]:


try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    import pandas as pd 
    import seaborn as sns 
    import numpy as np 
    import os
    import matplotlib.pyplot as plt
    print('module imported')
except:
    raise ModuleNotFoundError


# In[ ]:


df=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')


# ***CHECKING FOR THE OUTLIERS***

# In[ ]:


def check_outlier1():
    plt.hist(df['fixed acidity'])
    lower_bound=0.1
    upper_bound=0.95
    res=df['fixed acidity'].quantile([lower_bound,upper_bound])
    print(res)
    true_index=(res.loc[lower_bound]<df['fixed acidity'].values) &         (df['fixed acidity'].values<res.loc[upper_bound])
    df['fixed acidity']=df['fixed acidity'][true_index]
check_outlier1()
def check_outlier2():
    plt.hist(df['volatile acidity'])
    lower_bound=0.1
    upper_bound=0.95
    res=df['volatile acidity'].quantile([lower_bound,upper_bound])
    print(res)
    true_index=(res.loc[lower_bound]<df['volatile acidity'].values) &         (df['volatile acidity'].values<res.loc[upper_bound])
    df['volatile acidity']=df['volatile acidity'][true_index]
check_outlier2()

def check_outlier3():
    plt.hist(df['citric acid'])
    lower_bound=0.1
    upper_bound=0.95
    res=df['citric acid'].quantile([lower_bound,upper_bound])
    print(res)
    true_index=(res.loc[lower_bound]<df['citric acid'].values) &         (df['citric acid'].values<res.loc[upper_bound])
    df['citric acid']=df['citric acid'][true_index]
check_outlier3()
def check_outlier4():
    plt.hist(df['residual sugar'])
    lower_bound=0.1
    upper_bound=0.95
    res=df['residual sugar'].quantile([lower_bound,upper_bound])
    print(res)
    true_index=(res.loc[lower_bound]<df['residual sugar'].values) &         (df['residual sugar'].values<res.loc[upper_bound])
    df['residual sugar']=df['residual sugar'][true_index]
check_outlier4()
def check_outlier5():
    plt.hist(df['chlorides'])
    lower_bound=0.1
    upper_bound=0.95
    res=df['chlorides'].quantile([lower_bound,upper_bound])
    print(res)
    true_index=(res.loc[lower_bound]<df['chlorides'].values) &         (df['chlorides'].values<res.loc[upper_bound])
    df['chlorides']=df['chlorides'][true_index]
check_outlier5()
def check_outlier6():
    plt.hist(df['free sulfur dioxide'])
    lower_bound=0.1
    upper_bound=0.95
    res=df['free sulfur dioxide'].quantile([lower_bound,upper_bound])
    print(res)
    true_index=(res.loc[lower_bound]<df['free sulfur dioxide'].values) &         (df['free sulfur dioxide'].values<res.loc[upper_bound])
    df['free sulfur dioxide']=df['free sulfur dioxide'][true_index]
check_outlier6()

def check_outlier7():
    plt.hist(df['total sulfur dioxide'])
    lower_bound=0.1
    upper_bound=0.95
    res=df['total sulfur dioxide'].quantile([lower_bound,upper_bound])
    print(res)
    true_index=(res.loc[lower_bound]<df['total sulfur dioxide'].values) &         (df['total sulfur dioxide'].values<res.loc[upper_bound])
    df['total sulfur dioxide']=df['total sulfur dioxide'][true_index]
check_outlier7()

def check_outlier8():
    plt.hist(df['density'])
    lower_bound=0.1
    upper_bound=0.95
    res=df['density'].quantile([lower_bound,upper_bound])
    print(res)
    true_index=(res.loc[lower_bound]<df['density'].values) &         (df['density'].values<res.loc[upper_bound])
    df['density']=df['density'][true_index]
check_outlier8()

def check_outlier9():
    plt.hist(df['pH'])
    lower_bound=0.1
    upper_bound=0.95
    res=df['pH'].quantile([lower_bound,upper_bound])
    print(res)
    true_index=(res.loc[lower_bound]<df['pH'].values) &         (df['pH'].values<res.loc[upper_bound])
    df['pH']=df['pH'][true_index]
check_outlier9()

def check_outlier10():
    plt.hist(df['sulphates'])
    lower_bound=0.1
    upper_bound=0.95
    res=df['sulphates'].quantile([lower_bound,upper_bound])
    print(res)
    true_index=(res.loc[lower_bound]<df['sulphates'].values) &         (df['sulphates'].values<res.loc[upper_bound])
    df['sulphates']=df['sulphates'][true_index]
check_outlier10()

def check_outlier11():
    plt.hist(df['alcohol'])
    lower_bound=0.1
    upper_bound=0.95
    res=df['alcohol'].quantile([lower_bound,upper_bound])
    print(res)
    true_index=(res.loc[lower_bound]<df['alcohol'].values) &         (df['alcohol'].values<res.loc[upper_bound])
    df['alcohol']=df['alcohol'][true_index]
check_outlier11()


# ***FILLING ALL THE NULL VALUES***

# In[ ]:


df['fixed acidity'].fillna(8,inplace=True)
df['volatile acidity'].fillna(0.535,inplace=True)
df['citric acid'].fillna(0.27,inplace=True)
df['residual sugar'].fillna(2.2,inplace=True)
df['chlorides'].fillna(0.8,inplace=True)
df['free sulfur dioxide'].fillna(15,inplace=True)
df['total sulfur dioxide'].fillna(40,inplace=True)
df['density'].fillna(0.9968,inplace=True)
df['pH'].fillna(3.32,inplace=True)
df['sulphates'].fillna(0.63,inplace=True)
df['alcohol'].fillna(10.3,inplace=True)


# In[ ]:


bins = (1, 6.5, 10)
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)


# ***SPLITTING THE DATASET INTO X AND Y***

# In[ ]:


x=df.iloc[:,:-1]
y=df.iloc[:,11]


# ***ENCODING THE Y WHICH IS JUST SPLITTING AND WHICH IS HOLDING THE TARGET VARIABLE***

# In[ ]:


from sklearn.preprocessing import LabelEncoder , OneHotEncoder
label = LabelEncoder()
y = label.fit_transform(y)


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=12)


# ***IMPORTING THE STANDARDSCALER FOR TRANSFORMING THE DATA INTO VAULES BETWEEN -1 TO 1***

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


# ***USING RANDOMFORESTCLASSIFIER****

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy',max_depth=5)
rfc.fit(x_train,y_train)
y_predict=rfc.predict(x_test)
score3=accuracy_score(y_test,y_predict)
print(score3)


# ***CHECKING THE ACCURACY USING CROSS VALIDATION***

# In[ ]:


from sklearn.model_selection import cross_val_score
CV=cross_val_score(rfc,x_train,y_train,cv=5)
print(CV.mean())


# In[ ]:




