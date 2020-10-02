#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv("/kaggle/input/wine-quality/winequalityN.csv")


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df.isnull().sum()


# In[ ]:


df["fixed acidity"].value_counts()


# In[ ]:


## df["fixed acidity"].fillna(mean,inplace=True)


# In[ ]:


mean = df["fixed acidity"].mean()
df["fixed acidity"].fillna(mean,inplace=True)
df["fixed acidity"].isnull().sum()


# In[ ]:


mean2 = df["volatile acidity"].mean()
df["volatile acidity"].fillna(mean,inplace=True)
df["volatile acidity"].isnull().sum()


# In[ ]:


df["citric acid"].value_counts()


# In[ ]:


mean3 = df["citric acid"].mean()
df["citric acid"].fillna(mean,inplace=True)
df["citric acid"].isnull().sum()


# In[ ]:


df["residual sugar"].value_counts()


# In[ ]:


mean4 = df["residual sugar"].mean()
df["residual sugar"].fillna(mean,inplace=True)
df["residual sugar"].isnull().sum()


# In[ ]:


mean4 = df["chlorides"].mean()
df["chlorides"].fillna(mean,inplace=True)
df["chlorides"].isnull().sum()


# In[ ]:



mean5 = df["pH"].mean()
df["pH"].fillna(mean,inplace=True)
df["pH"].isnull().sum()


# In[ ]:


mean6 = df["sulphates"].mean()
df["sulphates"].fillna(mean,inplace=True)
df["sulphates"].isnull().sum()


# In[ ]:


df.isnull().sum()


# # Let's Visualize the Data

# In[ ]:


plt.figure(figsize=(10,7))
plt.scatter(x="alcohol",y="fixed acidity",data =df,marker= 'o',c="m")
plt.xlabel("alcohol",fontsize=15)
plt.ylabel("fixed_acidity",fontsize=15)
plt.show()


# In[ ]:


sns.lmplot(x="alcohol",y="fixed acidity",data=df)
plt.plot()


# In[ ]:


plt.figure(figsize=(10,7))
plt.scatter(x="volatile acidity",y="alcohol",data =df,marker= 'o',c="m")
plt.xlabel("volatile_acidity",fontsize=15)
plt.ylabel("alcohol",fontsize=15)
plt.show()


# In[ ]:


sns.set(style="darkgrid")
sns.countplot(df["quality"],hue="type",data=df)
plt.show()


# In[ ]:


sns.set()
sns.distplot(df["quality"],bins=10)
plt.show()


# In[ ]:


plt.figure(figsize=(10,7))
sns.regplot(x="citric acid",y="chlorides",data =df,marker= 'o',color="m")
plt.show()


# In[ ]:


sns.set()
sns.pairplot(df)
plt.show()


# In[ ]:


sns.set()
plt.figure(figsize=(20,10))
sns.boxplot(data=df,palette="Set3")
plt.show()


#  # We can see that there are Some outliers.So now let's remove those Outliers

# In[ ]:


lower_limit = df["free sulfur dioxide"].mean() - 3*df["free sulfur dioxide"].std()
upper_limit = df["free sulfur dioxide"].mean() + 3*df["free sulfur dioxide"].std()


# In[ ]:


print(lower_limit,upper_limit)


# In[ ]:


df2 = df[(df["free sulfur dioxide"] > lower_limit) & (df["free sulfur dioxide"] < upper_limit)]


# In[ ]:


df.shape[0] - df2.shape[0]


# In[ ]:


lower_limit = df2['total sulfur dioxide'].mean() - 3*df2['total sulfur dioxide'].std()
upper_limit = df2['total sulfur dioxide'].mean() + 3*df2['total sulfur dioxide'].std()
print(lower_limit,upper_limit)


# In[ ]:


df3 = df2[(df2['total sulfur dioxide'] > lower_limit) & (df2['total sulfur dioxide'] < upper_limit)]
df3.head()


# In[ ]:


df2.shape[0] - df3.shape[0]


# In[ ]:


lower_limit = df3['residual sugar'].mean() - 3*df3['residual sugar'].std()
upper_limit = df3['residual sugar'].mean() + 3*df3['residual sugar'].std()
print(lower_limit,upper_limit)


# In[ ]:


df4 = df3[(df3['residual sugar'] > lower_limit) & (df3['residual sugar'] < upper_limit)]
df4.head()


# In[ ]:


df3.shape[0] - df4.shape[0]


# In[ ]:


df4.isnull().sum()


# In[ ]:


dummies = pd.get_dummies(df4["type"],drop_first=True)


# In[ ]:


df4 = pd.concat([df4,dummies],axis=1)


# In[ ]:


df4.drop("type",axis=1,inplace=True)


# In[ ]:


df4.head()


# In[ ]:


df4.quality.value_counts()


# # Now lets Change the Categorical 'String' Variables into Numerical Variables

# In[ ]:


quaity_mapping = { 3 : "Low",4 : "Low",5: "Medium",6 : "Medium",7: "Medium",8 : "High",9 : "High"}
df4["quality"] =  df4["quality"].map(quaity_mapping)


# In[ ]:


df4.quality.value_counts()


# In[ ]:


df4.head()


# In[ ]:


mapping_quality = {"Low" : 0,"Medium": 1,"High" : 2}
df4["quality"] =  df4["quality"].map(mapping_quality)


# In[ ]:


df4.head()


# # Lets Select the best Features for our Model

# In[ ]:


x = df4.drop("quality",axis=True)
y = df4["quality"]


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x,y)


# In[ ]:


print(model.feature_importances_)


# In[ ]:


feat_importances = pd.Series(model.feature_importances_,index =x.columns)
feat_importances.nlargest(9).plot(kind="barh")
plt.show()


# # Now Let's select the best model for our Dataset

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[ ]:


model_params  = {
    "svm" : {
        "model":SVC(gamma="auto"),
        "params":{
            'C' : [1,10,20],
            'kernel':["rbf"]
        }
    },
    
    "decision_tree":{
        "model": DecisionTreeClassifier(),
        "params":{
            'criterion':["entropy","gini"],
            "max_depth":[5,8,9]
        }
    },
    
    "random_forest":{
        "model": RandomForestClassifier(),
        "params":{
            "n_estimators":[1,5,10],
            "max_depth":[5,8,9]
        }
    },
    "naive_bayes":{
        "model": GaussianNB(),
        "params":{}
    },
    
    'logistic_regression' : {
        'model' : LogisticRegression(solver='liblinear',multi_class = 'auto'),
        'params': {
            "C" : [1,5,10]
        }
    }
    
}


# In[ ]:


score=[]
for model_name,mp in model_params.items():
    clf = GridSearchCV(mp["model"],mp["params"],cv=8,return_train_score=False)
    clf.fit(x,y)
    score.append({
        "Model" : model_name,
        "Best_Score": clf.best_score_,
        "Best_Params": clf.best_params_
    })


# In[ ]:


df5 = pd.DataFrame(score,columns=["Model","Best_Score","Best_Params"])


# In[ ]:


df5


# # So we can see that, we are getting 93% accuracy for SVM & Random Forest

# In[ ]:


from sklearn.model_selection import cross_val_score
clf_svm = SVC(kernel="rbf",C=1)
scores = cross_val_score(clf_svm,x,y,cv=8,scoring="accuracy")


# In[ ]:


scores


# In[ ]:


scores.mean()


# # So we are getting 93% Accuracy for predicting the Quality of Wine

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[ ]:


clf_svm1 = SVC(kernel="rbf",C=1)
clf_svm1.fit(x_train,y_train)


# In[ ]:


y_pred = clf_svm1.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)


# In[ ]:


accuracy


# # Now Lets see the Real value and Predicted Value

# In[ ]:


accuracy_dataframe = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})


# In[ ]:


accuracy_dataframe.head()


# 

# In[ ]:





# 

# 
