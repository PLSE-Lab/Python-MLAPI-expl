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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_data = pd.read_csv("/kaggle/input/novartis-data/Train.csv")
test_data = pd.read_csv("/kaggle/input/novartis-data/Test.csv")


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# In[ ]:


train_data.describe()


# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# ## We can see that we have some null values .So let's remove these null values

# In[ ]:


train_data["X_12"] = train_data["X_12"].ffill()
test_data["X_12"] = test_data["X_12"].ffill()
train_data["X_12"] = train_data["X_12"].bfill()
test_data["X_12"] = test_data["X_12"].bfill()


# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# ## Now we can see that, we have no Null Values.So we have successfully remove all the Null Values

# ## Now let's drop the first two column(INCIDENT_ID & DATE) because we don't need these two column

# In[ ]:


train_data.drop(["INCIDENT_ID","DATE"],axis=1,inplace=True)


# In[ ]:


test_data.drop(["INCIDENT_ID","DATE"],axis=1,inplace=True)


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


sns.set()
train_data.hist(figsize=(20,10),bins=15,color="purple")
plt.title("Distribution of Features")
plt.show()


# In[ ]:


sns.set()
plt.figure(figsize=(20,10))
sns.boxplot(data= train_data,palette = "Set3")
plt.show()


# In[ ]:


sns.set()
plt.figure(figsize=(20,10))
sns.boxplot(data= test_data,palette = "Set3")
plt.show()


# ## Here we can see that our dataset has a lot of outliers,Let's remove some of the outliers, becouse outliers are gonna make some trouble for us to create a Good Model

# ## Let's remove these Outliers using Standard Deviation

# In[ ]:


lower_limit = train_data["X_8"].mean() - 3*train_data["X_8"].std()
upper_limit = train_data["X_8"].mean() + 3*train_data["X_8"].std()


# In[ ]:


df_train1 = train_data[(train_data["X_8"] > lower_limit) & (train_data["X_8"] < upper_limit)]


# In[ ]:


train_data.shape[0] - df_train1.shape[0]


# In[ ]:


lower_limit = test_data["X_8"].mean() - 3*test_data["X_8"].std()
upper_limit = test_data["X_8"].mean() + 3*test_data["X_8"].std()


# In[ ]:


df_test1 = test_data[(test_data["X_8"] > lower_limit) & (test_data["X_8"] < upper_limit)]


# In[ ]:


test_data.shape[0] - df_test1.shape[0]


# In[ ]:


lower_limit = df_train1["X_10"].mean() - 3*df_train1["X_10"].std()
upper_limit = df_train1["X_10"].mean() + 3*df_train1["X_10"].std()


# In[ ]:


df_train2 = df_train1[(df_train1["X_10"] > lower_limit) & (df_train1["X_10"] < upper_limit)]


# In[ ]:


df_train1.shape[0] - df_train2.shape[0]


# In[ ]:


lower_limit = df_test1["X_10"].mean() - 3*df_test1["X_10"].std()
upper_limit = df_test1["X_10"].mean() + 3*df_test1["X_10"].std()


# In[ ]:


df_test2 = df_test1[(df_test1["X_8"] > lower_limit) & (df_test1["X_8"] < upper_limit)]


# In[ ]:


df_test1.shape[0] - df_test2.shape[0]


# In[ ]:


lower_limit = df_train2["X_11"].mean() - 3*df_train2["X_11"].std()
upper_limit = df_train2["X_11"].mean() + 3*df_train2["X_11"].std()


# In[ ]:


df_train3 = df_train2[(df_train2["X_11"] > lower_limit) & (df_train2["X_11"] < upper_limit)]


# In[ ]:


df_train2.shape[0] - df_train3.shape[0]


# In[ ]:


lower_limit = df_test2["X_11"].mean() - 3*df_test2["X_11"].std()
upper_limit = df_test2["X_11"].mean() + 3*df_test2["X_11"].std()


# In[ ]:


df_test3 = df_test2[(df_test2["X_11"] > lower_limit) & (df_test2["X_11"] < upper_limit)]


# In[ ]:


df_test2.shape[0] - df_test3.shape[0]


# In[ ]:


lower_limit = df_train3["X_12"].mean() - 3*df_train3["X_12"].std()
upper_limit = df_train3["X_12"].mean() + 3*df_train3["X_12"].std()


# In[ ]:


df_train4 = df_train3[(df_train3["X_12"] > lower_limit) & (df_train3["X_12"] < upper_limit)]


# In[ ]:


df_train3.shape[0] - df_train4.shape[0]


# In[ ]:


lower_limit = df_test3["X_12"].mean() - 3*df_test3["X_12"].std()
upper_limit = df_test3["X_12"].mean() + 3*df_test3["X_12"].std()


# In[ ]:


df_test4 = df_test3[(df_test3["X_12"] > lower_limit) & (df_test3["X_12"] < upper_limit)]


# In[ ]:


df_test3.shape[0] - df_test4.shape[0]


# In[ ]:


lower_limit = df_train4["X_13"].mean() - 3*df_train4["X_13"].std()
upper_limit = df_train4["X_13"].mean() + 3*df_train4["X_13"].std()


# In[ ]:


df_train5 = df_train4[(df_train4["X_13"] > lower_limit) & (df_train4["X_13"] < upper_limit)]


# In[ ]:


df_train4.shape[0] - df_train5.shape[0]


# In[ ]:


lower_limit = df_test4["X_13"].mean() - 3*df_test4["X_13"].std()
upper_limit = df_test4["X_13"].mean() + 3*df_test4["X_13"].std()


# In[ ]:


df_test5 = df_test4[(df_test4["X_13"] > lower_limit) & (df_test4["X_13"] < upper_limit)]


# In[ ]:


df_test4.shape[0] - df_test5.shape[0]


# In[ ]:


df_train5.head()


# In[ ]:


df_test5.head()


# In[ ]:


df_train5.info()


# ## Here we can see that we have totally removed (23856 - 22628) = 1228 outliers from train data

# In[ ]:


df_test5.info()


# ## Here we can see that we have totally removed (15903  - 14981 ) = 922 outliers from test data

# ## Now Let's see  our Dataset has any null values or not

# In[ ]:


n = msno.bar(df_train5,color="purple")


# ## So we can see that our Dataset has no Null Values

# In[ ]:


x = df_train5.drop("MULTIPLE_OFFENSE",axis=1)
y = df_train5["MULTIPLE_OFFENSE"]


# ## So ,let's now scale our Dataset

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)


# In[ ]:


x


# ## Now the precious time has come to see which parameter is best to create a Good Model

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# In[ ]:


model_params ={
    "svm": {
        "model" : SVC(gamma="auto"),
        "params": {
            "C" : [1,10,20],
            "kernel": ["rbf"],
            "random_state":[0,10,100]
        }
    },
    
    "decision_tree":{
        "model": DecisionTreeClassifier(),
        "params":{
            "criterion": ["entropy","gini"],
            "max_depth": [5,8,9],
            "random_state":[0,10,100]
        }
    },
    "random_forest":{
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators" : [1,5,10],
            "max_depth" : [5,8,9],
            "random_state":[0,10,100]
        }
    },
    "naive_bayes":{
        "model": GaussianNB(),
        "params": {}
    },
    
    "logistic_regression":{
        "model" : LogisticRegression(solver='liblinear',multi_class = 'auto'),
        "params":{
            "C" : [1,5,10],
            "random_state":[0,10,100]
        }
    },
    "knn" : {
        "model" : KNeighborsClassifier(),
        "params": {
            "n_neighbors" : [5,12,13]
        }
    }
    
    
}


# In[ ]:


scores =[]
for model_name,mp in model_params.items():
    clf = GridSearchCV(mp["model"],mp["params"],cv=12,return_train_score=False)
    clf.fit(x,y)
    scores.append({
        "Model" : model_name,
        "Best_Score": clf.best_score_,
        "Best_Params": clf.best_params_
    })


# In[ ]:


result_score = pd.DataFrame(scores, columns = ["Model","Best_Score","Best_Params"])


# In[ ]:


result_score


# ## Wow, we can see that "Random Forest" & "Decision Tree" gives us almost 99% Accuracy

# In[ ]:


from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# In[ ]:


clf_dt = DecisionTreeClassifier(criterion= "gini",max_depth = 9,random_state=0)


# In[ ]:


clf_dt.fit(x_train,y_train)


# In[ ]:


y_pred = clf_dt.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# ## Let's see our Actual vs Predicted Values,Which says,if a server will be Hack or Not

# In[ ]:


result = pd.DataFrame({"Actual_Value": y_test, "Predicted_Value": y_pred})
result


# ## *** --------If you like it,then please do UpVote-------- ***

# In[ ]:




