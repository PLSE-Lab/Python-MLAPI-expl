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
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("/kaggle/input/lower-back-pain-symptoms-dataset/Dataset_spine.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# ## Here is a column which is unnamed,let's see whats this column contains

# In[ ]:


df["Unnamed: 13"][:30]


# ## so this column hold the names of the other columns,so let's named those columns & we don't need this column for further purposes,so first drop this column 

# In[ ]:


df.drop("Unnamed: 13",axis=1,inplace=True)


# In[ ]:


df.columns=["pelvic_incidence","pelvic_tilt","lumbar_lordosis_angle","sacral_slope",
            "pelvic_radius","degree_spondylolisthesis","pelvic_slope","Direct_tilt",
            "thoracic_slope","cervical_tilt","sacrum_angle","scoliosis_slope","state"]


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# ## Let's Visualize the Data

# In[ ]:


sns.set()
sns.regplot(x="pelvic_incidence", y = "pelvic_tilt",data = df)
plt.show()


# In[ ]:


sns.set()
sns.regplot(x = "pelvic_tilt",y="lumbar_lordosis_angle", data = df)
plt.show()


# ## Let's convert the string value of "state" column into numerical value

# In[ ]:


df["state"].value_counts()


# In[ ]:


mapping ={"Abnormal": 0,"Normal": 1}
df["state"] = df["state"].map(mapping)


# In[ ]:


df["state"].value_counts()


# In[ ]:


df.head()


# In[ ]:


sns.set()
sns.pairplot(df,hue="state")
plt.show()


# In[ ]:


sns.set()
sns.countplot(x = "state",data=df)
plt.show()


# In[ ]:


sns.set()
sns.distplot(df["pelvic_incidence"])
plt.show()


# In[ ]:


sns.set()
sns.distplot(df["lumbar_lordosis_angle"])
plt.show()


# In[ ]:


sns.set()
df.hist(figsize=(20,10),bins = 15,color="purple")
plt.title("Distribution of Features")
plt.show()


# ## Looks like Most of the Data are Not Normally Distributed

# In[ ]:


plt.figure(figsize=(20,10))
sns.set()
sns.boxplot(data=df,palette= "Set3")
plt.show()


# In[ ]:


df.columns


# ## So there are some Outlier,Let's remove those outliers

# In[ ]:


lower_limit = df["pelvic_incidence"].mean() - 3*df["pelvic_incidence"].std()
upper_limit = df["pelvic_incidence"].mean() + 3*df["pelvic_incidence"].std()


# In[ ]:


df2 = df[(df["pelvic_incidence"] > lower_limit) & (df["pelvic_incidence"] < upper_limit)]


# In[ ]:


df.shape[0] - df2.shape[0]


# In[ ]:


lower_limit = df2["pelvic_tilt"].mean() - 3*df2["pelvic_tilt"].std()
upper_limit = df2["pelvic_tilt"].mean() + 3*df2["pelvic_tilt"].std()


# In[ ]:


df3 = df2[(df2["pelvic_tilt"] > lower_limit) & (df2["pelvic_tilt"] < upper_limit)]


# In[ ]:


df2.shape[0] - df3.shape[0]


# In[ ]:


lower_limit = df3["lumbar_lordosis_angle"].mean() - 3*df3["lumbar_lordosis_angle"].std()
upper_limit = df3["lumbar_lordosis_angle"].mean() + 3*df3["lumbar_lordosis_angle"].std()


# In[ ]:


df4 = df3[(df3["pelvic_tilt"] > lower_limit) & (df3["pelvic_tilt"] < upper_limit)]


# In[ ]:


df3.shape[0] - df4.shape[0]


# In[ ]:


lower_limit = df4["degree_spondylolisthesis"].mean() - 2*df4["degree_spondylolisthesis"].std()
upper_limit = df4["degree_spondylolisthesis"].mean() + 2*df4["degree_spondylolisthesis"].std()


# In[ ]:


df5 = df4[(df4["degree_spondylolisthesis"] > lower_limit) & (df4["degree_spondylolisthesis"] < upper_limit)]


# In[ ]:


df4.shape[0] - df5.shape[0]


# In[ ]:


df5.head()


# In[ ]:


df5.info()


# In[ ]:


import missingno as msno
n = msno.bar(df5,color='purple')


# ## We can see that there is no Null Values

# In[ ]:


x = df5.drop("state",axis=1)
y = df5["state"]


# ## Now let's scale our Dataset

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)


# In[ ]:


x


# ## Let's find the best model for our Dataset

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


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state= 10)


# In[ ]:


clf_rf = RandomForestClassifier(max_depth = 8,n_estimators =10,random_state=0)
clf_rf.fit(x_train,y_train)


# In[ ]:


y_pred = clf_rf.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:


clf_lr = LogisticRegression(C=12,random_state = 100)
clf_lr.fit(x_train,y_train)


# In[ ]:


y_pred = clf_lr.predict(x_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# ## So we can see that, by using Logistic Regression we are getting 84% accuracy

# ## Now let's see our Actual vs Predicted Value

# In[ ]:


result = pd.DataFrame({"Actual_Value": y_test, "Predicted_Value": y_pred})


# In[ ]:


result


# In[ ]:




