#!/usr/bin/env python
# coding: utf-8

# Importing necessary libs

# In[ ]:


import pandas as pd
import tensorflow
import numpy
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler


# I ran this code on Google Colab with mounted Drive so specify path of dataset if running on local. If using Kaggle specify kaggle library.

# This Dataset is of a server log for week1 and classifies sessions as suspicious or not. We are trying to see what factors are responsible for malicious activities. I have done some basic preprocessing steps followed by EDA.

# In[ ]:


df = pd.read_csv("../input/server-logs-suspicious/CIDDS-001-external-week1.csv")


# In[ ]:


df.head()


# In[ ]:


df.columns


# Checking for NULL values below. Some classification models on sklearn do not work well with missing values therefore it is necessary to remove or impute them with another well fitted statistical measure

# In[ ]:


df.isnull().sum()


# In[ ]:


df.dtypes


# We have the class variable which we need to predict later on. It has following unique values.

# In[ ]:


df["class"].unique()


# In[ ]:


data = df.copy()
data.shape


# We are using seaborn plots to visualize our data

# We can see that majority of the data is suspicious sessions so our model will learn better on classifiying suspicious class.

# In[ ]:


sns.set(style = "darkgrid")
sns.countplot(x = "class",data=data)


# The Flags field is a string of some flag variables which are necessary in the networking field. Wherever a certain flag is not set a "." is used. We need to break this columns into individual variables for flags as it will help the model learn faster and establish better relations. 

# In[ ]:


df["Flags"].unique()


# In[ ]:


df["A"]=0
df["P"]=0
df["S"]=0
df["R"]=0
df["F"]=0
df["x"]=0


# In[ ]:


def set_flag(data,check):
    val=0;
    if(check in list(data["Flags"])):
        val = 1 ;
    return val;


# In[ ]:


df.columns


# In[ ]:


df["A"] = df.apply(set_flag,check ="A", axis = 1)
df["P"] = df.apply(set_flag,check = "P" ,axis = 1)
df["S"] = df.apply(set_flag,check ="S",axis = 1)
df["R"] = df.apply(set_flag,check="R" ,axis = 1)
df["F"] = df.apply(set_flag,check ="F" ,axis = 1)
df["x"] = df.apply(set_flag,check ="x" ,axis = 1)


# Checking here the individual flag variables and impact of each variable on class. You can change the variable name in the below plot and see the impact.

# In[ ]:


sns.countplot(x="S",hue = "class",data=df)


# In[ ]:


sns.countplot(x = "Proto",hue = "class",data = df)


# Dropping some unnecessary columns and columns having a single value like flows and tos

# In[ ]:


df=df.drop(columns = ["Date first seen","attackType","attackID","attackDescription","Flows","Tos","Flags"])


# In[ ]:


df.head()


# The Bytes variable was an object as seen in head command and the model would not recognize it as a number. Therefore we convert the M in the number to a multiplication of 1M with the number part. This has been simply done using regex

# In[ ]:


import re
def convtonum(data):
    num1=data["Bytes"]
    if "M" in data["Bytes"]:
        num=re.findall("[0-9.0-9]",data["Bytes"])
        num1 = float("".join(num))*100000
    num1 = float(num1)
    return num1


# In[ ]:


df["Bytes"] = df.apply(convtonum,axis = 1)


# In[ ]:


df.head()


# Label Encoding categorical values.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
col = ["Proto","class","Src IP Addr","Dst IP Addr"]
enc = LabelEncoder()
for col_name in col:
    df[col_name]=enc.fit_transform(df[col_name])


# Correlation Heatmap shows how each variable is correlated with class variable which we will try to predict.

# In[ ]:


data1 = df.copy()
plt.figure(figsize=(18,5))
sns.heatmap(data1.corr(),annot=True,cmap = "RdYlGn")


# In[ ]:


data_y = data1["class"]
data_x = data1.drop(columns = ["class"])


# Breaking dataset into train and test sets randomly

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=1)


# In[ ]:


#decision-tree-classifier - single-tree-classifier  // using all features

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion="entropy", max_depth=10) # you can use GINI index also here as a critirion 
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


# We got a 99% accuracy on the first go we can check further if it is overfitting and also see important variable in the model 

# In[ ]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.show()


# In[ ]:


# if you want to select most important features from an algorithm use recursive feature elimination and run algorithm on that

from sklearn.feature_selection import RFE

m = DecisionTreeClassifier(criterion="entropy", max_depth=10)
rfe = RFE(m,8)
fit=rfe.fit(X_train,y_train)

print(X_train.columns)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)


# Please upvote the kernel if it was helpful

# In[ ]:






# In[ ]:




