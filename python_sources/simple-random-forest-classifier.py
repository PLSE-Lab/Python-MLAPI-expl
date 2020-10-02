#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 
from sklearn.ensemble import RandomForestClassifier


# In[ ]:



train = pd.read_csv("../input/learn-together/train.csv") # training data set
test = pd.read_csv("../input/learn-together/test.csv") # testing data set


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.head(20)


# In[ ]:


test.head()


# In[ ]:


train.info()


# In[ ]:





# In[ ]:


train.isnull().sum()


# In[ ]:





# In[ ]:


#colormap = plt.cm.RdBu cmap=colormap,
plt.figure(figsize=(50,35))
cor=train.corr()
sns.heatmap(cor,  square=True,  linecolor='white', annot=True ,  cmap="viridis")


# In[ ]:


#Correlation with output variable
cor_target = abs(cor["Cover_Type"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.1]
relevant_features


# In[ ]:


sns.barplot(x="Cover_Type", y="Horizontal_Distance_To_Roadways" , data=train)


# In[ ]:


sns.barplot(x="Cover_Type", y="Wilderness_Area1" , data=train)


# In[ ]:


sns.barplot(x="Cover_Type", y="Wilderness_Area3" , data=train)


# In[ ]:


sns.scatterplot(x="Cover_Type", y="Soil_Type39" , data=train)


# > from the relevant_features we can determine the most correlated features, so we will use these features in our algorithms

# In[ ]:


X=train[["Horizontal_Distance_To_Roadways","Wilderness_Area1",
        "Wilderness_Area3","Soil_Type10","Soil_Type12","Soil_Type22",
        "Soil_Type23","Soil_Type24","Soil_Type29","Soil_Type32","Soil_Type35"
        ,"Soil_Type38","Soil_Type39","Soil_Type40"]]
y=train["Cover_Type"]


# In[ ]:


X.head()


# In[ ]:


X.info()


# there is not any null values in selected features and also the features are integers 
# 

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=44, shuffle =True)


# In[ ]:


rc=RandomForestClassifier(n_estimators=100, max_depth=15)
rc.fit(X_train,y_train)


# In[ ]:


print("rc score is",rc.score(X_train,y_train))
print(":::::::::::::::::::::::")
print("rc score is",rc.score(X_val,y_val))


# In[ ]:


X_test=test[["Horizontal_Distance_To_Roadways","Wilderness_Area1",
        "Wilderness_Area3","Soil_Type10","Soil_Type12","Soil_Type22",
        "Soil_Type23","Soil_Type24","Soil_Type29","Soil_Type32","Soil_Type35"
        ,"Soil_Type38","Soil_Type39","Soil_Type40"]]


# In[ ]:


X_test.head()


# In[ ]:


y_predict =rc.predict(X_test)


# In[ ]:


print(y_predict)


# In[ ]:


output = pd.DataFrame({'Id': test.Id,
                       'Cover_Type': y_predict})
output.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




