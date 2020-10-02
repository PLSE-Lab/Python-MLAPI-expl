#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# In[ ]:


data = pd.read_csv("../input/iris/Iris.csv")


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.groupby("Species").agg(["min", "max", "std", "mean"])


# In[ ]:


data.isna().sum() #checking nulls parameter


# ## Visualizations

# In[ ]:


data.head()


# In[ ]:


for column in data.columns[1:-1]:
    sns.scatterplot(data=data, x="Id", y=column, hue="Species")
    plt.show()


# # Outlier Detection

# ## 3 Sigma

# In[ ]:


for column in data.columns[1:-1]:
    for spec in data["Species"].unique():
        selected_spec = data[data["Species"] == spec]
        selected_column = selected_spec[column]
        
        std = selected_column.std()
        avg = selected_column.mean()
        
        three_sigma_plus = avg + (3 * std)
        three_sigma_minus =  avg - (3 * std)
        
        outliers = selected_column[((selected_spec[column] > three_sigma_plus) | (selected_spec[column] < three_sigma_minus))].index
        data.drop(index=outliers, inplace=True)
        print(column, spec, outliers)


# In[ ]:


for column in data.columns[1:-1]:
    sns.scatterplot(data=data, x="Id", y=column, hue="Species")
    plt.show()


# ## IQR - Quantile

# In[ ]:


for column in data.columns[1:-1]:
    for spec in data["Species"].unique():
        selected_spec = data[data["Species"] == spec]
        selected_column = selected_spec[column]
        
        q1 = selected_column.quantile(0.25)
        q3 = selected_column.quantile(0.75)
        
        iqr = q3 - q1
        
        minimum = q1 - (1.5 * iqr)
        maximum = q3 + (1.5 * iqr)
        
        print(column, spec, "| min= ", minimum, "max= ", maximum)
        
        max_idxs = data[(data["Species"] == spec) & (data[column] > maximum)].index
        print(max_idxs)
        min_idxs = data[(data["Species"] == spec) & (data[column] < minimum)].index
        print(min_idxs)
        
        data.drop(index=max_idxs, inplace=True)
        data.drop(index=min_idxs, inplace=True)


# In[ ]:


for column in data.columns[1:-1]:
    sns.scatterplot(data=data, x="Id", y=column, hue="Species")
    plt.show()


# In[ ]:


data.to_csv("final_data.csv") #prepocessing ends here


# In[ ]:


data = pd.read_csv("final_data.csv")


# In[ ]:


data.head()


# In[ ]:


data.drop("Unnamed: 0", axis=1, inplace=True)
data.head()


# ## Label Encoding 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[ ]:


le = LabelEncoder()


# In[ ]:


target = le.fit_transform(data["Species"]) 


# In[ ]:


data["Species"] = target
data.head()


# In[ ]:


data.isna().sum()  #conrolling data


# In[ ]:


data.dtypes #conrolling data


# In[ ]:


data.drop("Id", axis=1, inplace=True) # we do not need "Id" column


# In[ ]:


data.head()


# ## Train and Test Split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3)


# In[ ]:


display(X_train.shape)
X_test.shape


# In[ ]:


y_train.value_counts()


# In[ ]:


y_test.value_counts()


# ## Building Our Model

# In[ ]:


import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix


# In[ ]:


xgb_cls = xgb.XGBClassifier(objective="multi:softmax", num_class=3)


# In[ ]:


xgb_cls.fit(X_train, y_train)


# In[ ]:


preds = xgb_cls.predict(X_test)


# In[ ]:


accuracy_score(y_test, preds)


# In[ ]:


confusion_matrix(y_test, preds)


# In[ ]:




