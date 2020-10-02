#!/usr/bin/env python
# coding: utf-8

# # Employee Attrition Rate. Will your employee leave you?

# ## Importing Necessary Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error


# #### Let's find out the path of input files.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Reading Dataset

# In[ ]:


train = pd.read_csv("/kaggle/input/Train.csv") 
test = pd.read_csv("/kaggle/input/Test.csv")


# In[ ]:


train.columns


# In[ ]:


print(train.shape)
train.head()


# In[ ]:


train.describe()


# In[ ]:


train.info()


# #### Checking if there are some missing values in the data or not

# In[ ]:


train.isna().any()


# ## Data Visualization and Preprocessing

# #### Plotting the correlation matrix for the dataset

# In[ ]:


#Using Pearson Correlation
plt.figure(figsize=(18,10))
cor = train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Accent)
plt.show()
plt.savefig("main_correlation.png")


# ### Labels and Features

# In[ ]:


label = ["Attrition_rate"]
features = ['VAR7','VAR6','VAR5','VAR1','VAR3','growth_rate','Time_of_service','Time_since_promotion','Travel_Rate','Post_Level','Education_Level']


# In[ ]:


featured_data = train.loc[:,features+label]
featured_data = featured_data.dropna(axis=0)
featured_data.shape


# In[ ]:


X = featured_data.loc[:,features]
y = featured_data.loc[:,label]


# ### Splitting the Dataset

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.55)


# ## Learning Model

# In[ ]:


df = Ridge(alpha=0.000001)
df.fit(X_train,y_train)
y_pred = df.predict(X_test)
c=[]
for i in range(len(y_pred)):
    c.append((y_pred[i][0].round(5)))
pf=c[:3000]

print(len(c),len(pf),c[0])


# In[ ]:


score = 100* max(0, 1-mean_squared_error(y_test, y_pred))
print(score)


# ## Data Preprocessing Test file

# In[ ]:


selected_test = test.loc[:,features]
#selected_test.info()
mean_values = np.mean(selected_test)
selected_test[features].replace(mean_values,np.nan,inplace=True)
for i,val in enumerate(features):
    selected_test[val] = selected_test[val].fillna(mean_values[i])
    
selected_test.head()


# ## Prediction

# In[ ]:


#Predicting
import pandas as pd
dff = pd.DataFrame({'Employee_ID':test['Employee_ID'],'Attrition_rate':pf})
#Converting to CSV
dff.to_csv("Predictions.csv",index=False)


# #### The final test submission score.

# ![final test submission score](https://raw.githubusercontent.com/blurred-machine/HackerEarth-Machine-Learning-Challenge/master/final_test_score.PNG)
