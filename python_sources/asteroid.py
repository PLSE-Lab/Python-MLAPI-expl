#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


nRowRead = 3000 # specify 'None' if want to read whole file
original_dataset = pd.read_csv("/kaggle/input/prediction-of-asteroid-diameter/Asteroid_Updated.csv", delimiter = ',', nrows = nRowRead)
dataset = original_dataset.copy()


# In[ ]:



dataset.head(10)


# ## Describe Data

# In[ ]:


dataset.describe()


# Display Data Information

# In[ ]:


dataset.info()


# In[ ]:


dataset.hist(bins = 50, figsize = (20,15))


# In[ ]:


## Convert diameter To float
convertDict = {'diameter' : float}
dataset = dataset.astype(convertDict) 


# In[ ]:


corr_matrix = dataset.corr()
corr_matrix.columns
corr_matrix['diameter'].sort_values(ascending = False)


# In[ ]:


dataset.isnull().sum()


# # Data Cleaning
# ## Removing Numerical Columns
# As We can see that Columns:
# 'extent' has 10 non null values out of 3000,hence it will be illogical to fill Nan, because it does show any relation with diameter
# 'GM' has 11 non null values out of 3000,hence it will be illogical to fill Nan, because it does show any relation with diameter
# 'G' has 113 non null values out of 3000,hence it will be illogical to fill Nan, because it does show any relation with diameter
# 'IR' has 0 non null values
# (extent : 10/3000,GM : 11/3000, 113/3000, 'G' : 113/3000,'IR' : 0 /3000)
# Hence They should be removed from Dataframe

# In[ ]:


#(extent : 10/3000,GM : 11/3000, 113/3000, 'G' : 113/3000, IR : 0 /3000) Thse rows have maximun null value
dropColumn = ['extent','GM','G','IR']
dataset = dataset.drop(dropColumn, axis = 1)


# In[ ]:


dataset.info()


# In[ ]:


dataset['diameter'].describe()


# In[ ]:


dataset['diameter'].median()


# In[ ]:


# As per Analysis of  columns diameter, we should feel this column with its mean value
#dataset['diameter'].filna(dataset['diameter'].mean())
dataset['diameter'].fillna(dataset['diameter'].mean(), inplace=True)


# In[ ]:


dataset['diameter'].describe()


# In[ ]:


# As per Analysis of  columns albedo, we should feel this column with its median value
dataset['albedo'].fillna(dataset['albedo'].median(), inplace=True)
dataset['albedo'].describe()


# In[ ]:


# As per Analysis of  columns rot_per, we should feel this column with its mean value
dataset['rot_per'].fillna(dataset['rot_per'].mean(), inplace=True)
dataset['rot_per'].describe()


# In[ ]:


# As per Analysis of  columns BV,UB, we should feel this column with its mean value
dataset['BV'].fillna(dataset['BV'].mean(), inplace=True)
dataset['UB'].fillna(dataset['UB'].mean(), inplace=True)


# ## We have filled Numerical data
# Now lets analyse thse data with diameter columns values

# In[ ]:


dataset.info()


# # Analysys of Numerical data with Diameter

# In[ ]:


# Looking for Coorelation
corr_matrix = dataset.corr()
corr_matrix['diameter'].sort_values(ascending = False)


# In[ ]:


# dataset.plot(kind = 'scatter', x = 'rot_per',y = 'diameter', alpha = 0.6)
import seaborn as sns
#dataset.info()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_data = dataset.select_dtypes(include=numerics)
#num_data.info()
plt.subplots(figsize=(15,12))
sns.heatmap(num_data.corr(),annot=True,annot_kws={'size':10})
#num_data.corr()


# In[ ]:


# After analysing HeatMap we can element some columns which have no multicolinearity
#e,i,w, condition_cofde, n_obs_use,albedo,not_per,ma
dropNumColumn = ['e','i','w','condition_code','n_obs_used','rot_per','ma']
dataset = dataset.drop(dropNumColumn, axis = 1)


# In[ ]:


plt.subplots(figsize=(15,12))
num_data = dataset.select_dtypes(include=numerics)
sns.heatmap(num_data.corr(),annot=True,annot_kws={'size':10})


# ## Lets Play With Categorical Data

# In[ ]:


#corr_matrix.columns
dataset.head(10)


# In[ ]:


dataset.columns


# In[ ]:


categoricalData = dataset.select_dtypes(include=['object']).copy()
categoricalData.head(5)


# In[ ]:


categoricalData.isnull().sum()


# ## Fill Missing data in categorical

# In[ ]:


#categoricalData['spec_B'].value_counts()
categoricalData = categoricalData.fillna(categoricalData['spec_B'].value_counts().index[0])
categoricalData = categoricalData.fillna(categoricalData['spec_T'].value_counts().index[0])


# In[ ]:


# Columns wise Distribution
print(categoricalData.isnull().sum())


# ## Represent Data

# In[ ]:


# As we can see that
class_count = categoricalData['class'].value_counts()
sns.set(style="darkgrid")
sns.barplot(class_count.index, class_count.values, alpha=0.9)
plt.title('Frequency Distribution of Class')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Carrier', fontsize=12)
plt.show()


# In[ ]:


#dataset['neo'].value_counts()
categoricalData['pha'].value_counts()


# In[ ]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEnc = LabelEncoder()
categoricalData['neo'] = labelEnc.fit_transform(categoricalData['neo'])
categoricalData['pha'] = labelEnc.fit_transform(categoricalData['pha'])


# In[ ]:


categoricalData.head()


# In[ ]:


# Now do one hot encoder
categoricalData = pd.get_dummies(categoricalData, columns=['neo','pha'])
categoricalData.head()


# In[ ]:


from sklearn.preprocessing import LabelBinarizer
#categoricalDataClass = categoricalDataCopy.copy()
lb = LabelBinarizer()
lb_results = lb.fit_transform(categoricalData['class'])
lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)


# In[ ]:


categoricalData = pd.concat([categoricalData, lb_results_df], axis=1)
categoricalData.head()


# In[ ]:


categoricalData['class'].value_counts()


# In[ ]:



categoricalData['spec_B'] = labelEnc.fit_transform(categoricalData['spec_B'])
categoricalData['spec_T'] = labelEnc.fit_transform(categoricalData['spec_T'])


# In[ ]:


categoricalData.head()


# In[ ]:


# Now Drob Class column beacse it has been converted into LabelBinarizor
# Drop name column it jus a name
categoricalData.drop(['name','class'], inplace = True, axis = 1)


# In[ ]:


categoricalData.head(3)


# In[ ]:


num_data.head(3)


# ## Now Add Numerical and Categorical data which we hace cleaned and transformed

# In[ ]:


cleanDataset = pd.concat([categoricalData,num_data],axis = 1)


# In[ ]:


cleanDataset.head()


# #Split Data into features and target

# In[ ]:


#Split Data into features and target
y = cleanDataset['diameter']
X = cleanDataset.drop(['diameter'],axis = 1)


# In[ ]:


X = X.iloc[:,:].values
X.shape


# ## Feature Scaling
# Primarily, there are two types of feature scaling method:
# 1. min-max scaling(Normalization)
# (values -min) / (max - min)  # Lies 0-1
# 2. Standardization:
# (values - mean / std)
# for this sklearn provide class standardScaler
# 

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
     ('std_scaler', StandardScaler()),
    # Add as many as you can
])


# In[ ]:


X_std = my_pipeline.fit_transform(X)


# In[ ]:


X_std.shape


# # Split Data set into Training and test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.2, random_state = 0)


# ## Selecting a Desired Model for Our Project

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


model.predict(X_test)


# ## Evaluating Models

# In[ ]:


from sklearn.metrics import mean_squared_error
diameterPrediction  = model.predict(X_test)
lin_mse = mean_squared_error(y_test, diameterPrediction)
lin_mse = np.sqrt(lin_mse)


# In[ ]:


lin_mse


# # R-Square
# R-squared is a statistical measure that represents the goodness of fit of a regression model. The ideal value for r-square is 1. The closer the value of r-square to 1, the better is the model fitted.

# In[ ]:


from sklearn.metrics import r2_score
r2 = r2_score(y_test,diameterPrediction)
print("R2 : ",r2)


# ## Using Better Eveluation Techniques:
# How it works:
# 1 2 3 4 5 : it create  5 group(cv : fold) (example it may more)
# it trains 2 3 4 5 and test 1
# agian it trains 1 3 4 5 and test 2
#  and so on , finalyy returns score

# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, scoring = "neg_mean_squared_error", cv = 10)


# In[ ]:


rm_error = np.sqrt(-scores)# - because sqrt does not calculate negative value
rm_error


# In[ ]:


def print_score(score):
    print("Score: ", score)
    print("Mean: ", score.mean())
    print("Std: ", score.std())


# In[ ]:


print_score(rm_error)


# In[ ]:




