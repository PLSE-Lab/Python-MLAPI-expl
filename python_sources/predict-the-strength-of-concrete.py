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


# <h2 style="color:blue" align="left"> 1. Import necessary Libraries </h2>

# In[ ]:


# Read Data
import numpy as np     # Linear Algebra (calculate the mean and standard deviation)
import pandas as pd    # manipulate data, data processing, load csv file I/O (e.g. pd.read_csv)

# Visualization
import seaborn as sns                  # Visualization using seaborn
import matplotlib.pyplot as plt        # Visualization using matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly                          # Visualization using Plotly
import plotly.express as px
import plotly.graph_objs as go

# style
plt.style.use("fivethirtyeight")       # Set Graphs Background style using matplotlib
sns.set_style("darkgrid")              # Set Graphs Background style using seaborn

# ML model building; Pre Processing & Evaluation
from sklearn.model_selection import train_test_split                     # split  data into training and testing sets
from sklearn.linear_model import LinearRegression, Lasso, Ridge          # Linear Regression, Lasso and Ridge
from sklearn.tree import DecisionTreeRegressor                           # Decision tree Regression
from sklearn.ensemble import RandomForestRegressor                       # this will make a Random Forest Regression
from sklearn import svm                                                  # this will make a SVM classificaiton
from sklearn.svm import SVC                                              # import SVC from SVM
from sklearn.metrics import confusion_matrix, classification_report      # this creates a confusion matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import roc_curve,auc                                # ROC
from sklearn.preprocessing import StandardScaler                         # Standard Scalar
from sklearn.model_selection import GridSearchCV                         # this will do cross validation
from sklearn.decomposition import PCA                                    # to perform PCA to plot the data

import warnings                                                          # Ignore Warnings
warnings.filterwarnings("ignore")


# <h2 style="color:blue" align="left"> 2. Load data </h2>

# In[ ]:


# Import first 5 rows
df = pd.read_csv("/kaggle/input/concrete-compressive-strength-data-set/compresive_strength_concrete.csv")
df.head()


# In[ ]:


# checking dimension (num of rows and columns) of dataset
df.shape


# #### Checking for Numerical and Categorical features

# In[ ]:


# check dataframe structure like columns and its counts, datatypes & Null Values
df.info()


# In[ ]:


# check the datatypes
df.dtypes


# - Observed that **there is no categorical features** in this dataset. Only have **numerical features of int64 & float64**.

# In[ ]:


# Gives number of data points in each variable
df.count()


# <h2 style="color:blue" align="left"> 3. EDA (Exploratory Data Analysis) </h2>
# 
# - EDA is a way of **Visualizing, Summarizing and interpreting** the information that is **hidden in rows and column** format.
# 
# - Find Unwanted Columns
# - Find Missing Values
# - Find Features with one value
# - Explore the Categorical Features
# - Find Categorical Feature Distribution
# - Relationship between Categorical Features and Label
# - Explore the Numerical Features
# - Find Discrete Numerical Features
# - Relation between Discrete numerical Features and Labels
# - Find Continous Numerical Features
# - Distribution of Continous Numerical Features
# - Relation between Continous numerical Features and Labels
# - Find Outliers in numerical features
# - Explore the Correlation between numerical features

# ### 1. Find Unwanted Columns

# - There is no unwanted column present in given dataset to remove.
# 
#      EX: ID, S.No etc

# ### 2. Find Missing Values

# #### Mising Value Checking
# - I use five function for mising value checking
#    - insull()
#       - If any value is null return True
#       - Otherewise return False
#       
#    - isnull().any()
#       - If any columns have null value return True
#       - Otherewise return False
#       
#    - isnull().sum()
#       - If any columns have null value return how many null values have
#       - If no null value present return 0
#       
#    - missingno()
#       - Showing values by bar graph
#       
#    - Heatmap()
#       - Showing values by graph

# In[ ]:


# isnull() check null value
df.isnull()


# In[ ]:


# Listing Number of missing values by feature column wise.
df.isnull().sum()


# In[ ]:


# any() check null values by columns
df.isnull().any()


# - It shows all columns as False means no NULL Values present in dataset.

# In[ ]:


# Missing value representation by Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.xticks(fontsize=14)
plt.title('Count of Missing Values by Heat Map', fontsize=20, fontweight = 'bold')
plt.show()


# - From graph understood that there is **no missing values** in this dataset.
