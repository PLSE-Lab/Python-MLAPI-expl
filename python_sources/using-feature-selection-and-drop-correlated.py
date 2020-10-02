#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1450]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier


# In[1451]:


import os
os.listdir('../input')


# ### Util Functions

# In[1452]:


# Print the bar graph from data
def bar(acumm_data):
    # Do plot
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    ax = sns.barplot(x=acumm_data.index, y=acumm_data.values, palette='tab20b', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)    
    return ax

def corr_drop(corr_m, factor=.9):
    """
    Drop correlated features maintaining the most relevant.
    
    Parameters
    ----------
    corr_m : pandas.DataFrame
        Correlation matrix
    factor : float
        Min correlation level   
    
    Returns
    ----------
    pandas.DataFrame
        Correlation matrix only with most relevant features
    """
    global cm
    cm = corr_m
    # Get correlation score, as high as this score, more chances to be dropped.
    cum_corr = cm.applymap(abs).sum()
    def remove_corr():
        global cm
        for col in cm.columns:
            for ind in cm.index:
                if (ind in cm.columns) and (col in cm.index):
                    # Compare if are high correlated.
                    if (cm.loc[ind,col] > factor) and (ind!=col):
                        cum = cum_corr[[ind,col]].sort_values(ascending=False)
                        cm.drop(cum.index[0], axis=0, inplace=True)
                        cm.drop(cum.index[0], axis=1, inplace=True)
                        # Do recursion until the last high correlated.
                        remove_corr()
        return cm
    return remove_corr()


# # Reading data

# In[1453]:


df = pd.read_csv('../input/indian_liver_patient.csv')


# In[1454]:


df.info()


# In[1455]:


df.head()


# ## First exploratory analysis

# In[1456]:


df.describe()


# # Pre-proccessing

# ### Check for non-valid values.

# In[1457]:


flat_data = df.values.flatten()
count=0
for value in flat_data:
    if value is not None and value != np.nan and value != 'NaN' :
        continue
    count+= 1
pct_nan = round(100*count/len(flat_data))
print(f'{pct_nan}% of data are non-valid.')


# ### Convert Gender to binary.

# In[1458]:


df.Gender = df.Gender.replace({'Male': 0, 'Female': 1})
df.Dataset = df.Dataset-1


# In[1459]:


df.Albumin_and_Globulin_Ratio = df.Albumin_and_Globulin_Ratio.apply(pd.to_numeric)


# # Preparation and Analysis

# ### Analysing Age

# Visualizing data.

# In[1460]:


bar(df.Age.value_counts())
plt.show()


# __Obs__: Almost all ages represented, we can use than better if we bin it.

# Make 5 bins for Age.

# In[1461]:


df.Age = pd.cut(df.Age,
       [0, 25, 50,75,100],
       labels=[
           'lower than 25',
           '25-49',
           '50-74',
           'greater than 75'           
       ]      
)


# Visualizing data after create bins.

# In[1462]:


bar(df.Age.value_counts())
plt.show()


# Pass data to numeric.

# In[1463]:


labels=['lower than 25', '25-49', '50-74', 'greater than 75']
values = [0, 1, 2, 3]
dct_rpl = dict(zip(labels, values))
df.Age = df.Age.replace(dct_rpl)


# ### Separation features and classes

# In[1464]:


X = df.drop('Dataset', axis=1)
y = df.Dataset


# ### Scalling numeric values

# In[1465]:


num_features = [
    'Age',
    'Total_Bilirubin',
    'Direct_Bilirubin',
    'Alkaline_Phosphotase',
    'Alamine_Aminotransferase',
    'Aspartate_Aminotransferase',
    'Total_Protiens',
    'Albumin'
]


# In[1466]:


scaler = MinMaxScaler()
scaler.fit(X[num_features])
X[num_features] = scaler.transform(X[num_features])


# # Exploratory analysis

# In[1467]:


continium_features = num_features = [
    'Total_Bilirubin',
    'Direct_Bilirubin',
    'Alkaline_Phosphotase',
    'Alamine_Aminotransferase',
    'Aspartate_Aminotransferase',
    'Total_Protiens',
    'Albumin',
    'Albumin_and_Globulin_Ratio'
]

Xy = X.copy()
Xy['Class'] = y
sns.pairplot(Xy, hue="Class", size=3, diag_kind='kde', vars=continium_features)
plt.show()


# # Feature selection

# ### Select using univariate statistics tests

# In[1469]:


sel = SelectKBest(chi2,k=4)
sel.fit(X.iloc[:,:-1],y)

idxs = sel.get_support(indices=True)
choosen_features_1 = X.columns[idxs]
#pd.DataFrame(list(zip(choosen_features_1, sel.scores_, sel.pvalues_)), columns=['features', 'score', 'p-value'])


# ### Drop correlated

# In[1470]:


choosen_features_2 = corr_drop(X[choosen_features_1].corr(), 0.80).columns


# # Balance

# In[1471]:


bar(y.value_counts())
plt.show()


# # Modeling

# In[1472]:


X = X[choosen_features_2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[1473]:


knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
knn.score(X_test, y_test)


# In[1474]:


lr = LogisticRegression().fit(X_train, y_train)
lr.score(X_test, y_test)


# In[1475]:


rfc = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
rfc.score(X_test, y_test)


# In[1476]:


gbc = GradientBoostingClassifier(n_estimators=100).fit(X_train, y_train)
gbc.score(X_test, y_test)

