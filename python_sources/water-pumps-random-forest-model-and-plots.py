#!/usr/bin/env python
# coding: utf-8

# **Tanzania's Faulty Water Pumps Predictions using Random Forest and Plots**
# 
# This is an interesting dataset. The target variable consisted of three classes of water pumps: Functional, Non-function, and the ones that require repair work. This is a classic classification problem of how to accurately predict the classes. The challenge was if we can predict which pumps are functional.
# The dataset is decent size but requires some serious cleaning. Except for a few numerical variables, most of the features are categorical. The numeric features aren't very helpful given the number of missing values they have. However, they are important enough not to be thrown away. On the other hand, categorical variables don't have many missing values in comparison to numerical features, but almost each of them represents at least hundreds of categories. For example: the feature 'wpt_name' has 45600 types of names.In this notebook you will see:
# 1.Manual cleaning because I wanted to hand pick the number of categories which represented the most information (also to prevent my laptop from crashing).
# 2. pandas dummy encoding of categorical variables
# 3. Logistic Regression
# 4. Decision tree model
# 5. Random forests model
# 6. Cool visualizations!
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions


#let's import the warning before running any sophisticated methods
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train_features = pd.read_csv('../input/train_features.csv')
df_train_labels = pd.read_csv("../input/train_labels.csv")
df_test_features = pd.read_csv("../input/test_features.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


df_train_features.shape, df_test_features.shape, df_train_labels.shape


# In[ ]:


df_train_features.head()


# In[ ]:


df_train_labels.status_group.value_counts()


# In[ ]:


df_train_labels.status_group.value_counts(normalize=True)


# **Let's visualize the target variable**

# In[ ]:


import seaborn as sns
plt.figure(figsize=(14,7))
plt.title("Water Pumps by Functionality",fontsize=16, fontweight='bold')
sns.countplot(x=df_train_labels['status_group'], data=df_train_labels, palette="Greens_d");


# In[ ]:


df_pivot = pd.concat([df_train_features, df_train_labels], axis=1)
piv_df= df_pivot[['basin','status_group','construction_year']]
piv_table = piv_df.pivot_table(index='basin',
                           columns='status_group', aggfunc='count')
piv_table


# In[ ]:


df_pivot = pd.concat([df_train_features, df_train_labels], axis=1)
piv_df= df_pivot[['water_quality','status_group','basin']]
piv_table = piv_df.pivot_table(index='water_quality',
                           columns='status_group', aggfunc='count')
piv_table


#  **Merge Train and Test sets**
# it's the same formula, just don't mention axis=1
# The last row of our df_train_features dataset is 59394

# In[ ]:


#Let's merge train and test
full_df = pd.concat([df_train_features, df_test_features])


# In[ ]:


full_df.shape


# In[ ]:


59400 + 14358


# In[ ]:


full_df.head()


# The date recorded columns might indicate some inforamtion, however it's not usable in this format. It's a good idea to convert it into a column representing months. 

# In[ ]:


full_df['date_recorded_months'] = [(pd.to_datetime(date)-pd.to_datetime('2000-01-01')).days/30 for date in full_df['date_recorded']]


# I duplicated scheme_name feature to experiment with it. 

# In[ ]:


full_df['scheme_name_duplicate'] = full_df['scheme_name']


# The code below is taking only the categories whose frequency/occurance is more than 250 times in the column. Anything category with less than  250 frequency will be treated as NaN. That we I ensured we have only limited/managable ~10-30 categories in comparision to thousands.

# In[ ]:


full_df = full_df.apply(lambda x: x.mask(x.map(x.value_counts())<250, 'NaN') if x.name=='scheme_name_duplicate' else x)


# In[ ]:


full_df.scheme_name_duplicate.value_counts()


# In[ ]:


sum(full_df.gps_height.value_counts()>50)


# I repeated the process for the following additional categorical columns after checking the value counts for eachof them. 

# In[ ]:


full_df = full_df.apply(lambda x: x.mask(x.map(x.value_counts())<250, 'NaN') if x.name=='funder' else x)


# In[ ]:


full_df = full_df.apply(lambda x: x.mask(x.map(x.value_counts())<250, 'NaN') if x.name=='installer' else x)


# In[ ]:


full_df = full_df.apply(lambda x: x.mask(x.map(x.value_counts())<150, 'NaN') if x.name=='subvillage' else x)


# For the following numerical columns, before filling them with averages, I created the boolean columns to keep track of the NaNs. 

# In[ ]:


full_df['public_meeting_missing'] = full_df['public_meeting'].isna()


# In[ ]:


full_df['public_meeting'] = full_df['public_meeting'].fillna(full_df['public_meeting'].mode()[0])


# In[ ]:


full_df['scheme_management_missing'] = full_df['scheme_management'].isna()


# In[ ]:


full_df['scheme_management'] = full_df['scheme_management'].fillna(full_df['scheme_management'].mode()[0])


# In[ ]:


full_df['permit_missing'] = full_df['permit'].isna()


# In[ ]:


full_df['permit'] = full_df['permit'].fillna(full_df['permit'].mode()[0])


# There are some misisng values coded diferently in the dataset like the one below:

# In[ ]:


full_df = full_df.replace('none', np.NaN)


# In[ ]:


full_df = full_df.apply(lambda x: x.mask(x.map(x.value_counts())<100, 'NaN') if x.name=='wpt_name' else x)


# There are also NaNs coded as zeros 

# In[ ]:


full_df = full_df.replace('0', np.NaN)


# In[ ]:


full_df = full_df.apply(lambda x: x.mask(x.map(x.value_counts())<150, 'NaN') if x.name=='ward' else x)


# In[ ]:


full_df['construction_year_missing'] = (full_df['construction_year'] ==0)*1


# In[ ]:


#before filling the null, let's keep track of them

#to fill missing dates, we can use: mean, median or the oldest
mean_year = full_df[full_df['construction_year']>0]['construction_year'].mean()
full_df.loc[full_df['construction_year']==0, 'construction_year'] = int(mean_year)


# In[ ]:


full_df['gps_height_missing'] = full_df['gps_height'].isna()


# In[ ]:


mean_gps_height = full_df[full_df['gps_height']>0]['gps_height'].mean()
full_df.loc[full_df['gps_height']==0, 'gps_height'] = int(mean_gps_height)


# In[ ]:


full_df['num_private_missing'] = full_df['num_private'].isna()


# In[ ]:


mean_num_private = full_df[full_df['num_private']>0]['num_private'].mean()
full_df.loc[full_df['num_private']==0, 'num_private'] = int(mean_num_private)


# In[ ]:


full_df['population_missing'] = full_df['population'].isna()


# In[ ]:


mean_population = full_df[full_df['population']>0]['population'].mean()
full_df.loc[full_df['population']==0, 'population'] = int(mean_population)


# In[ ]:


full_df['amount_tsh_missing'] = full_df['amount_tsh'].isna()


# In[ ]:


mean_amount = full_df[full_df['amount_tsh']>0]['amount_tsh'].mean()
full_df.loc[full_df['amount_tsh']==0, 'amount_tsh'] = int(mean_amount)


# In[ ]:


full_df_selected_columns = full_df.drop(columns=['scheme_name','date_recorded','lga','recorded_by', 'waterpoint_type_group','source', 'quality_group',
                                                'payment_type', 'management_group', 'extraction_type',
                                                'extraction_type_group', 
                                                 ]) 


# In[ ]:


full_df_selected_columns.head()


# **Hot Encoding the categorical features**

# In[ ]:


#full_df_selected_columns['average_amount'] = full_df_selected_columns['amount_tsh']/ full_df_selected_columns['population']


# In[ ]:


import pandas as pd
df_main = pd.get_dummies(full_df_selected_columns)
#pd.set_option('display.max_columns', None)


# In[ ]:


df_main.head()


# In[ ]:


df_main.shape


# **Splitting the data back into the shape it was originally**

# In[ ]:


#split the data back
X_cleaned = df_main[:-14358]
X_test_main_cleaned = df_main[-14358:]
y = df_train_labels['status_group']


# In[ ]:


X_cleaned.shape, X_test_main_cleaned.shape, y.shape


# **Splitting train set into train and test for model training and predictions**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y, test_size=0.25, random_state=42, shuffle=True)


# In[ ]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


X_train.head()


# **First Try Decision Tree**

# In[ ]:


from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(X=X_train, y=y_train)
clf.feature_importances_ 
clf.score(X=X_test, y=y_test) # 1.0


# **Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.cluster import hierarchy as hc


# In[ ]:


m = RandomForestClassifier(n_estimators=200,min_samples_leaf=3 ,n_jobs=-1,max_features=0.25)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
y_pred= m.predict(X_test)
accuracy_score(y_test, y_pred)


# **Finally, let's plot more plots!**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
from statsmodels.graphics.mosaicplot import mosaic


# In[ ]:


df_viz = pd.concat([df_train_features, df_train_labels['status_group']], axis=1)


# In[ ]:


df_viz.shape


# In[ ]:


df_viz[df_viz['longitude']>0] [df_viz['latitude']<0][df_viz['construction_year']>0].plot(kind='scatter', x="longitude", y="latitude", alpha=0.4,
s=df_viz["population"]/10, label="population", figsize=(14,10),

c="construction_year", cmap=plt.get_cmap("jet"), colorbar=True,
sharex=False);
plt.title("Population Size, Construction Years, & Locations of Waterpumps in Tanzania", 
         fontsize =16, fontweight='bold')
plt.legend;


# The above plot is Tanzania's actaul map. Up North, West, and South West have three major rivers (googel maps). Seems like that waterpumps were not installed near lakes. May be the idea was to make water more accessible to the areas with less water. 

# In[ ]:


plt.figure(figsize=(14, 7))
sns.distplot(df_viz['construction_year'][df_viz['construction_year']>0]);
plt.title("Water Pump Construction by Years", fontsize=16, fontweight='bold')


# In[ ]:


corr_table = df_viz.corr()
plt.figure(figsize=(14,13))
sns.heatmap(corr_table, square=True, annot=True, cbar=False);


# In[ ]:


plt.figure(figsize=(16,14))
sns.set(style='ticks')
sns.pairplot(df_viz[['population', 'num_private', 'amount_tsh', 'status_group']],            hue='status_group', diag_kind='kde');


# In[ ]:


sns.set(style='ticks')
sns.pairplot(df_viz[['longitude', 'latitude', 'gps_height', 'status_group']],            hue='status_group', diag_kind='kde');


# In[ ]:




