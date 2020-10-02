#!/usr/bin/env python
# coding: utf-8

# # Index:
# I [Import necessary packages & files](#first-bullet)
# 
# II [Understanding the problem](#second-bullet)
# 
# III [EDA](#third-bullet)
# 
#     A. Looking at size of data
#     
#     B. Properties of the target variable
#     
#     C. Properties of the features: Finding some peculirarities and dependencies between features and target variable
#     
#     D. Generate ideas for feature engineering and future hypothesis

# # I. Importing necessary packages & files

# #### A. Importing packages

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import numpy as np


# #### B. Importing files

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:



train_dataset = pd.read_csv("/kaggle/input/learn-together/train.csv")
test_dataset = pd.read_csv("/kaggle/input/learn-together/test.csv")


# In[ ]:


train_dataset.head()


# # II. Understand the problem

# Data Type: Tabluar data
# 
# Problem Type: Classification
# 
# Competition Metric: Categorization Accuracy, percentage of correct predictions

# In[ ]:


def evaluate_metric_score(y_true, y_pred):
    if y_true.shape[0] != y_pred.shape[0]:
        raise Exception("Sizes do not match")
        return 0
    else:
        size = y_true.shape[0]
        matches = 0
        y_true_array = np.array(list(y_true))
        y_pred_array = np.array(list(y_pred))
        for i in range(0,size):
            if y_true_array[i]==y_pred_array[i]:
                matches = matches + 1
        return matches/size


# # III. Initial EDA
# 
# Goals of EDA:
# * Size of the data
# * Properties of the target variable (check for issues like high class imbalance, skewed distribution in a regression)
# * Properties of the features: Finding some peculirarities and dependencies between features and target variable is always useful
# * Generate ideas for feature engineering and future hypothesis

# ## A. Size of the data

# In[ ]:


train_dataset.shape


# In[ ]:


test_dataset.shape


# In[ ]:


X = train_dataset.copy()
X = X.drop(columns=['Cover_Type'])
y = train_dataset[['Cover_Type']]


# In[ ]:


X.head()


# In[ ]:


X.columns


# In[ ]:


train_dataset.describe()


# In[ ]:


train_dataset.info()


# In[ ]:


test_dataset.describe()


# In[ ]:


test_dataset.info()


# .describe() & .info() gave the constant count at 15120 (train dataset) and 565892 (test dataset) across all columns
# Thus no null values

# ## B. Properties of target variable

# In[ ]:


y['Cover_Type'].value_counts()


# #### => There is an equal class distribution

# ## C. Properties of features

# Start with general understanding of features
# 1. Categorical
# 2. Continuous
# 3. Categorical v/s Continuous

# Checking 'Id' Column whether it is unique with entries

# In[ ]:


len(X['Id'].unique()) == len(X)


# #### Based on description: 
# 
# #### Below columns seem to have continuous numeric data 
# 'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm','Horizontal_Distance_To_Fire_Points'
# 
# 
# #### Below columns seem to have categorical data 
# Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
# Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
# Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation

# In[ ]:


#List of continuous numeric data
list_contfeatures=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm','Horizontal_Distance_To_Fire_Points']


# In[ ]:


len(list_contfeatures)


# In[ ]:


check = train_dataset.copy()
check_test = test_dataset.copy()


# In[ ]:


#Setting values to string type
check['Cover_Type']=check['Cover_Type'].astype(str)


# #### Looking at how these features are distributed & how these vary with Cover Type (using Boxplot)

# In[ ]:


for col in list_contfeatures:
    sns.catplot(x='Cover_Type',y=col,data=check,kind='box')


# #### Creating a new column to compress one-hot encoded features in wilderness area for visualization purposes
# We know every observation corresponds to 30mx30m patch. We are now checking whether each 30mx30m has unique:
# 1. Cover_Type 
# 2. Wilderness_Area
# 3. Soil_Type
# 
# We know that for 1. Cover_Type, this column contains an integer value thus every 30x30 patch is mapped to a cover type
# But for Wilderness Area  we are unsure as these are one-hot encoded. 
# These being binary values, we will sum it across the different values

# Checking Hypothesis for 2. Wilderness_area in training & testdata just to be sure

# In[ ]:


list_feat_wildarea=[]
str1 = "Wilderness_Area"
for i in range(1,5):
    str2=str1+str(i)
    list_feat_wildarea.append(str2)
check['Wilderness_Area_sum']=check[list_feat_wildarea].sum(axis=1)
check_test['Wilderness_Area_sum']=check_test[list_feat_wildarea].sum(axis=1)


# In[ ]:


#Checking in train data whether it contains only 1 unique value and it should be 1
check['Wilderness_Area_sum'].unique()


# In[ ]:


#Checking in test data whether it contains only 1 unique value and it should be 1
check_test['Wilderness_Area_sum'].unique()


# #### => Confirmed every 30mx30m belongs to only one Wildereness_Area
# Now next to Soil_Type

# In[ ]:


list_feat_soiltype=[]
str1 = "Soil_Type"
for i in range(1,41):
    str2=str1+str(i)
    list_feat_soiltype.append(str2)
check['Soil_Type_sum']=check[list_feat_soiltype].sum(axis=1)
check_test['Soil_Type_sum']=check_test[list_feat_soiltype].sum(axis=1)


# In[ ]:


#Checking in train data whether it contains only 1 unique value and it should be 1
check['Soil_Type_sum'].unique()


# In[ ]:


#Checking in test data whether it contains only 1 unique value and it should be 1
check_test['Soil_Type_sum'].unique()


# #### => Confirmed every 30mx30m contains only one Soil Type

# Lets create 2 columns for:
#     1. 4 feature columns corresponding to Wilderness Areas (naming it 'Wilderness_Area')
#     2. 40 feature columns corresponding to Soil Types (naming it 'Soil_Type')
# => In essence we have to analyze these categorical features (compressing 44 cols to 2 cols).This will prove useful when for EDA purposes
# However for model building we preserve the one-hot encoded features

# In[ ]:


str1="Wilderness_Area"
for i in range(1,5):
    str2=str1+str(i)
    check.loc[(check[str2]==1),str1]=str2
    check_test.loc[(check_test[str2]==1),str1]=str2


# In[ ]:


#Lets check the uniqueness - should have only 4 categories
check['Wilderness_Area'].unique()


# In[ ]:


check_test['Wilderness_Area'].unique()


# In[ ]:


str1="Soil_Type"
not_present =[]
for i in range(1,41):
    str2=str1+str(i)
    check.loc[(check[str2]==1),str1] = str2
    if len(check.loc[(check[str2]==1),str1]) ==0:
        not_present.append(str2)
    check_test.loc[(check_test[str2]==1),str1]=str2


# In[ ]:


check['Soil_Type'].unique()


# In[ ]:


not_present


# #### => Soil types 7 & 15 are absent in train dataset

# In[ ]:


len(check_test['Soil_Type'].unique())==40


# ### Now lets analyze the relationship of numerical & categorical features
# Just to recap for our EDA, we have:
# 
# A. 3 categorical variables: 2 features detailing 
#     1. Wilderness_Area & 
#     2. Soil_Type 
#     3. and (1 Target feature:) Cover_Type
# B. 10 numerical variables (all of which are features currently):
#     1. 'Elevation',
#     2. 'Aspect', 
#     3. 'Slope', 
#     4. 'Horizontal_Distance_To_Hydrology', 
#     5. 'Vertical_Distance_To_Hydrology', 
#     6. 'Horizontal_Distance_To_Roadways', 
#     7. 'Hillshade_9am', 
#     8. 'Hillshade_Noon', 
#     9. 'Hillshade_3pm',
#     10. 'Horizontal_Distance_To_Fire_Points'

# ### 1. Only on Categorical Features (Categorical v/s Categorical)

# #### 1.A.  Wilderness Area(Countplot)

# In[ ]:


sns.catplot(x='Wilderness_Area',data=check,kind='count',order=['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'])
plt.xticks(rotation=90)


# #### => Wilderness Area 3 has higher number of number of observations compared to Wilderness Area 2

# #### 1.B.  Soil Type (Countplot)

# In[ ]:


plt.figure(figsize=(50,10))
g= sns.catplot(x='Soil_Type',data=check,kind='count',height=4,aspect=2)
plt.xticks(rotation=90)


# Understanding the distribution of points within these numerical features wrt cover type

# #### 2.A. Cover Type v/s Numerical Features (for different Wilderness_Areas) Histogram

# In[ ]:


fig,ax = plt.subplots()
ax.hist(check['Elevation'])
fig.set_size_inches([5,5])
plt.show()


# ### 2. Continuous Features

# #### 2.A. Histograms

# In[ ]:


for col in list_contfeatures:
    fig,ax=plt.subplots()
    ax.hist(check[col])
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Number of occurrences")
    fig.set_size_inches([5,5])
    plt.show()


# #### These observations are majorly within the range of 2000 - 3500 meters elevation

# #### 2.B. Calculation of Correlation values

# In[ ]:


check[list_contfeatures].corr()


# In[ ]:


check_corr=check[list_contfeatures].corr().abs()


# In[ ]:


# Generate a mask for the upper triangle
mask = np.zeros_like(check_corr, dtype=np.bool)
# To extract values corresponding to upper(u) triangle(tri) of mask use np.triu_indices_from
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(check_corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5)


# #### 2.C. Scatter plots

# I dont think it will be useful to create scatter plots blindly. 
# Here we had 10 numerical features
# So this would account for 10C2 (45) plots
# So lets take hint from correlation metrics

# In[ ]:


xdf=check_corr.mask(check_corr<0.4).mask(check_corr==1)


# In[ ]:


xdf


# In[ ]:


xdf.to_csv("correlation_values_with0_5mask.csv")


# #### 2.C.i) For Elevation, Horizontal Distance is the highly correlated feature with correlation at 0.578659

# In[ ]:


sns.regplot(x="Elevation",y="Horizontal_Distance_To_Roadways",data=check)


# In[ ]:


sns.relplot(x="Elevation",y="Horizontal_Distance_To_Roadways",data=check,row="Cover_Type",kind='scatter')


# #### 2.C.ii) For Aspect, Hillshade_3pm is the highly correlated feature with correlation at 0.635022

# In[ ]:


sns.regplot(x="Aspect",y="Hillshade_3pm",data=check)


# In[ ]:


sns.relplot(x="Aspect",y="Hillshade_3pm",data=check,row="Cover_Type",kind='scatter')


# #### 2.C.iii) For Slope, Hillshade_noon is the highly correlated feature with correlation at 0.612613

# In[ ]:


sns.regplot(x='Slope',y='Hillshade_Noon',data=check)


# In[ ]:


sns.relplot(x='Slope',y='Hillshade_Noon',data=check,row='Cover_Type')


# #### 2.C.v) For Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology is the highly correlated feature with correlation at 0.652142

# In[ ]:


sns.regplot(x='Horizontal_Distance_To_Hydrology',y='Vertical_Distance_To_Hydrology',data=check)


# In[ ]:


sns.relplot(x='Horizontal_Distance_To_Hydrology',y='Vertical_Distance_To_Hydrology',data=check,row='Cover_Type')


# 2.C.vi) For Vertical_Distance_To_Hyrdology, Vertical_Distance_To_Hydrology is the highly correlated features (v)

# 2.C.vi) For Horizontal_Distance_To_Roadways, Elevation is the highly correlated feature with correlation (i)

# #### 2.C.vii) For Hillshade_9am,Hillshade_3pm is the highly correlated feature with correlation at 0.779965

# In[ ]:


sns.regplot(x='Hillshade_9am',y='Hillshade_3pm',data=check)


# In[ ]:


sns.relplot(x='Hillshade_9am',y='Hillshade_3pm', data=check,kind='scatter',row='Cover_Type')


# 2.C.viii) For Hillshade_Noon, Elevation is the highly correlated feature with correlation at 0.578 (i)

# 2.C.ix) For Hillshade_3pm, Hillshade_Noon is the highly correlated feature with correlation at 0.7799647 (vii)

# #### 2.C.x) For Horizontal_Distance_To_Fire_Points, Hillshade_9am is the highly correlated feature with correlation at 0.486385

# In[ ]:


sns.regplot(x='Horizontal_Distance_To_Fire_Points',y='Hillshade_9am',data=check)


# In[ ]:


sns.relplot(x='Horizontal_Distance_To_Fire_Points',y='Hillshade_9am',data=check,row='Cover_Type')


# ### 3. Categorical v/s Numerical Features

# #### 3.A. Cover Type v/s Numerical Features (for different Wilderness_Areas) Boxplot

# In[ ]:


for col in list_contfeatures:
    fig1,ax1 = plt.subplots(figsize=(20,10))
    g=sns.boxplot(ax=ax1,x='Cover_Type',y=col,data=check,hue='Wilderness_Area',hue_order=['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'])
    ax1.set_title(col,fontsize=24)
    plt.show()


# #### 3.B. Cover Type v/s Numerical Features (for different Wilderness_Areas) Swarmplot

# In[ ]:


for col in list_contfeatures:
    fig1,ax1 = plt.subplots(figsize=(20,10))
    g=sns.swarmplot(ax=ax1,x='Cover_Type',y=col,data=check,hue='Wilderness_Area',hue_order=['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'])
    ax1.set_title(col,fontsize=24)
    plt.show()


# #### 3.C. Cover Type v/s Numerical Features to understand distribution for different Wilderness_Areas (ViolinPlot)

# In[ ]:


for col in list_contfeatures:
    fig1,ax1 = plt.subplots(figsize=(20,10))
    g=sns.violinplot(ax=ax1,x='Cover_Type',y=col,data=check,hue='Wilderness_Area',hue_order=['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'])
    ax1.set_title(col,fontsize=24)
    plt.show()


# In[ ]:


for col in list_contfeatures:
    fig1,ax1 = plt.subplots(figsize=(20,10))
    g=sns.boxplot(ax=ax1,x='Wilderness_Area',y=col,data=check,hue='Cover_Type',hue_order=['1','2','3','4','5','6','7'])
    ax1.set_title(col,fontsize=24)
    plt.show()


# Checking whether 1 Wilderness Area can have multiple Cover Types

# In[ ]:


check.groupby(['Wilderness_Area','Cover_Type']).size()


# #### Yes, 1 wilderness area can have multiple Cover Types

# In[ ]:


check.groupby(['Wilderness_Area','Soil_Type']).size()


# #### Yes, 1 wilderness area can have multiple Soil Types

# In[ ]:


check_test.shape


# #### Lets also try analyzing wilderness areas

# In[ ]:


wilderness_area1_dataset = pd.concat([check.loc[check['Wilderness_Area']=="Wilderness_Area1",check.columns!='Cover_Type'],check_test.loc[check_test['Wilderness_Area']=="Wilderness_Area1"]],axis=0)


# In[ ]:


wilderness_area1_dataset.describe()


# In[ ]:


wilderness_area2_dataset = pd.concat([check.loc[check['Wilderness_Area']=="Wilderness_Area2",check.columns!='Cover_Type'],check_test.loc[check_test['Wilderness_Area']=="Wilderness_Area2"]],axis=0)


# In[ ]:


wilderness_area2_dataset.describe()


# In[ ]:


wilderness_area3_dataset = pd.concat([check.loc[check['Wilderness_Area']=="Wilderness_Area3",check.columns!='Cover_Type'],check_test.loc[check_test['Wilderness_Area']=="Wilderness_Area3"]],axis=0)


# In[ ]:


wilderness_area3_dataset.describe()


# In[ ]:


wilderness_area4_dataset = pd.concat([check.loc[check['Wilderness_Area']=="Wilderness_Area4",check.columns!='Cover_Type'],check_test.loc[check_test['Wilderness_Area']=="Wilderness_Area4"]],axis=0)


# In[ ]:


wilderness_area4_dataset.describe()


# #### D. Generate ideas for feature engineering and formulating any possible hypothesis 
# We visualized different features, their distribution and dependencies with one another and target variable, which helped uncover many patterns such as elevation distribution for cover type 7 is on the higher side and so on
# We found strong correlations (only taking magnitude) between features such as:
# 1. Hillshade @ 9 am and Hillshade @ 3pm 
# 2. Horizontal Distance to Hydrology & Vertical Distance Hydrology
# ....
# 
# While building models, we can use these insights. With a baseline model ready, we keep on piling these and see whether there is any improvement
# Plus we will be using feature importance from RF, PCA to further drill down feature importance.
# 
# ### Next up is start building models

# In[ ]:




