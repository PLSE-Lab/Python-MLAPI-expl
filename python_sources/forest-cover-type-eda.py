#!/usr/bin/env python
# coding: utf-8

# # Classify forest types based on information in the area

# In this competition,we are asked to predict the type of trees there are in an area based on the various geographic features.The seven types are:
# 
# 1 - Spruce/Fir
# 2 - Lodgepole Pine
# 3 - Ponderosa Pine
# 4 - Cottonwood/Willow
# 5 - Aspen
# 6 - Douglas-fir
# 7 - Krummholz

# ### Loading the required libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ### Loading the data

# In[ ]:


kaggle=1

if kaggle==0:
    train=pd.read_csv("train.csv")
    test=pd.read_csv("test.csv")
    sample_submission=pd.read_csv("sample_submission.csv")
else:
    train=pd.read_csv("../input/learn-together/train.csv")
    test=pd.read_csv("../input/learn-together/test.csv")
    sample_submission=pd.read_csv("../input/learn-together/sample_submission.csv")


# ### Glimpse of the data

# In[ ]:


print(f'Shape of train {train.shape} and Shape of test {test.shape}')


# We see that the test dataset is large and has more rows than the train dataset.

# ### Categorical and numerical features

# In[ ]:


train.dtypes


# From the datatype we understand that the following are the numerical features:
# 
# **Elevation** - Elevation in meters
# 
# **Aspect** - Aspect in degrees azimuth
# 
# **Slope** - Slope in degrees
# 
# **Horizontal_Distance_To_Hydrology** - Horz Dist to nearest surface water features
# 
# **Vertical_Distance_To_Hydrology** - Vert Dist to nearest surface water features
# 
# **Horizontal_Distance_To_Roadways** - Horz Dist to nearest roadway
# 
# **Hillshade_9am** - Hillshade index at 9am, summer solstice (0 to 255 index)
# 
# **Hillshade_Noon**  - Hillshade index at noon, summer solstice(0 to 255 index)
# 
# **Hillshade_3pm**  - Hillshade index at 3pm, summer solstice(0 to 255 index)
# 
# **Horizontal_Distance_To_Fire_Points** - Horz Dist to nearest wildfire ignition points
# 
# The following columns are categorical features:
# 
# **Wilderness Area** - Cardinality of 4.
# 
# **Soil Types** - Cardinality of 40

# In[ ]:


test.dtypes


# ### Exploratory Data Analysis

# In[ ]:


#Convert the datatypes to categorical,
cat_columns=train.columns[11:55]
for col in cat_columns:
    print(f'Converting {col} as categorical')
    train[col]=train[col].astype('category')
    test[col]=test[col].astype('category')


# In[ ]:


#Check if the data is balanced or unbalanced,
train['Cover_Type'].value_counts()


# The target values are equally distributed in the dataset.

# In[ ]:


#Check if the dataset has any missing values
missing_train=train.isnull().sum()
missing_test=test.isnull().sum()


# In[ ]:


missing_train[missing_train>0].index


# In[ ]:


missing_test[missing_test>0].index


# Thus both the training and test dataset do not have any missing values.

# ### Elevation

# In[ ]:


train['Elevation'].describe()


# In[ ]:


test['Elevation'].describe()


# In[ ]:


train.groupby('Cover_Type')['Elevation'].median()


# Lets plot the boxplot of elevation with covertype and see if there is a significant difference between the cover types.

# In[ ]:


plt.figure(figsize=(8,8))
sns.boxplot(train['Cover_Type'],train['Elevation'])
plt.title("Boxplot of Elevation with CoverType in train dataset")
plt.xlabel("Cover Type")


# From the boxplot we find that there is a significant difference between the covertype and elevation.This variable could play an important role in differentiating between the different cover types.

# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(train['Elevation'].values, bins=50, kde=False, color="red")
plt.title("Histogram of Elevation")
plt.xlabel('Elevation', fontsize=12)
plt.show()


# We see that the histogram of elevation is multimodal with some elevations going beyong 3750.Lets just get the count of such rows.

# In[ ]:


(train['Elevation']>3750).sum()


# There are 29 rows.Lets check the distribution for elevation in the test set.

# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(test['Elevation'].values, bins=50, kde=False, color="red")
plt.title("Histogram of Elevation")
plt.xlabel('Elevation', fontsize=12)
plt.show()


# Here we see that the distribution is skewed towards left.

# Now , since we have close to 9 numerical features,instead of rewritting code to plot the distribution,we define a function and invoke the function everytime

# In[ ]:


def plot_numerical(variable):
    plt.figure(figsize=(16,6))
    plt.subplot(121)
    sns.distplot(train[variable].values, bins=50, kde=False, color="red")
    plt.title(f'Histogram of {variable} in train')
    plt.xlabel(f'{variable}', fontsize=12)
    plt.subplot(122)
    sns.distplot(test[variable].values, bins=50, kde=False, color="red")
    plt.title(f'Histogram of {variable} in test')
    plt.xlabel(f'{variable}', fontsize=12)
    plt.show()
    
def plot_boxplot(variable):
    #print(f'Plotting boxplot for {variable}\n')
    plt.figure(figsize=(8,8))
    sns.boxplot(train['Cover_Type'],train[variable])
    plt.title(f'Boxplot of {variable} with CoverType in train dataset')
    plt.xlabel("Cover Type")
    


# ### Aspect:

# In[ ]:


plot_numerical('Aspect')


# In[ ]:


plot_boxplot('Aspect')


# 1.We see that the distribution of aspect in both train and test data is similar.
# 
# 2.Boxplot indicates that the median values between the different cover types with aspect is significantly different.

# ### Slope

# In[ ]:


plot_numerical('Slope')


# In[ ]:


train['Slope'].describe()


# In[ ]:


test['Slope'].describe()


# We see that there are quite a few outliers in train and test data for slope.

# In[ ]:


plot_boxplot('Slope')


# The boxplot is significantly different between different cover types.

# In[ ]:


distance_cols=['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']


# In[ ]:


for col in distance_cols:
    print(f'Plotting Histogram for {col}\n')
    plot_numerical(col)
   


# Almost the distribution is the same in both the train and test dataset.

# In[ ]:


for col in distance_cols:
    plot_boxplot(col)
   


# From the boxplots,it is seen that each variable is significantly different for each cover type.

# In[ ]:


hillshade_cols=['Hillshade_9am','Hillshade_Noon','Hillshade_3pm']


# In[ ]:


for col in hillshade_cols:
    print(f'Plotting for {col}\n')
    plot_numerical(col)


# While the hillshare at 9am and  noon is skewed towards left,the distribution plot for 3pm is spread out and almost normal.

# In[ ]:


for col in hillshade_cols:
    plot_boxplot(col)


# From the boxplot distribution , it is understood that the Hillshade 9am and noon values are above 200 for all the cover types whereas at 3pm the values in 75 % of the dataset are lesser than 90.There is a significant difference between each of the covertypes.

# Lets check the cardinality of each categorical variables.

# In[ ]:


#Check the unique values for each categorical variables :
for col in cat_columns:
    print(f'{col} has {train[col].nunique()} unique values in train\n')
    print(f'{col} has {test[col].nunique()} unique values in test\n')


# It is seen that Soil type 7 in train has only 1 constant value where as in test there are 2 values.Similarly for Soil type 15.
