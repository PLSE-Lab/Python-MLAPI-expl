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


# # Import and Review Data

# In[ ]:


# Import training data - review shape and columns
import pandas as pd
train = pd.read_csv('/kaggle/input/learn-together/train.csv')
train.shape
train.columns


# # Reshape data frame for analysis
# The binary columns for Wild Area, Soil Type are not easy to analyse. It is better to convert them into categorical values for a more general Variable - Wild Area and Soil Type

# In[ ]:


# Reverse the one hot encoding for Wilderness Area and Soil Type to create one column each for Wilderness Areas and Soil Types which contain all all possible category values
train_1 = train.copy()
train_1['Wild_Area'] = train_1.iloc[:,11:15].idxmax(axis=1)
train_1['Soil_Type'] = train_1.iloc[:,16:55].idxmax(axis=1)

# Drop the one hot encoded columns to a get a data frame only with numeric and the categorical columns
col_to_drop = np.arange(11,55)
col_to_drop
train_1.drop(train_1.columns[col_to_drop], axis = 1, inplace = True)
train_1.columns
train_1.head()


# In[ ]:


# check the distribution of Wild Area
train_1.Wild_Area.value_counts()


# In[ ]:


# check the distribution of Forest Cover Types
train_1.Cover_Type.value_counts()


# The training data classes are quite evenly spread.

# In[ ]:


# check the distribution of Soil Types
train_1.Soil_Type.value_counts()


# Note that starting from the bottom of the list, there are many soil type columns which appear in very few rows of the training data. Having ver few entries like even 100 in a training data size of 15000, may mean they might not influence the prediction in a major way. however if we do remove them, there is likely less chance of overfitting based on the these columns.
# what is a good threshold count to remove the columns from feature space is a tricky question. we will address this later when we get to fit models.
# 
# Lets start with 100 as a threshold to weed out potentially non impactful features.

# In[ ]:


# check the avail_soil types which have counts less than 100. 
less_100 = train_1.Soil_Type.value_counts() < 100
less_100_soil = train_1.Soil_Type.value_counts()[less_100].index
less_100_soil


# For Soil Type binary columns, not that the number of Soil types that are listed above are only 37/40. hence 3 of the columns are only filled with 0. Hence they do not effectively particpate in predicting the cover type.
# 
# lets find out which of the 3 soil types are missing

# In[ ]:


# list of non zero soil types available in training data
avail_soil = train_1.Soil_Type.value_counts().index
avail_soil

# Get all possible Soil Types available as features in the training data as columns
all_soil = train.columns[15:55]
all_soil
# Check the missing soil types from training data by comparing all_soil and avail_soil
miss_soil = np.setdiff1d(all_soil,avail_soil)
miss_soil


# These 3 columns can certainly be removed from our list of feature columns for prediction

# Now let us look at all the Soil Type binary columns that can be removed - zero count columns plus lesser count columns

# In[ ]:


# Create list of Soil type columns to be removed - missing soil types + less than 50 counts soil types
remove_soil = list(miss_soil) + list(less_100_soil)
remove_soil


# Note that when we remove these columns from training data, same needs to be done for the test data as well before predicting the test set cover type labels.

# In[ ]:


# Create box plots for all numeric variables - Elevation, Aspect, Slope, Horizontal Distance to Hydrology
#import seaborn as sns
#import matplotlib.pyplot as plt
#sns.set(rc={'figure.figsize':(15,15)})
#fig,ax = plt.subplots(5,2)
#sns.boxplot(x = 'Cover_Type', y = 'Elevation', data = train_1, ax = ax[0,0])
#sns.boxplot(x = 'Cover_Type', y = 'Aspect', data = train_1, ax = ax[0,1])
#sns.boxplot(x = 'Cover_Type', y = 'Slope', data = train_1,ax = ax[1,0])
#sns.boxplot(x = 'Cover_Type', y = 'Horizontal_Distance_To_Hydrology', data = train_1,ax = ax[1,1])
#sns.boxplot(x = 'Cover_Type', y = 'Vertical_Distance_To_Hydrology', data = train_1,ax = ax[2,0])
#sns.boxplot(x = 'Cover_Type', y = 'Horizontal_Distance_To_Roadways', data = train_1,ax = ax[2,1])
#sns.boxplot(x = 'Cover_Type', y = 'Horizontal_Distance_To_Fire_Points', data = train_1,ax = ax[3,0])
#sns.boxplot(x = 'Cover_Type', y = 'Hillshade_9am', data = train_1,ax = ax[3,1])
#sns.boxplot(x = 'Cover_Type', y = 'Hillshade_Noon', data = train_1,ax = ax[4,0])
#sns.boxplot(x = 'Cover_Type', y = 'Hillshade_3pm', data = train_1,ax = ax[4,1])
#plt.close(4,1)
#plt.close(3)
#plt.close(4)
#plt.close(5)
#plt.yticks(rotation = 90)


# # Analyze numerical features 
# Box plots can help understand the distribution of data at each level of cover type. With this we can see if the specific variable values are good enough to linearly separate the classes - atleast some if not all.

# In[ ]:


# Plot box plots of all numeric variables by Cover Type
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(10,5)})
for column in train_1.columns[1:10]:
    sns.boxplot(x = 'Cover_Type', y = column, data = train_1)
    plt.show()


# By looking at the distribution of values for each of the numeric variables with the box plot, we can see that Elevation is able to find different range of values for different cover types and is more able to linearly separate some of the classes

# In[ ]:


# Plot Hillshade variables
import seaborn as sns
#train_1['Cover_Type'] = train_1['Cover_Type'].astype('object')
sns.scatterplot(x='Hillshade_9am', y = 'Hillshade_3pm', data = train_1, hue = 'Cover_Type', palette=sns.color_palette("Set1", 7))
plt.show()
sns.scatterplot(x='Hillshade_9am', y = 'Hillshade_Noon', data = train_1, hue = 'Cover_Type', palette=sns.color_palette("Set1", 7))
plt.show()
sns.scatterplot(x='Hillshade_Noon', y = 'Hillshade_3pm', data = train_1, hue = 'Cover_Type', palette=sns.color_palette("Set1", 7))
plt.show()
#train_1.plot(x='Hillshade_3pm', y = 'Hillshade_Noon', kind = 'scatter', c = 'Cover_Type')


# Hillshade_3pm is correlated with both the other two hill shade variables. But the classes are not clearly separated by any of the individual hill shade variables itself.

# # Summary Pair Plots
# plotting all pair plots gives us a little more understanding of how each of the numerical variable is related to the other. we can place them in the context of each cover type to see if the cover types are separated by any pair of variables.

# In[ ]:


# construct scatter plots to check for 2 variable combinations that can separate the classes
sns.pairplot(train_1, hue = 'Cover_Type')


# It is pretty clear that Elevation has a big impact on cover type but other variables do not have a obvious single impact on the classification, but the interactions between the variables will probably lead to the classification
# 
# Hillside 3 pm has a strong positive correlation with Hillside_Noon and a strong negative correlation with Hillside_9am. we could probably choose to incorporate only Hillside_3pm leaving out the other 2 variables.
# Vertical distance to hydrology is also strongly correlated with horizontal distance to hydrology. but horizontal distance to hydrology shows more variation across cover types and hence may be a better indicator. so we will choose only horizontal distance to hydrology.
# Slope seems to have strong negative correlation with both hillshade 9 am and hill shade noon. so we can choose only slope instead of other two. this is also consistent with leaving out them due to strong correlation with hillshade 3 pm.
# 
# 

# # Conclusions
# As a first trial, we will remove the below features for initial modeling
# 1. Hillshade_9 am
# 2. Hillshade_Noon
# 3. Vertical_Distance_To_Hydrology
# 4. Soil_Type1
# 5. Soil_Type15
# 6. Soil_Type7
# 7. Soil_Type18
# 8. Soil_Type26
# 9. Soil_Type19
# 10. Soil_Type37
# 11. Soil_Type34
# 12. Soil_Type21
# 13. Soil_Type27
# 14. Soil_Type36
# 15. Soil_Type9
# 16. Soil_Type28
# 17. Soil_Type25
# 18. Soil_Type8

# In[ ]:


#sns.scatterplot(x='Aspect', y = 'Horizontal_Distance_To_Fire_Points', data = train_1, hue = 'Cover_Type', palette=sns.color_palette("Set1", 7))
#sns.scatterplot(x='Hillshade_Noon', y = 'Elevation', data = train_1, hue = 'Cover_Type', palette=sns.color_palette("Set1", 7))


# To be continued. will update the kernel with model results based on the feature reduction.
