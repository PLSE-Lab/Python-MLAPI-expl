#!/usr/bin/env python
# coding: utf-8

# In this tutorial our goal is to predict the forest cover type. This can be achieved by analysing the dataset provided. We will be using pandas library to help with determining the main factors that are correlated with forest cover type.

# In[ ]:



import zipfile
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from collections import Counter
import matplotlib.pyplot as plt
#import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()

from sklearn.decomposition import PCA
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
   for filename in filenames:
       print(os.path.join(dirname, filename))

print ("Setup Complete")


# In[ ]:


# Path of the file to read
train_filepath = 'kaggle/input/Forest/train.csv'
test_filepath = 'kaggle/input/Forest/train.csv'


import zipfile
zf = zipfile.ZipFile('/kaggle/input/forest-cover-type-kernels-only/train.csv.zip')
zf1 = zipfile.ZipFile('/kaggle/input/forest-cover-type-kernels-only/test.csv.zip')
train= pd.read_csv(zf.open('train.csv'),index_col=0)
test = pd.read_csv(zf1.open('test.csv'),index_col=0)


# #  Data Exploration and Feature Engineering

# First it might be useful to get an idea of the size of dataset available.
# As shown, our train dataFrame has around 15000 records split across 56 different columns. 
# We will also examine the cells by printing the first 5 rows.

# In[ ]:


print("The number of traning examples(data points) = %i " % train.shape[0])
print("The number of features we have = %i " % train.shape[1])


# In[ ]:


train.head()


# It seems that the cover type is assigned a value. We can determine the range of values that the cover type can take. We will figure that the values range between 1 and 7.

# In[ ]:


Distinct_Val_in_CoverType= train.Cover_Type.unique()


# In[ ]:


print("Minimum_Number_Assigned:",train['Cover_Type'].min(), "Maximum_Number_Assigned:",train['Cover_Type'].max())


# Now we can check the count of each element in Cover_Type column.

# In[ ]:


train.Cover_Type.value_counts()


# The results above may indicate that each Cover Type has equal chance, however further examination is needed.

# Perhaps examining other columns might assist in finding a pattern.

# In[ ]:


train.describe(include = 'all')


# In[ ]:



col_uni_val={}
for i in train.columns:
    col_uni_val[i] = len(train[i].unique())

import pprint
pprint.pprint(col_uni_val)


# As we can see, it seems that all Soil Type columns can take up to two values only. This might be a sign that the type of soil is of boolean type, either 'yes' or 'no'. Further investigation is needed here to determine whether a forest can be of several types and whether a certain combination of soil types can affect Forest Cover Type.

# Another important factor to check for is whether there are any null values. Null values need to be filled with other reasonable of values or removed if they are very few.

# In[ ]:


train.isnull().sum()


# Alright, so there are no null values!

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

corr = train.corr()
f, ax = plt.subplots(figsize=(25, 25))
#cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr,linewidths=.5)


# This is an overview of the correlation values between all the features available in the data set. Our main focus is on the column 'Cover Type' as we need to determine which features have the greatest effect on the Cover Type.

# I will start with the Soil_Type columns in the data set and see if it is possible to deduce any strong relationship between the 'Cover_Type' and some Soil Types.

# Obviously, Soil_Type7 and Soil_Type15 do not seem to have any effect on the Cover_Type(no correlation). 
# Soil_Type38 and Soil_Type39 have the largest +ve correlation, while Soil_Type22 and Soil_Type29 have the lowest -ve correlation between Cover Type.
# However, we need to check the actual correlation values to avoid making wrong assumptions.

# In[ ]:



Soil_Type_Corr= corr.loc[corr.index[14:54], 'Cover_Type']
print( Soil_Type_Corr)


# In[ ]:


print("The above values proved that Soil_Type7 and Soil_Type15 have NaN values (null)")
print(Soil_Type_Corr.idxmax(),"has the maximum positive correlation value of ",Soil_Type_Corr.max())
print(Soil_Type_Corr.idxmin(),"has the maximum positive correlation value of ",Soil_Type_Corr.min())


# Removing the columns for Soil_Type7 and Soil_Type15 would be the best way to deal with them in order to eliminate unuseful data. As for the correlation values of Soil_Type38 and Soil_Type29, the values are small (might not be very significant ) but we might need them later on for further exploration.

# In[ ]:


train.drop(["Soil_Type7", "Soil_Type15"], axis = 1, inplace = True)


# After having finished analysing the direct relationship between Soil Type and Cover Type, we can move on to the other features.
# According to the overview graph shown earlier, there isn't a strong correlation between one specific feature of the remaining features (Soil type not included) and Cover Type, hence it might be wise to check whether there is a correlation between the remaining features with each other and see if we can remove some features completely or remove some features but replace them with a new feature that combines several features together to improve the performance of the model.

# Our focus this time will be on distances: Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, and Horizontal_Distance_To_Fire_points.

# First, I will check the relation between Horizontal_Distance_To_Hydrology and the remaining distances.

# In[ ]:


import matplotlib.pyplot as plt
classes = np.array(list(train.Cover_Type.values))

def plotRelation(first_feature, sec_feature):

    plt.scatter(first_feature, sec_feature, c = classes, s=10)
    plt.xlabel(first_feature.name)
    plt.ylabel(sec_feature.name)
    
f = plt.figure(figsize=(15,15))
f.add_subplot(331)
plotRelation(train.Horizontal_Distance_To_Hydrology, train.Horizontal_Distance_To_Fire_Points)
f.add_subplot(332)
plotRelation(train.Horizontal_Distance_To_Hydrology, train.Horizontal_Distance_To_Roadways)
f.add_subplot(333)
plotRelation(train.Horizontal_Distance_To_Hydrology, train.Vertical_Distance_To_Hydrology)


# It seems that there is a positive correlation between Vertical_Distance_To_Hydrology and Horizontal_Distance_To_Hydrology. It might be useful to note that Vertical Distance has some negative values, hence taking the absolute of this column might give better results.

# In[ ]:


train["Vertical_Distance_To_Hydrology"] = abs(train['Vertical_Distance_To_Hydrology'])

sns.FacetGrid(train, hue="Cover_Type", size=10).map(plt.scatter, "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology").add_legend()


# We can still see that there is some correlation between the two variables/ features.

# In[ ]:


print("The correlation value between Vertical_Distance_To_Hydrology and Horizontal_Distance_To_Hydrology is ",corr.loc['Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Hydrology'])


# Now we'll do the same for the other distance features to finish comparing all of the distance features together.

# In[ ]:


f = plt.figure(figsize=(15,15))
f.add_subplot(331)
plotRelation( train.Horizontal_Distance_To_Fire_Points,train.Horizontal_Distance_To_Roadways)
f.add_subplot(333)
plotRelation(train.Horizontal_Distance_To_Fire_Points,train.Vertical_Distance_To_Hydrology )
f.add_subplot(332)
plotRelation( train.Horizontal_Distance_To_Roadways,train.Vertical_Distance_To_Hydrology)


# It seems that the only significant correlation was between Horizontal_Distance_To_Hydrology and Vertical_Distance_To_Hydrology, so we can drop one of them.

# The next step would be to search for outliers; this is best done through boxplots.

# In[ ]:


train.boxplot( fontsize=15, grid=True, figsize=(25,25),vert=False)


# Here I will focus on the features that have larger values such as the distance features and check the nature of their outliers. The outliers can be a result of a mistake during data collection or it can be just an indication of variance in the data, so I will remove the rows where there are outliers in distance columns. I chose to remove the rows where Horizontal_Distance_To_Fire_Points have outliers because this is the column which has the largest number of outliers and its outliers are also overlapping with that in the Horizontal_Distance_To_Roadways column.

# In[ ]:



Q1 = train['Horizontal_Distance_To_Fire_Points'].quantile(0.25)
Q3 = train['Horizontal_Distance_To_Fire_Points'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 



train.drop(train[(train['Horizontal_Distance_To_Fire_Points'] >= Q1 - 1.5 * IQR) & (train['Horizontal_Distance_To_Fire_Points'] <= Q3 + 1.5 *IQR)].index)


# # Feature Engineering

# Now I will add some extra features that combine several of the existing features and the next step would be to determine whether this new feature is reasonable enough that it could be replace the features that it was created from.

# In[ ]:


train.head()
## ( addition and subtraction)
train['HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points'])
train['Neg_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])

train['HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
train['Neg_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])

train['HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
train['Neg_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])

##(mean)
train['Mean_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points'])/2
train['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])/2

train['Mean_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])/2
train['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])/2

train['Mean_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])/2
train['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])/2

## mean of all three distances 
train['Mean_Fire_Hydrology_Roadways']=(train['Horizontal_Distance_To_Fire_Points'] + train['Horizontal_Distance_To_Hydrology'] + train['Horizontal_Distance_To_Roadways']) / 3

## mean of hillshade 
train['mean_hillshade'] = (train['Hillshade_9am']  + train['Hillshade_Noon']  + train['Hillshade_3pm'] ) / 3

train.head()
## ( addition and subtraction)
test['HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points'])
test['Neg_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])

test['HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])
test['Neg_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])

test['HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])
train['Neg_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])

##(mean)
test['Mean_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points'])/2
test['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])/2

test['Mean_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])/2
test['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])/2

test['Mean_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])/2
test['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])/2

## mean of all three distances 
test['Mean_Fire_Hydrology_Roadways']=(test['Horizontal_Distance_To_Fire_Points'] + test['Horizontal_Distance_To_Hydrology'] + test['Horizontal_Distance_To_Roadways']) / 3

test['mean_hillshade'] = (test['Hillshade_9am']  + test['Hillshade_Noon']  + test['Hillshade_3pm'] ) / 3


# Note that in the code above I have not mentioned Vertical_Distance_To_Hydrology because I used Horizontal_Distance_To_Hydrology which is highly correlated with it and so I will drop it.

# The three most popular tools used by data scientists are decision trees, random forest and gradient boosting. 
# I tried Gradient Boosting first. Gradient boosting also combine decision trees, but unlike random forests, they start the combining process at the beginning, instead of at the end.

# In[ ]:


from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
X, y = make_classification(random_state=0) 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0,test_size=0.05,)
clf = GradientBoostingClassifier(random_state=0)
clf.fit(X_train, y_train)
print('Accuracy of classifier on training set: {:.2f}'.format(clf.score(X_train, y_train) * 100))
print('Accuracy of classifier on test set: {:.2f}'.format(clf.score(X_test, y_test) * 100))


# The result of the gradient boosting algorithm can be further enhanced if we tuned the parameters more.
