#!/usr/bin/env python
# coding: utf-8

# Vehicle Recognition

# #Overview
# Welcome to my kernel!
# Data Description: The data contains features extracted from the silhouette of vehicles in different angles. Four "Corgie" model vehicles were used for the experiment: a double decker bus, Cheverolet van, Saab 9000 and an Opel Manta 400 cars. This particular combination of vehicles was chosen with the expectation that the bus, van and either one of the cars would be readily distinguishable, but it would be more difficult to distinguish between the cars.

# #Objective:
# The objective is to classify a given silhouette as one of three types of vehicle, using a set of features extracted from the silhouette. The vehicle may be viewed from one of many different angles.

# #Importing the Libraries and Basic EDA

# In[ ]:


#import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
from scipy.stats import zscore
from sklearn.model_selection import train_test_split


# In[ ]:


#load the csv file and make the data frame
vehicle_df = pd.read_csv('/kaggle/input/vehicle/vehicle.csv')


# In[ ]:


#display the first 5 rows of dataframe
vehicle_df.head()


# In[ ]:


print("The dataframe has {} rows and {} columns".format(vehicle_df.shape[0],vehicle_df.shape[1]))


# In[ ]:


#display the information of dataframe
vehicle_df.info()


# From above we can see that except 'class' column all columns are numeric type and there are null values in some columns.
# class column is our target column.

# In[ ]:


#display in each column how many null values are there
vehicle_df.apply(lambda x: sum(x.isnull()))


# From above we can see that max null values is 6 which are in two columns 'radius_ratio', 'skewness_about'.
# so we have two options either we will drop those null values or we will impute those null values.
# Dropping null values is not a good way because we will lose some information.but we will go with both options then we will see what's the effect on model.

# In[ ]:


#display 5 point summary of dataframe
vehicle_df.describe().transpose()


# In[ ]:


sns.pairplot(vehicle_df,diag_kind='kde')
plt.show()


# From above pair plots we can see that many columns are correlated and many columns have long tail so that is the indication of outliers.we will see down the line with the help of correlation matrix what's the strength of correlation and outliers are there or not.

# From above we can see that our data has missing values in some column. so before building any model we have to handle missing values. we have two option either we will drop those missing values or we will impute missing values. we will go with both options and see what's the effect on model. so first we will drop the missing values. Before dropping missing values we will create another dataframe and copy the original dataframe data into that. It's a good practice to keep the original dataframe as it is and make all modifications to the new dataframe.

# #Dropping Missing Values

# In[ ]:


#copy the dataframe to another dataframe and drop null/missing values from the newly created dataframe
new_vehicle_df = vehicle_df.copy()


# so now we have new dataframe called new_vehicle_df and we will make changes in this new dataframe.

# In[ ]:


#display the first 5 rows of new dataframe
new_vehicle_df.head()


# In[ ]:


#display the shape of dataframe
print("Shape of newly created dataframe:",new_vehicle_df.shape)


# In[ ]:


#drop the null vaues from the new dataframe
new_vehicle_df.dropna(axis=0,inplace=True)


# In[ ]:


#now we will see what is the shape of dataframe
print("After dropping missing values shape of dataframe:",new_vehicle_df.shape)


# In[ ]:


#display 5 point summary of new dataframe
new_vehicle_df.describe().transpose()


# #Analysis of each column with the help of plots

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(new_vehicle_df['compactness'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(new_vehicle_df['compactness'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in compactness column and it's looks like normally distributed.

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(new_vehicle_df['circularity'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(new_vehicle_df['circularity'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in circularity column and it's looks like normally distributed

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(new_vehicle_df['distance_circularity'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(new_vehicle_df['distance_circularity'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in distance_circularity column but in distribution plot we can see that there are two peaks and we can see that there is right skewness because long tail is at the right side(mean>median)

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(new_vehicle_df['radius_ratio'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(new_vehicle_df['radius_ratio'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are outliers in radius_ratio column and there is right skewness because long tail is at the right side(mean>median)

# In[ ]:


#check how many outliers are there in radius_ratio column
q1 = np.quantile(new_vehicle_df['radius_ratio'],0.25)
q2 = np.quantile(new_vehicle_df['radius_ratio'],0.50)
q3 = np.quantile(new_vehicle_df['radius_ratio'],0.75)
IQR = q3-q1
print("Quartie1::",q1)
print("Quartie2::",q2)
print("Quartie3::",q3)
print("Inter Quartie Range::",IQR)
#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR
print("radius_ratio above",new_vehicle_df['radius_ratio'].quantile(0.75)+(1.5 * IQR),"are outliers")
print("The Outliers in radius_ratio column are",new_vehicle_df[new_vehicle_df['radius_ratio']>276]['radius_ratio'].shape[0])


# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(new_vehicle_df['pr.axis_aspect_ratio'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(new_vehicle_df['pr.axis_aspect_ratio'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are outliers in pr.axis_aspect_ratio column and there is right skewness because long tail is at right side(mean>median)

# In[ ]:


#check how many outliers are there in pr.axis_aspect_ratio column
q1 = np.quantile(new_vehicle_df['pr.axis_aspect_ratio'],0.25)
q2 = np.quantile(new_vehicle_df['pr.axis_aspect_ratio'],0.50)
q3 = np.quantile(new_vehicle_df['pr.axis_aspect_ratio'],0.75)
IQR = q3-q1
print("Quartie1::",q1)
print("Quartie2::",q2)
print("Quartie3::",q3)
print("Inter Quartie Range::",IQR)
#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR
print("pr.axis_aspect_ratio above",new_vehicle_df['pr.axis_aspect_ratio'].quantile(0.75)+(1.5 * IQR),"are outliers")
print("The Outliers in pr.axis_aspect_ratio column are",new_vehicle_df[new_vehicle_df['pr.axis_aspect_ratio']>77]['pr.axis_aspect_ratio'].shape[0])


# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(new_vehicle_df['max.length_aspect_ratio'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(new_vehicle_df['max.length_aspect_ratio'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are outliers in max.length_aspect_ratio and there is a right skewness because long tail is at right side(mean>median)

# In[ ]:


#check how many outliers are there in pr.axis_aspect_ratio column
q1 = np.quantile(new_vehicle_df['max.length_aspect_ratio'],0.25)
q2 = np.quantile(new_vehicle_df['max.length_aspect_ratio'],0.50)
q3 = np.quantile(new_vehicle_df['max.length_aspect_ratio'],0.75)
IQR = q3-q1
print("Quartie1::",q1)
print("Quartie2::",q2)
print("Quartie3::",q3)
print("Inter Quartie Range::",IQR)
#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR
print("max.length_aspect_ratio above",new_vehicle_df['max.length_aspect_ratio'].quantile(0.75)+(1.5 * IQR),"are outliers")
print("max.length_aspect_ratio below",new_vehicle_df['max.length_aspect_ratio'].quantile(0.25)-(1.5 * IQR),"are outliers")
print("The above Outliers in max.length_aspect_ratio column are",new_vehicle_df[new_vehicle_df['max.length_aspect_ratio']>14.5]['max.length_aspect_ratio'].shape[0])
print("The below Outliers in max.length_aspect_ratio column are",new_vehicle_df[new_vehicle_df['max.length_aspect_ratio']<2.5]['max.length_aspect_ratio'].shape[0])


# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(new_vehicle_df['scatter_ratio'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(new_vehicle_df['scatter_ratio'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in scatter_ratio column and there are two peaks in distribution plot and there is right skewness because long tail is at right side(mean>median) 

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(new_vehicle_df['elongatedness'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(new_vehicle_df['elongatedness'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in elongatedness column and there are two peaks in distribution plot and there is left skewness because long tail is at left side(mean<median) 

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(new_vehicle_df['pr.axis_rectangularity'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(new_vehicle_df['pr.axis_rectangularity'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in pr.axis_rectangularity column and there are two peaks in distribution plot and there is right skewness because long tail is at right side(mean>median)

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(new_vehicle_df['max.length_rectangularity'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(new_vehicle_df['max.length_rectangularity'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in max.length_rectangularity column and there are two peaks in distribution plot and there is right skewness because long tail is at right side(mean>median)

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(new_vehicle_df['scaled_variance'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(new_vehicle_df['scaled_variance'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are outliers in scaled_variance column and there are two peaks in distribution plot and there is right skewness because long tail is at right side(mean>median)

# In[ ]:


#check how many outliers are there in scaled_variance column
q1 = np.quantile(new_vehicle_df['scaled_variance'],0.25)
q2 = np.quantile(new_vehicle_df['scaled_variance'],0.50)
q3 = np.quantile(new_vehicle_df['scaled_variance'],0.75)
IQR = q3-q1
print("Quartie1::",q1)
print("Quartie2::",q2)
print("Quartie3::",q3)
print("Inter Quartie Range::",IQR)
#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR
print("scaled_variance above",new_vehicle_df['scaled_variance'].quantile(0.75)+(1.5 * IQR),"are outliers")
print("The Outliers in scaled_variance column are",new_vehicle_df[new_vehicle_df['scaled_variance']>292]['scaled_variance'].shape[0])


# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(new_vehicle_df['scaled_variance.1'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(new_vehicle_df['scaled_variance.1'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are outliers in scaled_variance.1 column and there are two peaks in distribution plot and there is right skewness because long tail is at right side(mean>median)

# In[ ]:


#check how many outliers are there in scaled_variance.1 column
q1 = np.quantile(new_vehicle_df['scaled_variance.1'],0.25)
q2 = np.quantile(new_vehicle_df['scaled_variance.1'],0.50)
q3 = np.quantile(new_vehicle_df['scaled_variance.1'],0.75)
IQR = q3-q1
print("Quartie1::",q1)
print("Quartie2::",q2)
print("Quartie3::",q3)
print("Inter Quartie Range::",IQR)
#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR
print("scaled_variance.1 above",new_vehicle_df['scaled_variance.1'].quantile(0.75)+(1.5 * IQR),"are outliers")
print("The Outliers in scaled_variance.1 column are",new_vehicle_df[new_vehicle_df['scaled_variance.1']>988]['scaled_variance.1'].shape[0])


# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(new_vehicle_df['scaled_radius_of_gyration'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(new_vehicle_df['scaled_radius_of_gyration'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in scaled_radius_of_gyration column and there is right skewness because long tail is at right side(mean>median)

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(new_vehicle_df['scaled_radius_of_gyration.1'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(new_vehicle_df['scaled_radius_of_gyration.1'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are outliers in scaled_radius_of_gyration.1 column and there is right skewness because long tail is at right side(mean>median)

# In[ ]:


#check how many outliers are there in scaled_radius_of_gyration.1 column
q1 = np.quantile(new_vehicle_df['scaled_radius_of_gyration.1'],0.25)
q2 = np.quantile(new_vehicle_df['scaled_radius_of_gyration.1'],0.50)
q3 = np.quantile(new_vehicle_df['scaled_radius_of_gyration.1'],0.75)
IQR = q3-q1
print("Quartie1::",q1)
print("Quartie2::",q2)
print("Quartie3::",q3)
print("Inter Quartie Range::",IQR)
#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR
print("scaled_radius_of_gyration.1 above",new_vehicle_df['scaled_radius_of_gyration.1'].quantile(0.75)+(1.5 * IQR),"are outliers")
print("The Outliers in scaled_radius_of_gyration.1 column are",new_vehicle_df[new_vehicle_df['scaled_radius_of_gyration.1']>87]['scaled_radius_of_gyration.1'].shape[0])


# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(new_vehicle_df['skewness_about'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(new_vehicle_df['skewness_about'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are outliers in skewness_about column and there is right skewness because long tail is at right side(mean>median)

# In[ ]:


#check how many outliers are there in skewness_about column
q1 = np.quantile(new_vehicle_df['skewness_about'],0.25)
q2 = np.quantile(new_vehicle_df['skewness_about'],0.50)
q3 = np.quantile(new_vehicle_df['skewness_about'],0.75)
IQR = q3-q1
print("Quartie1::",q1)
print("Quartie2::",q2)
print("Quartie3::",q3)
print("Inter Quartie Range::",IQR)
#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR
print("skewness_about above",new_vehicle_df['skewness_about'].quantile(0.75)+(1.5 * IQR),"are outliers")
print("The Outliers in skewness_about column are",new_vehicle_df[new_vehicle_df['skewness_about']>19.5]['skewness_about'].shape[0])


# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(new_vehicle_df['skewness_about.1'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(new_vehicle_df['skewness_about.1'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are outliers in skewness_about.1 column and there is right skewness because long tail is at right side(mean>median)

# In[ ]:


#check how many outliers are there in skewness_about.1 column
q1 = np.quantile(new_vehicle_df['skewness_about.1'],0.25)
q2 = np.quantile(new_vehicle_df['skewness_about.1'],0.50)
q3 = np.quantile(new_vehicle_df['skewness_about.1'],0.75)
IQR = q3-q1
print("Quartie1::",q1)
print("Quartie2::",q2)
print("Quartie3::",q3)
print("Inter Quartie Range::",IQR)
#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR
print("skewness_about.1 above",new_vehicle_df['skewness_about.1'].quantile(0.75)+(1.5 * IQR),"are outliers")
print("The Outliers in skewness_about.1 column are",new_vehicle_df[new_vehicle_df['skewness_about.1']>38.5]['skewness_about.1'].shape[0])


# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(new_vehicle_df['skewness_about.2'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(new_vehicle_df['skewness_about.2'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in skewness_about.2 column and there is left skewness because long tail is at left side(mean<median)

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(new_vehicle_df['hollows_ratio'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(new_vehicle_df['hollows_ratio'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in hollows_ratio column and there is left skewness because long tail is at left side(mean<median)

# In[ ]:


#display how many are car,bus,van. 
new_vehicle_df['class'].value_counts()


# In[ ]:


sns.countplot(new_vehicle_df['class'])
plt.show()


# From above we can see that cars are most followed by bus and then vans.

# so by now we analyze each column and we found that there are outliers in some column. now our next step is to know whether these outliers are natural or artificial. if natural then we have to do nothing but if these outliers are artificial then we have to handle these outliers.
# we have 8 columns in which we found outliers:
# ->radius_ratio
# ->pr.axis_aspect_ratio
# ->max.length_aspect_ratio
# ->scaled_variance
# ->scaled_variance.1
# ->scaled_radius_of_gyration.1
# ->skewness_about
# ->skewness_about.1

# after seeing the max values of above outliers column. it's looks like outliers in above columns are natural not a typo mistake or artificial.
# Note: It's my assumption only. as there is no way to prove whether these outliers are natural or artificial.
# As we know that mostly algorithms are affected by outliers and outliers may affect the model.as we will apply SVM on above data which is affected by outliers. so better to drop those outliers.

# #Fix Outliers after dropping missing values

# In[ ]:


#radius_ratio column outliers
new_vehicle_df.drop(new_vehicle_df[new_vehicle_df['radius_ratio']>276].index,axis=0,inplace=True)


# In[ ]:


#pr.axis_aspect_ratio column outliers
new_vehicle_df.drop(new_vehicle_df[new_vehicle_df['pr.axis_aspect_ratio']>77].index,axis=0,inplace=True)


# In[ ]:


#max.length_aspect_ratio column outliers
new_vehicle_df.drop(new_vehicle_df[new_vehicle_df['max.length_aspect_ratio']>14.5].index,axis=0,inplace=True)
new_vehicle_df.drop(new_vehicle_df[new_vehicle_df['max.length_aspect_ratio']<2.5].index,axis=0,inplace=True)


# In[ ]:


#scaled_variance column outliers
new_vehicle_df[new_vehicle_df['scaled_variance']>292]


# from above we can see that scaled_variance column outliers has been removed

# In[ ]:


#scaled_variance.1 column outliers
new_vehicle_df.drop(new_vehicle_df[new_vehicle_df['scaled_variance.1']>988].index,axis=0,inplace=True)


# In[ ]:


#scaled_radius_of_gyration.1 column outliers
new_vehicle_df.drop(new_vehicle_df[new_vehicle_df['scaled_radius_of_gyration.1']>87].index,axis=0,inplace=True)


# In[ ]:


#skewness_about column outliers
new_vehicle_df.drop(new_vehicle_df[new_vehicle_df['skewness_about']>19.5].index,axis=0,inplace=True)


# In[ ]:


#skewness_about.1 column outliers
new_vehicle_df.drop(new_vehicle_df[new_vehicle_df['skewness_about.1']>38.5].index,axis=0,inplace=True)


# In[ ]:


#now what is the shape of dataframe
print("after removing outliers shape of dataframe:",new_vehicle_df.shape)


# In[ ]:


#find the correlation between independent variables
plt.figure(figsize=(20,5))
sns.heatmap(new_vehicle_df.corr(),annot=True)
plt.show()


# so our objective is to reocgnize whether an object is a van or bus or car based on some input features.
# so our main assumption is there is little or no multicollinearity between the features.
# if two features is highly correlated then there is no use in using both features.in that case, we can drop one feature. 
# so heatmap gives us the correlation matrix there we can see which features are highly correlated.
# From above correlation matrix we can see that there are many features which are highly correlated. if we see carefully then scaled_variance.1 and scatter_ratio has 1 correlation and many other features also there which having more than 0.9 correlation
# so we will drop those columns whose correlation is +-0.9 or above.
# so there are 8 such columns:
# ->max.length_rectangularity
# ->scaled_radius_of_gyration
# ->skewness_about.2
# ->scatter_ratio
# ->elongatedness
# ->pr.axis_rectangularity
# ->scaled_variance
# ->scaled_variance.1

# now, again we have two option we will drop those above eight columns manually or we will apply pca and let pca to be decided how it will explain above data which is in high dimension with smaller number of variables.
# we will see both approaches.

# Principal Component Analysis is an unsupervised learning class of statistical techniques used to explain data in high dimension using small number of variables called the principal components. Principal components are the linear combinations of the original variables in the dataset. As it will explain high dimension data with small number of variables. The big disadvantage is we cannot do interpretation with the model.In other words model with pca will become blackbox.   
# In pca first we have to find the covariance matrix after that from that covariance matrix we have to find eigen vectors and eigen values. There is mathematical way to find eigen vectors and eigen values. i will attach the link of how to find the eigen value and eigen vector. Corresponding to each eigen vector there is eigen value. after that we have to sort the eigen vector by decreasing eigen values and choose k eigen vectors with the largest eigen value. 

# #With Principal Component Analysis(PCA) 

# In[ ]:


#now separate the dataframe into dependent and independent variables
new_vehicle_df_independent_attr = new_vehicle_df.drop('class',axis=1)
new_vehicle_df_dependent_attr = new_vehicle_df['class']
print("shape of new_vehicle_df_independent_attr::",new_vehicle_df_independent_attr.shape)
print("shape of new_vehicle_df_dependent_attr::",new_vehicle_df_dependent_attr.shape)


# In[ ]:


#now sclaed the independent attribute and replace the dependent attr value with number
new_vehicle_df_independent_attr_scaled = new_vehicle_df_independent_attr.apply(zscore)
new_vehicle_df_dependent_attr.replace({'car':0,'bus':1,'van':2},inplace=True)


# In[ ]:


#make the covariance matrix and we have 18 independent features so aur covariance matrix is 18*18 matrix
cov_matrix = np.cov(new_vehicle_df_independent_attr_scaled,rowvar=False)
print("cov_matrix shape:",cov_matrix.shape)
print("Covariance_matrix",cov_matrix)


# In[ ]:


#now with the help of above covariance matrix we will find eigen value and eigen vectors
pca_to_learn_variance = PCA(n_components=18)
pca_to_learn_variance.fit(new_vehicle_df_independent_attr_scaled)


# In[ ]:


#display explained variance ratio
pca_to_learn_variance.explained_variance_ratio_


# In[ ]:


#display explained variance
pca_to_learn_variance.explained_variance_


# In[ ]:


#display principal components
pca_to_learn_variance.components_


# In[ ]:


plt.bar(list(range(1,19)),pca_to_learn_variance.explained_variance_ratio_)
plt.xlabel("eigen value/components")
plt.ylabel("variation explained")
plt.show()


# In[ ]:


plt.step(list(range(1,19)),np.cumsum(pca_to_learn_variance.explained_variance_ratio_))
plt.xlabel("eigen value/components")
plt.ylabel("cummalative of variation explained")
plt.show()


# From above we can see that 8 dimension are able to explain 95%variance of data. so we will use first 8 principal components

# In[ ]:


#use first 8 principal components
pca_eight_components = PCA(n_components=8)
pca_eight_components.fit(new_vehicle_df_independent_attr_scaled)


# In[ ]:


#transform the raw data which is in 18 dimension into 8 new dimension with pca
new_vehicle_df_pca_independent_attr = pca_eight_components.transform(new_vehicle_df_independent_attr_scaled)


# In[ ]:


#display the shape of new_vehicle_df_pca_independent_attr
new_vehicle_df_pca_independent_attr.shape


# now before apply pca with 8 dimension which are explaining more than 95% variantion of data we will make model on raw data after that we will make model with pca and then we will compare both models.

# In[ ]:


#now split the data into 80:20 ratio
rawdata_X_train,rawdata_X_test,rawdata_y_train,rawdata_y_test = train_test_split(new_vehicle_df_independent_attr_scaled,new_vehicle_df_dependent_attr,test_size=0.20,random_state=1)
pca_X_train,pca_X_test,pca_y_train,pca_y_test = train_test_split(new_vehicle_df_pca_independent_attr,new_vehicle_df_dependent_attr,test_size=0.20,random_state=1)


# In[ ]:


print("shape of rawdata_X_train",rawdata_X_train.shape)
print("shape of rawdata_y_train",rawdata_y_train.shape)
print("shape of rawdata_X_test",rawdata_X_test.shape)
print("shape of rawdata_y_test",rawdata_y_test.shape)
print("--------------------------------------------")
print("shape of pca_X_train",pca_X_train.shape)
print("shape of pca_y_train",pca_y_train.shape)
print("shape of pca_X_test",pca_X_test.shape)
print("shape of pca_y_test",pca_y_test.shape)


# In[ ]:


#now we will train the model with both raw data and pca data with new dimension
svc = SVC() #instantiate the object


# In[ ]:


#fit the model on raw data
svc.fit(rawdata_X_train,rawdata_y_train)


# In[ ]:


#predict the y value
rawdata_y_predict = svc.predict(rawdata_X_test)


# In[ ]:


#now fit the model on pca data with new dimension
svc.fit(pca_X_train,pca_y_train)


# In[ ]:


#predict the y value
pca_y_predict = svc.predict(pca_X_test)


# In[ ]:


#display accuracy score of both models
print("Accuracy score with raw data(18 dimension)",accuracy_score(rawdata_y_test,rawdata_y_predict))
print("Accuracy score with pca data(8 dimension)",accuracy_score(pca_y_test,pca_y_predict))


# From above we can see that by reducing 10 dimension we are achieving 94% accuracy

# In[ ]:


#display confusion matrix of both models
print("Confusion matrix with raw data(18 dimension)\n",confusion_matrix(rawdata_y_test,rawdata_y_predict))
print("Confusion matrix with pca data(8 dimension)\n",confusion_matrix(pca_y_test,pca_y_predict))


# #With dropping the above mentioned columns Manually

# In[ ]:


#drop the columns
new_vehicle_df_independent_attr_scaled.drop(['max.length_rectangularity','scaled_radius_of_gyration','skewness_about.2','scatter_ratio','elongatedness','pr.axis_rectangularity','scaled_variance','scaled_variance.1'],axis=1,inplace=True)


# In[ ]:


#display the shape of new dataframe
new_vehicle_df_independent_attr_scaled.shape


# In[ ]:


dropcolumn_X_train,dropcolumn_X_test,dropcolumn_y_train,dropcolumn_y_test = train_test_split(new_vehicle_df_independent_attr_scaled,new_vehicle_df_dependent_attr,test_size=0.20,random_state=1)


# In[ ]:


print("shape of dropcolumn_X_train",dropcolumn_X_train.shape)
print("shape of dropcolumn_y_train",dropcolumn_y_train.shape)
print("shape of dropcolumn_X_test",dropcolumn_X_test.shape)
print("shape of dropcolumn_y_test",dropcolumn_y_test.shape)


# In[ ]:


#fit the model on dropcolumn_X_train,dropcolumn_y_train
svc.fit(dropcolumn_X_train,dropcolumn_y_train)


# In[ ]:


#predict the y value
dropcolumn_y_predict = svc.predict(dropcolumn_X_test)


# In[ ]:


#display the accuracy score and confusion matrix
print("Accuracy score with dropcolumn data(10 dimension)",accuracy_score(dropcolumn_y_test,dropcolumn_y_predict))
print("Confusion matrix with dropcolumn data(10 dimension)\n",confusion_matrix(dropcolumn_y_test,dropcolumn_y_predict))


# #Imputing missing values

# First let's create a new dataframe and then we will impute the missing values.

# In[ ]:


#create a new dataframe
impute_vehicle_df = vehicle_df.copy()


# In[ ]:


#display the first 5 rows of dataframe
impute_vehicle_df.head()


# In[ ]:


#display the shape of dataframe
impute_vehicle_df.shape


# In[ ]:


#display the information of dataframe
impute_vehicle_df.info()


# From above we can see that there are null values in some column.now we will impute those null values.

# In[ ]:


#display 5 point summary
impute_vehicle_df.describe().transpose()


# From above 5 point summary it's looks like we can impute with median.again by imputing the missing values with median we are changing the shape of distribution and introducing bias.but it's might be better than drpping missing values.

# In[ ]:


impute_vehicle_df.fillna(impute_vehicle_df.median(),axis=0,inplace=True)


# In[ ]:


#display the info of dataframe
impute_vehicle_df.info()


# From above we can see that there are no null values in each column

# In[ ]:


#display 5 point summary after imputation 
impute_vehicle_df.describe().transpose()


# #Analysis of each column with the help of plots

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(impute_vehicle_df['compactness'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(impute_vehicle_df['compactness'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in compactness column and it's looks like normally distributed.

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(impute_vehicle_df['circularity'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(impute_vehicle_df['circularity'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in circularity column and it's looks like normally distributed

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(impute_vehicle_df['distance_circularity'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(impute_vehicle_df['distance_circularity'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in distance_circularity column but in distribution plot we can see that there are two peaks and we can see that there is right skewness because long tail is at the right side(mean>median)

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(impute_vehicle_df['radius_ratio'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(impute_vehicle_df['radius_ratio'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are outliers in radius_ratio column and there is right skewness because long tail is at the right side(mean>median)

# In[ ]:


#check how many outliers are there in radius_ratio column
q1 = np.quantile(impute_vehicle_df['radius_ratio'],0.25)
q2 = np.quantile(impute_vehicle_df['radius_ratio'],0.50)
q3 = np.quantile(impute_vehicle_df['radius_ratio'],0.75)
IQR = q3-q1
print("Quartie1::",q1)
print("Quartie2::",q2)
print("Quartie3::",q3)
print("Inter Quartie Range::",IQR)
#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR
print("radius_ratio above",impute_vehicle_df['radius_ratio'].quantile(0.75)+(1.5 * IQR),"are outliers")
print("The Outliers in radius_ratio column are",impute_vehicle_df[impute_vehicle_df['radius_ratio']>276]['radius_ratio'].shape[0])


# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(impute_vehicle_df['pr.axis_aspect_ratio'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(impute_vehicle_df['pr.axis_aspect_ratio'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are outliers in pr.axis_aspect_ratio column and there is right skewness because long tail is at right side(mean>median)

# In[ ]:


#check how many outliers are there in pr.axis_aspect_ratio column
q1 = np.quantile(impute_vehicle_df['pr.axis_aspect_ratio'],0.25)
q2 = np.quantile(impute_vehicle_df['pr.axis_aspect_ratio'],0.50)
q3 = np.quantile(impute_vehicle_df['pr.axis_aspect_ratio'],0.75)
IQR = q3-q1
print("Quartie1::",q1)
print("Quartie2::",q2)
print("Quartie3::",q3)
print("Inter Quartie Range::",IQR)
#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR
print("pr.axis_aspect_ratio above",impute_vehicle_df['pr.axis_aspect_ratio'].quantile(0.75)+(1.5 * IQR),"are outliers")
print("The Outliers in pr.axis_aspect_ratio column are",impute_vehicle_df[impute_vehicle_df['pr.axis_aspect_ratio']>77]['pr.axis_aspect_ratio'].shape[0])


# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(impute_vehicle_df['max.length_aspect_ratio'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(impute_vehicle_df['max.length_aspect_ratio'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are outliers in max.length_aspect_ratio and there is a right skewness because long tail is at right side(mean>median)

# In[ ]:


#check how many outliers are there in max.length_aspect_ratio column
q1 = np.quantile(impute_vehicle_df['max.length_aspect_ratio'],0.25)
q2 = np.quantile(impute_vehicle_df['max.length_aspect_ratio'],0.50)
q3 = np.quantile(impute_vehicle_df['max.length_aspect_ratio'],0.75)
IQR = q3-q1
print("Quartie1::",q1)
print("Quartie2::",q2)
print("Quartie3::",q3)
print("Inter Quartie Range::",IQR)
#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR
print("max.length_aspect_ratio above",impute_vehicle_df['max.length_aspect_ratio'].quantile(0.75)+(1.5 * IQR),"are outliers")
print("max.length_aspect_ratio below",impute_vehicle_df['max.length_aspect_ratio'].quantile(0.25)-(1.5 * IQR),"are outliers")
print("The above Outliers in max.length_aspect_ratio column are",impute_vehicle_df[impute_vehicle_df['max.length_aspect_ratio']>14.5]['max.length_aspect_ratio'].shape[0])
print("The below Outliers in max.length_aspect_ratio column are",impute_vehicle_df[impute_vehicle_df['max.length_aspect_ratio']<2.5]['max.length_aspect_ratio'].shape[0])


# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(impute_vehicle_df['scatter_ratio'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(impute_vehicle_df['scatter_ratio'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in scatter_ratio column and there are two peaks in distribution plot and there is right skewness because long tail is at right side(mean>median) 

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(impute_vehicle_df['elongatedness'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(impute_vehicle_df['elongatedness'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in elongatedness column and there are two peaks in distribution plot and there is left skewness because long tail is at left side(mean<median) 

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(impute_vehicle_df['pr.axis_rectangularity'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(impute_vehicle_df['pr.axis_rectangularity'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in pr.axis_rectangularity column and there are two peaks in distribution plot and there is right skewness because long tail is at right side(mean>median)

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(impute_vehicle_df['max.length_rectangularity'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(impute_vehicle_df['max.length_rectangularity'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in max.length_rectangularity column and there are two peaks in distribution plot and there is right skewness because long tail is at right side(mean>median)

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(impute_vehicle_df['scaled_variance'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(impute_vehicle_df['scaled_variance'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are outliers in scaled_variance column and there are two peaks in distribution plot and there is right skewness because long tail is at right side(mean>median)

# In[ ]:


#check how many outliers are there in scaled_variance column
q1 = np.quantile(impute_vehicle_df['scaled_variance'],0.25)
q2 = np.quantile(impute_vehicle_df['scaled_variance'],0.50)
q3 = np.quantile(impute_vehicle_df['scaled_variance'],0.75)
IQR = q3-q1
print("Quartie1::",q1)
print("Quartie2::",q2)
print("Quartie3::",q3)
print("Inter Quartie Range::",IQR)
#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR
print("scaled_variance above",impute_vehicle_df['scaled_variance'].quantile(0.75)+(1.5 * IQR),"are outliers")
print("The Outliers in scaled_variance column are",impute_vehicle_df[impute_vehicle_df['scaled_variance']>292]['scaled_variance'].shape[0])


# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(impute_vehicle_df['scaled_variance.1'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(impute_vehicle_df['scaled_variance.1'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are outliers in scaled_variance.1 column and there are two peaks in distribution plot and there is right skewness because long tail is at right side(mean>median)

# In[ ]:


#check how many outliers are there in scaled_variance.1 column
q1 = np.quantile(impute_vehicle_df['scaled_variance.1'],0.25)
q2 = np.quantile(impute_vehicle_df['scaled_variance.1'],0.50)
q3 = np.quantile(impute_vehicle_df['scaled_variance.1'],0.75)
IQR = q3-q1
print("Quartie1::",q1)
print("Quartie2::",q2)
print("Quartie3::",q3)
print("Inter Quartie Range::",IQR)
#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR
print("scaled_variance.1 above",impute_vehicle_df['scaled_variance.1'].quantile(0.75)+(1.5 * IQR),"are outliers")
print("The Outliers in scaled_variance.1 column are",impute_vehicle_df[impute_vehicle_df['scaled_variance.1']>989.5]['scaled_variance.1'].shape[0])


# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(impute_vehicle_df['scaled_radius_of_gyration'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(impute_vehicle_df['scaled_radius_of_gyration'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in scaled_radius_of_gyration column and there is right skewness because long tail is at right side(mean>median)

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(impute_vehicle_df['scaled_radius_of_gyration.1'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(impute_vehicle_df['scaled_radius_of_gyration.1'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are outliers in scaled_radius_of_gyration.1 column and there is right skewness because long tail is at right side(mean>median)

# In[ ]:


#check how many outliers are there in scaled_radius_of_gyration.1 column
q1 = np.quantile(impute_vehicle_df['scaled_radius_of_gyration.1'],0.25)
q2 = np.quantile(impute_vehicle_df['scaled_radius_of_gyration.1'],0.50)
q3 = np.quantile(impute_vehicle_df['scaled_radius_of_gyration.1'],0.75)
IQR = q3-q1
print("Quartie1::",q1)
print("Quartie2::",q2)
print("Quartie3::",q3)
print("Inter Quartie Range::",IQR)
#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR
print("scaled_radius_of_gyration.1 above",impute_vehicle_df['scaled_radius_of_gyration.1'].quantile(0.75)+(1.5 * IQR),"are outliers")
print("The Outliers in scaled_radius_of_gyration.1 column are",impute_vehicle_df[impute_vehicle_df['scaled_radius_of_gyration.1']>87]['scaled_radius_of_gyration.1'].shape[0])


# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(impute_vehicle_df['skewness_about'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(impute_vehicle_df['skewness_about'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are outliers in skewness_about column and there is right skewness because long tail is at right side(mean>median)

# In[ ]:


#check how many outliers are there in skewness_about column
q1 = np.quantile(impute_vehicle_df['skewness_about'],0.25)
q2 = np.quantile(impute_vehicle_df['skewness_about'],0.50)
q3 = np.quantile(impute_vehicle_df['skewness_about'],0.75)
IQR = q3-q1
print("Quartie1::",q1)
print("Quartie2::",q2)
print("Quartie3::",q3)
print("Inter Quartie Range::",IQR)
#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR
print("skewness_about above",impute_vehicle_df['skewness_about'].quantile(0.75)+(1.5 * IQR),"are outliers")
print("The Outliers in skewness_about column are",impute_vehicle_df[impute_vehicle_df['skewness_about']>19.5]['skewness_about'].shape[0])


# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(impute_vehicle_df['skewness_about.1'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(impute_vehicle_df['skewness_about.1'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are outliers in skewness_about.1 column and there is right skewness because long tail is at right side(mean>median)

# In[ ]:


#check how many outliers are there in skewness_about.1 column
q1 = np.quantile(impute_vehicle_df['skewness_about.1'],0.25)
q2 = np.quantile(impute_vehicle_df['skewness_about.1'],0.50)
q3 = np.quantile(impute_vehicle_df['skewness_about.1'],0.75)
IQR = q3-q1
print("Quartie1::",q1)
print("Quartie2::",q2)
print("Quartie3::",q3)
print("Inter Quartie Range::",IQR)
#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR
print("skewness_about.1 above",impute_vehicle_df['skewness_about.1'].quantile(0.75)+(1.5 * IQR),"are outliers")
print("The Outliers in skewness_about.1 column are",impute_vehicle_df[impute_vehicle_df['skewness_about.1']>40]['skewness_about.1'].shape[0])


# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(impute_vehicle_df['skewness_about.2'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(impute_vehicle_df['skewness_about.2'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in skewness_about.2 column and there is left skewness because long tail is at left side(mean<median)

# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(20,4)
sns.distplot(impute_vehicle_df['hollows_ratio'],ax=ax1)
ax1.set_title("Distribution Plot")

sns.boxplot(impute_vehicle_df['hollows_ratio'],ax=ax2)
ax2.set_title("Box Plot")


# From above we can see that there are no outliers in hollows_ratio column and there is left skewness because long tail is at left side(mean<median)

# In[ ]:


impute_vehicle_df['class'].value_counts()


# In[ ]:


sns.countplot(impute_vehicle_df['class'])
plt.show()


# From above we can see that cars are most followed by bus and then vans.

# so by now we analyze each column and we found that there are outliers in some column. now our next step is to know whether these outliers are natural or artificial. if natural then we have to do nothing but if these outliers are artificial then we have to handle these outliers.
# we have 8 columns in which we found outliers:
# ->radius_ratio
# ->pr.axis_aspect_ratio
# ->max.length_aspect_ratio
# ->scaled_variance
# ->scaled_variance.1
# ->scaled_radius_of_gyration.1
# ->skewness_about
# ->skewness_about.1

# after seeing the max values of above outliers column. it's looks like outliers in above columns are natural not a typo mistake or artificial.
# as we will apply SVM on above data which is affected by outliers. so better to drop those outliers.

# #Fix Outliers after imputing missing values

# In[ ]:


#radius_ratio column outliers
impute_vehicle_df.drop(impute_vehicle_df[impute_vehicle_df['radius_ratio']>276].index,axis=0,inplace=True)


# In[ ]:


#pr.axis_aspect_ratio column outliers
impute_vehicle_df.drop(impute_vehicle_df[impute_vehicle_df['pr.axis_aspect_ratio']>77].index,axis=0,inplace=True)


# In[ ]:


#max.length_aspect_ratio column outliers
impute_vehicle_df.drop(impute_vehicle_df[impute_vehicle_df['max.length_aspect_ratio']>14.5].index,axis=0,inplace=True)
impute_vehicle_df.drop(impute_vehicle_df[impute_vehicle_df['max.length_aspect_ratio']<2.5].index,axis=0,inplace=True)


# In[ ]:


#scaled_variance column outliers
impute_vehicle_df[impute_vehicle_df['scaled_variance']>292]


# from above we can see that scaled_variance column outliers has been removed

# In[ ]:


#scaled_variance.1 column outliers
impute_vehicle_df.drop(impute_vehicle_df[impute_vehicle_df['scaled_variance.1']>989.5].index,axis=0,inplace=True)


# In[ ]:


#scaled_radius_of_gyration.1 column outliers
impute_vehicle_df.drop(impute_vehicle_df[impute_vehicle_df['scaled_radius_of_gyration.1']>87].index,axis=0,inplace=True)


# In[ ]:


#skewness_about column outliers
impute_vehicle_df.drop(impute_vehicle_df[impute_vehicle_df['skewness_about']>19.5].index,axis=0,inplace=True)


# In[ ]:


#skewness_about.1 column outliers
impute_vehicle_df.drop(impute_vehicle_df[impute_vehicle_df['skewness_about.1']>40].index,axis=0,inplace=True)


# In[ ]:


#display the shape of data frame
print("after fixing outliers shape of dataframe:",impute_vehicle_df.shape)


# In[ ]:


plt.figure(figsize=(20,4))
sns.heatmap(impute_vehicle_df.corr(),annot=True)
plt.show()


# so our objective is to reocgnize whether an object is a van or bus or car based on some input features.
# so our main assumption is there is little or no multicollinearity between the features.
# if two features is highly correlated then there is no use in using both features.in that case, we can drop one feature. 
# so heatmap gives us the correlation matrix there we can see which features are highly correlated.
# From above correlation matrix we can see that there are many features which are highly correlated. if we see carefully then scaled_variance.1 and scatter_ratio has 1 correlation and many other features also there which having more than 0.9 correlation
# so we will drop those columns whose correlation is +-0.9 or above.
# so there are 8 such columns:
# ->max.length_rectangularity
# ->scaled_radius_of_gyration
# ->skewness_about.2
# ->scatter_ratio
# ->elongatedness
# ->pr.axis_rectangularity
# ->scaled_variance
# ->scaled_variance.1

# now, again we have two option we will drop those above eight columns manually or we will apply pca and let pca to be decided how it will explain above data which is in high dimension with smaller number of variables.
# we will see both approaches.

# #With Principal Component Analysis(PCA)

# In[ ]:


#now separate the dataframe into dependent and independent variables
impute_vehicle_df_independent_attr = impute_vehicle_df.drop('class',axis=1)
impute_vehicle_df_dependent_attr = impute_vehicle_df['class']
print("shape of impute_vehicle_df_independent_attr::",impute_vehicle_df_independent_attr.shape)
print("shape of impute_vehicle_df_dependent_attr::",impute_vehicle_df_dependent_attr.shape)


# In[ ]:


#now sclaed the independent attribute and replace the dependent attr value with number
impute_vehicle_df_independent_attr_scaled = impute_vehicle_df_independent_attr.apply(zscore)
impute_vehicle_df_dependent_attr.replace({'car':0,'bus':1,'van':2},inplace=True)


# In[ ]:


#make the covariance matrix and we have 18 independent features so aur covariance matrix is 18*18 matrix
impute_cov_matrix = np.cov(impute_vehicle_df_independent_attr_scaled,rowvar=False)
print("Impute cov_matrix shape:",impute_cov_matrix.shape)
print("Impute Covariance_matrix",impute_cov_matrix)


# In[ ]:


#now with the help of above covariance matrix we will find eigen value and eigen vectors
impute_pca_to_learn_variance = PCA(n_components=18)
impute_pca_to_learn_variance.fit(impute_vehicle_df_independent_attr_scaled)


# In[ ]:


#display explained variance ratio
impute_pca_to_learn_variance.explained_variance_ratio_


# In[ ]:


#display explained variance
impute_pca_to_learn_variance.explained_variance_


# In[ ]:


#display principal components
impute_pca_to_learn_variance.components_


# In[ ]:


plt.bar(list(range(1,19)),impute_pca_to_learn_variance.explained_variance_ratio_)
plt.xlabel("eigen value/components")
plt.ylabel("variation explained")
plt.show()


# In[ ]:


plt.step(list(range(1,19)),np.cumsum(impute_pca_to_learn_variance.explained_variance_ratio_))
plt.xlabel("eigen value/components")
plt.ylabel("cummalative of variation explained")
plt.show()


# From above we can see that 8 dimension are able to explain 95%variance of data. so we will use first 8 principal components

# In[ ]:


#use first 8 principal components
impute_pca_eight_components = PCA(n_components=8)
impute_pca_eight_components.fit(impute_vehicle_df_independent_attr_scaled)


# In[ ]:


#transform the impute raw data which is in 18 dimension into 8 new dimension with pca
impute_vehicle_df_pca_independent_attr = impute_pca_eight_components.transform(impute_vehicle_df_independent_attr_scaled)


# In[ ]:


#display the shape of new_vehicle_df_pca_independent_attr
impute_vehicle_df_pca_independent_attr.shape


# now before apply pca with 8 dimension which are explaining more than 95% variantion of data we will make model on raw data after that we will make model with pca and then we will compare both models.

# In[ ]:


#now split the data into 80:20 ratio
impute_rawdata_X_train,impute_rawdata_X_test,impute_rawdata_y_train,impute_rawdata_y_test = train_test_split(impute_vehicle_df_independent_attr_scaled,impute_vehicle_df_dependent_attr,test_size=0.20,random_state=1)
impute_pca_X_train,impute_pca_X_test,impute_pca_y_train,impute_pca_y_test = train_test_split(impute_vehicle_df_pca_independent_attr,impute_vehicle_df_dependent_attr,test_size=0.20,random_state=1)


# In[ ]:


print("shape of impute_rawdata_X_train",impute_rawdata_X_train.shape)
print("shape of impute_rawdata_y_train",impute_rawdata_y_train.shape)
print("shape of impute_rawdata_X_test",impute_rawdata_X_test.shape)
print("shape of impute_rawdata_y_test",impute_rawdata_y_test.shape)
print("--------------------------------------------")
print("shape of impute_pca_X_train",impute_pca_X_train.shape)
print("shape of impute_pca_y_train",impute_pca_y_train.shape)
print("shape of impute_pca_X_test",impute_pca_X_test.shape)
print("shape of impute_pca_y_test",impute_pca_y_test.shape)


# In[ ]:


#fit the model on impute raw data
svc.fit(impute_rawdata_X_train,impute_rawdata_y_train)


# In[ ]:


#predict the y value
impute_rawdata_y_predict = svc.predict(impute_rawdata_X_test)


# In[ ]:


#now fit the model on pca data with new dimension
svc.fit(impute_pca_X_train,impute_pca_y_train)


# In[ ]:


#predict the y value
impute_pca_y_predict = svc.predict(impute_pca_X_test)


# In[ ]:


#display accuracy score of both models
print("Accuracy score with impute raw data(18 dimension)",accuracy_score(impute_rawdata_y_test,impute_rawdata_y_predict))
print("Accuracy score with impute pca data(8 dimension)",accuracy_score(impute_pca_y_test,impute_pca_y_predict))


# In[ ]:


#display confusion matrix of both models
print("Confusion matrix with impute raw data(18 dimension)\n",confusion_matrix(impute_rawdata_y_test,impute_rawdata_y_predict))
print("Confusion matrix with impute pca data(8 dimension)\n",confusion_matrix(impute_pca_y_test,impute_pca_y_predict))


# #With Dropping the above mentioned columns manually

# In[ ]:


#drop the columns
impute_vehicle_df_independent_attr_scaled.drop(['max.length_rectangularity','scaled_radius_of_gyration','skewness_about.2','scatter_ratio','elongatedness','pr.axis_rectangularity','scaled_variance','scaled_variance.1'],axis=1,inplace=True)


# In[ ]:


#display the shape of new dataframe
impute_vehicle_df_independent_attr_scaled.shape


# In[ ]:


impute_dropcolumn_X_train,impute_dropcolumn_X_test,impute_dropcolumn_y_train,impute_dropcolumn_y_test = train_test_split(impute_vehicle_df_independent_attr_scaled,impute_vehicle_df_dependent_attr,test_size=0.20,random_state=1)


# In[ ]:


print("shape of impute_dropcolumn_X_train",impute_dropcolumn_X_train.shape)
print("shape of impute_dropcolumn_y_train",impute_dropcolumn_y_train.shape)
print("shape of impute_dropcolumn_X_test",impute_dropcolumn_X_test.shape)
print("shape of impute_dropcolumn_y_test",impute_dropcolumn_y_test.shape)


# In[ ]:


#fit the model on dropcolumn_X_train,dropcolumn_y_train
svc.fit(impute_dropcolumn_X_train,impute_dropcolumn_y_train)


# In[ ]:


#predict the y value
impute_dropcolumn_y_predict = svc.predict(impute_dropcolumn_X_test)


# In[ ]:


#display the accuracy score and confusion matrix
print("Accuracy score with impute dropcolumn data(10 dimension)",accuracy_score(impute_dropcolumn_y_test,impute_dropcolumn_y_predict))
print("Confusion matrix with impute dropcolumn data(10 dimension)\n",confusion_matrix(impute_dropcolumn_y_test,impute_dropcolumn_y_predict))


# #Conclusion:
# From above we can see that pca is doing a very good job.Accuracy with pca is approx 94% and with raw data approx 96% but note that pca 94% accuracy is with only 8 dimension where as rawdata has 18 dimension.But every thing has two sides, disadvantage of pca is we cannot do interpretation with the model.it's blackbox.

# Thanks for reading the kernel!
# Happy Learning:)
