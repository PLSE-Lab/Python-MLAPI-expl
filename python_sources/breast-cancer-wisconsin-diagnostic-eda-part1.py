#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#columns = ["ID","diagnosis","radius","texture","perimeter","area","smoothness","compactness","concavity","concave points","symmtery","fractal dimension",""]
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",sep=",",header=None)
df.columns = ["ID","Diagnosis","radius","texture","perimeter","area","smoothness","compactness","concavity","concave_points","symmtery","fractal_dimension",
              "radius_se", "texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave_point_se","symmetry_se","fractal_dimension_se",
              "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_point_worst","symmetry_worst","fractal_dimension_worst"]
df.head()


# In[ ]:


#Separating dependent and independent variables
drop_cols = ["Diagnosis","ID"]
X = df.drop(drop_cols, axis=1)
y = df["Diagnosis"]
X.head()


# In[ ]:


y.head()


# In[ ]:


# To know if the dataset is imbalance or not using visualizations
ax = sns.countplot(y,label = "count")   # countplot counts the no.of values of different classes in a column
B, M = y.value_counts()
print("Number of Benign Tumors :", B)
print("Number of Malignant Tumors :", M)


# The above histogram clearly shows that the data is imabalanced as the number of Benign cases are higher when compared to Malignant. So, it shows that we have to handle the imbalanced data before modelling

# In[ ]:


X.describe()   # descriptive statistics


# The above tables describes the mean, standard deviation, min value, maximum value etc. For example, the range of max value of area and smoothness differs alot, that shows that we have to use some normalisation or standartization techniques before we move on to feature selection and classification.

# Violin plot is a method of plotting numeric data.   
# Violin plots are similar to box plots except that they show the density of the data at different values.    
# Before creating Violin plots and swarm plots we need standardize the data.

# In[ ]:


data = X
data_standard = (data - data.mean())/data.std()    # standardizing the data


# Before Creating Violin plots, we saw that we have 30 columns, 
# so it would be heavy to plot it out at once, so we shall divide it in to 3 groups where each group has 10 features. This way we would be able to observe the data better.

# In[ ]:


data = pd.concat([y, data_standard.iloc[:, 0:10]], axis=1) # Creating a new dataframe with only 10 features
data = pd.melt(data, id_vars = "Diagnosis", var_name = "features", value_name= "value") # Making the data from in to right format and structure for plotting
plt.figure(figsize=(10,10))  # Using matplotlib to set the figure size
sns.violinplot(x ="features", y ="value", hue="Diagnosis",data= data, split= True, inner= "quart")
plt.xticks(rotation=45);


# In[ ]:


data = pd.concat([y, data_standard.iloc[:, 10:20]], axis=1) # Creating a new dataframe with only 10 features
data = pd.melt(data, id_vars = "Diagnosis", var_name = "features", value_name= "value") # Making the data from in to right format and structure for plotting
plt.figure(figsize=(10,10))  # Using matplotlib to set the figure size
sns.violinplot(x ="features", y ="value", hue="Diagnosis",data= data, split= True, inner= "quart")
plt.xticks(rotation=45);


# In[ ]:


data = pd.concat([y, data_standard.iloc[:, 20:30]], axis=1) # Creating a new dataframe with only 10 features
data = pd.melt(data, id_vars = "Diagnosis", var_name = "features", value_name= "value") # Making the data from in to right format and structure for plotting
plt.figure(figsize=(10,10))  # Using matplotlib to set the figure size
sns.violinplot(x ="features", y ="value", hue="Diagnosis",data= data, split= True, inner= "quart")
plt.xticks(rotation=45);


# For example, in the above plot, it looks like the concavity_worst and concave_point_worst are similar. For now, we are not sure if they are corelated or not. The best practice is, if they are corelated, it a best way to reduce the redundacy by dropping one of those features.

# In[ ]:


# Box plots helps in identifying the outliers.
data = pd.concat([y, data_standard.iloc[:, 0:10]], axis=1) # Creating a new dataframe with only 10 features
data = pd.melt(data, id_vars = "Diagnosis", var_name = "features", value_name= "value")
plt.figure(figsize=(10,10))
sns.boxplot(x ="features", y="value", hue="Diagnosis",data=data)
plt.xticks(rotation=45);


# In[ ]:


data = pd.concat([y, data_standard.iloc[:, 10:20]], axis=1) # Creating a new dataframe with only 10 features
data = pd.melt(data, id_vars = "Diagnosis", var_name = "features", value_name= "value")
plt.figure(figsize=(10,10))
sns.boxplot(x ="features", y="value", hue="Diagnosis",data=data)
plt.xticks(rotation=45);


# In[ ]:


data = pd.concat([y, data_standard.iloc[:, 20:30]], axis=1) # Creating a new dataframe with only 10 features
data = pd.melt(data, id_vars = "Diagnosis", var_name = "features", value_name= "value")
plt.figure(figsize=(10,10))
sns.boxplot(x ="features", y="value", hue="Diagnosis",data=data)
plt.xticks(rotation=45);


# In the violin plot we found that the 2 features looks similar to each other. The issue here is that the 2 corelated columns can negatively impact the predictive accuracy of your classifier. So we would drop one of the corelated columns.
# 
# 
# Joint Plots help us in the same. It is very much helpful to dig deeper in to 2 specific features.

# In[ ]:


sns.jointplot(X.loc[:, "concavity_worst"], 
              X.loc[:, "concave_point_worst"], 
              kind="regg",
              color= "#ce1414"
              );


# So, by looking at the above graph, we can straightly confirm that those 2 features are highly corelated.

# In[ ]:


# Swarm plot will show us all the data points while stacking up with the similar values
# It helps in observing the distribution of the values 
sns.set(style="whitegrid", palette="muted")
data = X
data_standard = (data - data.mean())/data.std()
data = pd.concat([y, data_standard.iloc[:, 0:10]], axis=1)
data = pd.melt(data, id_vars="Diagnosis",
               var_name="features",
               value_name="value")
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="Diagnosis", data=data)
plt.xticks(rotation=45);


# In[ ]:


sns.set(style="whitegrid", palette="muted")
data = X
data_standard = (data - data.mean())/data.std()
data = pd.concat([y, data_standard.iloc[:, 10:20]], axis=1)
data = pd.melt(data, id_vars="Diagnosis",
               var_name="features",
               value_name="value")
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="Diagnosis", data=data)
plt.xticks(rotation=45);


# In[ ]:


sns.set(style="whitegrid", palette="muted")
data = X
data_standard = (data - data.mean())/data.std()
data = pd.concat([y, data_standard.iloc[:, 20:30]], axis=1)
data = pd.melt(data, id_vars="Diagnosis",
               var_name="features",
               value_name="value")
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="Diagnosis", data=data)
plt.xticks(rotation=45);


# So, we can tell which feature is better for classification by looking at the variance and how well separated they are. For example from the second plot we can say that the perimeter_worst feature is much more than the smoothness_worst feature
# 
# 
# 
# So till now we have done all the plots using batchwise to find out the corelation. Now lets look at how we can get the corelation  matrix for all the features using heatmaps.

# In[ ]:


f, ax = plt.subplots(figsize=(20,20))
sns.heatmap(X.corr(), annot=True, linewidth= .5, fmt ='.0%', ax=ax);


# The above heatmap shows the corelation between all the features which would be very helpful in selecting the important features.For example, the black colour here shows that they are very negatively corelated and white colour cells have a very high corelation.
