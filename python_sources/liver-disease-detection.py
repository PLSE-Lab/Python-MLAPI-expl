#!/usr/bin/env python
# coding: utf-8

# # Importing Packages and looking into the dataset
# 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Reading the file as saving to dataframe(df)


# In[ ]:


df=pd.read_csv('../input/liver.csv')
df.head()


# In[ ]:


# Checking dataframe to see if there are any missing values. There are two missing values


# In[ ]:


sns.heatmap(df.isnull(), cmap='coolwarm',xticklabels=True,yticklabels=False,cbar=False)


# In[ ]:


# Impute missing values by importing the Imputer class from sklearn.preprocessing


# In[ ]:


from sklearn.preprocessing import Imputer


# In[ ]:


imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)


# In[ ]:


imputer = imputer.fit(df.iloc[:,9:10])


# In[ ]:


df.iloc[:,9:10]= imputer.transform(df.iloc[:,9:10])


# In[ ]:


# Checking for missing data again


# In[ ]:


sns.heatmap(df.isnull(), cmap='coolwarm',xticklabels=True,yticklabels=False,cbar=False)


# In[ ]:


# Get some info on the dataset


# In[ ]:


df.info()


# In[ ]:


df.drop('Selector',axis=1).describe()


# In[ ]:


# Checking if data is skewed


# In[ ]:


df.skew()


# # Data Visualization

# In[ ]:


# As the heatmap shows there are no missing values. Now let's do some visalization.


# In[ ]:


#plt.figure(figsize= (6
#df.hist()


# In[ ]:


sns.pairplot(df)


# In[ ]:


# Looks like there may be some linear correlations between some of the features. More data visualizations


# In[ ]:


sns.barplot(x='Selector',y='Age',data =df)


# In[ ]:


# Mean Age is roughly the same for both selctors


# In[ ]:


sns.jointplot(x='Selector',y='Age',data =df)


# In[ ]:


sns.distplot(df['Age'])


# In[ ]:


# Age looks almost normally distributed


# In[ ]:


sns.countplot(x='Gender',data=df)


# In[ ]:


# More Males than Females in the dataset


# In[ ]:


sns.countplot(x='Gender',data=df,hue='Selector')


# In[ ]:


# The percentage of females falling under category 2 is higher than that of of males when compared to the total of
#their gender.


# In[ ]:


sns.violinplot(x='Gender',y='Age',hue='Selector',data=df)


# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


# Some of the features are highly correlated


# # Preparing data for Machine Learning Algorithms

# In[ ]:


df.head()


# # Using pd.get_dummies instead to turn categorical data into inegers

# In[ ]:


# Encoding gender 


# In[ ]:


Gender = pd.get_dummies(df.iloc[: ,1], drop_first=True)


# In[ ]:


df = pd.concat([df,Gender],axis=1)


# In[ ]:


df.head()


# In[ ]:


df.drop('Gender',axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


#Encoding Selector and Renaming it as Prognosis


# In[ ]:


Result = pd.get_dummies(df['Selector'],drop_first=True)


# In[ ]:


df=pd.concat([df,Result],axis=1)


# In[ ]:


df.head(10)


# In[ ]:


df.drop('Selector',axis=1,inplace=True)


# In[ ]:


# This turned the categories in the Selector column: Category 2 is now category 1 and Category 1 is now category 0


# In[ ]:


df.head()


# In[ ]:


#renaming column 2 to Prognosis


# In[ ]:


df['Prognosis'] = df[2]


# In[ ]:


df.drop(2,axis=1,inplace=True)


# In[ ]:


df.head(10)


# In[ ]:


#checking if target variable is imbalanced


# In[ ]:


df['Prognosis'].value_counts()


# # Data is unbalanced - Balancing it by Up-sampling the minority class

# In[ ]:


from sklearn.utils import resample


# In[ ]:


# Creating 2 different dataframes df_majority and df_minority


# In[ ]:


df_majority = df[df['Prognosis']==0]


# In[ ]:


df_minority = df[df['Prognosis']==1]


# In[ ]:


# Upsample minority class


# In[ ]:


df_minority_upsampled = resample(df_minority,replace=True,n_samples=416, random_state=123)


# In[ ]:


# Combine majority class with upsampled minority class


# In[ ]:


df_upsampled = pd.concat([df_majority,df_minority_upsampled])


# In[ ]:


df_upsampled['Prognosis'].value_counts()


# # Spliting the dataset into independent variables X and dependent variable y and into test and train sets

# In[ ]:


X =df_upsampled.drop('Prognosis', axis=1)


# In[ ]:


y = df_upsampled['Prognosis']


# # Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


sc_X = StandardScaler()


# In[ ]:


X = sc_X.fit_transform(X)


# # Spliting Data into Test and Train Data

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# # 9. Deep Neural Networks

# In[ ]:


import keras


# In[ ]:


from keras.models import Sequential


# In[ ]:


from keras.layers import Dense


# In[ ]:


# Initializing the Network


# In[ ]:


nn_classifier = Sequential()


# In[ ]:


# Adding the first input layer and the first hidden layer


# In[ ]:


nn_classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=10))


# In[ ]:


# Adding second Layer


# In[ ]:


nn_classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))


# In[ ]:


# Adding output layer


# In[ ]:


nn_classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))


# In[ ]:


# Compiling Neural Network


# In[ ]:


nn_classifier.compile(optimizer='adam', loss = 'binary_crossentropy',metrics=['accuracy'])


# In[ ]:


nn_classifier.fit(X_train,y_train,batch_size=10,epochs=1000)


# In[ ]:


# Neural Network gives us an 80-81% accuracy

