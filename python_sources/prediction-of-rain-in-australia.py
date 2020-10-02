#!/usr/bin/env python
# coding: utf-8

# ## 1. Import Libraries <a class = "anchor" id = "1"></a>

# In[ ]:


import numpy as np #linear algebra
import pandas as pd #data processing

#import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# importing data
df = pd.read_csv("../input/weather-dataset-rattle-package/weatherAUS.csv")


# ## 2. Exploratory data analysis<a class = "anchor" id = "2"></a>
# * As we have imported the data.
# * Now, its time to explore the data to get insigths about it.

# In[ ]:


#preview the dataset
df.head()


# In[ ]:


#dimentions of dataset
df.shape


# In[ ]:


#viewing column names
df.columns


# In[ ]:


#drop RISK_MM variable (it is given in the description to drop the feature)
df.drop(["RISK_MM"], axis=1, inplace=True)


# In[ ]:


#viewing the summary of dataset
df.info()


# In[ ]:


#view statistical properties of dataset
df.describe()


# ## 3.Univariate Analysis<a class = "anchor" id = "3"></a>

# In[ ]:


#Explore "RainTomorrow" target variable
#check for missing values
df["RainTomorrow"].isnull().sum()


# In[ ]:


#view number of unique values
df["RainTomorrow"].nunique()


# In[ ]:


#view the unique values
df["RainTomorrow"].unique()


# In[ ]:


#view the frequency distribution of values
df['RainTomorrow'].value_counts()


# In[ ]:


#view percentage of frequency distribution of values
df["RainTomorrow"].value_counts()/len(df)


# ### Findings of Univariate Analysis
# * The number of unique values in "RainTomorrow" is 2 ie "Yes" or "No".
# * Out of total number of "RainTomorrow" values, No appears 77.58% times and Yes appears 22.42% times.

# ## 6.Bivariate Analysis<a class = "anchor" id = "6"></a>
# In this section we explore two categories : Categorial Variables and Numerical Variables. 

# ### Exploring Categorical Variables

# In[ ]:


#find categorical values

categorical = [var for var in df.columns if df[var].dtype=='O']
print("There are {} categorical values\n".format(len(categorical)))
print("The categorical variavles are : ", categorical)


# In[ ]:


#view categorical variables
df[categorical].head()


# In[ ]:


#check missing values in categorical variables
df[categorical].isnull().sum()


# In[ ]:


#view frequency count of categorical variables
for var in categorical:
    print(df[var].value_counts())


# In[ ]:


#check for cardinality in categorical variables
for var in categorical:
    print(var, " contains ",len(df[var].unique()), " labels")


# Date variable needs to be preprocessed as it has **High cardinality**. All the other variables contain relatively smaller number of variables.
# 
# **Feature Engineering of Date variable.**

# In[ ]:


df['Date'].dtypes


# We can see that the data type of Date variable is object. I will parse the Date currently coded as object into datetime format.

# In[ ]:


#parse the dates, currently coded as strings, into datetime format
df["Date"] = pd.to_datetime(df['Date'])


# In[ ]:


#extract year from date
df['Year'] = df['Date'].dt.year
df['Year'].head()


# In[ ]:


#extract month from date
df['Month'] = df['Date'].dt.month
df['Month'].head() 


# In[ ]:


#extract day from date
df['Day'] = df['Date'].dt.day
df['Day'].head()


# In[ ]:


#again viewing the summary of the dataset
df.info()


# In[ ]:


#As there are three additional columns from Date variable, I will drop the original Date variable.
df.drop('Date', axis = 1, inplace = True)


# In[ ]:


df.head()


# In[ ]:


#find categorical values
categorical = [var for var in df.columns if df[var].dtype=='O']
df[categorical].isnull().sum()


# ### Explore `Location` variable

# In[ ]:


#print number of labels in Location variable
print('Location contains', len(df.Location.unique()), 'labels')


# In[ ]:


#check labels in location variable
df.Location.unique()


# In[ ]:


#check frequency distribution of values in Location variabe
df.Location.value_counts()


# In[ ]:


# let's do One Hot Encoding of Location variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method

pd.get_dummies(df.Location, drop_first = True).head()


# ### Explore `WindGustDir` variable

# In[ ]:


#print number of labels in WindGustDir variable
print('WindGustDir contains',len(df.WindGustDir.unique()),'labels')


# In[ ]:


#check labels in WindGustDir variable
df['WindGustDir'].unique()


# In[ ]:


#check frequency distribution of values in WindGustDir variable
df.WindGustDir.value_counts()


# In[ ]:


# let's do One Hot Encoding of WindGustDir variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method
pd.get_dummies(df.WindGustDir, drop_first = True, dummy_na = True).head()


# In[ ]:


# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category
pd.get_dummies(df.WindGustDir, drop_first = True, dummy_na = True).sum(axis=0)


# ### Explore `WindDir9am` variable

# In[ ]:


#check number of labels in WindDir9am variable
print('WindDir9am contains', len(df.WindDir9am.unique()),'labels')


# In[ ]:


#check lables in WindDir9am variable
df['WindDir9am'].unique()


# In[ ]:


#check frequency distribution of values in WindDir9am variable
df['WindDir9am'].value_counts()


# In[ ]:


# let's do One Hot Encoding of WindDir9am variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()


# In[ ]:


# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)


# ### Explore `WindDir3pm` variable

# In[ ]:


#print number of lables in WindDir3pm variable
print('WindDir3pm contains',len(df.WindDir3pm.unique()),'labels')


# In[ ]:


#check labels in WindDir3pm variable
df['WindDir3pm'].unique()


# In[ ]:


#check for frequency distribution of values in WindDir3pm variable
df['WindDir3pm'].value_counts()


# In[ ]:


# let's do One Hot Encoding of WindDir3pm variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()


# In[ ]:


# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)


# ### Explore `RainToday` variable

# In[ ]:


# print number of labels in RainToday variable
print('RainToday contains', len(df['RainToday'].unique()), 'labels')


# In[ ]:


# check labels in WindGustDir variable
df['RainToday'].unique()


# In[ ]:


# check frequency distribution of values in WindGustDir variable
df.RainToday.value_counts()


# In[ ]:


# let's do One Hot Encoding of RainToday variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()


# In[ ]:


# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0)


# ### Explore Numerical variables

# In[ ]:


# find numerical variables
numerical = [var for var in df.columns if df[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)


# In[ ]:


#view the numberical variables
df[numerical].head()


# #### Explore problems within numerical variables

# In[ ]:


#check missing values in numerical variable
df[numerical].isnull().sum()


# We can see that there are 16 numerical variable containing missing values.

# In[ ]:


#view summary statistics in numerical variables
print(round(df[numerical].describe()),2)


# On closer inspection, we can see that the `Rainfall`, `Evaporation`, `WindSpeed9am` and `WindSpeed3pm` columns may contain outliers.
# 
# Let's draw boxplot to visualize outliers.

# In[ ]:


plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_label('Rainfall')

plt.subplot(2,2,2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_label('Evaporation')

plt.subplot(2,2,3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_label('WindSpeed9am')

plt.subplot(2,2,4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_label('WindSpeed3pm')


# The above boxplot confirms that there are lot of outliers in these variables.

# ## Check the distribution of variables
# * Now, I will plot histograms to check distributions to find out if the are normal or skewed.
# * If the variable follows normal distribution, then I will do `Extreme value Analysis` otherwise if they are skewed, i will find IQR(Interquantile range).

# In[ ]:


# plot historams to check distribution
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')

plt.subplot(2,2,2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')

plt.subplot(2,2,3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')

plt.subplot(2,2,4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')


# We can see that all the four variables are skewed. So, I will use IQR to find outliers.

# In[ ]:


# find outliers for Rainfall variable

IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)

print('Rainfall outlier values are < {} or > {}'.format(lower_fence, upper_fence))


# For `Rainfall`, the minimum and maximum values are 0.0 and 371.0. So, the outliers are values > 3.2

# In[ ]:


# find outliers for Evaporation variable

IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)

print('Evaporation outliers values are < {} or > {}'.format(lower_fence, upper_fence))


# For `Evaporation`, the minimum and maximum values are 0.0 and 145.0. so, the outliers are values > 21.8

# In[ ]:


# find outliers for WindSpeed9am variable

IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outlier values are < {} or > {}'.format(lower_fence, upper_fence))


# For `WindSpeed9am`, the minimum and maximum values are 0.0 and 130.0. so, the outiers are values > 55.0

# In[ ]:


# find outliers for WindSpeed3pm variable

IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers values are < {} or > {}'.format(lower_fence, upper_fence))


# For `WindSpeed3pm`, the minimum and maximum values are 0.0 and 87.0. so, the outliers are values > 57.0

# ## 7. Multivariate Analysis
# * An important spet in EDA is to discover patterns and relationships between variables in the dataset.
# * I will use heat map and pair plt to discover the patterns and relationsip in the dataset.
# * First of all, I will draw a heat map.

# In[ ]:


correlation = df.corr()


# ### HeatMap

# In[ ]:


plt.figure(figsize=(16, 12))
plt.title('Correlation Heatmap of Rain in Australia Dataset')
ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f',linecolor='white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)

plt.show()


# **Interpretation** : 
# From the above correlation map, we can conclude that the variables which are highly positively correlated are - 
# * `MinTemp` and `MaxTemp` (cc = 0.74)
# * `MinTemp` and `Temp3pm` (cc = 0.71)
# * `MinTemp` and `Temp9am` (cc = 0.90)
# * `MaxTemp` and `Temp9am` (cc = 0.89)
# * `MaxTemp` and `Temp3pm` (cc = 0.98)
# * `WindGustSpeed` and `WindSpeed3pm` (cc = 0.69)
# * `Pressure9am` and `Pressure3pm` (cc = 0.96)
# * `Temp9am` and `Temp3pm` (cc = 0.86)

# ### Pair Plot
# First of all, I will define extract the variables which are highly positivety correlated.

# In[ ]:


num_var = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm', 'WindGustSpeed', 'WindSpeed3pm', 'Pressure9am', 'Pressure3pm']


# In[ ]:


#sns.pairplot(df[num_var], kind='scatter', diag_kind='hist', palette='Rainbow')
#plt.show()


# **Interpretation** 
# * I have defined a variable num_var which consists of `MinTemp`, `MaxTemp`, `Temp9am`, `Temp3pm`, `WindGustSpeed`, `WindSpeed3pm`, `Pressure9am` and `Pressure3pm` variables.

# ## 8. Declare feature vector and target variable

# In[ ]:


X = df.drop(['RainTomorrow'], axis=1)
y = df['RainTomorrow']


# ## 9. Split data into seprate training and test set

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


X_train.shape, X_test.shape


# ## 10. Feature Engineering
# **Feature Engineering** is the process of transforming raw data into useful features that helps us to understand our model better and increase its predictive power. I will carry out our feature engineering on different types of variables.
# First, I will display the categorical and numberical variables again separately.

# In[ ]:


# check data types in x_train
X_train.dtypes


# In[ ]:


# display categorical variables
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
categorical


# In[ ]:


# display numerical variables
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
numerical


# ### Engineering missing values in numerical variables

# In[ ]:


X_train[numerical].isnull().sum()


# In[ ]:


X_test[numerical].isnull().sum()


# In[ ]:


# print percentage of missing values in the numerical variables in train set
for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, round(X_train[col].isnull().mean(), 4))


# **Assumption**
# > I assume that the data are missing completely at random (MCAR). There are two methods which can be used to impute missing values. One is mean or median imputation and other one is random sample imputation. When there are outliers in the dataset, we should use median imputation. So, I will use median imputation because median imputation is robust to outliers.
# 
# > I will impute missing values with the appropriate statistical measures of the data, in this case median. Imputation should be done over the training set, and then propagated to the test set. It means that the statistical measures to be used to fill missing values both in train and test set, should be extracted from the train set only. This is to avoid overfitting.

# In[ ]:


# inpute missing values in X_train and X_test with respective column meadian in X_train
for df1 in [X_train, X_test]:
    for col in numerical:
        col_median = X_train[col].median()
        df1[col].fillna(col_median, inplace = True)


# In[ ]:


X_train[numerical].isnull().sum()


# In[ ]:


X_test[numerical].isnull().sum()


# Now, we can see that there are no missing values in the numerical columns of training and test set.

# ### Engineering missing values in categorical variables

# In[ ]:


# print percentage of missing values in the categorical variables in training set
X_train[categorical].isnull().mean()


# In[ ]:


#inpute missing categorical variables with most frequent value
for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)


# In[ ]:


X_train[categorical].isnull().sum()


# In[ ]:


X_test[categorical].isnull().sum()


# We can see that are no missing values in X_train and X_test.

# ### Engineering outliers in numerical variables
# > We have seen that the `Rainfall`, `Evaporation`, `WindSpeed9am` and `WindSpeed3pm` columns contain outliers. I will use top-coding approach to cap maximum values and remove outliers from the above variables.

# In[ ]:


def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)


# In[ ]:


X_train.Rainfall.max(), X_test.Rainfall.max()


# In[ ]:


X_train.Evaporation.max(), X_test.Evaporation.max()


# In[ ]:


X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()


# In[ ]:


X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()


# In[ ]:


X_train[numerical].describe()


# We can now see that the outliers in `Rainfall`, `Evaporation`, `WindSpeed9am` and `WindSpeed3pm` columns are capped.

# ### Encode categorical variables

# In[ ]:


# print categorical variables
categorical


# In[ ]:


X_train[categorical].head()


# In[ ]:


#encode RainToday variable
import category_encoders as ce
encoder = ce.BinaryEncoder(cols=['RainToday'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)


# In[ ]:


X_train.head()


# In[ ]:


X_train = pd.concat([X_train[numerical], X_train[['RainToday_0','RainToday_1']],pd.get_dummies(X_train.Location), pd.get_dummies(X_train.WindGustDir), pd.get_dummies(X_train.WindDir9am), pd.get_dummies(X_train.WindDir3pm)],axis=1)


# In[ ]:


X_train.head()


# In[ ]:


X_test = pd.concat([X_test[numerical], X_test[['RainToday_0','RainToday_1']],pd.get_dummies(X_test.Location), pd.get_dummies(X_test.WindGustDir), pd.get_dummies(X_test.WindDir9am), pd.get_dummies(X_test.WindDir3pm)],axis=1)


# In[ ]:


X_test.head()


# We now have training and testing set ready for model building. Before that, we should map all the feature variables onto the same sacle. It is call `Feature Scaling`. I will do it as follows:

# ## 11. Feature Scaling

# In[ ]:


X_train.describe()


# In[ ]:


cols = X_train.columns


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


X_train = pd.DataFrame(X_train, columns=[cols])


# In[ ]:


X_test = pd.DataFrame(X_test, columns=[cols])


# In[ ]:


X_train.describe()


# We now have `X_train` dataset ready to be fed into the Logistic Regression classifier. I will do it as follows:

# ## 12. Model Training

# In[ ]:




