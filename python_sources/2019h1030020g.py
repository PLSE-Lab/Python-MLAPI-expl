#!/usr/bin/env python
# coding: utf-8

# # Data Analysis
# Adapted from IBM Cognitive Class Series

# ### Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', 100)


# ### Reading the data set from the URL and adding the related headers.

# In[ ]:


filename = "/kaggle/input/eval-lab-1-f464-v2/train.csv"


#  Python list <b>headers</b> containing name of headers 

# In[ ]:


# headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
#          "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
#          "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
#          "peak-rpm","city-mpg","highway-mpg","price"]


# Use the Pandas method <b>read_csv()</b> to load the data from the web address. Set the parameter  "names" equal to the Python list "headers".

# In[ ]:


df = pd.read_csv(filename) #, names = headers)


#  Use the method <b>head()</b> to display the first 10 rows of the dataframe. 

# In[ ]:


df.head(n=10)


# Use <b>info()</b> method to see basic information about the dataset 

# In[ ]:


df.info()


# ## Data Wrangling

# Data Wrangling is the process of converting data from the initial format to a format that may be better for analysis.

# As we can see, several question marks appeared in the dataframe; those are missing values which may hinder our further analysis. 
# So, how do we identify all those missing values and deal with them?
# 
# 
# **Steps for working with missing data:**
# <ol>
#     <li>Identify missing data</li>
#     <li>Deal with missing data</li>
#     <li>Correct data format</li>
# </ol>

# ### Identifying missing values
# #### Convert "?" to NaN
# In the car dataset, missing data comes with the question mark "?".
# We replace "?" with NaN (Not a Number), which is Python's default missing value marker, for reasons of computational speed and convenience. Here we use the function: 
#  <pre>.replace(A, B, inplace = True) </pre>
# to replace A by B

# In[ ]:


missing_values = df.isnull().sum()
missing_values[missing_values>0]
# replace "?" with NaN
df.replace("?", np.nan, inplace = True)
#df.head(5)


# #### Count missing values in each column
# The missing values are converted to Python's default. We use Python's built-in functions to identify these missing values. There are two methods to detect missing data:
# <ol>
#     <li><b>.isnull()</b></li>
#     <li><b>.notnull()</b></li>
# </ol>
# The output is a boolean value indicating whether the value that is passed into the argument is in fact missing data.

# In[ ]:


df.isnull().any().any()


# "True" stands for missing value, while "False" stands for not missing value.

# In[ ]:


missing_count = df.isnull().sum()
missing_count[missing_count > 0]


# Based on the summary above, each column has 205 rows of data, seven columns containing missing data:
# <ol>
#     <li>"normalized-losses": 41 missing data</li>
#     <li>"num-of-doors": 2 missing data</li>
#     <li>"bore": 4 missing data</li>
#     <li>"stroke" : 4 missing data</li>
#     <li>"horsepower": 2 missing data</li>
#     <li>"peak-rpm": 2 missing data</li>
#     <li>"price": 4 missing data</li>
# </ol>

# ### Dealing with missing data
# **How to deal with missing data?**
# 
# <ol>
#     <li>drop data<br>
#         a. drop the whole row<br>
#         b. drop the whole column
#     </li>
#     <li>replace data<br>
#         a. replace it by mean<br>
#         b. replace it by frequency<br>
#         c. replace it based on other functions
#     </li>
# </ol>

# Whole columns should be dropped only if most entries in the column are empty. In our dataset, none of the columns are empty enough to drop entirely.
# We have some freedom in choosing which method to replace data; however, some methods may seem more reasonable than others. We will apply each method to many different columns:
# 
# <b>Replace by mean:</b>
# <ul>
#     <li>"normalized-losses": 41 missing data, replace them with mean</li>
#     <li>"stroke": 4 missing data, replace them with mean</li>
#     <li>"bore": 4 missing data, replace them with mean</li>
#     <li>"horsepower": 2 missing data, replace them with mean</li>
#     <li>"peak-rpm": 2 missing data, replace them with mean</li>
# </ul>
# 
# <b>Replace by frequency:</b>
# <ul>
#     <li>"num-of-doors": 2 missing data, replace them with "four". 
#         <ul>
#             <li>Reason: 84% sedans is four doors. Since four doors is most frequent, it is most likely to occur</li>
#         </ul>
#     </li>
# </ul>
# 
# <b>Drop the whole row:</b>
# <ul>
#     <li>"price": 4 missing data, simply delete the whole row
#         <ul>
#             <li>Reason: price is what we want to predict. Any data entry without price data cannot be used for prediction; therefore any row now without price data is not useful to us</li>
#         </ul>
#     </li>
# </ul>

# In[ ]:


# Calculate mean for column normalized-losses
# df["normalized-losses"].mean()


# #### Correct data format
# 
# In Pandas, we use 
# <p><b>.dtype()</b> to check the data type</p>
# <p><b>.astype()</b> to change the data type</p>

# **Lets list the data types and number of unique values for each column**

# In[ ]:


df_dtype_nunique = pd.concat([df.dtypes, df.nunique()],axis=1)
df_dtype_nunique.columns = ["dtype","unique"]
df_dtype_nunique


# <p>As we can see above, some columns are not of the correct data type. Numerical variables should have type 'float' or 'int', and variables with strings such as categories should have type 'object'. For example, 'bore' and 'stroke' variables are numerical values that describe the engines, so we should expect them to be of the type 'float' or 'int'; however, they are shown as type 'object'. We have to convert data types into a proper format for each column using the "astype()" method.</p> 

# In[ ]:


df.head()


# #### Convert data types to proper format

# In[ ]:


#numerical_features = [""]
#df[numerical_features] = df[numerical_features].astype("float")


# #### Let us list the columns after the conversion

# In[ ]:


df.dtypes


# #### Dropping rows with "NaN"

# Let's drop all rows that do not have price data:

# In[ ]:


# simply drop whole row with NaN in "price" column
# df.dropna(subset=["price"], 0axis=0, inplace=True)

# reset index, because we droped two rows
# df.reset_index(drop=True, inplace=True)


# <h4>Calculate the average of the "normalized-losses" column </h4>

# In[ ]:


#avg = df["normalized-losses"].mean()
#print("Average of normalized-losses:", avg_norm_loss)


# #### Replace "NaN" by mean value in "normalized-losses" column

# In[ ]:


# df["normalized-losses"].fillna(value=avg_norm_loss, inplace=True)
# OR
# df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)


# #### Replacing "NaN" by mean value for all numeric features in one go

# In[ ]:


df.fillna(value=df.mean(),inplace=True)
df.isnull().any().any()


# #### Replacing "NaN" with mode (most frequent) for categorical features

# To see which values are present in a particular column, we can use the ".value_counts()" method:

# In[ ]:


# df['num-of-doors'].value_counts()


# We can see that four doors are the most common type. We can also use the ".idxmax()" method to calculate for us the most common type automatically:

# In[ ]:


# df['type'].value_counts()


# The replacement procedure is very similar to what we have seen previously

# In[ ]:


#replace the missing 'num-of-doors' values by the most frequent 
# df["num-of-doors"].replace(np.nan, "four", inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.isnull().any().any()


# <b>Good!</b> Now, we obtain the dataset with no missing values.

# In[ ]:





# ## Data Analysis

# ### Descriptive Statistical Analysis

# <p>Let's first take a look at the variables by utilizing a description method.</p>
# 
# <p>The <b>describe</b> function automatically computes basic statistics for all continuous variables. Any NaN values are automatically skipped in these statistics.</p>
# 
# This will show:
# <ul>
#     <li>the count of that variable</li>
#     <li>the mean</li>
#     <li>the standard deviation (std)</li> 
#     <li>the minimum value</li>
#     <li>the IQR (Interquartile Range: 25%, 50% and 75%)</li>
#     <li>the maximum value</li>
# <ul>

#  We can apply the method "describe" as follows:

# In[ ]:


df.describe()


#  The default setting of "describe" skips variables of type object. We can apply the method "describe" on the variables of type 'object' as follows:

# In[ ]:


df.describe(include='object')


# ### Grouping

# <p>The "groupby" method groups data by different categories. The data is grouped based on one or several variables and analysis is performed on the individual groups.</p>
# 
# <p>For example, let's group by the variable "drive-wheels". We see that there are 3 different categories of drive wheels.</p>

# In[ ]:


# df['drive-wheels'].unique()


# <p>If we want to know, on average, which type of drive wheel is most valuable, we can group "drive-wheels" and then average them.</p>
# 
# <p>We can select the columns 'drive-wheels', 'body-style' and 'price', then assign it to the variable "df_group_one".</p>

# In[ ]:


# df_group = df[['drive-wheels','body-style','price']]


# We can then calculate the average price for each of the different categories of data.

# In[ ]:


# Use groupby to calculate average price for each category of drive-wheels
# grouped_test1 = df_group.groupby(['drive-wheels'],as_index=False).mean()
# grouped_test1


# <p>From our data, it seems rear-wheel drive vehicles are, on average, the most expensive, while 4-wheel and front-wheel are approximately the same in price.</p>
# 
# <p>You can also group with multiple variables. For example, let's group by both 'drive-wheels' and 'body-style'. This groups the dataframe by the unique combinations 'drive-wheels' and 'body-style'. We can store the results in the variable 'grouped_test1'.</p>

# ### Data Visualization
# <p>When visualizing individual variables, it is important to first understand what type of variable you are dealing with. This will help us find the right visualization method for that variable.</p>
# 

# In[ ]:


# List the data types for each column
print(df.dtypes)


# #### Continuous numerical variables: 
# 
# <p>Continuous numerical variables are variables that may contain any value within some range. Continuous numerical variables can have the type "int64" or "float64". A great way to visualize these variables is by using scatterplots with fitted lines.</p>
# 
# <p>In order to start understanding the (linear) relationship between an individual variable and the price. We can do this by using "regplot", which plots the scatterplot plus the fitted regression line for the data.</p>

#  Let's see several examples of different linear relationships:

# **Positive linear relationship**

# Let's find the scatterplot of "engine-size" and "price" 

# In[ ]:


# Engine size as potential predictor variable of price
sns.boxplot(x="type", y="rating", data=df)


# <p>As the engine-size goes up, the price goes up: this indicates a positive direct correlation between these two variables. Engine size seems like a pretty good predictor of price since the regression line is almost a perfect diagonal line.</p>

#  We can examine the correlation between 'engine-size' and 'price' and see it's approximately  0.86

# In[ ]:


#df[["engine-size", "price"]].corr()


# Highway mpg is a potential predictor variable of price 

# In[ ]:


#sns.regplot(x="highway-mpg", y="price", data=df)


# <p>As the highway-mpg goes up, the price goes down: this indicates an inverse/negative relationship between these two variables. Highway mpg could potentially be a predictor of price.</p>

# We can examine the correlation between 'highway-mpg' and 'price' and see it's approximately  -0.7

# In[ ]:


#df[['highway-mpg', 'price']].corr()


# **Weak Linear Relationship**

# Let's see if "Peak-rpm" as a predictor variable of "price".

# In[ ]:


#sns.regplot(x="", y="rating", data=df)
# import matplotlib.pyplot as plt
# plt.plot(df2['rating'])
# plt.ylabel('some numbers')
# plt.show()


# <p>Peak rpm does not seem like a good predictor of the price at all since the regression line is close to horizontal. Also, the data points are very scattered and far from the fitted line, showing lots of variability. Therefore it's it is not a reliable variable.</p>

# We can examine the correlation between 'peak-rpm' and 'price' and see it's approximately -0.1

# In[ ]:


# df[['peak-rpm','price']].corr()


# In[ ]:


# Find the correlation between x="stroke", y="price"
# df[["stroke","price"]].corr()


# In[ ]:


# Given the correlation results between "price" and "stroke" do you expect a linear relationship?
# Verify your results using the function "regplot()".
sns.regplot(x="id", y="feature4", data=df)


# #### Categorical variables
# 
# <p>These are variables that describe a 'characteristic' of a data unit, and are selected from a small group of categories. The categorical variables can have the type "object" or "int64". A good way to visualize categorical variables is by using boxplots.</p>

# Let's look at the relationship between "body-style" and "price".

# In[ ]:


# sns.boxplot(x="type", y="rating", data=df)


# <p>We see that the distributions of price between the different body-style categories have a significant overlap, and so body-style would not be a good predictor of price. Let's examine engine "engine-location" and "price":</p>

# In[ ]:


# sns.boxplot(x="", y="price", data=df)


# <p>Here we see that the distribution of price between these two engine-location categories, front and rear, are distinct enough to take engine-location as a potential good predictor of price.</p>

#  Let's examine "drive-wheels" and "price".

# In[ ]:


# drive-wheels
# sns.boxplot(x="drive-wheels", y="price", data=df)


# <p>Here we see that the distribution of price between the different drive-wheels categories differs; as such drive-wheels could potentially be a predictor of price.</p>

# ### Correlation

# <p><b>Correlation</b>: a measure of the extent of interdependence between variables.</p>
# 
# <p><b>Causation</b>: the relationship between cause and effect between two variables.</p>
# 
# <p>It is important to know the difference between these two and that correlation does not imply causation. Determining correlation is much simpler  the determining causation as causation may require independent experimentation.</p>

# We can calculate the correlation between variables  of type "int64" or "float64" using the method "corr":

# In[ ]:


df.corr()


# The diagonal elements are always one

# In[ ]:


# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()


# ### Conclusion: Important Variables

# <p>We now have a better idea of what our data looks like and which variables are important to take into account when predicting the car price. We have narrowed it down to the following variables:</p>
# 
# Continuous numerical variables:
# <ul>
#     <li>Length</li>
#     <li>Width</li>
#     <li>Curb-weight</li>
#     <li>Engine-size</li>
#     <li>Horsepower</li>
#     <li>City-mpg</li>
#     <li>Highway-mpg</li>
#     <li>Wheel-base</li>
#     <li>Bore</li>
# </ul>
#     
# Categorical variables:
# <ul>
#     <li>Drive-wheels</li>
#     <li>Engine-location</li>
# </ul>
# 
# <p>As we now move into building machine learning models to automate our analysis, feeding the model with variables that meaningfully affect our target variable will improve our model's prediction performance.</p>

# ## A Few More Steps

# ### Feature Selection

# In[ ]:


X = df[["id","feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","feature10","feature11"]].copy()
y = df[['rating']].copy()


# In[ ]:


# y.head()
X


# In[ ]:


# numerical_features = ["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","feature10","feature11"]
# categorical_features = ["type"]


# ### Feature Scaling

# In[ ]:


# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X[numerical_features])

# X_scaled


# ### One-hot encoding of categorical attributes

# In[ ]:


# X_encoded = pd.get_dummies(X[categorical_features])
# X_encoded.head()


# In[ ]:


# X_new = np.concatenate([X_scaled,X_encoded.values],axis=1)
# # X_new


# ### Create Training and Validation Data Split

# In[ ]:


# from sklearn.model_selection import train_test_split

# # X_train.head()
# a_train = X.iloc[:,0:13]
# a_train.head()

# b_train = y.iloc[:,0:1]
# b_train.head()


# ### Model Training

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.20,random_state=42)

from sklearn.linear_model import LinearRegression

reg_lr = LinearRegression().fit(X_train,y_train)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X_train)

# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# # from sklearn.neighbors import KNeighborsClassifier

# # classifier = KNeighborsClassifier(n_neighbors = 5)#,leaf_size=1,algorithm="auto",n_jobs=-1,p =30, weights="distance",metric="euclidean")
# # classifier.fit(X_train, y_train)

# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors=50)
# classifier.fit(X_train, np.ravel(y_train,order='C'))


# 
# ### Model Evaluation

# In[ ]:




y_pred = reg_lr.predict(X_val)
y_pred

# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))


# In[ ]:


#root mean square error
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_pred,y_val))
print(rmse)


# In[ ]:


df2 = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")
# df2.head()
# freq = df2['feature11'].isnull().sum()
# freq
# df2.isnull().any().any()
df2.fillna(value=df2.mean(),inplace=True)
df2.isnull().any().any()


# In[ ]:


X_t = df2[["id","feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","feature10","feature11"]].copy()
y_t_pred = reg_lr.predict(X_t)
y_t_pred


# In[ ]:


df2.info()


# In[ ]:


df2['rating'] = y_t_pred


# In[ ]:


df2.info()


# In[ ]:


final_pred = df2[['id','rating']].copy()
final_pred['rating'] = final_pred['rating'].round(decimals=0)
final_pred['rating'] = final_pred['rating'].astype("int")
final_pred


# In[ ]:


final_pred.to_csv("eval_1_v2_pred2.csv",index=False,encoding='utf-8')


# In[ ]:





# In[ ]:




