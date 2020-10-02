#!/usr/bin/env python
# coding: utf-8

# <center><h1> NYC Property Sales</h1></center>
# This is a data set about property sales in New York City. It's a nice dataset to train our regression skills. In this analysis we will explore some important topics like
# <ul>
#     <li> Linear Regression </li>
#     <li> Lasso, Ridge and elasticNet regressions</li>
#     <li> Least Angle Regression (LARS) </li>
#     <li> Decision Tree Regression </li>
#     <li> AdaBoost Regressor </li>
#     <li> Gradient Boosting </li>
#     <li> Cros Validation </li>
#     <li> Principle Component Analysis (PCA)</li>
#     <li> Data encoding (integer encoding and one-hot encoding)</li>
#     <li> Outlier detection with Isolation forests</li>
# </ul>
# and the usual exploratory data analysis and data visualization.
# <hr>
# <hr>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# <h2> Libraries and data importation </h2>

# In[ ]:


df = pd.read_csv('/kaggle/input/nyc-property-sales/nyc-rolling-sales.csv', index_col = 0)


# <h1> Cleaning and regularizing the data set</h1>
#     <p> Firstly we will see the overall aspects of the data. Identify (and solve) possible issues. The target feature in this case is quite easy to indentify (SALE VALUE), but in many real case scenarios this might pose a challenge.</p>

# In[ ]:


df.info()


# Its always nice to check for NaNs immidiatelly. In this case, as we will see, they are not propper empty observations but '-' or whitespaces, making cleaning a bit harder

# In[ ]:


df.isna().sum()


# <h3>Categorical features </h3>
# Dealing with categorical data is usualy the hardest part of treating any data set, so lets start by it.

# In[ ]:


categorical = df.select_dtypes(include=['object'])
categorical.head().transpose() # Transposing make visualization easier for big datasets


# Some things stand out already. 'EASE-MENT' and 'APARTMENT NUMBER' seem to have empty entries. 'SALE PRICE', our target feature, also has problems as some observations have a '-'. 
# 
# A quick summary of the statistics of the dataset can be obtained as follows

# In[ ]:


categorical.describe().transpose()


# Some of these features should be numeric, so lets convert then. Some have '-' and blackspaces that we need to fix

# In[ ]:


df['SALE PRICE'] = df['SALE PRICE'].apply(lambda s: int(s) if not "-" in s else 0)
df['LAND SQUARE FEET'] = df['LAND SQUARE FEET'].apply(lambda s: int(s) if not '-' in s else 0)
df['GROSS SQUARE FEET'] = df['GROSS SQUARE FEET'].apply(lambda s: int(s) if not '-' in s else 0)
df['SALE PRICE'] = df['SALE PRICE'].apply(lambda s: s if type(s) == int else 0)


# The SALE DATE column should be converted to a propper datetime format.

# In[ ]:


df['SALE DATE'] = pd.to_datetime(df['SALE DATE'], format= '%Y-%m-%d %H:%M:%S')


# The EASE-MENT column has no data in it so we will just drop it from the dataframe.
# 
# The APPARTMENT NUMBER column is 50% empty. One could use it to split the data in two groups, since the appartment number might hold a very important information: the building floor. For now we will drop it to avoid over complicating things.
# 
# The ADDRESS columns could be used for geographical data analysis, but doing such analysis is not in the scope of this project. It also seems retundant, as the ZIP code should hold the same information.

# In[ ]:


df.drop(["EASE-MENT", "ADDRESS", "APARTMENT NUMBER"], axis = 1, inplace = True)


# <h3>Numerical data</h3>

# In[ ]:


numerical = df.select_dtypes(exclude=['object'])
numerical.info()


# In[ ]:


numerical.describe().transpose()


# Some of this data just don't make any sense. 0 values are unaceptable values as ZIP CODE, TOTAL UNITS, GROSS SQUARE FEET and YEAR BUILT.
# 
# Our target feature SALE PRICE also have some 0 values. We will just eliminate these observations, but there are some other options. It's possible to use these as the test set (might not be a good idea as this subset can be different from the rest i.e. comparing sold to non sold properties). One could also try to make a classification analisys to decide if a new property will be sold or not.
# 
# Depending on the ammount of missing data we might want to delete the observations from the dataframe (as deleting few observation won't affect the statistics) or the entire feature (if most of the data is missing and deleting the observations it would cause severe information loss). Be aware that a feature can be VERY important, and sometimes its worth loosing half your data to keep it. I think that this is the case for the GROSS SQUARE feature, as this is a very important factor on the price of a property. This will drastically reduce the amount of availabe observations.
# 
# Yet another possibility is to use data imputation techniques (e.g. filling missing observations with the overall average), but I personaly don't like it as it feels wrong filling the target feature "by hand".

# In[ ]:


weird_zeros_cols = [
    "ZIP CODE",
    "GROSS SQUARE FEET",
    "YEAR BUILT",
    "SALE PRICE"
   ]

l = len(df)
for col in weird_zeros_cols:
    print(f"{col:.10}\t{len(df[df[col] == 0])/l:0.2f}% missing")


# In[ ]:


for col in weird_zeros_cols:
    df = df[df[col] != 0]


# <h1>Exploratory Data Analysis</h1>
# There are simply too many things to explore in each data set. I like to see wich categories hold most observations, plot the relationship of the features to the target and look for outliers.

# In[ ]:


categorical = df.select_dtypes(include=['object'])
categorical.describe().transpose()


# In[ ]:


sns.countplot(
    x="TAX CLASS AT PRESENT",
    data = df,
    order = df["TAX CLASS AT PRESENT"].value_counts().index,
)
plt.show()


# In[ ]:


pivot = df.pivot_table(index='TAX CLASS AT PRESENT',
                       values='SALE PRICE',
                       aggfunc=np.sum,).sort_values("SALE PRICE")

pivot.plot(
    kind='bar',
    color='orange',
    title="Total Sale Price per Tax Class"
)


# In[ ]:


pivot = df.pivot_table(index='TAX CLASS AT PRESENT',
                       values='SALE PRICE',
                       aggfunc=np.mean).sort_values("SALE PRICE")
pivot.plot(kind='bar',
           color='black',
           title="Average Price per Tax Class"
          )


# In[ ]:


g = sns.countplot(
    x='BUILDING CLASS CATEGORY',
    data = df,
    order = df["BUILDING CLASS CATEGORY"].value_counts().index,
)
g.set_yscale('log')
g.set_xticklabels(g.get_xticklabels(), rotation = 90)
plt.show()


# In[ ]:


pivot = df.pivot_table(index='BUILDING CLASS CATEGORY',
                       values='SALE PRICE',
                       aggfunc=np.sum).sort_values("SALE PRICE")
pivot.plot(kind='bar', color = 'green')


# In[ ]:


df['BUILDING CLASS CATEGORY'].value_counts().head(6)


# To reduce the complexity of our model, lets grab only the most representative BUIDING CLASS CATEGORY

# In[ ]:


top_vals = df['BUILDING CLASS CATEGORY'].value_counts().index[:5]
df = df[df["BUILDING CLASS CATEGORY"].isin(top_vals)]


# <hr>

# <h3>Numerical features</h3>

# In[ ]:


numerical = df.select_dtypes(exclude=['object', 'datetime'])
numerical.describe().transpose()


# In[ ]:


sns.heatmap(numerical.corr()) #, annot= True)


# We see a huge correlation between LAND SQUARE FEET and GROSS SQUARE FEET and between RESIDENTIAL UNITS and TOTAL UNITS. Correlated features in general don't improve models so we will keep only one of these features.

# In[ ]:


df.drop(["RESIDENTIAL UNITS", "LAND SQUARE FEET"], axis = 1, inplace = True)


# In[ ]:


plt.hist(
    df["SALE PRICE"],
    bins = 20,
    log = True,
    rwidth = 0.8
)
plt.show()


# It's clear that we have some outliers. Lets remove then.

# In[ ]:


df = df[(df['SALE PRICE'] < 5e8) & (df['SALE PRICE'] > 1e5)]


# If you want to make a pretier plot (for a executive report for example), it would be interesting to add colors to the histogram. It is good to remember that this is just "perfumery", as adding colors won't add any new information to the plot and might just confuse the reader in most casses (as people will try to undertand the meaning of the colors, wich they don't have)

# In[ ]:


# Plot histogram.
n, bins, patches = plt.hist(
    df["SALE PRICE"],
    bins = 20,
    log = True,
    rwidth = 0.8
) 
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)

cm = plt.cm.get_cmap('plasma')
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))


#  The colors in the last histogram scale with the x-axis, but we could scale the color by the y axis. This can be achived by odering the patches by desceding (or ascending) order of counts in each bin (given by the 'n' list) 

# In[ ]:


n, bins, patches = plt.hist(
    df["SALE PRICE"],
    bins = 20,
    log = True,
    rwidth = 0.8
) 

bin_centers = 0.5 * (bins[:-1] + bins[1:])

sorted_patches = [p for _,p in sorted(zip(n,patches), key=lambda pair: pair[0])] #sorts patches in respect to n
sorted_centers = [c for _,c in sorted(zip(n, bin_centers), key=lambda pair: pair[0])] #sorts bin_centers in respect to n

# scale values to interval [0,1]
col = sorted_centers - min(sorted_centers)
col /= max(col)
col = sorted(col)
cm = plt.cm.get_cmap('plasma')
for c, p in zip(col, sorted_patches):
    plt.setp(p, 'facecolor', cm(c))

plt.show()


# This features is very skewed, in numbers:

# In[ ]:


df["SALE PRICE"].skew()


# As we will try to fit a simetrical model to the data, it is interesting to remove this skewness. After we explore our numerical features we will remove the skewness from all of then

# In[ ]:


price = np.log(df["SALE PRICE"])
print(price.skew())


# In[ ]:


plt.hist(price, bins =20)
plt.show()


# In[ ]:


df.groupby("BOROUGH").mean()['SALE PRICE'].plot(kind='bar') # An alternative to pivot tables


# In[ ]:


df['TOTAL UNITS'].value_counts()


# In[ ]:


sns.countplot(x='TOTAL UNITS', data = df, log=True)
plt.show()


# Lets keep just the main values for TOTAL UNITS to simplify our model

# In[ ]:


df = df[df['TOTAL UNITS'] < 50]


# The Tax Class ad Borough could be used as categorical data, as a borough of 5 is not 'larger' that a borough of 1, they are just categories.

# In[ ]:


df['TAX CLASS AT TIME OF SALE'].value_counts()


# In[ ]:


df['BOROUGH'].value_counts()


# <h3>Temporal data</h3>
# The only temporal feature is SALE DATE. Lets check structure of the time series

# In[ ]:


plt.hist(df['SALE DATE'], bins=20)
plt.show()


# It is interesting to look at the weekly and monthly sales

# In[ ]:


sns.countplot(df["SALE DATE"].dt.dayofweek)


# In[ ]:


df['day'] = df["SALE DATE"].dt.dayofweek
df = df[df["day"] < 5 ]
df.drop(["day"], axis =1, inplace = True)


# In[ ]:


sns.countplot(df["SALE DATE"].dt.day)


# We can plot the monthly sales in the shape of an actual calendar

# In[ ]:


month = np.empty(5 * 7)
for day, count in df["SALE DATE"].dt.day.value_counts().iteritems():
    month[int(day) -1] = count
month = month.reshape((5,7))
sns.heatmap(month)


# I want to make a fancy data visualization out of this (just for the sake of doing it).
# 
# Seaborn has a nice FacedGrid method that allows for ridge plots, but I find it quite non intuitive, and prefer to use the joyplot library because I find it more intuitive and has a syntax similar to matplotlib. A nice challege would be to implement this with pure matplotlib functions.
# 

# In[ ]:


import joypy
month_df = pd.DataFrame(month)
joypy.joyplot(month_df, overlap=2, colormap=plt.cm.OrRd_r, linecolor='w', linewidth=.5)
plt.show()


# As a last visualization, the time series itself.

# In[ ]:


axes = df["SALE PRICE"].plot(
    marker='.',
    alpha=0.5,
    linestyle='',
    figsize=(11, 9),
    subplots=True
)


# <h1>Data Preparation</h1>
# Preperaing the data for regression

# In[ ]:


df.info()


# To reduce he dimensionality, we're going to delete some features that feel redundant.
# 
# BUIDING CLASS AT PRESENT and BUILDING CLASS AT TIME OF SALE feel redundant with CUILDING CLASS CATEGORY, so we will keep only the latter.
# 
# All "geological" information should be in the ZIP CODE, so we will drop all other related features.
# 
# We will chosse only one of TAX CLASS AT PRESENT and TAX CLASS AT TIME OF SALE

# In[ ]:


df.drop([
    "BUILDING CLASS AT PRESENT",
    "BUILDING CLASS AT TIME OF SALE",
    "NEIGHBORHOOD",
    'TAX CLASS AT PRESENT'
], axis = 1, inplace = True)


# In[ ]:


df['TAX CLASS AT TIME OF SALE'].value_counts()


# Transforming temporal data to float

# In[ ]:


df["SALE DATE"] = pd.to_numeric(df["SALE DATE"])


# Its always in good practice to remove skwness from the numerical data and scale it (its not mandatory, but since some of our models will be simetric, removing the skewness from all features should help quite a lot)

# In[ ]:


numeric_cols = df.select_dtypes(exclude=['object']).columns.tolist()
numeric_cols.remove("SALE DATE")

# Removing exceding skewness from features
for col in numeric_cols:
    if df[col].skew() > 1:
        df[col] = np.log1p(df[col])


# Now we encode all categorical features. This can be done in two ways (mainly): One-hot encoding and integer encoding. One-hot is best used when there are only a few unique values of a given feature, as it creates a new column per unique value (i.e. True or False for each value). If we have too many unique values (and a big data set) integer encoding becomes a possibility.

# In[ ]:


one_hot = ['BUILDING CLASS CATEGORY','TAX CLASS AT TIME OF SALE']
dummies = pd.get_dummies(df[one_hot])
dummies = pd.concat([dummies, pd.get_dummies(df["BOROUGH"])], axis=1) #BOROUGH are integers, so we need to do it seperately
dummies.info(verbose=True, memory_usage=True) #Its nice to check how much memory the dummies will require


# In[ ]:


df.drop(['BUILDING CLASS CATEGORY', 'TAX CLASS AT TIME OF SALE', 'BOROUGH'], axis = 1, inplace = True)
df = pd.concat([df, dummies], axis =1)
df.info()


# The final step is to separate the train and test data samples

# In[ ]:


from sklearn.model_selection import train_test_split

features = df.drop(["SALE PRICE"], axis = 1)
target = df["SALE PRICE"]

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.3)


# <h3>Outlier detection with Isolation Forests</h3>

# In[ ]:


# Outlier detection
from sklearn.ensemble import IsolationForest
clf = IsolationForest(max_samples = 1000)
clf.fit(x_train)

outliers = clf.predict(x_train)
np.unique(outliers, return_counts = True)


# <h1>Models</h1>
# There are just too many models one could use to make a regression. I tried a lot of then and I picked the best performing.

# <h4> Linear Regression </h4>

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr = LinearRegression(normalize = True)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(lr.score(x_test, y_test))
print(mean_squared_error(y_test, y_pred) ** 0.5)


# It is important to look at the residuals (or reduced residuals) to check if there are any patterns

# In[ ]:


sns.residplot(y_test, y_pred, color="orange", scatter_kws={"s": 3})


# <h4>LARS</h4>

# In[ ]:


from sklearn.linear_model import Lars
lars = Lars()
lars.fit(x_train, y_train)
y_pred = lars.predict(x_test)
print(lars.score(x_test, y_test))
print(mean_squared_error(y_test, y_pred))


# In[ ]:


sns.residplot(y_test, y_pred, color="g", scatter_kws={"s": 3})


# <h4>Decision Tree Regression</h4>

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_depth = 100)
dtr.fit(x_train, y_train)
y_pred = dtr.predict(x_test)
print(dtr.score(x_test,y_test))
print(mean_squared_error(y_test, y_pred))


# In[ ]:


sns.residplot(y_test, y_pred, color="g", scatter_kws={"s": 3})


# <h4>AdaBoost Regression</h4>
# This was the best performing method, and its also the slowest to train.

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
adar = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators = 600)
adar.fit(x_train, y_train)
y_pred = adar.predict(x_test)
adar.score(x_train, y_train)


# In[ ]:


sns.residplot(y_test, y_pred, color="purple", scatter_kws={"s": 3}) #The best result is plotted in royal purple


# In[ ]:


fig, ax = plt.subplots(1)
ax.scatter(y_test, y_pred, s=2)
ax.plot([min(y_test.to_list()), max(y_test.to_list())], [min(y_test.to_list()), max(y_test.to_list())], ls='--', c='black', lw=2)
plt.show()


# <h4> Gradient Boosting </h4>
# Another boosting method

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.05, max_depth = 1, loss='ls')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(clf.score(x_test, y_test))
print(mean_squared_error(y_test, y_pred))


# In[ ]:


sns.residplot(y_test, y_pred, color="blue", scatter_kws={"s": 3})


# <h4>Cross validation</h4>
# Cross validation is a very powerful and simple technique that can be used to improve the overall quality of a regression.

# In[ ]:


from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

lreg = LinearRegression(normalize = True)
y_pred = cross_val_predict(lreg, x_test, y_test, cv=50)
sns.residplot(y_test, y_pred, color="pink", scatter_kws={"s": 3})
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))


# <h3>PCA and Scaling</h3>
# As a closing note we will discuss briefly PCA as it is a very interesting approach to dimensionality reduction.

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(features)
feat_pca = pca.transform(features)


# In[ ]:


fig, ax = plt.subplots(1, figsize=(15,7))
ax.scatter(feat_pca[:,0], feat_pca[:,1], cmap='plasma', c= target, alpha = 0.5, s=2)
#ax.set_ylim(-1e1, 3e1)
plt.show()


# The color is used to test if prices are regular.

# In[ ]:


pca = PCA(n_components = 3)
pca.fit(features)
feat_pca = pca.transform(features)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(feat_pca[:,0], feat_pca[:,1],feat_pca[:,2],cmap='plasma', c = target, alpha = 0.5)
ax.view_init(30,30)


# This is a very interesting result. It sugests that there are internal structures that could be used to make a segmentation of the data set, obtaining better regressions

# <h1> Further analysis </h1>
# Usually the dataset must be segmented the quality of each regression. This could be done with WOE binning or using a very simple decision tree. With this segmentation we could explore some of the features that we left out such as APARTMENT NUMBER or make a separate analysis for luxury properties, commercial and residential,etc..
#  
# # Thanks for reading!! 

# In[ ]:




