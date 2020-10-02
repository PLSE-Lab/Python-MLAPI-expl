#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <h2 style="color:blue" align="left"> 1. Import necessary Libraries </h2>

# In[ ]:


# Read Data
import numpy as np                     # Linear Algebra (calculate the mean and standard deviation)
import pandas as pd                    # manipulate data, data processing, load csv file I/O (e.g. pd.read_csv)

# Visualization
import seaborn as sns                  # Visualization using seaborn
import matplotlib.pyplot as plt        # Visualization using matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# style
import plotly.io as pio
pio.templates.default = "plotly_dark"
plt.style.use("fivethirtyeight")       # Set Graphs Background style using matplotlib
sns.set_style("darkgrid")              # Set Graphs Background style using seaborn

# ML model building; Pre Processing & Evaluation
from sklearn.model_selection import train_test_split                     # split  data into training and testing sets
from sklearn.linear_model import LinearRegression, Lasso, Ridge          # Linear Regression, Lasso and Ridge
from sklearn.linear_model import LogisticRegression                      # Logistic Regression
from sklearn.tree import DecisionTreeRegressor                           # Decision tree Regression
from sklearn.ensemble import RandomForestClassifier                      # this will make a Random Forest Classifier
from sklearn import svm                                                  # this will make a SVM classificaiton
from sklearn.svm import SVC                                              # import SVC from SVM
import xgboost
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report      # this creates a confusion matrix
from sklearn.metrics import roc_curve,auc                                # ROC
from sklearn.preprocessing import StandardScaler                         # Standard Scalar
from sklearn.model_selection import GridSearchCV                         # this will do cross validation

import warnings                        # Ignore Warnings
warnings.filterwarnings("ignore")


# <h2 style="color:blue" align="left"> 2. Load data </h2>

# In[ ]:


# Import first 5 rows
cover = pd.read_csv("/kaggle/input/forest-cover-type-prediction/train.csv")
test = pd.read_csv("/kaggle/input/forest-cover-type-prediction/test.csv")


# In[ ]:


display(cover.head())
display(test.head())


# In[ ]:


# checking dimension (num of rows and columns) of dataset
print("Training data shape (Rows, Columns):",cover.shape)
print("Training data shape (Rows, Columns):",test.shape)


# In[ ]:


cover['Cover_Type'].value_counts()


# ### Checking for Numerical and Categorical features

# In[ ]:


# check dataframe structure like columns and its counts, datatypes & Null Values
cover.info()


# In[ ]:


cover.dtypes.value_counts()


# - Our dataset features consists of only integers

# In[ ]:


# Gives number of data points in each variable
cover.count()


# <h2 style="color:blue" align="left"> 3. EDA (Exploratory Data Analysis) </h2>
# 
# - EDA is a way of **Visualizing, Summarizing and interpreting** the information that is **hidden in rows and column** format.
# 
# ### Steps involved in EDA:
# 1. Find Unwanted Columns
# - Find Missing Values
# - Find Features with one value
# - Explore the Categorical Features
# - Find Categorical Feature Distribution
# - Relationship between Categorical Features and Label
# - Explore the Numerical Features
# - Find Discrete Numerical Features
# - Relation between Discrete numerical Features and Labels
# - Find Continous Numerical Features
# - Distribution of Continous Numerical Features
# - Relation between Continous numerical Features and Labels
# - Find Outliers in numerical features
# - Explore the Correlation between numerical features

# #### 1. Find Unwanted Columns
# 
# - There is no unwanted column present in given dataset to remove.
# 
#      EX: ID

# #### 2. Find Missing Values
# 
# - Checking missing values by below methods:
# 
#      1. df.isnull().sum()
#         - It returns null values for each column
#           
#      2. isnull().any()
#         - It returns True if column have NULL Values
#         - It returns False if column don't have NULL Values
#           
#      3. Heatmap()
#         - Missing value representation using heatmap.
#           
#      4. Percentage of Missing values

# In[ ]:


# Listing Number of missing values by feature column wise
cover.isnull().sum()


# In[ ]:


# any() check null values by columns
cover.isnull().any()


# In[ ]:


plt.figure(figsize=(17,10))
sns.heatmap(cover.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# - There is no missing values in dataset.

# #### 3. Find Features with one value

# In[ ]:


for column in cover.columns:
    print(column,cover[column].nunique())


# - Since there is no Categorical Features, we can skip steps from 4 to 6.

# #### 7. Explore the Numerical Features

# In[ ]:


numerical_features = cover.select_dtypes(exclude='object')
numerical_features


# #### 8. Find Discrete Numerical Features

# In[ ]:


discrete_feature=[feature for feature in numerical_features if len(cover[feature].unique())<25]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# #### 9. Find Continous Numerical Features

# In[ ]:


continuous_features=[feature for feature in numerical_features if feature not in discrete_feature+['Cover_Type']]
print("Continuous feature Count {}".format(len(continuous_features)))


# In[ ]:


continuous_features


# #### 10. Distribution of Continous Numerical Features

# In[ ]:


fig, ax = plt.subplots(3,4, figsize=(14,9))
sns.distplot(cover.Elevation, bins = 20, ax=ax[0,0]) 
sns.distplot(cover.Aspect, bins = 20, ax=ax[0,1]) 
sns.distplot(cover.Slope, bins = 20, ax=ax[0,2]) 
sns.distplot(cover.Horizontal_Distance_To_Hydrology, bins = 20, ax=ax[0,3])
sns.distplot(cover.Vertical_Distance_To_Hydrology, bins = 20, ax=ax[1,0]) 
sns.distplot(cover.Horizontal_Distance_To_Roadways, bins = 20, ax=ax[1,1]) 
sns.distplot(cover.Hillshade_9am, bins = 20, ax=ax[1,2]) 
sns.distplot(cover.Hillshade_Noon, bins = 20, ax=ax[1,3])
sns.distplot(cover.Hillshade_3pm, bins = 20, ax=ax[2,0])
sns.distplot(cover.Horizontal_Distance_To_Fire_Points, bins = 20, ax=ax[2,1])
plt.show()


# - it seems all continuous features are not normally distributed
# 
# - **Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Horizontal_Distance_To_Fire_Points** are **right skewed**
# 
# - **Elevation, Hillshade_9am, Hillshade_3pm** is **left skewed**.

# #### 11. Relation between Continous numerical Features and Labels

# In[ ]:


plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for feature in continuous_features:
    data=cover.copy()
    ax = plt.subplot(12,3,plotnumber)
    plt.scatter(cover[feature], cover['Cover_Type'])
    plt.xlabel(feature)
    plt.ylabel('Cover_Type')
    plt.title(feature)
    plotnumber+=1
plt.show()


# #### 12. Find Outliers in numerical features

# In[ ]:


# boxplot on numerical features to find outliers
plt.figure(figsize=(18,15), facecolor='white')
plotnumber =1
for numerical_feature in numerical_features:
    ax = plt.subplot(19,3,plotnumber)
    sns.boxplot(cover[numerical_feature])
    plt.xlabel(numerical_feature)
    plotnumber+=1
plt.show()


# - all features have outliers

# #### 13. Explore the Correlation between numerical features

# #### Correlation Heat Map

# In[ ]:


plt.figure(figsize = (14,12))
plt.title('Correlation of Numeric Features with Sale Price', y=1, size=16)
sns.heatmap(cover.corr(), square = True, vmax=0.8)


# #### Selected HeatMap

# In[ ]:


corr = cover.drop('Cover_Type', axis=1).corr()
plt.figure(figsize=(17, 14))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


# #### Correlation requires continuous data. Hence, ignore Wilderness_Area and Soil_Type as they are binary values

# In[ ]:


corrmat = cover.iloc[:,:10].corr()
f, ax = plt.subplots(figsize = (12,8))
sns.heatmap(corrmat, cmap='viridis', vmax=0.8, annot=True, square=True);


# #### 14. Descriptive statistics

# In[ ]:


# descriptive statistics (numerical columns)
pd.set_option('display.max_columns', None)
cover.describe()


# - Count is 581012 for each column, so no data point is missing.
# - Wilderness_Area and Soil_Type are one hot encoded. Hence, they could be converted back for some analysis.
# - Scales are not the same for all. Hence, rescaling and standardisation may be necessary for some algos.

# <h2 style="color:green" align="left"> 5. Data Visualization </h2>
# 
# - Used below **visualisation libraries**
# 
#      1. Matplotlib
#      2. Seaborn (statistical data visualization)
#      
# ### 1. Categorical
# 
# - Categorical data :
# 
#      1. Numerical Summaries
#      2. Histograms
#      3. Pie Charts
# 
# 
# ### 2. Univariate Analysis
# 
# - Univariate Analysis : data consists of **only one variable (only x value)**.
# 
#      1. Line Plots / Bar Charts
#      2. Histograms
#      3. Box Plots 
#      4. Count Plots
#      5. Descriptive Statistics techniques
#      6. Violin Plot

# #### Histogram

# In[ ]:


# Histogram for "Elevation"
plt.figure(figsize=(5,4))
sns.distplot(cover.Elevation,rug=True)


# In[ ]:


sns.distplot(cover.Aspect)
plt.grid()


# In[ ]:


# Histogram for "Elevation"
sns.boxplot(cover['Elevation'])


# In[ ]:


sns.boxplot(cover.Vertical_Distance_To_Hydrology)
plt.title('Vertical_Distance_To_Hydrology')


# In[ ]:


# Histogram for "All Features"
cover.hist(figsize=(16, 20), bins=50, xlabelsize=7, ylabelsize=7);


# ### 3. Bivariate Analysis
# 
# - **Bivariate Analysis** : data involves **two different variables**.
# 
#      1. Bar Charts
#      2. Scatter Plots
#      3. FacetGrid
#      
# 
# -  There are **three** types of bivariate analysis
# 
#      1. Numerical & Numerical
#      2. Categorical & Categorical
#      3. Numerical & Categorical

# In[ ]:


sns.violinplot(x=cover['Cover_Type'],y=cover['Elevation'])
plt.grid()


# In[ ]:


sns.violinplot(x=cover['Cover_Type'],y=cover['Aspect'])
plt.grid()


# In[ ]:


# vertical distance to the hydrology column
sns.violinplot(x=cover.Cover_Type, y=cover.Vertical_Distance_To_Hydrology)


# ### 1. Line Plot

# #### Aspect Vs Cover_Type

# In[ ]:


# Line Plot between "Aspect" and "Cover_Type"
plt.figure(figsize=(7,6))
sns.lineplot(x=cover['Aspect'], y=cover['Cover_Type'])

plt.xlabel('Aspect', fontsize=15, fontweight='bold')
plt.ylabel('Cover_Type', fontsize=15, fontweight='bold')

plt.title('Aspect Vs Cover_Type', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# #### Slope Vs Cover_Type

# In[ ]:


# Line Plot between "Slope" and "Cover_Type"
plt.figure(figsize=(7,6))
sns.lineplot(x=cover['Slope'], y=cover['Cover_Type'])

plt.xlabel('Slope', fontsize=15, fontweight='bold')
plt.ylabel('Cover_Type', fontsize=15, fontweight='bold')

plt.title('Slope Vs Cover_Type', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# ### 2. Scatter Plot

# #### Elevation Vs HD Roadways

# In[ ]:


fig = px.scatter(cover, x='Elevation', y= 'Horizontal_Distance_To_Roadways', color='Cover_Type', width=800, height=400)
fig.show()


# - We can see a positive correlation between Elevation and Distance to Roadways.

# #### Aspect  Vs Hillshade_3pm

# In[ ]:


# Scatter Plot between "GrLivArea" and "SalePrice"
plt.figure(figsize=(7,6))
sns.scatterplot(cover.Aspect, cover.Hillshade_3pm)

plt.xlabel('Aspect', fontsize=15, fontweight='bold')
plt.ylabel('Hillshade_3pm', fontsize=15, fontweight='bold')

plt.title('Aspect Vs Hillshade_3pm', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# #### Horizontal_Distance_To_Hydrology Vs Vertical_Distance_To_Hydrology

# In[ ]:


# Scatter Plot between "Horizontal_Distance_To_Hydrology" and "Vertical_Distance_To_Hydrology" variable
plt.figure(figsize=(7,6))
sns.scatterplot(cover['Horizontal_Distance_To_Hydrology'], cover['Vertical_Distance_To_Hydrology'])

plt.xlabel('Horizontal_Distance_To_Hydrology', fontsize=15, fontweight='bold')
plt.ylabel('Vertical_Distance_To_Hydrology', fontsize=15, fontweight='bold')

plt.title('Horizontal_Distance_To_Hydrology Vs Vertical_Distance_To_Hydrology', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# #### Hillshade_Noon Vs Hillshade_3pm

# In[ ]:


# Scatter Plot between "Hillshade_Noon" and "Hillshade_3pm"
fig = px.scatter(cover,x='Hillshade_Noon',y= 'Hillshade_3pm',color='Cover_Type',width=800,height=400)
fig.show()


# ####  Aspect Vs Hillshade_9am

# In[ ]:


# Scatter Plot between "Aspect" and "Hillshade_9am"
fig = px.scatter(cover,x='Aspect',y= 'Hillshade_9am',color='Cover_Type',width=800,height=400)
fig.show()


# #### Hillshade_9am vs Hillshade_3pm

# In[ ]:


# Scatter Plot between "Hillshade_9am" and "Hillshade_3pm"
fig = px.scatter(cover,x='Hillshade_9am',y= 'Hillshade_3pm',color='Cover_Type',width=800,height=400)
fig.show()


# #### Slope Vs Hillshade_Noon

# In[ ]:


# Scatter Plot between "Slope" and "Hillshade_Noon"
fig = px.scatter(cover,x='Slope',y= 'Hillshade_Noon',color='Cover_Type',width=800,height=400)
fig.show()


# ### 4. Count Plot

# In[ ]:


# Count Plot for "Cover_Type"
plt.figure(figsize = (15, 9))
sns.countplot(x = 'Cover_Type', data = cover)
xt = plt.xticks(rotation=45)


# ### 5. Violin Plot

# In[ ]:


# A violin plot is a hybrid of a box plot and a kernel density plot, which shows peaks in the data.
cols = cover.columns
size = len(cols) - 1 # We don't need the target attribute
# x-axis has target attributes to distinguish between classes
x = cols[size]
y = cols[0:size]

for i in range(0, size):
    sns.violinplot(data=cover, x=x, y=y[i])
    plt.show()


# - Aspect plot contains couple of normal distribution for several classes
# - Hillshade 9am and 12pm displays left skew (long tail towards left)
# - Wilderness_Area3 gives no class distinction.
# - Soil_Type, 1,5,8,9,12,14,18-22, 25-30 and 35-40 offer class distinction as values are not present for many classes

# ### 3. Multivariate Analysis
# 
# - 1. Pair Plot

# #### Pair Plot between 'SalePrice' and correlated variables

# In[ ]:


sns.set()
columns = cover.iloc[:,:10]
sns.pairplot(columns, kind ='scatter', diag_kind='kde')


# ### Drop Features have more missing values

# In[ ]:


# Checking the value count for different soil_types
for i in range(10, cover.shape[1]-1):
    j = cover.columns[i]
    print (cover[j].value_counts())


# <h2 style="color:blue" align="left"> 7. Check & Reduce Skewness </h2>

# - Skewness tells us about the symmetry in a distribution.
# 
# * If the **skewness** is **between -0.5 to +0.5** then we can say data is **fairly symmetrical**.
#   
# * If the **skewness** is **between -1 to -0.5 or 0.5 to 1** then data is **moderately skewed**.
#   
# * If the **skewness** is **less than -1 and greater than +1** then our data is **heavily skewed**.

# In[ ]:


cover.iloc[:,:10].skew()


# - "Aspect" & "Hillshade_3pm" are in between -0.5 and +0.5. Fairly skewed.
# - In our above data,
#     1. Slope
#     2. Horizontal_Distance_To_Hydrology
#     3. Vertical_Distance_To_Hydrology
# 
# - Are highly positively, right skewed.
# 
#     1. Elevation
#     2. Hillshade_9am
#     3. Hillshade_Noon
#     
# - Are highly negitively, left skewed.

# ### a. Checking Skewness for feature "Horizontal_Distance_To_Hydrology"

# In[ ]:


# Checking the skewness of "LotArea" attributes
sns.distplot(cover['Horizontal_Distance_To_Hydrology'])
Skew_Horizontal_Distance_To_Hydrology = cover['Horizontal_Distance_To_Hydrology'].skew()
plt.title("Skew:"+str(Skew_Horizontal_Distance_To_Hydrology))


# In[ ]:


# calculating the square for the column df['LotArea'] column
sns.distplot(np.sqrt(cover['Horizontal_Distance_To_Hydrology']))
Skew_Horizontal_Distance_To_Hydrology_sqrt = np.sqrt(cover['Horizontal_Distance_To_Hydrology']+1).skew()
plt.title("Skew:"+str(Skew_Horizontal_Distance_To_Hydrology_sqrt))


# - skewness is close to zero means normally distributed.

# ### b. Checking Skewness for feature "Vertical_Distance_To_Hydrology"

# In[ ]:


# Checking the skewness of "Vertical_Distance_To_Hydrology" attributes
sns.distplot(cover['Vertical_Distance_To_Hydrology'])
Skew_Vertical_Distance_To_Hydrology = cover['Vertical_Distance_To_Hydrology'].skew()
plt.title("Skew:"+str(Skew_Vertical_Distance_To_Hydrology))


# ### c. Checking Skewness for feature "Horizontal_Distance_To_Roadways"

# In[ ]:


# Checking the skewness of "Horizontal_Distance_To_Roadways" attributes
sns.distplot(cover['Horizontal_Distance_To_Roadways'])
Skew_Horizontal_Distance_To_Roadways = cover['Horizontal_Distance_To_Roadways'].skew()
plt.title("Skew:"+str(Skew_Horizontal_Distance_To_Roadways))


# In[ ]:


# calculating the square for the column df['Horizontal_Distance_To_Roadways'] column
sns.distplot(np.sqrt(cover['Horizontal_Distance_To_Roadways']))
Skew_Horizontal_Distance_To_Roadways_sqrt = np.sqrt(cover['Horizontal_Distance_To_Roadways']).skew()
plt.title("Skew:"+str(Skew_Horizontal_Distance_To_Roadways_sqrt))


# - The **skewness** is **between -0.5 to +0.5** then we can say data is **fairly symmetrical**.

# ### d. Checking Skewness for feature "Hillshade_9am"

# In[ ]:


# Checking the skewness of "Hillshade_9am" attributes
sns.distplot(cover['Hillshade_9am'])
Skew_Hillshade_9am = cover['Hillshade_9am'].skew()
plt.title("Skew:"+str(Skew_Hillshade_9am))


# In[ ]:


# calculating the square for the column df['Hillshade_9am'] column
sns.distplot(np.power(cover['Hillshade_9am'],5))
Skew_Hillshade_9am_power = np.power(cover['Hillshade_9am'],5).skew()
plt.title("Skew:"+str(Skew_Hillshade_9am_power))


# - The **skewness** is **between -0.5 to +0.5** then we can say data is **fairly symmetrical**.

# ### e. Checking Skewness for feature "Hillshade_Noon"

# In[ ]:


# Checking the skewness of "Hillshade_Noon" attributes
sns.distplot(cover['Hillshade_Noon'])
Skew_Hillshade_Noon = cover['Hillshade_Noon'].skew()
plt.title("Skew:"+str(Skew_Hillshade_Noon))


# In[ ]:


# calculating the square for the column df['Hillshade_9am'] column
sns.distplot(np.power(cover['Hillshade_Noon'],5))
Skew_Hillshade_Noon_power = np.power(cover['Hillshade_Noon'],5).skew()
plt.title("Skew:"+str(Skew_Hillshade_Noon_power))


# - The **skewness** is **between -0.5 to +0.5** then we can say data is **fairly symmetrical**.

# ### f. Checking Skewness for feature "Horizontal_Distance_To_Fire_Points"

# In[ ]:


# Checking the skewness of "Horizontal_Distance_To_Fire_Points" attributes
sns.distplot(cover['Horizontal_Distance_To_Fire_Points'])
Skew_Horizontal_Distance_To_Fire_Points = cover['Horizontal_Distance_To_Fire_Points'].skew()
plt.title("Skew:"+str(Skew_Horizontal_Distance_To_Fire_Points))


# In[ ]:


# calculating the square for the column df['Horizontal_Distance_To_Fire_Points'] column
sns.distplot(np.cbrt(cover['Horizontal_Distance_To_Fire_Points']))
Skew_Horizontal_Distance_To_Fire_Points_cube = np.cbrt(cover['Horizontal_Distance_To_Fire_Points']).skew()
plt.title("Skew:"+str(Skew_Horizontal_Distance_To_Fire_Points_cube))


# - The **skewness** is **between -0.5 to +0.5** then we can say data is **fairly symmetrical**.

# ### g. Checking Skewness for feature "Slope"

# In[ ]:


# Checking the skewness of "Slope" attributes
sns.distplot(cover['Slope'])
Skew_Slope = cover['Slope'].skew()
plt.title("Skew:"+str(Skew_Slope))


# In[ ]:


# calculating the square for the column df['Slope'] column
sns.distplot(np.sqrt(cover['Slope']))
Skew_Slope_sqrt = np.sqrt(cover['Slope']).skew()
plt.title("Skew:"+str(Skew_Slope_sqrt))


# - The **skewness** is **between -0.5 to +0.5** then we can say data is **fairly symmetrical**.

# In[ ]:


cover['dist_hydr'] = np.sqrt(cover['Vertical_Distance_To_Hydrology']**2 + cover['Horizontal_Distance_To_Hydrology']**2)
test['dist_hydr'] = np.sqrt(cover['Vertical_Distance_To_Hydrology']**2 + cover['Horizontal_Distance_To_Hydrology']**2)


# In[ ]:


sns.distplot(cover['dist_hydr'], color='green')


# <h2 style="color:blue" align="left"> 7. Model building and Evaluation </h2>

# In[ ]:


cover.head()


# In[ ]:


test.head()


# In[ ]:


# standardizing the columns except "soil type and wilderness_area" since they are binary  

cover_new = cover.iloc[:,:11]
cover_new['dist_hydr'] = cover['dist_hydr']
cover_new.info()


# ### Scaling
# - Standardizing the data i.e. to rescale the features to have a **mean of zero** and **standard deviation of 1.**

# In[ ]:


sc = StandardScaler()
sc.fit(cover_new)
cover_new = sc.transform(cover_new)


# In[ ]:


cover_new[:10,1]


# In[ ]:


cover.iloc[:,1:11] = cover_new[:,0:10]


# In[ ]:


cover['dist_hydr'] = cover_new[:,10]


# In[ ]:


# Correlation of "independant features" with "target" feature
# Drop least correlated features; since we have hign dimmensional data 
cover_corr = cover.corr()
cover_corr['Cover_Type'].abs().sort_values(ascending=False)


# In[ ]:


# Independant variable
X = cover.drop(columns='Cover_Type',axis=1)
# Dependant variable
y = cover['Cover_Type']


# In[ ]:


# split  data into training and testing sets of 70:30 ratio
# 20% of test size selected
# random_state is random seed
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=4)


# In[ ]:


# shape of X & Y test / train
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[ ]:


clf_accuracy=[]


# ### 1. Logistic Regression

# In[ ]:


LogReg = LogisticRegression(max_iter=1000)
LogReg.fit(X_train, y_train)


# In[ ]:


y_pred_LogReg = LogReg.predict(X_test)
clf_accuracy.append(accuracy_score(y_test, y_pred_LogReg))
print(accuracy_score(y_test, y_pred_LogReg))


# In[ ]:


print("Train Score {:.2f} & Test Score {:.2f}".format(LogReg.score(X_train, y_train), LogReg.score(X_test, y_test)))


# In[ ]:


print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2")
print("""Logistic Regression \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_LogReg)),
            mean_squared_error(y_test, y_pred_LogReg),
            mean_absolute_error(y_test, y_pred_LogReg),
            r2_score(y_test, y_pred_LogReg)))


# ### 2. Decision Tree

# In[ ]:


DTR = DecisionTreeRegressor()
DTR.fit(X_train, y_train)


# In[ ]:


y_pred_DTR = DTR.predict(X_test)


# In[ ]:


clf_accuracy.append(accuracy_score(y_test, y_pred_DTR))
print(accuracy_score(y_test, y_pred_DTR))


# In[ ]:


print("Train Score {:.2f} & Test Score {:.2f}".format(DTR.score(X_train, y_train), DTR.score(X_test, y_test)))


# In[ ]:


print("Model\t\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2")
print("""Decision Tree Regressor \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_DTR)),
            mean_squared_error(y_test, y_pred_DTR),
            mean_absolute_error(y_test, y_pred_DTR),
            r2_score(y_test, y_pred_DTR)))

plt.scatter(y_test, y_pred_DTR)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

plt.xlabel("Predicted")
plt.ylabel("True")

plt.title("Decision Tree Regressor")

plt.show()


# ### 3. Random Forest

# In[ ]:


rf = RandomForestClassifier()
rf.fit(X_train,y_train)


# In[ ]:


pred_rf = rf.predict(X_test)


# In[ ]:


clf_accuracy.append(accuracy_score(y_test, pred_rf ))
print(accuracy_score(y_test, pred_rf ))


# In[ ]:


print("Train Score {:.2f} & Test Score {:.2f}".format(rf.score(X_train, y_train), rf.score(X_test, y_test)))


# In[ ]:


print("Model\t\t RMSE \t\t MSE \t\t MAE \t\t R2")
print("""Random Forest \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, pred_rf )),
            mean_squared_error(y_test, pred_rf ),
            mean_absolute_error(y_test, pred_rf ),
            r2_score(y_test, pred_rf )))


# ### 4. KNN (K Nearest Neighbors)

# In[ ]:


KNN = KNeighborsClassifier()

l=[i for i in range(1,11)]
accuracy=[]

for i in l:
    KNN = KNeighborsClassifier(n_neighbors=i, weights='distance')
    KNN.fit(X_train, y_train)
    pred_knn = KNN.predict(X_test)
    accuracy.append(accuracy_score(y_test, pred_knn))

plt.plot(l,accuracy)
plt.title('knn_accuracy plot')
plt.xlabel('neighbors')
plt.ylabel('accuracy')
plt.grid()

print(max(accuracy))

clf_accuracy.append(max(accuracy))


# In[ ]:


print("Model\t\t RMSE \t\t MSE \t\t MAE \t\t R2")
print("""Random Forest \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, pred_rf )),
            mean_squared_error(y_test, pred_rf ),
            mean_absolute_error(y_test, pred_rf ),
            r2_score(y_test, pred_rf )))


# ### 5. XGBoost

# In[ ]:


import xgboost
reg_xgb = xgboost.XGBClassifier(max_depth=7)
reg_xgb.fit(X_train,y_train)


# In[ ]:


# predicting X_test
y_pred_xgb = reg_xgb.predict(X_test)


# In[ ]:


print("Train Score {:.2f} & Test Score {:.2f}".format(reg_xgb.score(X_train,y_train),reg_xgb.score(X_test,y_test)))


# In[ ]:


clf_accuracy.append(accuracy_score(y_test, y_pred_xgb))
print(accuracy_score(y_test, y_pred_xgb))


# In[ ]:


print("Model\t\t RMSE \t\t MSE \t\t MAE \t\t R2")
print("""XGBClassifier \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, pred_rf )),
            mean_squared_error(y_test, pred_rf ),
            mean_absolute_error(y_test, pred_rf ),
            r2_score(y_test, pred_rf )))


# ### 6. Naive Bayes Classifier

# In[ ]:


nb = GaussianNB()
nb.fit(X_train,y_train)


# In[ ]:


pred_nb = nb.predict(X_test)


# In[ ]:


clf_accuracy.append(accuracy_score(y_test, pred_nb))
print(accuracy_score(y_test, pred_nb))


# In[ ]:


print("Train Score {:.2f} & Test Score {:.2f}".format(nb.score(X_train,y_train),nb.score(X_test,y_test)))


# In[ ]:


print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2")
print("""Naive Bayes Classifier \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, pred_nb )),
            mean_squared_error(y_test, pred_nb ),
            mean_absolute_error(y_test, pred_nb ),
            r2_score(y_test, pred_nb )))


# ### classification Report

# In[ ]:


# classification Report
print(classification_report(y_test, pred_rf))


# ### Confusion Matrix

# In[ ]:


# Confusion Matrix
cf_matrix = confusion_matrix(y_test, pred_rf)
print('Confusion Matrix \n',cf_matrix)


# ### Confusion Matrix Heatmap

# In[ ]:


plt.figure(figsize=(7,6))
sns.heatmap(cf_matrix, cmap='coolwarm', annot=True, linewidth=1, fmt="d")
plt.show()


# ### Score Summary :

# In[ ]:


classifier_list=['Logistic Regression','Decision Tree','Random Forest','KNN','xgboost','nbayes']
clf_accuracy1 = [0.6488095238095238,0.781415343915344,0.8621031746031746,0.6458333333333334,0.8753306878306878,0.6504629629629629]


# In[ ]:


plt.figure(figsize=(7,6))
sns.barplot(x=clf_accuracy1, y=classifier_list)
plt.grid()
plt.xlabel('accuracy')
plt.ylabel('classifier')
plt.title('classifier vs accuracy plot')


# In[ ]:


models = [LogReg, DTR, rf, KNN, reg_xgb, nb]
names = ['Logistic Regression','Decision Tree','Random Forest','KNN','xgboost','nbayes']
rmses = []

for model in models:
    rmses.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

x = np.arange(len(names)) 
width = 0.3

fig, ax = plt.subplots(figsize=(10,7))
rects = ax.bar(x, rmses, width)
ax.set_ylabel('RMSE')
ax.set_xlabel('Models')

ax.set_title('RMSE with Different Algorithms')

ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45)

fig.tight_layout()


# #### Logistic Regression
# - Accuracy on Test Data Set with Logistic Regression : 64%
# 
# #### Decision Tree
# - Accuracy on Test Data Set with DecisionTree Regression : 78%
# 
# #### Random Forest
# - Accuracy on Test Data Set with Random Forest Classifier : 86%
# 
# #### KNN
# - Accuracy on Test Data Set with K Nearest Neighbours : 64%
# 
# #### XGBoost Model
# - Accuracy on Test Data Set with XGBoost Classifier : 87%
# 
# #### Naive Bayes Classifier
# - Accuracy on Test Data Set with Naive Bayes Classifier : 65%
# 
# 
# - So far **GBoost Model** proved to be the best performing model with **87% accuracy**.

# ### Submission

# In[ ]:


y_pred_test = reg_xgb.predict(test)


# In[ ]:


submission = pd.DataFrame({'Id': test['Id'], 'Cover_Type': y_pred_test})
submission.to_csv('Forest Covetype.csv', index=False)


# <h3 style="color:green" align="left"> Hypothesis Testing </h3>

# In[ ]:


from scipy import stats
from scipy.stats import f_oneway
from scipy.stats import ttest_ind


# In[ ]:


stats.ttest_1samp(cover['Elevation'],0)


# ### Chi-Square Test-
# - The test is applied when you have two categorical variables from a single population. It is used to determine whether there is a significant association between the two variables.

# In[ ]:


street_table = pd.crosstab(cover['Elevation'], cover['Cover_Type'])
print(street_table)


# In[ ]:


street_table.values 


# In[ ]:


# Observed Values
Observed_Values = street_table.values 
print("Observed Values :-\n",Observed_Values)


# In[ ]:


val = stats.chi2_contingency(street_table)
val


# In[ ]:


Expected_Values = val[3]


# In[ ]:


no_of_rows = len(street_table.iloc[0:2,0])
no_of_columns = len(street_table.iloc[0,0:2])
ddof = (no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",ddof)
alpha = 0.05


# In[ ]:


from scipy.stats import chi2
chi_square = sum([(o-e)**2./e for o,e in zip(Observed_Values, Expected_Values)])
chi_square_statistic = chi_square[0]+chi_square[1]


# In[ ]:


print("chi-square statistic:-",chi_square_statistic)


# In[ ]:


critical_value = chi2.ppf(q=1-alpha,df=ddof)
print('critical_value:',critical_value)


# In[ ]:


# p-value
p_value = 1-chi2.cdf(x=chi_square_statistic, df=ddof)
print('p-value:', p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',ddof)
print('p-value:', p_value)


# In[ ]:


if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# In[ ]:


import statsmodels.api as sms
model = sms.OLS(y,X).fit()
model.summary()

