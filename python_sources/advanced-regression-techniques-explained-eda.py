#!/usr/bin/env python
# coding: utf-8

# This is my 2nd kernel (Actually, this my 1st one, the earlier one I posted for the sake of getting used to the kaggle way) and personally feel its a bit long. Kindly forgive me for that!!!! I have given reference to all the materials that have helped me in writing this kernel. Kindly upvote if you believe its useful

# ## INTRODUCTION
# kaggle is one of the great platforms in the world of Data Science to get exposed to the various kinds of problems that ML/DS could solve. This is one of the challenges that puts skills to test as the data is mulivariate in nature(81 [columns] under consideration) and in addition to that kaggle recommends to use Advanced Regression Techniques such as Random Forest, Light Gradient Boosting, Extreme Gradient Boosting Techniques to solve the problem. I personally expec myself to have learnt advanced predictive modelling once completing the kernel. Let's get it Started!!!!
# 
# Kaggle states the problem as follows:
# 
# "Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home."

# ## CRISP DM METHODOLOGY
# 
# The methodology that I am going to follow to address the problem is called as **CRISP-DM**. CRISP-DM stands	for*	Cross Industry Standard Process	for	Data Mining*. It	is a data mining process model that	describes commonly used	approaches that	expert data miners use to tackle problems. Polls conducted in 2002, 2004, and 2007 show that it is the leading methodology used by data miners. The only other data mining standards named in these polls was SEMMA. However, 3-4 times as many people reported using CRISP-DM. A review and critique of data mining process models in 2009 called the CRISP-DM the "de facto standard for developing data mining and knowledge discovery projects. 
# 

# ### SIX PHASES OF CRISP-DM
# * **Business Understanding**
# * **Data Understanding**
# * **Data Preparation**
# * **Modelling**
# * **Evaluation**
# * **Deployment (This stage is out of scope as we are not integrating or deloying a model for production)**
# <img src="https://www.kdnuggets.com/wp-content/uploads/crisp-dm-4-problems-fig1.png" width="400px">

# ### PHASE 1: BUSINESS UNDERSTANDING
# I strongly believe that the first objective of any Data Scientist should be to throughly understand the problem from a business perspective, obtain the project objectives and requirements from the client and then converting this knowledge into a data mining problem definition. Often the client has many competing objectives and contraints that must be properly balanced. A possible consequence of neglecting this step is to expend a great deal of time, effort and resources in producing the answers to the wrong questions.
# 
#  Here, I am going to assume and put myself in a postion as a Data Scientist, facing a client running an online website which generates revenue by acting as a broker / online consultancy platform, providing solutions and recommendations to its customers intersted in buying mortgages and properties. Effectively, data collection has been done and one clear business objective is set. Building an Machine Learning Model that gets integrated into a web application (Assumption) that helps customer gain insights about the real estate market and gives recommendations on the salesprice of the various property across the city/region.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#importing supportive libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #Data Visualization
import matplotlib.pyplot as plt #Data Visualization
from scipy.stats import skew #Function to Determine skewness associated with variables in the data
from scipy.stats.stats import pearsonr #To find Correlation coefficient
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

#Models for Prediction problem
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet,Ridge, BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.pipeline import make_pipeline


#from sklearn.preprocessing import RobustScaler
#from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone



#functions to support data splitting, Data Transformation and evaluation metrics
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error, mean_squared_log_error 
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Reading the Data into variables

#training data
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

#test data
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# # DATA UNDERSTANDING / DATA EXPLORATION

# Most of the time this phase of a data mining project is skipped or the importance is undermined. Data Exploration is the most important step in any data mining project as it can uncover many hidden insights and can act as a land mine detector while we were walking towards our end goal. Exploration often leads to the discovery unexpected trends, patterns, missing data issues, outliers and other significant problems, if not addressed could potentially result in inaccurate or misleading results and conclusions.

# <img src="https://dataingovernment.blog.gov.uk/wp-content/uploads/sites/46/2016/08/Data-Science-Process-5-620x309.png" width="700px">

# In[ ]:


#check the dimension of the training data
train.info()


# <font size = "3">The data contains 1460 instances (records) and 81 columns. Data is of pretty decent size for analysis, however, in reality, more the data, better the results would be.The Data contains variables of type integer, float and object types, however, we can expect many categorical variables. Due to the reson that we have more variables under study, it would ask anyone going through this kernel to take a look at this [page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) to understand the data varaibles better. :)
#     </font>

# After taking a deeper look at the variables, out of all the variables, few variables seemed promising. Though I was a finance graduate and have a penchant towards investment and risk management, I dont really have as much interest in Real Estate as I do have in stock markets. I was thinking what a common man would analyze if he/she is going to buy a property?. How does he determine the value of the same? I came to a conclusion that if I am going to buy a property, I would definitely be interested in the following:
# 
# * area of the property?
# 
# * How old or new the property is?
# 
# * Is it brand new or a renovated one?
# 
# * Loaction of the property - City, Town, Sub-urbs or rural. (on a broader view, however, this can be broken down further)
# 
# Hence, just to have a quick sneakpeak, **I proceeded further to visualize the data in Tableau in order to see if that has any correlation with the SalePrice**. The below is the simple dashboard that I have generated using Tableau. Feel free to look at the dashboard by clicking this [link](https://public.tableau.com/views/AdvancedHousingPriceDashboard/Dashboard1?:embed=y&:display_count=yes&publish=yes)

# In[ ]:


from IPython.display import Image
Image("../input/dashboard-advreg/Advanced_Regression.png")


# It  became apparent that the variables **TotalBsmtSF, LotArea, and YearBuilt** have** *positive correlation with the outcome or target variable SalePrice***. However, we cannot restrict and build model out of these variables as it will lead to bias and most importantly considering all other features and building a better model is the ultimate aim of this notebook. Also, in the bottom most area chart, it is evident that from 1940 till 2010 the average SalePrice is in an upward trend and if you can notice that sale price plummeting after 2008 before rising. Obviously, this may be due to the 2008 crisis, where most of the mortgages went default and there is huge loss incurred to AIG. If you want to know more about the crisis, here is a short and crisp video explained by an investment legend Warren Buffett [link](https://www.youtube.com/watch?v=k2VSSNECLTQ). 

# Also, another variable I would assume that could greatly influence the price would be the Quality of the property or the material used to build the property. Fortunately, there is a variable named as **OverallQual** that depicts the same. Let's do some charts to explore that as well. Since the OverallQual is a categorical variable (actually here is it in an integer format but it describes the category of the quality on a scale of 1 to 10), we would compare OverallQual vs SalePrice to see if any interesting insights we could get.

# In[ ]:


#Box plot using seaborn
f, ax = plt.subplots(nrows = 1, ncols = 1,figsize=(12, 8))
fig = sns.boxplot(x="OverallQual", y="SalePrice", data=train)
fig.axis(ymin=0, ymax=1000000);


# <font size="3"> Clearly, as the **quality goes up** (1 being the lowest, and 10 being the highest), the **median of the SalePrice goes up**. </font> 

# Certainly, There are some other auxilliary features that one would look at while buying a property might include:  
# **Bedroom:** no of bedrooms and other associated attributes such as the finish, quality, no of bedrooms etc  
# **Kitchen:** type of kitchen, kitchen spacea and other assoicated attributes such as kitchen exterior, no of kitchens etc   
# **PoolArea:** Pool Area and other associated attributes  
# **GarageCars:** Garage Area and other assoicated attributes  
# 
# However, There were other characteristics which I felt were totally unnnecessary to be considered (such as ***Alley, LandSlope, LotShape*** etc),but, however, it is very subjective in nature and the preferences change from person to person. Hence considering all the economic classes in the society, we need to drill further to get an idea of how these secondary features could add value to our Machine Learning model.
# 

# Most of Machine Learning Models assume that the data is normally distributed. However, that's not the often. In our case, we have considered the 4 variables *TotalBsmtSF, LotArea, YearBuilt and OverallQual* to be the driving features (assuming that they are, however, we may face surprising events down the line), It is very imperative to understand the distribution of all these variables and transform it if necessary (Here is where the transformations come into play) 

# ## CORRELATIONS!!!

# I decided to create a correlation matrix and then a simple visualization showing the correlations between these feature variables and our target variable *SalePrice* in order to easily inspect +ve and -ve correlations. we can use the corr function to find the pearson's coefficient
# 

# In[ ]:


#generate correlation matrix for all the pair of variables
corr_matrix = train.corr()


# In[ ]:


#Inspecting the top 10 variables which has higher postive correlation with the SalePrice (1st one is itself a SalePrice)
corr_matrix["SalePrice"].sort_values(ascending = False)[:11]


# In[ ]:


#visualizing the missed out GrLivArea variable
sns.jointplot(x='GrLivArea',y='SalePrice',data=train,kind='reg')


# Fantastic!!! Earlier I mentioned 4 variables (*TotalBsmtSF, LotArea, YearBuilt and OverallQual* ) that could be the major predictor for the SalePrice and a big Yes from the correlation results!!! 3 of the 4 variables are in the top 10 predictor variables that has higher positive correlations with the SalePrice(Target).  To my surprise, LotArea is missing
# 
# There are also other significant variables that I missed during the earlier sneakpeak and that's the beauty of **EDA**. They are:   
# ***GrLivArea*** (Living Area)   
# ***GarageCars*** (no of cars the garage can hold and space)  
# ***1stFlrSF*** (first floor square feet, Yes of course I love a vast first floor area)  
# **FullBath** (who doesn't love a big fat bath tub and bath space to drench themselves like a hippo!!!. Moreover, I would love a surround and an audio system - Its a personal choice though)  
# YearRemodAdd (Re-modelled year else if its a new property then the value is same as the Built year for that observation)

# In[ ]:


#Inspecting the top 15 variables which has higher postive correlation with the SalePrice
corr_matrix["SalePrice"].sort_values(ascending = False)[:15]


# Let's Build our correlation matrix!!!! Note that we are not building the heatmap for all the variables under study but for the top 15 variables and visually inspect the relationship.

# In[ ]:


#correlation matrix for top 10 variables
from matplotlib import rcParams
rcParams['figure.figsize'] = 10,10
column_ind = corr_matrix.nlargest(11, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[column_ind].values.T)
sns.set(font_scale=1.1)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 14}, yticklabels=column_ind.values, xticklabels=column_ind.values)


# ### Some findings from the correlation heatmap are as follows:
# 1. It is very obvious that diagonal points of a correlation matrix are highly correlated (1 always) as it is the same column it is correlating with
# 2. GarageArea with GarageCars. (it is well known that no of cars that can be parked in th garage increase with the increase in space
# 3. 1stFlrSF increases as TotalBsmtSF increases(seems very logical to me)  
# 4.YearBuilt vs GarageArea/GarageCars (0.48/0.54) (*this seems interesting to think as things modernize,spending increases, lifestyle changes and transportation and convenience changes as each individual prefer having seperate cars for their convenience, thereby requiring more space in Garage.*)
# 
# Some unusual correlations that seems like a causation to me rather than a correlation are:  
# .1.TotalRmsAbvGrd vs GrLivArea (0.83)  
# 2.GrLivArea vs FullBath (0.63)  
# 3.TotalRmsAbvGrd vs FullBath (0.55)
# 
# The best insight that I gained is yet to come.There is multicollinearity between the square feet varibles (TotalBsmtSF and 1stFlrSF) and between the (GarageCars and GarageArea). This greatly helps in ignoring features and better feature selection, because they do not add any new variance or non-linearity to our model. This is where you are able to see through things that can never be seen or understood that easily without an exploratory study.In addition to that, I strongly suggest you to go back to the correlation matrix above and check the second row that relates OverallQual with all other variables. A groundbreaking insight that's been playing hide and seek with us now brought to light. OverallQual has strong positive correlationship with all the other 9 variables out of top 10 variables that has high positive correlationship with SalePrice.
# 
# For instance, let's consider OverallQual vs price or investment to buy or build a property. The more you spend the better the product quality will be!!!!!
# 

# Just to try out as a traditional approach (Though I dont believe that I could get any meaningful insights out of it), I am going to run a Correlation Heat Map for all the variables(Except Id variable - is simply ridiculous and a lurking variable.Also, when included in the study it feels like running a correlation between student ID in the unversity and their grade)

# In[ ]:


#let's just try for good sake if any patterns could be found in the below all in all variable corr_matrix
sns.heatmap(corr_matrix[1:], vmax=.8, square=True,yticklabels=corr_matrix.columns[1:], xticklabels=corr_matrix.columns[1:])


# One thing that was pinching me in my was the price movement. When I was back working in one of the world's largest investment bank, I had a habit of checking the historical price movements of the securities that's been traded by our cleints. I always check how the prices have significantly moved over years and would think, If I could travel back in time and buy those securities at that time so that I don't want to work ever and JUST travel Europe(cheap and an impossible-to-happen wish I should say). As soon as I knew that there was a price column, my unusual behaviour started kicking in. Let's check the SalePrice movements over the years!!!!!

# In[ ]:


#
rcParams["figure.figsize"] = 15,10
fig = sns.boxplot(x = "YearBuilt", y = "SalePrice", data  = train,palette = "Greens")
plt.xticks(rotation = 90)


# Last but not the least, Inorder to proceed further in data cleaning and transformations, It is always of prime importance to check the distribution of all the numeric variables involved in the study, most importantly the target variable SalePrice. I am not going to make my pairplot uglier and less intuitive by including all the variables. I am going to only include and infer the top 10 variables that have high correlation with SalePrice (and even with that top 10, I am going to ignore one out of the two variables that exhibit multi-collinearity.

# In[ ]:


sns.set()
sns.pairplot(train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath','TotRmsAbvGrd', 'YearBuilt','YearRemodAdd']], size = 3)
plt.show();


# It's so Fascinating to know that one single plot could lead to numerous insights and proves some assumptions wrong. Firstly, I did this plot to see the skewness in the promising numerical predictor variables and our most important SalePrice Variable.
# 
# Some notable points to carry forward are as follows:
# 
# * SalePrice - has Right skewness or Positive skewness because of the fact that only rich can afford expensive properties
# 
# * GrLiveArea, TotalBsmtSF, TotRmsAbvGrd, YearBuilt has skewness associated with them (with only YearBuilt having left skewness and other having Right Skewness)
# 
# * Clear cut outlier points can be noted in TotalBsmtSF and GrLivArea (May be need to check on that during Data Prep)
# 
# ***GrLivArea vs SalePrice*** - There are two outlier on the minimum side with a large living area and lower sale price. (May be those properties are reportedly have paranormal activity, this is why I hate large houses - Actually I can afford those though)
# 
# ***TotalBsmtSF vs SalePrice*** - There is an outlier with Large Basement Surface Area and very low SalePrice
# 
# * Many data clouds possess linearity (Nothing new as we know this is going to happen as told by our sooth sayer corr_matrix)
# 
# * **SalePrice vs YearBuilt looks Polynomial** (this is very interesting though) (SalePrice of the properties built in the late 2000's seems to have increased exponentially!!! Life always seems to play unfair with 90's kids)
# 
# 
# 
# *YearBuilt vs YearRemodAdd* (a strong linear trend occurrence can be seen for the newly built homes or properties. The data is collected in such a way that any property that was built and does not undergo any renovation or remodel, the remodel date is same as the YearBuilt date
# 

# pufff!!!!!! so many vizzes and very difficult to keep track of them. I decided to conclude my exploration. Next moving on to Data Preparation. One of the vital phases which could significantly improve our model's performance!!!. All those valuable and gold-standard insights gonna help us slay and lay the foundation for building our model. 

# # DATA PREPARATION

# The Data preparation phase covers all activities to construct the final dataset (data that will be fed into modelling phase) from the initial raw data. Data Preparation tasks are likely to be performed multiple times and not in any specific order.Tasks Cleaning, munging and manipulation of the data to cater our end goal, Transformations, Feature selection and Feature engineering to aid and improve the accuracy and performance of our model. 
# 
# Some of the Data Preparation steps that will notebook covers are as follows:
# Data Cleaning includes:   
# 1.Removing unncessary variables such as ID  
# 2.Handling Missing Values  
# 3.Outlier Removal  
# 
# 
# There are many reasons for Data Transformation. Most common reasons include but not limited to:  
# 1.Convenience (original values to percentages or degrees, Standardization, Normalization)  
# 2.Reducing Skewness for numeric variables(Right Skewness - log or root transformation)(Left Skewness - square transformation)  
# 3.Equal Spreads (Though variables have different mean, it is always easy to handle them if they have equal spead or variation)(Homoscedasticity)  
# 4.Create Polynomial Features (also can be included in feature engineering)  
# 5.Creating dummies for Categorical variables (as machines can only 0 and 1 better  - For example, even as a Human being, it is difficult for me to understand which neighbourhood is expensive to dwell in just by the name of the neighbourhood
# 
# I found an amazing blog that perfectly explains what these transformations are, what problems does they address, and to which situations they can be applied. I recommend everyone taking a look at it [http://fmwww.bc.edu/repec/bocode/t/transint.html](http://fmwww.bc.edu/repec/bocode/t/transint.html)
# 
# 
# 
# 

# ### DATA CLEANING

# <img src="http://www.dumpaday.com/wp-content/uploads/2013/07/Love-doing-laundry-Dump-E-card1.jpg" width="700px">

# well said!!!! Everyone like to work with data when it is clean and no painstaking efforts needed to clean and transform it. In any ML project, cleaning and transforming the data takes about 70 percent of the time. However, you need to know when to say no to certain things and when not to skip a process.

# In[ ]:


#making a copy of both train and test set for future reference and reduce redundancy in loading the data again
train_copy = train.copy()
test_copy = test.copy()

#Dropping the ID variable
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#getting output values for later to be used for training in the model


# In[ ]:


#printing the dimensions of the data
print("Shape of the training data copy:{}".format(train_copy.shape))
print("Shape of the Original train data {}".format(train.shape))
print("-"*160)
print("Shape of the test data copy:{}".format(test_copy.shape))
print("Shape of the Original test data {}".format(test.shape))


# In[ ]:


#Concatenate train and test
fulldata = pd.concat([train, test])
fulldata = fulldata.reset_index(drop = True)


# ## Missing Values

# I personally consider dealing with missing values is very prominent as it can significantly affect the size of the data from the ML model perspective. Ignoring missing values can significantly impact the size of the data as well as the quality of the results. My client was very eager, curious and in a haste to implement the model into his website and see result. It is my duty now to make him/her understand that data collection shoud be precise and ignoring a structured and quality data collection process could cost him and hurt the model !!!!
# 
# I learned about data cleaning process from the famous ["Hands-On Machine Learning with Scikit Learn and Tensorflow"](https://www.googleadservices.com/pagead/aclk?sa=L&ai=DChcSEwi8oLievcbgAhVvk-0KHaeyBTEYABAGGgJkZw&ohost=www.google.com&cid=CAESEeD2dCXiIBMclNmhdWYsVX_T&sig=AOD64_2fdY9qzvEDU3X4phSrlKjIn6mnAQ&ctype=5&q=&ved=0ahUKEwjc6rGevcbgAhWTonEKHYTnDAUQ9A4IpwE&adurl=). Data cleaning concepts were clearly explained in the book. The author has explicitly recommends 3 methods to deal missing values:  
# 1. Get rid of the corresponding row - using dropna() function  
# 2. Get rid of the whole attribute if necessary - using drop() function  
# 3. Set the values to some value (zero, mean, median, mode)  -  using fillna() function  
# 
# Let's Check if our dataset has any

# In[ ]:


missing_data = train.isnull().apply(sum).sort_values(ascending = False)
missing_col_name = missing_data[missing_data > 0]
print(missing_col_name)
print("There are {} variables with missing values".format(len(missing_col_name)))


# In[ ]:


#Fetching the names of the attributes that has missing values
missing_col_name = missing_data[missing_data > 0]
corr_matrix_missing = train[missing_col_name.index]
corr_matrix_missing["SalePrice"] = train["SalePrice"]


# In[ ]:


missing_corr_values = corr_matrix_missing.corr()


# In[ ]:


#checking numerical variables correlation as well as finding out how many categorical variables have missing value.
missing_corr_values["SalePrice"].sort_values(ascending = False)


# This is the step that almost changed my thinking on how to clean the missing values. Though the features having too many missing values, I felt hesitant to drop the features, though they are not strongly correlated with our target variable. Firstly, I almost removed the columns that had missing values and almost bulit the pipeline for models and this is where everything changed. I took a look at the documentation provided and it gave me goosebumps. This is what I have found:
# 
# ***Alley: Type of alley access to property  
# Grvl Gravel  
# Pave Paved  
# NA No Alley access***
# 
# NA means No Alley Access. Really? dont you get any other words or characters such as None, 0 or any mayan language character to get filled with :(. Not to my surprise, most missing value columns have this convention followed
# 
# I really want to know who created and collected this dataset, and was thinking to say a big thank you to him/her for making me waste my precious time.  

# After a good amount of homework, I came to a conclusion to impute the values rather than dropping the columns. I could have just used the training set data with cross validation to write the entire kernel as we don't have Y labels for the test set to compare and compute the error. These additional transformation I am making to make a competition submission. :))

# In[ ]:


fulldata_missing = fulldata.isnull().sum().sort_values(ascending = False)
fulldata_missing_colname = fulldata_missing[fulldata_missing > 0]
print(fulldata_missing_colname)


# In[ ]:


#features that has same conventions as Alley Access - filled with None. Note - I am giving "None" instead of zero here because they are Qualitaive varaibles.
#The categories are class labels in text. We will later convert some of these variables using LabelEncoder.
NoneFill = ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu","GarageType", "GarageFinish", "GarageQual", "GarageCond","BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2","MasVnrType"]

for item in NoneFill:
    fulldata[item] = fulldata[item].fillna("None")
    


# In[ ]:


#These categorical Labels already are in numeric categorical form and hence do not require encoding, however, conversion to dummy variables can be done
ZeroFill = ['GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','MasVnrArea']

for item in ZeroFill:
    fulldata[item] = fulldata[item].fillna(0)


# In[ ]:


fulldata["LotFrontage"] = fulldata.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


# In[ ]:


ModeFill = ['MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType','Utilities']

for item in ModeFill:
    fulldata[item] = fulldata[item].fillna(fulldata[item].mode()[0])


# As per the documentation, Functional is a column describing the home functionality. It is clearly mentioned as below:  
# 
# **Home functionality (Assume typical unless deductions are warranted)**
# 
# There are two missing values in this column and I am going to give them the value for Typical ("Typ")

# In[ ]:


fulldata["Functional"] = fulldata["Functional"].fillna("Typ")


# In[ ]:


#Dropping the SalePirce variable from fulldata
fulldata.drop("SalePrice", axis = 1, inplace = True)


# In[ ]:


#Checking if Missing values still persist in our data
fulldata.isnull().any().any()


# # Outliers

# <img src="https://discourse-cdn-sjc1.com/business6/uploads/analyticsvidhya/original/2X/9/99bbfb4b85905c489099c374d37f89c6f79b3134.JPG" width="800px">

# Dealing with outliers requires knowledge about the outlier, the dataset and possibly domain knowledge. Given this, there are many options to handle outliers. Given the situation that there is a constraint on my knowledge in the Real Estate domain and the limited size of our dataset, I wish to remove only the most extreme points!!!!

# In[ ]:


#plots to visualize GrLivArea Outlier
f, ax = plt.subplots()
ax.scatter(x = train["GrLivArea"],y = train["SalePrice"])
ax.set_xlabel("Ground Living Area")
ax.set_ylabel("Sale Price")


# In[ ]:


#Plot to Visualize TotalBsmtSF outlier
f,ax = plt.subplots()
ax.scatter(x = train["TotalBsmtSF"], y = train["SalePrice"])
ax.set_xlabel("Total basement Square feet")
ax.set_ylabel("Sale Price")


# <font size = "3"> As you can see the 3 guys extremely isolated from the normal crowd (2 guys in the first plot and 1 guy in the second), and I am going to remove them

# In[ ]:


TotalBsmtSF_row = train.loc[(train["TotalBsmtSF"] > 6000) & (train["SalePrice"] < 200000)]
GrLivArea_row = train.loc[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000)]


# In[ ]:


TotalBsmtSF_row


# In[ ]:


GrLivArea_row


# We can now see that there are only 2 oultier data points (because one observation is repetitive in both of the cases. Now I am going to remove these points. I have to be very cautious in removing the data observations as I have already dropped my Y values from fulldata (concat of train and test data), I will delete the rows corresponding to these two rows. One very good lesson learnt is to restructure my ML model flow by starting with Oultier removal as the first step of Data Prep and Cleaning right before handling missing values and other mandatory steps
# 
# Points Removed:  
# TotBsSF   SalePrice  
# 1) 3138            184750  
# 2) 6110             160000  

# In[ ]:


#Remving the outliers from both the dataset - kindly note to follow the same order while removing the oulier.
#1st remove the outlier from fulldata dataset then remove from train dataset
#Because once you remove the outlier from train data, you wont get the index to remove it from fulldata.
fulldata = fulldata.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index)
fulldata = fulldata.reset_index(drop = True)
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index)
train = train.reset_index(drop = True)


# In[ ]:


print("Merged Full Data Shape:{}".format(fulldata.shape))
print("-"*160)
print("Training Data Shape:{}".format(train.shape))
print("-"*160)
print("Test Data Shape:{}".format(test.shape))


# #  Feature Engineering - Encoding and Dummy Variables - Data Transformation!!!

# There are some variables that I came across are categorical values.I used the below line of code to group all the variables according to its datatype

# In[ ]:


train.columns.to_series().groupby(train.dtypes).groups


# Out of all the above integer type variables, it can be easily seen that all the year related variables, OverallCond, and MSSubClass variable is actually should be categorical in nature but are in Numeric form. This could highly influence our model as computer language or even ML Models can only understand numbers than categories. In this case, let's say for example the year 2000 will be given less importance than year 2018 as we know numerically 2018 is greater than 2000. Inorder to avoid such bias, it should be converted to categorical.

# In[ ]:


#MSSubClass=The building class
fulldata['MSSubClass'] = fulldata['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
fulldata['OverallCond'] = fulldata['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
fulldata['YrSold'] = fulldata['YrSold'].astype(str)
fulldata['MoSold'] = fulldata['MoSold'].astype(str)
fulldata['YearBuilt'] = fulldata['YearBuilt'].astype(str)
fulldata['YearRemodAdd'] = fulldata["YearRemodAdd"].astype(str)


# I am quite dissappointed to find out that the variable "YearBuilt" to be a categorical variable, as It had some serious exponential relationship with the SalePrice. I thought of adding polynomial importance (raising to power of 2 or 3). I also tried if there is any variable that could possibly bring the prices down as it increase. I tried looking at the negative correlation between variables and found that no variable has strong negative correlation with SalePrice. KitchenAbvGr is the only variable with  -0.135907 (weak negative correlation)

# Practically, **the larger the area, the more the price of the property is.** As in our data, Garage and Area related variables have significant relationship with SalePrice Target Variable. I came to know that it would be good to make a feature engineering on that attribute

# In[ ]:


# Adding a new total feature 
fulldata['TotalSF'] = fulldata['TotalBsmtSF'] + fulldata['1stFlrSF'] + fulldata['2ndFlrSF']


# ## Label Encoding:
# I would recommend any who is new to ML and Data Science, to get a copy of this Book  ["Hands-On Machine Learning with Scikit Learn and Tensorflow"](https://www.googleadservices.com/pagead/aclk?sa=L&ai=DChcSEwi8oLievcbgAhVvk-0KHaeyBTEYABAGGgJkZw&ohost=www.google.com&cid=CAESEeD2dCXiIBMclNmhdWYsVX_T&sig=AOD64_2fdY9qzvEDU3X4phSrlKjIn6mnAQ&ctype=5&q=&ved=0ahUKEwjc6rGevcbgAhWTonEKHYTnDAUQ9A4IpwE&adurl=). In the second chapter, the author has broken down each phases of an ML Pipeline, into simple and intuitive steps. He has clearly described what LabelEncoder is and how does it works.

# In[ ]:


cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold','YearBuilt', "YearRemodAdd")
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(fulldata[c].values)) 
    fulldata[c] = lbl.transform(list(fulldata[c].values))

# shape        
print('Shape all_data: {}'.format(fulldata.shape))


# ## DATA TRANSFORMATIONS
# <img src = "https://www.brainyquote.com/photos_tr/en/w/waynedyer/154416/waynedyer1.jpg" width ="600px">

# Data transformation is very vital for any data that contains numeric variables as it may have Positive or Negative skewness. Since Most of the Machine Learning models assume the underlying data to be normally distributed, we will try to converge and address the skewness issue 

# In[ ]:


int_features = fulldata.dtypes[fulldata.dtypes == "int64"].index
float_features = fulldata.dtypes[fulldata.dtypes == "float64"].index

# Check the skew of all numerical features
skewed_int_feats = fulldata[int_features].apply(lambda x: skew(x.dropna()))
skewed_float_feats = fulldata[float_features].apply(lambda x: skew(x.dropna()))

skewed_features = pd.concat([skewed_int_feats,skewed_float_feats])

print("\nSkewness in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_features})
skewness = skewness.sort_values('Skew', ascending = False)
skewness.head(15)


# [Skewness Explained!!!](https://help.gooddata.com/display/doc/Normality+Testing+-+Skewness+and+Kurtosis?desktop=true&macroName=sv-translation) Always there is a confusion in the statistics world and ML world between begineers which is right or left (Positive or Negative in other sense) This article explains the picture perfectly!!!. As per the article, Skewness between -0.5 and 0.5 is consider approximately symmetric. I am applying this rule to adjust the skewness and give my model what it expects :)) . Also do check this video to understand how log and exponent are related and what these transformations mean.

# In[ ]:


skewness = skewness[abs(skewness) > 0.5]
print("There are {} skewed numerical features to log(x+1) transform".format(skewness.shape[0]))
skewed_features = skewness.index
for features in skewed_features:
    fulldata[features] = np.log1p(fulldata[features])


# ## Categorical to Dummy Variables

# Inorder to help my model better understand what data its going to face, I am converting all the encoded variables into dummy variables aka one-hot encoding.

# In[ ]:


fulldata = pd.get_dummies(fulldata)
print(fulldata.shape)


# Hmm!!!! 222 variables for a comparitively medium sized dataset. Usually, it is not recommended to have such proportions as it may lead to overfitting problems. However, as most of the variables are dummy encoding and represent information in a binary form. It's still fine and we are on the safer side.

# One Last Step before modelling - Transforming the SalePrice Variable. It is of Everest Importance to check the distribution of this variable as this one single hurdle will topple our fast moving car off the highway into the woods!!!. 

# In[ ]:


sns.distplot(train['SalePrice'])


# It is clearly visible from the above chart, that the data has Right / Positive Skewness. Hence we can make a log or Sqrt transformation to normalize the data to nomally distributed form

# In[ ]:


train["SalePrice"] = np.log1p(train["SalePrice"])


# In[ ]:


y_train = train.SalePrice.values


# In[ ]:


#splitting the data to train the model
x_train = fulldata[:train.shape[0]]


# In[ ]:


print("x_train shape:{}".format(x_train.shape))
print("y_train shape:{}".format(y_train.shape))


# # MODELING AND EVALUATION

# In[ ]:


models = [['DecisionTree :',DecisionTreeRegressor()],
           ['RandomForest :',RandomForestRegressor()],
           ['KNeighbours :', KNeighborsRegressor(n_neighbors = 2)],
           ['KernelRidge:',KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)],
           ['LassoLarsIC :',LassoLarsIC()],
           ['GradientBoostingClassifier: ', GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)],
           ['Lasso: ', Lasso(alpha =0.0005, random_state=1)],
           ['Ridge: ', Ridge()],
           ['BayesianRidge: ', BayesianRidge()],
           ['ElasticNet: ', ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)],
           ['XGB: ',xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)],['LGB: ',lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)]]


# In[ ]:


print("Score of Models...")

for name,model in models:
    ModelTemp = make_pipeline(StandardScaler(),model)
    rmse= np.sqrt(-cross_val_score(model, x_train.values, y_train, scoring="neg_mean_squared_error", cv = 10))
    print("Average {} cross validation score: ".format(name), np.mean(rmse))


# And we came to the end!!!! ElasticNet wins over other model. finally I will deliver it to my client and hope he will be happy. (assumption). 

# I realize that my Kernel is a bit long and that why I have added some memes and quotes to keep the reader active. I would try to revise and reiterate the kernel very soon. I saw some intersting ideas such as stackedRegression models and advanced feature engineering from amazing minds in our community. I would first learn those concepts, comprehend it and would see if any improvements can be made on that and will induce those ideas into my kernel. I thank everyone who took the time and patience to go through my work!!!!

# **If you have learnt, benefited or if you find this notebook helpful, Likes and forks would be much appreciated and that would motivate me re-iterate the kernel with new ideas. PCA, Stacked Models and more feature engineering coming soon!!!! Stay Tuned!!!!**

# In[ ]:




