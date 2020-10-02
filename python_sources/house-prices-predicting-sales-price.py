#!/usr/bin/env python
# coding: utf-8

# **House Prices: Advanced Regression Techniques**
# * Predict sales prices and practice feature engineering, RFs, and gradient boosting
# 
# **Forking and sharing plus credits if it has helped you readcarefully the comments and other text instructions and descriptions to understand this kernel hope will be helpful**

# In[ ]:



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
from scipy.stats import norm
import seaborn as sns;sns.set(style="whitegrid",color_codes=True) #data visualization and tune set params
import matplotlib.pyplot as plt;plt.figure(figsize=(15,15)) #data plotting/visualization
get_ipython().run_line_magic('matplotlib', 'inline')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# **STEP 1**
# * Read the test and the train datasets using pandas func **read_csv()**
# * We shall be reading our data from the train dataset and viewing missing values from the dataset

# In[ ]:



train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

print ("Train dataset has {0} rows and {1} columns".format(train.shape[0],train.shape[1]))
print ("Test dataset has {0} rows and {1} columns".format(test.shape[0],test.shape[1]))
#lets check for the missing values in the train dataset as missing values can really spoil your predictions
missing_values = train.columns[train.isnull().any()]
#we shall be using bash style color codes ***\033[1;36m*** to differentiate the ouput
print ("\033[1;36m"+"There are {0} missing values in the dataset Train\n---------------------------------------------------------".format(len(missing_values))+"\033[0m")
#We see that there are only 19 missing values only we can go ahead and see the column names with missing values
for missing_vals in missing_values:#loop througth the list and print all the column names
    print ("\033[1;32m"+missing_vals+"\033[0m")
print ("\033[1;36m"+"---------------------------------------------------------"+"\033[0m")


# **STEP 2**
# * Getting the percentage of the missing values in some of our columns
# * We can see variable PoolQC has 99.5% missing values followed by MiscFeature, Alley, and Fence
# * And then visualizing them 

# In[ ]:


#Then we shall be getting the percentage of the missing values in columns of our dataset like below
percentage_missing = train.isnull().sum()/len(train)
percentage_missing = percentage_missing[percentage_missing > 0]
percentage_missing.sort_values(inplace=True)#we use inplace=True to make changes to our columns
print(percentage_missing)
#lets plot to visualize the missing values
percentage_missing = percentage_missing.to_frame()
percentage_missing.columns=['Count']
percentage_missing.index.names = ['Name']
percentage_missing['Name'] = percentage_missing.index
plt.figure(figsize=(15,15))
sns.barplot(x="Name",y="Count",data=percentage_missing)
plt.xticks(rotation=90)


# **STEP 3**
# * We continue and check the distribution of our target variable **SalePrice**
# * And  Its skew if its right or left skewed we shall be tuning it in the next steps to be normally distributed using the **skew()** function

# In[ ]:


###Now lets check the distribution of the target variable SalePrice
plt.figure(figsize=(15,15))
sns.distplot(train["SalePrice"])
print ("SalePrice right-skewenes = {0}".format(train["SalePrice"].skew()))


# **STEP 4**
# * We see that the target variable SalePrice is right-skewed we need to log transform this variable so that it can be normally distributed. as Normally distributed variables help in better modelling the relationship between the target variable and independent variables

# In[ ]:


#We see that the target variable SalePrice is right-skewed we need to log transform this variable so that it can be normally distributed. as Normally distributed variables help in better modelling the relationship between the target variable and independent variables
normal_dist = np.log(train["SalePrice"])
"""Then we can see that atleast it has been nearly normally distributed"""
print ("SalePrice skew now = {0}".format(normal_dist.skew()))
plt.figure(figsize=(15,15))
sns.distplot(normal_dist)


# **STEP 5**
# * We need to separate both numerical and categorical data from our dataset
# * And then we are interested to learn about the correlation behavior of numeric variables. Out of 38 variables, we presume some of them must be correlated. If found, we can later remove these correlated variables as they won't provide any useful information to the model.
# * We can see the correlation of all variables against SalePrice. As we can see, some variables seem to be strongly correlated with the target variable. Here, a numerical correlation data will help us understand the graph at its best

# In[ ]:


#Now we need to separate Numerical and categorical data from the dataset each will be attacked and analyzed from defferent angles
numerical_data = train.select_dtypes(include=[np.number])
categorical_data = train.select_dtypes(exclude=[np.number])
print("The dataset has {0} numerical data and {1} categrical data".format(numerical_data.shape[1],categorical_data.shape[1]))
#we are interested to learn about the correlation behavior of numeric variables. Out of 38 variables, I presume some of them must be correlated. If found, we can later remove these correlated variables as they won't provide any useful information to the model.
corr = numerical_data.corr()
plt.figure(figsize=(15,15))
sns.heatmap(corr)


# In[ ]:


#a numerical correlation data 
print ("\033[1;34m"+"The Correlation of the first 15 variables\n-------------------------------")
print(corr["SalePrice"].sort_values(ascending=True)[:15])
print("\033[0m")
print ("\033[1;35m"+"The Correlation of the last 5 variables\n--------------------------------")
print(corr["SalePrice"].sort_values(ascending=True)[-5:])
print("\033[0m")


# **STEP 6**
# * We see that the feature OverallQual is 79% correlated with the target variable. OverallQual means the overall quality and material of a completed house and GrLivArea is 70% correlated with the target variable, GrLivArea refers to the size of the living room area in **sq. ft** above the ground and more adds most people care about a house with a garage, garage area and the basement size
# 
# * Finally we will have to check the median of the sale price of a house with respect to OverallQual, we shall be using median because our target variable is skewed of which skewed variables normally have outliers and median is robust to outliers

# In[ ]:


print(train['OverallQual'].unique())
#OverallQual is between a scale of 1 t0 10, 
#which we can fairly treat as an ordinal variable order.An ordinal value has an inherent order
#finally lets compute our median and visualize it in a plot
pivot = train.pivot_table(index="OverallQual",values="SalePrice",aggfunc=np.median)
#show a more sorted table
print(pivot.sort_values)
pivot.plot(kind='bar',color="magenta")


# * We can see by visualization that the great the quality of material used on a building the more its SalePrice
# * We go  further a head to plot the GrLivArea and visualize its behaviour

# In[ ]:


plt.figure(figsize=(15,15))
sns.jointplot(x=train['GrLivArea'],y=train['SalePrice'])


# As seen above, here also we see a direct correlation of living area with sale price. However, we can spot an outlier value GrLivArea > 5000. we shall visualize other variables later we'll move forward and explore categorical features. I have seen outliers spoiling the performance of a model performance we shall handle that later on removing outliers focus.
# 

# ** Now lets talk the route to Categorical features **
# * And the best way to understand categorical features using **.describe()** like below

# In[ ]:


#lets first understand whats in the categorical features
categorical_data.describe()


# **Now lets check the median of the sale price of a house based on its SaleCondition.
# SaleCodition explains the condition of sale**

# In[ ]:


#calculate median
cat_pivot = train.pivot_table(index="SaleCondition",values="SalePrice",aggfunc=np.median)
cat_pivot.sort_values


# In[ ]:


#we go ahead and visualize the median cat_pivot
cat_pivot.plot(kind="bar",color="blue")


# **We see that SaleCondition Partial has the highest mean sale price. Though, due to lack of information we can't generate many insights from this data. Moving forward, like we used correlation to determine the influence of numeric features on SalePrice. Similarly, we'll use the ANOVA test to understand the correlation between categorical variables and SalePrice. ANOVA test is a statistical technique used to determine if there exists a significant difference in the mean of groups**
# * Now, we'll define a function which calculates p values. From those p values, we'll calculate a disparity score. Higher the disparity score, better the feature in predicting sale price. 

# In[ ]:


#we use list compression to assign our return values to cat and for simplicity
#Read more about this technique and you will enjoy its power
cat=[f for f in train.columns if train.dtypes[f]=="object"]
#print(cat)
def anova(frame):
    anv = pd.DataFrame()
    anv['features'] = cat
    pvals=[]
    for c in cat:
        samples=[]
        for cls in frame[c].unique():
            s=frame[frame[c]==cls]['SalePrice'].values
            samples.append(s)
        pval=stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv["pval"]=pvals
    return anv.sort_values("pval")


# In[ ]:


plt.figure(figsize=(15,15))
categorical_data['SalePrice']=train.SalePrice.values
k=anova(categorical_data)
k['disparity']=np.log(1./k['pval'].values)
sns.barplot(data=k,x="features",y="disparity")
plt.xticks(rotation=90)
plt.show()


# **Here we see that among all categorical variables Neighborhood turned out to be the most important feature followed by ExterQual, KitchenQual, amonothers. 
# It means that people also consider the goodness of the neighborhood, the quality of the kitchen, the quality of the material used on the exterior walls. **
# * Finally, to get a quick glimpse of all variables in a data set, let's plot histograms for all numeric variables to determine if all variables are skewed. 
# * For categorical variables, we'll create a boxplot and understand their nature. 

# In[ ]:


#lets create numeric data plots
num=[f for f in train.columns if train[f].dtypes != "object"]
num.remove("Id")
c = {'color': ['r']}
nd = pd.melt(train,value_vars=num)
n1 = sns.FacetGrid(nd,col="variable", hue_kws=c,col_wrap=4,sharex=False,sharey=False)
n1.map(sns.distplot,"value")
n1


# **As you can see, most of the variables are right skewed. We'll have to transform them in the next stage. Now, let's create boxplots for visualizing categorical variables. **

# In[ ]:


#lets plot categorical data
def boxplot(x,y,**kwargs):
    sns.boxplot(x=x,y=y)
    x=plt.xticks(rotation=90)
cate = [f for f in train.columns if train[f].dtypes=="object"]
nd =pd.melt(train,id_vars="SalePrice",value_vars=cate)
np1 = sns.FacetGrid(nd,col="variable",col_wrap=2,sharex=False,sharey=False,size=5)
np1.map(boxplot,"value","SalePrice")


# ** EEEHWOOOO We move on now toooooooo DATA PRE-PROCESSING i Guess lets jumpin**
# ![](https://i.imgur.com/ijDul7d.png)
# * we'll deal with outlier values, encode variables, impute missing values, and take every possible initiative which can remove inconsistencies from the data set. If you remember, we discovered that the variable GrLivArea has outlier values. Precisely, one point crossed the 4000 mark. Let's remove that Guys

# In[ ]:


#Guys lets remove outliers
train.drop(train[train["GrLivArea"]>4000].index,inplace=True)
train.shape #seems and its true we remove 4 rows compared to our original data


# * In row 666, in the test data, it was found that information in variables related to 'Garage' (GarageQual, GarageCond, GarageFinish, GarageYrBlt) is missing. Let's impute them using the mode of these respective variables. Lets fix also that guys

# In[ ]:


#imputing using mode
test.loc[666, 'GarageQual'] = "TA" #stats.mode(test['GarageQual']).mode
test.loc[666, 'GarageCond'] = "TA" #stats.mode(test['GarageCond']).mode
test.loc[666, 'GarageFinish'] = "Unf" #stats.mode(test['GarageFinish']).mode
test.loc[666, 'GarageYrBlt'] = "1980" #np.nanmedian(test['GarageYrBlt'])


# * Yeah yeah do you also remember that row  row 1116 in the test data all garage variables were **NA** except for the garage type. lets us also mark it as **NA**. Here we go

# In[ ]:


#let us mark it as a NA variable
test.loc[1116,'GarageType']=np.nan


# * Now, we'll encode all the categorical variables. This is necessary because most ML algorithms do not accept categorical values, instead they are expected to be converted to numerical. **LabelEncoder** function from **sklearn** is used to encode variables, cite the sklearn documentation on how to use LabelEncoder even better that me **100%** and please make sure cause you will use it often and we shall need a method to encode that i call it **vectorize**

# In[ ]:


# import func LabelEncoder
from sklearn.preprocessing import LabelEncoder
lben = LabelEncoder()
def vectorize(data,var,fill_na=None):
    if fill_na is not None:
        data[var].fill_na(fill_na,inplace=True)
    lben.fit(data[var])
    data[var]=lben.transform(data[var])
    return data


# * This function above imputes the blank levels with mode values. The mode values are to be entered manually. Now, let's impute the missing values in **LotFrontage** variable using the median value of **LotFrontage** by **Neighborhood**. Such imputation strategies are built during **Data exploration**. I suggest you spend some more time on **Data exploration**. To do this, we should combine our train and test data so that we can modify both the data sets at once. And its time saving come to think of it

# In[ ]:


#combine the dataset
alldata = train.append(test)
#alldata.shape

#now lets impute LotFrontage by the median of Neighbourhood
lotf_by_neig=train['LotFrontage'].groupby(train["Neighborhood"])
for key,group in lotf_by_neig:
    idx=(alldata["Neighborhood"]==key)&(alldata['LotFrontage'].isnull())
    alldata.loc[idx,'LotFrontage'] = group.median()


# * With other numerical variables we will impute the missing values with zeros as below

# In[ ]:


#impute missing values
alldata["MasVnrArea"].fillna(0, inplace=True)
alldata["BsmtFinSF1"].fillna(0, inplace=True)
alldata["BsmtFinSF2"].fillna(0, inplace=True)
alldata["BsmtUnfSF"].fillna(0, inplace=True)
alldata["TotalBsmtSF"].fillna(0, inplace=True)
alldata["GarageArea"].fillna(0, inplace=True)
alldata["BsmtFullBath"].fillna(0, inplace=True)
alldata["BsmtHalfBath"].fillna(0, inplace=True)
alldata["GarageCars"].fillna(0, inplace=True)
alldata["GarageYrBlt"].fillna(0.0, inplace=True)
alldata["PoolArea"].fillna(0, inplace=True)


# * Variable names which have **quality** or **qual** in their names can be treated as ordinal variables, as mentioned above. Now, we'll convert the categorical variables into ordinal variables. To do this, we'll simply create a dictionary of key-value pairs and map it to the variable in the data set. 

# In[ ]:


qual_dict={np.nan:0,"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5}
name = np.array(['ExterQual','PoolQC' ,'ExterCond','BsmtQual','BsmtCond',                 'HeatingQC','KitchenQual','FireplaceQu', 'GarageQual','GarageCond'])
for i in name:
    alldata.head()


# In[ ]:




