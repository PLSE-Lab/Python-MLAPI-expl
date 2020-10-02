#!/usr/bin/env python
# coding: utf-8

# # I try and use the best and optimized techniques which have quality as well as save you some time. Check out my work below.
#  
# # Do upvote if you find the notebok useful.
#  
# # Also, please feel free to ask questions and do let me know about my work.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv('/kaggle/input/bigmart-sales-data/Train.csv')


# We shall take a look at the data first
# - shape
# - data types
# - columns

# In[ ]:


train.shape

#so the TRAIN data has 8523 rows and 12 columns


# In[ ]:


# get the list of all columns in TRAIN data
train.columns


# In[ ]:


# finding out the datatype of each column
train.info()


# Before getting on with EDA activity, we shall first segregate the train dataset on the basis of their datatypes. 
# - Cat_data (categorical data)
# - Num_data (Numeric data)  
# Note : The Year column will be included in Num_data

# In[ ]:


#we shall separate the categorical and numeric columns
cat_data = []
num_data = []

for i,c in enumerate(train.dtypes):
    print(i,c)
    if c == object or c == int:
        cat_data.append(train.iloc[:, i])
    else :
        num_data.append(train.iloc[:, i])

cat_data = pd.DataFrame(cat_data).transpose()
num_data = pd.DataFrame(num_data).transpose()
cat_data.head()


# Now, we shall use visualizations to understand and take meaning out of the data

# In[ ]:


import pandas_profiling as prof

#report = prof.ProfileReport(train)
#report


# From the analysis above, we have the following information :
# 1. Item_Weight has 1463 (17.2%) missing values
# 2. Outlet_Size has 2410 (28.3%) missing values
# 3. Item Fat content has a naming issue, we need to convert LF & low fat to Low Fat, and reg to Regular
# 4. Cramer's V statistics shows that Outlet_Size, Outlet_type, Outlet_location_type and Outlet Identifier are well correlated

# Treating the Item_Fat_Content column

# In[ ]:


cat_data.Item_Fat_Content.unique()
cat_data['Item_Fat_Content'] = cat_data['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat','reg': 'Regular'})
print(cat_data.Item_Fat_Content.unique())
cat_data.head()
##now we see that we now have only two unique Fat_content types

train = pd.concat([cat_data,num_data], axis = 1)
train.head()
train.shape


# # Lets do some Pre imputation visualization

# In[ ]:


## to start with a pairplot
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.pairplot(train)
plt.show()


# # I am unable to make anything but the Skewness in certain features on the num_data out of this pairplot .. Let us try and infer something from a correlation plot

# In[ ]:


num_data.corr()
sns.heatmap(num_data.corr(),annot=True)


# ## From the above Correlation heat map of Numerical data we infer that :
# -- The MRP of a product has quiet a strong affect on the Outlet sales of the products
# >     i.e. **The Total_Sales of Outlets is Price sensitive**

# # Lets perform some Univariate analysis

# In[ ]:


# lets get the basic statistical measures for num_data
num_data.describe()


# *In the above summary, I smell an opportunity to try and bin the Item_weight columns and even Item_visibility.
# We shall not bin Item_MRP because in our prior analysis we observed a good correlation between Tem_MRP and Outlet_Sales.*

# In[ ]:


cat_data.describe()


# In[ ]:


cat_data['Item_Identifier'].value_counts()
cat_data = cat_data.drop(['Item_Identifier'], axis=1)
cat_data.columns


# In[ ]:


for i in cat_data.columns:
    print()
    print(i)
    print(cat_data[i].value_counts())
    print()
    print()


# From the above output we can infer about the most common classes in each Categorical column

# # Creating Boxplots for each Numerical column to understand the spread of the data

# Relationship between 'Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales' and Outlet_Location_Type

# In[ ]:


cat_columns = cat_data.columns
cat_columns = list(cat_columns)
cat_columns.remove('Outlet_Establishment_Year')
cat_columns_new = cat_columns


for cat in cat_columns_new:
    print(cat)
    print()
    for i in num_data.columns:
        print(i, "vs", cat)
        sns.set(style="whitegrid")
        sns.boxplot(train[i], train[cat])
        plt.show()


# The above plots speak a lot of the data. Do give it a look and understand the data and its features well.

# Relationship between 'Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales' and Outlet_Location_Type

# In[ ]:


train.columns


# # Now, I shall list down the possible questions which we can answer using the 4 Numerical features and some Categorical variables
# 
# 1. Which class in each category corresponds to maximum sales
# 2. The Outlet corresponding to max and min sales and why
# 3. The number of outlets in a Location type
# 4. The relation between Outlet_Sales, Outlet Location type and Outlet size
# 5. The relation between Outlet_Sales, Outlet Location type and Outlet type
# 6.

# In[ ]:


## 1. Which class in each category corresponds to maximum sales

for cat in cat_data.columns:
    print("Item_Outlet_Sales in Thousands ('000)")
    print()
    print("-"*20 + cat + '  vs' + '  Item_Outlet_Sales' + "-"*20)
    output = train[[cat,'Item_Outlet_Sales']].groupby([cat]).apply(lambda x: x['Item_Outlet_Sales'].sum()/1000).sort_values(ascending=False)
    output = pd.DataFrame(output)
    output.columns = ['Item_Outlet_Sales']
    ax = sns.barplot(output.index,'Item_Outlet_Sales', data =output)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    for p in ax.patches:
        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width()/ 2., p.get_height()),ha='center', va='center', rotation=90, xytext=(0,40), textcoords='offset points')  #vertical bars
    plt.tight_layout()
    plt.show()
    print()
    print("Maximum Sales : ")
    print(output.head(1))
    print()
    print("-" *50)


# The visuals above gives us the information above the Total sales for each class of each category.
# 
# Some of the key information is stated below:
# 1. SuperMarket Type 1 correspond to Max. sales i.e 12917.34 (in thousands)
# 2. Tier 3 locations have max sales to the tune of 7636.75 (in thousands)
# 3. Outlets with medium size have max sales to the tune of 7489.72 (in thousands)
# 4. Outlet OUT027 has the max sales of 3453.9 (in thousands)
# 5. People purchase Fruits and Vegetables the most and second comes Snacks
# 6. Low fat goods sell the most

# # **Lets answer our 2nd ques.
# # "The Outlet corresponding to max and min sales and the possible reason for those sales"**

# In[ ]:


cat_data.describe()
train.groupby(['Outlet_Identifier','Outlet_Establishment_Year']).size().reset_index(name='Freq')
#train['Outlet_Identifier'].unique()
#train['Outlet_Establishment_Year'].unique()


# In[ ]:


train.columns
year_store_sales = train[['Outlet_Identifier','Outlet_Establishment_Year','Item_Outlet_Sales']].groupby(['Outlet_Identifier','Outlet_Establishment_Year']).apply(lambda x: x['Item_Outlet_Sales'].sum()/1000).sort_values(ascending=False)
year_store_sales = pd.DataFrame(year_store_sales)
year_store_sales.columns = ['Outlet_Sales']
year_store_sales


# From the above analysis we infer that Outlet OUT027 and OUT019 were established in the same year but 27 has max sales whereas 19 is at the bottom.
# 
# *Furthermore, we can see that OUT035,OUT017 has good sales given their Establishment year.* 
# 
# # Let us try and find out why ?

# In[ ]:


#To find out the reason for the above case we can act smart and subset data on the Outlet_Identifier and
#then remove duplicates (assuming that an outlet will be at only one location)

outlets = train[['Outlet_Identifier',"Outlet_Size","Outlet_Location_Type","Outlet_Type"]]
print(outlets.head())
print()
print("Before removing NA : " , outlets.shape)


# In[ ]:


# removing duplicates
outlets_new = outlets.drop_duplicates(subset=['Outlet_Identifier'])
outlets_new


# From the above analysis we infer that among Oultlet 17,19,35,17
# -- Other than Outlet 19 all the other 3 top Sales outlets are Supermarkets (2 of type 1 and 1 of Type 3)
# -- Also, along with this, we now have an the info that Outlet10, Outlet45, Outlet17  doesnt have a mentioned Outlet Size, and these correspond t0 2410 rows i.e around 28% of the total rows.
# 
# Now we have two options to solve this situation. Either we can drop Outlet_Size or try and impute the blank cells with some logical values
# 
# We shall tackle this while imputation

# In[ ]:


train[train.Outlet_Identifier.isin(['OUT010', "OUT045", "OUT017"])].shape[0] / train.shape[0]


# # # The 3rd question
# # # "The number of outlets in a Location type"

# In[ ]:


ld = train[["Outlet_Identifier","Outlet_Location_Type","Outlet_Size","Outlet_Type"]].groupby(['Outlet_Location_Type',"Outlet_Type","Outlet_Identifier"]).size()
ld = pd.DataFrame(ld)
ld.columns = ['Count']
ld


# The table above answers ques 3rd, 4th and 5th .. 
# It gives us the count of Outlets w.r.t. the Outlet_type and Location_type
# NOTE: I have ommited Outlet_Size because we dont have Outlet_size for Outlet 10,45 and 17

# # **It is a point to be noted that one can see many questions and try find an answer to those, but one should be smart enough to ask good question and seek their answers.
# 
# # **Now we shall move forward and attend the problem of missing values**

# # Null Value handling

# Lets take a look at the columns with null values

# In[ ]:


## finding the null values and treating them
heat = sns.heatmap(train.isnull(), cbar=False)
plt.show()
Null_percent = train.isna().mean().round(4)*100

Null_percent = pd.DataFrame({'Null_percentage' : Null_percent})
Null_percent.head()
Null_percent = Null_percent[Null_percent.Null_percentage > 0].sort_values(by = 'Null_percentage', ascending = False)
print("Percentage of Null cells : \n \n " , Null_percent)


# From the analysis above we understand that only 2 columns have null values.
# 
# Now we will have to take a decision either to keep them and impute or drop them from the data and move ahead.
# 
# I prefer dropping a feature if the Percentage of null values in it is greater than 15% . Also we have a small dataset at our disposal.
# So, I shall drop these two columns.

# In[ ]:


print(cat_data.columns)
print(num_data.columns)

cat_data_new = cat_data.drop(['Outlet_Size'], axis =1)
num_data_new = num_data.drop(['Item_Weight'], axis =1) 
cat_data_new.head()
num_data_new.head()

Null_percent_cat = cat_data_new.isna().mean().round(4)*100
print(Null_percent_cat)
Null_percent_num = num_data_new.isna().mean().round(4)*100
print(Null_percent_num)


# Now, we have Null free data. 
# 
# Now we shall encode our categorical variables and create dummy variables of them, to get the data ready for modelling.

# In[ ]:


## the columns we have to encode
## we shall not encode Outlet Establishment Year column
cat_data_new['Outlet_Establishment_Year'] = pd.to_numeric(cat_data_new['Outlet_Establishment_Year'])
cat_data_new.info()


# # Standard scaling and Label Encoding

# In[ ]:


## loading Standardscaler and OneHotEncoder from Sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Segregating train data into dependent and independent variable dataset

# In[ ]:


num_data_new.columns


# Scaling and encoding

# In[ ]:


## num_data has certain columns with some very high valued columns and some very low, thus we 
#should standardize the values of these columns

y = num_data_new['Item_Outlet_Sales']
num_data_new = num_data_new.drop(['Item_Outlet_Sales'], axis = 1)
print(num_data_new.columns)

for col in num_data_new.columns:
    num_data_new[col] = (num_data_new[col]-num_data_new[col].min())/(num_data_new[col].max() - num_data_new[col].min())
    
num_data_new.head()


# In[ ]:


##Label Encoding
from sklearn.preprocessing import LabelEncoder  
le = LabelEncoder()
cat_data_new.head()


# In[ ]:


# transform categorical columns columns

for i in cat_data_new.loc[:,~cat_data_new.columns.isin(['Outlet_Establishment_Year'])]:
    cat_data_new[i] = le.fit_transform(cat_data_new[i])

cat_data_new.head()
cat_data_new.shape


# # Now, we will have to create dummy variables of the cat data which we have

# In[ ]:


cat_data_new.columns
cat_data_new = pd.get_dummies(cat_data_new, columns=['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier','Outlet_Location_Type', 'Outlet_Type'])
cat_data_new.head()
cat_data_new.shape


# In[ ]:


x = pd.concat([num_data_new,cat_data_new], axis = 1)
x.head()
y


# In[ ]:


## I have an idea, why not convert the Establishment year 
#column to Years of existence by subtracting it by the current year

x['Outlet_Establishment_Year'] = 2020 - x['Outlet_Establishment_Year']

x.rename(columns = {"Outlet_Establishment_Year" : "Years_since"}, inplace = True)
 
x['Years_since'].describe()


# # Now we have the data ready to be modelled

# In[ ]:


data_dummy = pd.concat([x,y], axis =1)
data_dummy.columns


# # Applying Regression models

# In[ ]:


from sklearn.model_selection import train_test_split
train,test = train_test_split(data_dummy,test_size=0.20,random_state=2019)
print(train.shape)
print(test.shape)


# In[ ]:


train_label=train['Item_Outlet_Sales']
test_label=test['Item_Outlet_Sales']
del train['Item_Outlet_Sales']
del test['Item_Outlet_Sales']


# We shall apply the following Regression techniques
# 1. Linear Regression
# 2. Ridge Regression
# 3. Lasso Regression
# 4. Elastic net
# 5. Stochastic Gradient
# 6. SVR (Support vector regression)
# 7. Decision Tree
# 8. Random Forest
# 9. Bagging Regression
# 10. Adaptive Boosting (Ada Boost) 
# 11. Gradient Boosting
# 
# We shall apply Cross Validation to these

# In[ ]:


# algos to be used
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

# evaluating the model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


# In[ ]:


model_df = {'Name':['LR', 'Ridge', 'Lasso', 'E_Net','SVR','Dec_Tree','RF','Bagging_Reg','AdaBoost','Grad_Boost'],
             'Model' : [LinearRegression(), Ridge(alpha=0.05,solver='cholesky'), Lasso(alpha=0.01) ,ElasticNet(alpha=0.01,l1_ratio=0.5),
                     SVR(epsilon=15,kernel='linear'),DecisionTreeRegressor(),
                     RandomForestRegressor(),BaggingRegressor(max_samples=70),AdaBoostRegressor(),GradientBoostingRegressor()]}

model_df = pd.DataFrame(model_df)
model_df['Cross_val_score_mean'], model_df['Cross_val_score_STD'] = 0,0
model_df


# Now we have the dataframe of Model and model_names ready. We shall now proceed by training the data on each model using a for loop also in the process extract their cross val score (mean and std) to evaluate the model performance

# In[ ]:


for m in range(0,model_df.shape[0]):
    print(model_df['Name'][m])
    score=cross_val_score(model_df['Model'][m] , train , train_label , cv=10 , scoring='neg_mean_squared_error')
    score_cross=np.sqrt(-score)
    model_df['Cross_val_score_mean'][m] = np.mean(score_cross)
    model_df['Cross_val_score_STD'][m] = np.std(score_cross)
    
model_df


# In[ ]:


model_df.sort_values(by=['Cross_val_score_mean'])


# In the above dataframe we can see the result of our regression models.
# 
# - Gradient Boosting technique worked the best. Hence one can try and improve the model and further predict the values of the Test dataset.
# 
