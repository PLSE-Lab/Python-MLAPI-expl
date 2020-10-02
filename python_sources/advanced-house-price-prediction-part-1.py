#!/usr/bin/env python
# coding: utf-8

# ## Building Machine Learning Pipelines: Data Analysis Phase

# ## Kaggle Project: House Prices: Advanced Regression Techniques
# 
# The main aim of this project is to predict the house price based on various features which we will discuss as we go ahead
# 
# You can have a look at the project description and the data from below link
# 
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques

# ## Lifecycle or a Pipeline of a Data Science Projects
# 1. Data Analysis
# 2. Feature Engineering
# 3. Feature Selection
# 4. Model Building
# 5. Model Deployment

# ## Data Analysis
# 
# In this part 1 of the notebook, we will look at the various steps invloved in Data Analysis Phase where we involve more with the data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
## Display all the columns of the dataframe
pd.pandas.set_option('display.max_columns',None)


# In[ ]:


df_train=pd.read_csv('../input/advance-house-price-predicitons/train.csv')

## print shape of dataset with rows and columns
print(df_train.shape)


# In[ ]:


## print the top5 records
df_train.head()


# #### In Data Analysis involves finding out the below steps and analysing them
# 1. Missing Values
# 2. All The Numerical Variables
# 3. Distribution of the Numerical Variables
# 4. Categorical Variables
# 5. Cardinality of Categorical Variables
# 6. Outliers
# 7. Relationship between independent and dependent feature(SalePrice)
# 

# ## Missing Values

# In[ ]:


# Here we will check the percentage of nan values present in each feature
# 1. Get the list of features which has missing values
null_features = [ features 
                  for features in df_train.columns 
                      if df_train[features].isnull().sum()>1 ]

# 2. Print the feature name and the percentage of missing values
df_train_missing = pd.DataFrame(np.round(df_train[null_features].isnull().mean(), 4),
                                columns=['% missing values'])
df_train_missing


# ### Observation:
# 
# The columns Alley,PoolQC,Fence and MiscFeature have more than 80% of missing values,Now lets see whether there is a relationship between missing values and the target feature(SalesPrice) by plotting them

# In[ ]:


for feature in null_features:
    df_train_tmp = df_train.copy()
    
    # let's convert all the Nan values to 1, otherwise zero for easy plotting the relationship with the SalesPrice
    df_train_tmp[feature] = np.where(df_train_tmp[feature].isnull(), 1, 0)
    
    # let's calculate the mean SalePrice where the information is missing or present
    df_train_tmp.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()


# ### Observation
# Here With  the relation between the missing values and the dependent variable is clearly evident.We cannot remove these rows where NaN values are present as there is a dependency, so we have to replace these NaN values with something meaningful which we will do in the Feature Engineering section

# ### Styling using pandas
# 
# https://kanoki.org/2019/01/02/pandas-trick-for-the-day-color-code-columns-rows-cells-of-dataframe/

# In[ ]:


# # Highlight the entire row in Yellow where Column B value is greater than 1
# np.random.seed(24)
# df = pd.DataFrame({'A': np.linspace(1, 10, 10)})
 
# df = pd.concat([df, pd.DataFrame(np.random.randn(10, 4), columns=list('BCDE'))],
#                axis=1)
# df.iloc[0, 2] = np.nan
 
# def highlight_greaterthan(s,column):
#     is_max = pd.Series(data=False, index=s.index)
#     is_max[column] = s.loc[column] >= 1
#     return ['background-color: red' if is_max.any() else '' for v in is_max]
 
# def highlight_greaterthan_1(s):
#     if s.B > 1.0:
#         return ['background-color: yellow']*5
#     else:
#         return ['background-color: white']*5
 
 # df.style.apply(highlight_greaterthan_1, axis=1)


# # Color code the text having values less than zero in a row
# def color_negative_red(val):
#     color = 'red' if val &lt; 0 else 'black'  # here &lt corresponds to '<' and &gt for '>' is used html
#     return 'color: %s' % color

# df.style.applymap(color_negative_red)


# # Highlight the Specific Values or Cells
# import pandas as pd
# import numpy as np

# def highlight_max(s):    
#     is_max = s == s.max()
#     return ['background-color: red' if v else '' for v in is_max]

# df.style.apply(highlight_max)


# # Highlight all Nulls in My data
# df.style.highlight_null(null_color='green')

# # Styler.set_properties
# df.style.set_properties(**{'background-color': 'black',
#                             'color': 'lawngreen',
#                             'border-color': 'white'})

# # Bar charts in DataFrame
# df.style.bar(subset=['A', 'B'], color='#d65f5f')

# # Table style
# from IPython.display import HTML

# def hover(hover_color="#ffff99"):
#     return dict(selector="tr:hover",
#                 props=[("background-color", "%s" % hover_color)])

# styles = [
#     hover(),
#     dict(selector="th", props=[("font-size", "150%"),
#                                ("text-align", "center")]),
#     dict(selector="caption", props=[("caption-side", "bottom")])
# ]
# html = (df.style.set_table_styles(styles)
#           .set_caption("Hover to highlight."))
# html


# In[ ]:


df_train_html = df_train.head().copy()


# In[ ]:


from IPython.display import HTML

def hover(hover_color="#ffff99"):
    return dict(selector="tr:hover",
                props=[("background-color", "%s" % hover_color)])

styles = [
    hover(),
    dict(selector="th", props=[("font-size", "150%"),
                               ("text-align", "center")]),
    dict(selector="caption", props=[("caption-side", "bottom")])
]
html = (df_train_html.style.set_table_styles(styles)
          .set_caption("Hover to highlight."))
html


# ## Numerical Variables

# In[ ]:


# Lets see the data types present in our dataset
df_train.dtypes.unique()


# In[ ]:


# Here data type 'O' represents Object which includes categorical variables,
# Get the list of numerical variables
numerical_features = [feature 
                          for feature in df_train.columns 
                              if df_train[feature].dtypes != 'O']

print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
df_train[numerical_features].head()


# In[ ]:


def color_negative_red(col):
    col1 = col
    color = 'green' if 'Yr' in col1 or 'Year'in col1 else 'black'  # here &lt corresponds to '<' and &gt for '>' is used html
    return 'color: %s' % color

pd.DataFrame(df_train.columns).style.applymap(color_negative_red)


# we can see that there are columns(marked in ***green***) which consists of date, lets convert those columns to date data type

# #### Temporal Variables(Eg: Datetime Variables)
# 
# From the Dataset we have 4 year variables. We have to extract information from the datetime variables like no of years or no of days. One example in this specific scenario can be difference in years between the year the house was built and the year the house was sold. We will be performing this analysis in the Feature Engineering.

# In[ ]:


# list of variables that contain year information
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]

year_feature


# In[ ]:


# let's explore the content of these year variables
for feature in year_feature:
    print(feature, df_train[feature].unique(),end = '\n\n',sep='\n')


# In[ ]:


df_train[year_feature].describe()


# In[ ]:


# Lets analyze the Temporal Datetime Variables
# We will check the relationship between these year features and SalesPrice 
# and see how price is changing over the years
# Here mean and median are almost same, for a safer side lets do analysis by taking a median
for yr_feature in year_feature:
    df_train.groupby(yr_feature)['SalePrice'].median().plot()
    plt.xlabel(yr_feature)
    plt.ylabel('Median House Price')
    plt.title("House Price vs {}".format(yr_feature))
    plt.show()


# ### Oberservation
# 
# 1. There is increase in price, if the year of built is in the period 1940-2000. But there is almost same price(though there are spikes in some years) in the period 1880-1940.
# 2. There is an increase in price if the house is modified recently
# 3. Over the period of 1950-2000 there is demand for garage,if the garage is present then the house price increased over the years

# In[ ]:


## Here we will compare the difference between All years feature and year sold with SalePrice

data=df_train.copy()
for feature in year_feature:
    if feature!='YrSold':
        ## We will capture the difference between year variable and year the house was sold for
        data[feature]=data['YrSold']-data[feature]

        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


# ### Observation
# 
# 1. we can see that as the difference between the year features and the year sold increases then the house price decreases exponentially
# 

# In[ ]:


## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables

discrete_features=[feature for feature in numerical_features if len(data[feature].unique())<25 and feature not in year_feature+['Id']]
print("Total Discrete Variables: {}".format(len(discrete_features)))


# In[ ]:


discrete_features


# In[ ]:


data[discrete_features].head()


# In[ ]:


data[discrete_features].describe()


# In[ ]:


## Lets Find the realtionship between discrete variables and SalePice

for feature in discrete_features:
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# ### Observation
# 
# 1. we can see from the plots above that Some discrete variables have relationship with the SalesPrice and some seems to be constant with the SalesPrice

# #### Continuous Variable

# In[ ]:


continuous_feature=[feature for feature in numerical_features if feature not in discrete_features+year_feature+['Id']]
print("Continuous feature Count {}".format(len(continuous_feature)))


# In[ ]:


## Lets analyse the continuous values by creating histograms to understand the distribution

for feature in continuous_feature:
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# In[ ]:


df_train[continuous_feature].head()


# In[ ]:


## We will be using logarithmic transformation

data = df_train.copy()
for feature in continuous_feature:
# Here we skip the feature if it has unique value 0 as the log(0) is undefined
    if 0 in data[feature].unique() or feature in ['SalePrice']: 
        pass
    else:
        data[feature]=np.log(data[feature])
        data['SalePrice']=np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalesPrice')
        plt.title(feature)
        plt.show()


# ### Outliers

# In[ ]:


data=df_train.copy()
for feature in continuous_feature:
    if 0 in data[feature].unique(): # here we pass if the unique values
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# ### Categorical Variables

# In[ ]:


categorical_features = [feature for feature in df_train.columns if data[feature].dtypes=='O']
categorical_features


# In[ ]:


df_train[categorical_features].head()


# In[ ]:


cat_count = []
for feature in categorical_features:
    cat_count.append(len(df_train[feature].unique()))

data_cat = {'Feature':categorical_features, 'No of Categories':cat_count} 


# In[ ]:


data_cat = pd.DataFrame(data_cat) 


# In[ ]:


data_cat


# In[ ]:


## Lets see the relationship between categorical variable and dependent feature SalesPrice
data=df_train.copy()
for feature in categorical_features:
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()

