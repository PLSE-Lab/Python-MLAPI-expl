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


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as matplotlib
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams["figure.figsize"] = (20,10)


# In[ ]:


data = pd.read_csv("../input/bengaluru-house-price-data/Bengaluru_House_Data.csv")
data.head()


# In[ ]:


data.shape


# In[ ]:


data.columns


# ### Now let us first examine our first feature that is "area_type".

# ### We will be printing the count of data samples in each of the categories of the area_type feature.

# In[ ]:


data.groupby('area_type')['area_type'].agg('count')


# ### Now we are here proceeding for a simple model and hence we will drop some of the features we think don't have any effect on the price, which is our target variable.

# ### Thus, we are going to drop 'area_type', 'society' and 'availability'. We will assume that these features will not be having any effect on our target variable.

# In[ ]:


drop_columns = ['area_type', 'availability', 'society']
df = data.drop(drop_columns, axis = 1)
df.head()


# # ***DATA CLEANING PROCESS***

# * ##  Handling the null values.

# In[ ]:


df.isnull().sum(), df.shape


# ### Since we have 13,320 rows we can safely drop the rows having null values. 

# In[ ]:


df2 = df.dropna()
df2.isnull().sum(), df2.shape


# ### Now, if we see there are no NULL values which can be seen above. Also we see the size of the new dataset obtained after dropping the NULL values.

# In[ ]:


df2.head()


# ### Now if we observe the column 'size' we see that some values have values in 'x BHK' while some like 'x Bedroom'. Let us see how data entries in this column behave.

# In[ ]:


df2['size'].unique()


# ### So in order to tackle this problem, we will be creating a new column called 'bhk', which will show how many bedrooms are present in the house.

# In[ ]:


df2['bhk'] = df2['size'].apply(lambda x: (int)(x.split(' ')[0]))


# ### Now we can see that a new column has been added in our dataset.

# In[ ]:


df2.head()


# ### Also we can drop the column 'size' as it is of no use to us right now.

# In[ ]:


df3 = df2.drop(['size'], axis =1)
df3.head()


# In[ ]:


df3['bhk'].unique()


# ### We can see that some of the estate have as many as 43 bedrooms in it which clearly indicates it as an outlier. Let us observe more about these type of outliers.

# In[ ]:


df3[df3.bhk > 20]


# ### As we can see that the estate with 43 bedrooms has an area of 2400 sq. ft. which is absurd.

# ### To look more into these type of errors, let us explore the column 'total_sqft' into detail.

# In[ ]:


df3['total_sqft'].unique()


# ### We can see that sometimes we don't get a single value but a range of values like '1133 - 1384' which can be seen above.

# ### So let's remove these anomalies by taking the mean of the range and also check for other abnormalities. We will do that by checking whether the data entry can be converted into float type or not.

# In[ ]:


def isFloat(x):
    try:
        float(x)
    except:
        return False
    return True
    


# In[ ]:


df3[~( df3['total_sqft'].apply(isFloat) )].head(15)


# ### We can see apart from the range we also have other types of data entries, such as 34.46Sq. Meter in row 410 and so on.

# ### So in order to handle these non uniformities in data, we will be taking the mean if the entry is of range type and for any other type of entry in this column we will just simply drop the corresponding row.

# In[ ]:


def convertSqftToNum(x):
    values = x.split('-')
    if len(values)==2:
        return (float(values[0])+float(values[1]))/2
    try:
        return float(x)
    except:
        return None
    


# In[ ]:


df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convertSqftToNum)
df4.head(10)


# #  ***FEATURE ENGINEERING***

# In[ ]:


df5 = df4.copy()


# ### Now we all know that in the real estate business, price per sq ft of the estate is of utmost significance.

# ### Hence, we will be adding a new feature 'price_per_sqft' to observe our dataset into more details and also it will help us in cleansing our dataset of outliers and other anomalies.

# In[ ]:


df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']


# ### Since our price was given in lakhs, thus we have multiplied 100000 in the price of each state, which can be seen above.

# In[ ]:


df5.head()


# ### Now let us explore the column 'location' and see how many unique locations are present in our dataset and how many estates are in a particular location.

# In[ ]:


len(df5.location.unique())


# ### This is a very large number for locations. Now we convert the textual data in numerical by using 'ONE-HOT ENCODING'.

# ### But if we will consider all the locations,then we will be having more than 1250 columns, which will not be convenient. This increases the dimensionality too much, and hence we need to reduce the dimensions.

# ### So what we will do is that we will define a 'other' column, which will include all those locations where the data entries are only '1' and '2'.

# In[ ]:


#Let us first remove the leading white spaces from the location names.
df5.location = df5.location.apply(lambda x: x.strip())

location_statistics = df5.groupby('location')['location'].agg('count').sort_values(ascending = False)
location_statistics


# In[ ]:


location_statistics[location_statistics<10]


# In[ ]:


len(location_statistics[location_statistics<10])


# In[ ]:


location_less_than_10 = location_statistics[location_statistics<10]
len(df5.location.unique())


# In[ ]:


df5.location = df5.location.apply(lambda x: 'other' if x in location_less_than_10 else x)
len(df5.location.unique())


# In[ ]:


df5.head(20)


# ### Now let us look at the columns 'bath' and 'balcony' and filter out the anomalies related to them.

# In[ ]:


df5.groupby('balcony')['balcony'].agg('count')


# In[ ]:


df5.shape


# In[ ]:



df5[(df5.total_sqft/df.balcony < 300)]


# In[ ]:



df6 = df5[~(df5.total_sqft/df5.balcony < 300)]
df6.shape


# In[ ]:


df6[df6.total_sqft/df5.bhk < 300]


# In[ ]:


df7 = df6[~(df6.total_sqft/df5.bhk < 300)]
df7.shape


# ### After balcony and bath, let us now look at 'price_per_sqft' in detail. 

# In[ ]:


df7.price_per_sqft.describe()


# ### So we can see that the minimum and the maximum values for the column 'price_per_sqft' can be safely classified as outliers.

# ### Therefore we will be removing these extreme values on the basis of mean and standard deviation.

# ### Now mean and standard deviation will be calculated per location as some locations will have low prices while some will have higher prices.

# In[ ]:


def remove_extremes(df):
    df_out = pd.DataFrame()
    for key, dff in df.groupby('location'):
        m = np.mean(dff.price_per_sqft)
        sd = np.std(dff.price_per_sqft)
        reduced_df = dff[(dff.price_per_sqft>(m-sd)) & (dff.price_per_sqft<=(m+sd))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out
    
df8 = remove_extremes(df7)
df8.shape


# In[ ]:


df8.head()


# In[ ]:


df8.groupby('location')['location'].describe()


# ### Also if we look into our dataset we will find that some of the 2 bhk estates have more price than 3 bhk estates where total sq_ft area for both of them is same. It is a anomaly which we need to fix.
# 

# In[ ]:


pd.concat([df8[df8.bhk==2], df8[df8.bhk==3]], ignore_index= True)[35:70]
#df6[df6.total_sqft/df5.bhk < 300]


# ### In the index 63 and 64, 2 bhk has a price of 65 lakhs while 3 bhk has a price of 60 lakhs and total area sq_ft of both the estates are almost same.

# ### Therefore, we need to improvise again and remove such anomalies from our dataset. But first let's look at a scatter plot per location based on this ambiguity to identify which points are to be removed.

# In[ ]:


def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft, bhk2.price, marker='*', color='red', label='2 bhk', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='black', label='3 bhk', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price in Lakhs")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df8,"Rajaji Nagar")


# ### We can see from the above graph that price of 2bhk estate is higher than 3bhk when area is about 1700 sq ft.

# In[ ]:


plot_scatter_chart(df8,"Kanakpura Road")


# In[ ]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats={}
        
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
            
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    
    return df.drop(exclude_indices, axis='index')


df9 = remove_bhk_outliers(df8)
df9.shape
            


# ### Now plotting the same scatter plot which we plotted above but this time with the new dataset.

# In[ ]:


plot_scatter_chart(df9,"Rajaji Nagar")


# In[ ]:


plot_scatter_chart(df9,"Kanakpura Road")


# ### We can see that with the new dataset the anomalies have reduced to a very minimal value.

# ### Now let us plot a histogram to observe the number of estates in a particular range of price per sq ft.

# In[ ]:


matplotlib.rcParams['figure.figsize'] = (20,10)
plt.hist(df9.price_per_sqft, rwidth = 0.75)
plt.xlabel('Price per Square Feet')
plt.ylabel('Count')


# ### Also let us explore the bathroom feature or column 'bath' in our dataset.

# In[ ]:


df9.bath.unique()


# In[ ]:


df9[df9.bath>=10]


# In[ ]:


plt.hist(df9.bath, rwidth=0.75)
plt.xlabel('Number of Bathrooms')
plt.ylabel('Count')


# In[ ]:


df9[df9.bath>df9.bhk+2]


# In[ ]:


df10 = df9[df9.bath<df9.bhk+2]
df10.shape


# ### Now we can drop our column 'price_per_sqft' feature as we have exhausted its use. We used it to detect the outliers and did some data cleaning with its help. Now the column has no use for us and hence we can drop it.

# In[ ]:


df11 = df10.drop(['price_per_sqft'], axis = 'columns')
df11.head(), df11.shape


# ### Also we can see that the 'location' column of our dataset has a textual representaion and since it is categorical we can convert it into a numerical column.

# ### One of the ways to convert a text column into a numerical column is 'ONE-HOT ENCODING'. It is also called dummies method.

# In[ ]:


dummy = pd.get_dummies(df11.location)
dummy.head(10)


# In[ ]:


df12 = pd.concat([df11, dummy], axis='columns')
df12.head(10)


# In[ ]:


df13 = df12.drop(['other'], axis = 1)
df13.head(10)


# In[ ]:


df14 = df13.drop(['location'], axis = 1)
df14.head()


# In[ ]:


df14.shape


# In[ ]:


X = df14.drop(['price'], axis = 'columns')
X.head()


# In[ ]:


y = df13.price
y.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)


# ### Let us first try running our model on **Linear Regression algorithm** and see how the results pan out.

# In[ ]:


from sklearn.linear_model import LinearRegression
lrm = LinearRegression()
lrm.fit(X_train, y_train)
lrm.score(X_test, y_test)


# ## Typically, one should try a couple of different models with different parameters to come up with the best optimal model for prediction of price.

# ### And we are also going to do the same.

# ## We will first use a k-fold cross validation method for prediction.

# In[ ]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
cross_val_score(LinearRegression(), X, y, cv = cv)


# ### We can see that the majority of the time, we get a score of more than 80%.

# ### We also see that when we run Linear Regression on five full cross validation, we get a decent score. But we should also try other regression techniques.

# ## So we will be using Grid search CV method for using other regression techniques. It's an API from sklearn which can be used to run our model on different regressors and get the optimal algorithm for prediction.

# In[ ]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_optimum_model(X,y):
    algos = {
        'linear regression' : {
            'model' : LinearRegression(),
            'params' : {
                'normalize' : [True, False]
            }
        },
        
        'lasso': {
            'model' : Lasso(),
            'params' : {
                'alpha' : [1, 2],
                'selection' : ['random', 'cyclic']
            }
        },
        
        'decision_tree' : {
            'model' : DecisionTreeRegressor(),
            'params' : {
                'criterion' : ['mse', 'friedman_mse'],
                'splitter' : ['best', 'random']
            }
        }
        
    }
    
    scores = []
    cv  = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv = cv, return_train_score = False)
        gs.fit(X,y)
        
        scores.append(
            {
            'model' : algo_name,
            'best_score' : gs.best_score_,
            'best_params' : gs.best_params_
        }
        )
        
    return pd.DataFrame(scores, columns = ['model', 'best_score', 'best_params'])



find_optimum_model(X,y)        
        


# ## Since we can see from the above comparison, linear regression is giving the best and optimum results. So we will use linear regression only for the prediction of price for unknown data.

# In[ ]:


def predict(location, sqft, bath, balcony, bhk):
    location_index = np.where(X.columns == location)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = balcony
    x[3] = bhk
    
    if location_index >= 0:
        x[location_index] = 1
        
    return lrm.predict([x])[0]


# In[ ]:


predict('1st Phase JP Nagar', 1200, 2, 1, 3)


# In[ ]:


predict('1st Phase JP Nagar', 1200, 3, 1, 4)


# In[ ]:


predict('Indira Nagar', 1200, 2, 1, 3)


# In[ ]:


predict('Indira Nagar', 1200, 3, 1, 4)


# # Exporting our model

# In[ ]:


import pickle
with open('Bangalore_House_Prices_Model.pickle', 'wb') as f:
    pickle.dump(lrm, f)


# In[ ]:


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))

