#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams["figure.figsize"] = (20,10)


# In[ ]:


df1 = pd.read_csv("../input/bengaluru-house-price-data/Bengaluru_House_Data.csv")
df1.head()


# In[ ]:


df1.shape


# In[ ]:


df1.groupby('area_type')['area_type'].agg('count')


# In[ ]:


#Remove unnecessary columns 
df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
df2.head()


# #identify the null values
# df2.isnull().sum

# In[ ]:


df2.isnull().sum()


# In[ ]:


df3 = df2.dropna()
df3.isnull().sum()


# In[ ]:


df3.shape


# In[ ]:


#Finding the unique values and get only numeric values 
df3['size'].unique()


# In[ ]:


#Create bhk column to store values
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[ ]:


df3.head()


# In[ ]:


df3['bhk'].unique()


# In[ ]:


df3[df3.bhk > 20]


# In[ ]:


df3.total_sqft.unique()


# In[ ]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
    


# In[ ]:


df3[~df3['total_sqft'].apply(is_float)].head()


# In[ ]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[ ]:


convert_sqft_to_num('21345')


# In[ ]:


convert_sqft_to_num('2100-2456')


# In[ ]:


convert_sqft_to_num('34.56sq.Meter')


# In[ ]:


df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4.head(3)


# In[ ]:


df4.loc[30]


# In[ ]:


(2100+2850)/2


# In[ ]:


df4.head(3)


# In[ ]:


df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()


# In[ ]:


#dimentionality problems ....1 location has number of times 
len(df5.location.unique())


# In[ ]:


df5.location = df5.location.apply(lambda x: x.strip())

location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


# In[ ]:


len(location_stats[location_stats<=10])


# In[ ]:


location_stats_lessthan_10 = location_stats[location_stats<=10]
location_stats_lessthan_10


# In[ ]:


len(df5.location.unique())


# In[ ]:


#convert all the less than 10 locatioons to other...
df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_lessthan_10 else x )
len(df5.location.unique())


# In[ ]:


df5.head(10)


# In[ ]:


#600/6   sqft/bhk normally 300 sqft for 1 removing ouliers

df5[df5.total_sqft/df5.bhk<300].head()


# In[ ]:


df5.shape


# In[ ]:


#removing ouliers and copy into 6
df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape


# In[ ]:


df6.price_per_sqft.describe()


# In[ ]:


#find meand and sd
def remove_pps_outlier(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st= np.std(subdf.price_per_sqft)
        reduce_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <=(m+st))]
        df_out = pd.concat([df_out,reduce_df],ignore_index=True)
    return df_out
df7 = remove_pps_outlier(df6)
df7.shape


# In[ ]:


def plot_scatter_chart(df,location):   
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")


# In[ ]:


plot_scatter_chart(df7,"Hebbal")


# In[ ]:


get_ipython().run_cell_magic('HTML', '', "We should also remove properties where for same location, the price of (for example) 3 bedroom apartment is less than 2 bedroom apartment (with same square ft area). What we will do is for a given location, we will build a dictionary of stats per bhk, i.e.\n\n{\n    '1' : {\n        'mean': 4000,\n        'std: 2000,\n        'count': 34\n    },\n    '2' : {\n        'mean': 4300,\n        'std: 2300,\n        'count': 22\n    },    \n}\nNow we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment")


# In[ ]:



def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
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
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape


# In[ ]:


##Plot same scatter chart again to visualize price_per_sqft for 2 BHK and 3 BHK properties
plot_scatter_chart(df8,"Rajaji Nagar")


# In[ ]:


plot_scatter_chart(df8,"Hebbal")


# In[ ]:



import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[ ]:


#Outlier Removal Using Bathrooms Feature
df8.bath.unique()


# In[ ]:



plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[ ]:


df8[df8.bath>10]


# In[ ]:


#It is unusual to have 2 more bathrooms than number of bedrooms in a home

df8[df8.bath>df8.bhk+2]


# In[ ]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[ ]:


df9.head(2)


# In[ ]:


#drop for size and price per sqft
df10 = df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)


# In[ ]:


#Use One Hot Encoding For Location

dummies = pd.get_dummies(df10.location)
dummies.head(3)


# In[ ]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()


# In[ ]:


#drop location 

df12 = df11.drop('location',axis='columns')
df12.head(2)


# In[ ]:


#Build a Model Now...
df12.shape


# In[ ]:


#take price into y rest is x axis

X = df12.drop(['price'],axis='columns')
X.head(3)


# In[ ]:


X.shape


# In[ ]:


Y = df12.price
Y.head(3)


# In[ ]:


len(Y)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=10)


# In[ ]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[ ]:


#Use K Fold cross validation to measure accuracy of our LinearRegression model

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, Y, cv=cv)


# In[ ]:


#We can see that in 5 iterations we get a score above 80% all the time. This is pretty good but we want to test few other algorithms for regression to see if we can get even better score. We will use GridSearchCV for this purpose


# In[ ]:


#Find best model using GridSearchCV

#Find best model using GridSearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,Y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,Y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,Y)


# In[ ]:


#Based on above results we can say that LinearRegression gives the best score. Hence we will use that.

#Test the model for few properties

def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[ ]:


predict_price('1st Phase JP Nagar',1000, 2, 2)


# In[ ]:


predict_price('1st Phase JP Nagar',1000, 3, 3)


# In[ ]:


predict_price('Indira Nagar',1000, 2, 2)


# In[ ]:


predict_price('Indira Nagar',1000, 3, 3)


# In[ ]:


#Export the tested model to picke file..
import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# In[ ]:


#Export location and column information to a file that will be useful later on in our prediction application
import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))

