#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import matplotlib.dates as dates
from datetime import datetime
import matplotlib.pyplot as plt
import math
import tensorflow as tf
     
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import scipy


import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


row_data = pd.read_csv('/kaggle/input/ecommerce-events-history-in-cosmetics-shop/2019-Nov.csv') #import the dataset


if (row_data.isnull().values.any() == True):  # checking missing values
    print(row_data.isnull().sum())
else: 
    print("There is not any null number")


# In[ ]:


df = row_data
visitors  = df['user_id'].nunique()
# print("Number of visitors : {}".format(visitors))
df[df['user_session'] == -1]


# In[ ]:


date = df.loc[:,['event_time','user_id']]
date['event_time'] = date['event_time'].apply(lambda d: str(d)[0:10]) # changing event time value to :(2019-10-01  - 2019-10-31) format
df


# In[ ]:


# Daily visitors number

visitor_by_date = date.groupby(['event_time'])['user_id'].nunique()
visitor_by_date


# In[ ]:


x = pd.Series(visitor_by_date.index.values).apply(lambda s: datetime.strptime(s, '%Y-%m-%d').date())
y = visitor_by_date[:]

print(type(visitor_by_date))
plt.rcParams['figure.figsize'] = (25,12)
plt.plot(x,y)
plt.show()


# In[ ]:


purchase  = df.loc[df['event_type'] == 'purchase'] #getting only purchase event type 
purchase = purchase.dropna(axis='rows') # dropping rows that have  at least one missing value

top_brands = purchase.groupby(['brand'])['brand'].agg(['count']).sort_values(by=['count'],ascending=False)
top_brands.head(25) # [samsung, apple, xiaomi, huawei, ...]


#Dropping remove_from_cart event type

index = df[df['event_type'] == 'remove_from_cart'].index
df.drop(index=index,inplace=True)
df['user_session'] = df['user_session'].astype('category').cat.codes


#  ## Preprocessing dataset
#  
# *  ### Creating df_targets dataframe & creating new features
# 

# In[ ]:


# df_targets = df.loc[df["event_type"].isin(["cart","purchase"])].drop_duplicates(subset=['event_type', 'product_id','price', 'user_id','user_session'])
# df_targets["is_purchased"] = np.where(df_targets["event_type"]=="purchase",1,0)
# df_targets["is_purchased"] = df_targets.groupby(["user_session","product_id"])["is_purchased"].transform("max")
# df_targets = df_targets.loc[df_targets["event_type"]=="cart"].drop_duplicates(["user_session","product_id","is_purchased"])
# df_targets['event_weekday'] = df_targets['event_time'].apply(lambda s: str(datetime.strptime(str(s)[0:10], "%Y-%m-%d").weekday()))
# df_targets.dropna(how='any', inplace=True)
# df_targets["category_code_level1"] = df_targets["category_code"].str.split(".",expand=True)[0].astype('category')
# df_targets["category_code_level2"] = df_targets["category_code"].str.split(".",expand=True)[1].astype('category')
# df_targets.head()


# In[ ]:


# cart_purchase_users = df.loc[df["event_type"].isin(["cart","purchase"])].drop_duplicates(subset=['user_id'])
# cart_purchase_users.dropna(how='any', inplace=True)
# cart_purchase_users_all_activity = df.loc[df['user_id'].isin(cart_purchase_users['user_id'])]


# In[ ]:


# activity_in_session = cart_purchase_users_all_activity.groupby(['user_session'])['event_type'].count().reset_index()
# activity_in_session = activity_in_session.rename(columns={"event_type": "activity_count"})


# In[ ]:


# del date #free memory


# In[ ]:


# df_targets = df_targets.merge(activity_in_session, on='user_session', how='right')
# df_targets['activity_count'] = df_targets['activity_count'].fillna(0)


# In[ ]:



# df_targets.head()


# In[ ]:


# df_targets.to_csv('training_data.csv')


# In[ ]:


# df_targets.info()


# In[ ]:


# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from xgboost import plot_importance
# from sklearn.utils import resample
# from sklearn import metrics


# is_purcahase_set = df_targets[df_targets['is_purchased']== 1]
# is_purcahase_set.shape[0]


# In[ ]:


# not_purcahase_set = df_targets[df_targets['is_purchased']== 0]
# not_purcahase_set.shape[0]


# In[ ]:


# n_samples = 537
# is_purchase_downsampled = resample(is_purcahase_set,
#                                 replace = False, 
#                                 n_samples = n_samples,
#                                 random_state = 27)
# not_purcahase_set_downsampled = resample(not_purcahase_set,
#                                 replace = False,
#                                 n_samples = n_samples,
#                                 random_state = 27)


# In[ ]:


# downsampled = pd.concat([is_purchase_downsampled, not_purcahase_set_downsampled])
# downsampled['is_purchased'].value_counts()


# In[ ]:


# features = downsampled.loc[:,['brand', 'price', 'event_weekday', 'category_code_level1', 'category_code_level2', 'activity_count']]


# In[ ]:


# features.loc[:,'brand'] = LabelEncoder().fit_transform(downsampled.loc[:,'brand'].copy())
# features.loc[:,'event_weekday'] = LabelEncoder().fit_transform(downsampled.loc[:,'event_weekday'].copy())
# features.loc[:,'category_code_level1'] = LabelEncoder().fit_transform(downsampled.loc[:,'category_code_level1'].copy())
# features.loc[:,'category_code_level2'] = LabelEncoder().fit_transform(downsampled.loc[:,'category_code_level2'].copy())

# is_purchased = LabelEncoder().fit_transform(downsampled['is_purchased'])
# features.head()


# In[ ]:


# print(list(features.columns))
# X_train, X_test, y_train, y_test = train_test_split(features, 
#                                                     is_purchased, 
#                                                     test_size = 0.3, 
#                                                     random_state = 0)


# 
# ## Gini Impurity
# 
# Used by the CART (classification and regression tree) algorithm for classification trees, Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. The Gini impurity can be computed by summing the probability {\displaystyle p_{i}}p_{i} of an item with label {\displaystyle i}i being chosen times the probability {\displaystyle \sum _{k\neq i}p_{k}=1-p_{i}}{\displaystyle \sum _{k\neq i}p_{k}=1-p_{i}} of a mistake in categorizing that item. It reaches its minimum (zero) when all cases in the node fall into a single target category.

# In[ ]:


# from xgboost import XGBClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from catboost import CatBoostClassifier, Pool


# dtf = DecisionTreeClassifier(criterion="gini", max_depth=None)
# xgb = XGBClassifier(learning_rate=0.1)
# rf  = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
# cb = CatBoostClassifier(learning_rate=0.03,
#                            eval_metric='MAE')


#  ## Accuracy

# In[ ]:


# def print_acc(model):
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#     print("Precision:",metrics.precision_score(y_test, y_pred))
#     print("Recall:",metrics.recall_score(y_test, y_pred))
#     print("fbeta:",metrics.fbeta_score(y_test, y_pred, average='weighted', beta=0.5 )  ,"\n")


    
# print_acc(dtf)
# print_acc(xgb)
# print_acc(rf)


# In[ ]:




df3 = pd.DataFrame([[1,12,111], [1, 13,111], [1, 14,111],[2,21,112] ,[2,22 ,112], [2,23,112] ], columns=['user_id', 'product_id','user_session'])
df3 = df3.sort_values(by=['user_session'])
user_session_values_3 = df3['user_session'].values
df3


# In[ ]:


def take_column_products(multiarr,a):
    array1 = []
    for i in range(len(multiarr)):
        array1.append(multiarr[i][a])
    return array1


# In[ ]:


unique_sessions = np.unique(user_session_values)
products_for_each_user = []

for i in unique_sessions:
    products_for_each_user.append(df.loc[df['user_session'] == i, 'product_id'].values)
     
print(products_for_each_user)


# In[ ]:



def create_product_cols(sessions,df):
    products_for_each_user = []
    
    
    for i in sessions:
        products_for_each_user.append(df.loc[df['user_session'] == i, 'product_id'].values)
     
    
    print(products_for_each_user)
        

        
    column1 = take_column_products(products_for_each_user,0) #product_id1
    column2 = take_column_products(products_for_each_user,1) #product_id2
    column3  = take_column_products(products_for_each_user,2) #product_id3
    column4  = take_column_products(products_for_each_user,3) #product_id3
    column5  = take_column_products(products_for_each_user,4) #product_id3
    column6  = take_column_products(products_for_each_user,5) #product_id3       
    column7  = take_column_products(products_for_each_user,6) #product_id3
    column8  = take_column_products(products_for_each_user,7) #product_id3
    column9  = take_column_products(products_for_each_user,8) #product_id3
    column10  = take_column_products(products_for_each_user,9) #product_id3
    column11  = take_column_products(products_for_each_user,10) #product_id3
    column12  = take_column_products(products_for_each_user,11) #product_id3
   # column13  = take_column_products(products_for_each_user,12) #product_id3

    
    
    
    
    data = {'product_id1': column1, 
        'product_id2': column2, 
        'product_id3': column3,
        'product_id4': column4,
        'product_id5': column5,
        'product_id6': column6,
        'product_id7': column7,
        'product_id8': column8,
        'product_id9': column9,
        'product_id10': column10,
        'product_id11': column11,
        'product_id12': column12,
        } 
    
    new_df = pd.DataFrame(data)
    
    
    
    
#     df.insert(3,'product_id1',column1 )
#     df.insert(4,'product_id2', column2)
#     df.insert(5, 'product_id3' , column3)
#     df.insert(6, 'product_id4' , column4)
#     df.insert(7, 'product_id5' , column5)
#     df.insert(8, 'product_id6' , column6)
#     df.insert(9, 'product_id7' , column7)
#     df.insert(10, 'product_id8' , column8)
#     df.insert(11, 'product_id9' , column9)
#     df.insert(12, 'product_id10' , column10)
#     df.insert(13, 'product_id11' , column11)
#     df.insert(14, 'product_id12' , column12)
   # df.insert(15, 'product_id13' , column13)
   
    
        
     
    return new_df


# In[ ]:


# #products_for_each_basket_df =  create_product_cols(user_session_values_3,df3)
# print(products_for_each_basket_df)


# new_df = products_for_each_basket_df[['product_id1', 'product_id2', 'product_id3']] 
# unique_sessions = np.unique(user_session_values_3)
# target_values = []

# for i in unique_sessions:
#     arr = products_for_each_basket_df[products_for_each_basket_df['user_session'] == i][['product_id1','product_id2','product_id3']].values
#     target_values.append(arr)
    
    
# target_values = np.hstack(target_values)

# products_for_each_basket_df.insert(5,'target',target_values )


# In[ ]:


# products_for_each_basket_df = products_for_each_basket_df.drop(columns=['product_id'],axis=1)

# products_for_each_basket_df


# In[ ]:


import numpy as np

target_values = []

def generate_target(df , frequent_products):
        for index, row in df.iterrows():
            for i in range(1,13):
                    if row['product_id'+  str(i)] in frequent_products:
                        target_values.append(row['product_id'+  str(i)])
        return target_values


def convert_product_to_Nan(df):
    for index, row in df.iterrows():
        for i in range(1,13):
            if row['product_id'+  str(i)] == row['target']:
               df.loc[index,'product_id' + str(i)] = np.nan
            
    return df

           
           
        


# In[ ]:





# ## Applying to whole dataset

# In[ ]:


df = df.drop_duplicates(['user_session','product_id'])

users_interactions_count_df = df.groupby(['user_id', 'product_id' , 'user_session']).size().groupby('user_session').size()  #Ask question about event_time or product_id
print("Number of users: %d" % len(users_interactions_count_df))

users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df = 12].reset_index()[['user_session']]
users_with_enough_interactions_df
print(users_with_enough_interactions_df)
print('amount of  users with  13 interactions: %d' % len(users_with_enough_interactions_df))
print(round(len(users_with_enough_interactions_df) * 100 / len(users_interactions_count_df), 2) , "%")


# In[ ]:


interactions_from_selected_users_df = df.merge(users_with_enough_interactions_df, 
                                           how = 'right',
                                           left_on = 'user_session',
                                           right_on = 'user_session'
                                          )
print(interactions_from_selected_users_df)

print('# of interactions: %d' % len(df))
print('# of interactions from users with at least 13 interactions: %d' % len(interactions_from_selected_users_df))
interactions_from_selected_users_df = interactions_from_selected_users_df.drop(['event_type','event_time','category_id','brand', 'price'],axis=1)
interactions_from_selected_users_df = interactions_from_selected_users_df.drop(['category_code'],axis=1)


# In[ ]:


interactions_from_selected_users_df['user_session'] = interactions_from_selected_users_df['user_session'].astype('category').cat.codes
interactions_from_selected_users_df[-13: -1]


# In[ ]:



interactions_from_selected_users_df.drop(interactions_from_selected_users_df[interactions_from_selected_users_df['user_session'] == 0].index,inplace=True)
interactions_from_selected_users_df.drop(interactions_from_selected_users_df[interactions_from_selected_users_df['user_session'] ==-1].index,inplace=True)

print(interactions_from_selected_users_df[-13 : -1])

interactions_from_selected_users_df = interactions_from_selected_users_df.sort_values(by=['user_session'])
print(interactions_from_selected_users_df)
user_session_values = interactions_from_selected_users_df['user_session'].values
user_session_values[-14:-1]



# In[ ]:


df_new =interactions_from_selected_users_df.groupby(['product_id','user_session'],sort=True)['product_id'].count()
df_new=interactions_from_selected_users_df[['product_id']].apply(pd.Series.value_counts)
df_new = df_new.loc[df_new['product_id'] >=10].reset_index()
df_new.columns = ("product_id", 'count_of_products')
df_new


# In[ ]:


from collections import Counter

frequent_products =df_new.product_id.values
print(len(frequent_products))
print(frequent_products)
[item for item, count in Counter(frequent_products).items() if count > 1]


# In[ ]:


frequent_products_sessions_df = interactions_from_selected_users_df.merge(df_new, 
                                           how = 'right',
                                           left_on = 'product_id',
                                           right_on = 'product_id'
                                          )



print(frequent_products_sessions_df[:150])


frequent_sessions = frequent_products_sessions_df['user_session']
frequent_sessions = frequent_sessions.sort_values(ascending=True)

for i in frequent_sessions:
    if len(frequent_sessions[ frequent_sessions ==i]) < 12:
        index = frequent_sessions[frequent_sessions == i].index[0]
        frequent_sessions.drop(index= index,inplace=True)
        
        


# In[ ]:


print(len(frequent_sessions))


# In[ ]:


prod_for_basket_whole_df = create_product_cols(frequent_sessions,interactions_from_selected_users_df)


# In[ ]:


prod_for_basket_whole_df 


# In[ ]:


# product_arr= []
# #prod_for_basket_whole_df.drop(columns=['product_id'],inplace=True)

# print(len(prod_for_basket_whole_df))
# for i in range(1,13):
#     product_arr.append('product_id' + str(i))

# unique_sessions = np.unique(user_session_values)

# target_values = []

# for i in unique_sessions:
#     arr = prod_for_basket_whole_df[prod_for_basket_whole_df['user_session'] == i][product_arr].values
#     arr = np.unique(arr)
#     target_values.append(arr)


# target_values = np.hstack(target_values)
# print(len(target_values))



# In[ ]:


target_values = generate_target(prod_for_basket_whole_df,frequent_products)
print(len(target_values))


# In[ ]:


target_values = np.reshape(target_values, (468, 12))

targets = set()
newlist = []
for item in target_values:
    t = tuple(item)
    if t not in targets:
        newlist.append(item)
        targets.add(t)

        
newlist = np.hstack(newlist)
prod_for_basket_whole_df['target'] = newlist

prod_for_basket_whole_df


# In[ ]:


prod_for_basket_whole_df = convert_product_to_Nan(prod_for_basket_whole_df)
prod_for_basket_whole_df


# In[ ]:


print(target_values)
# prod_for_basket_whole_df['target'] = target_values
# prod_for_basket_whole_df


# ## Last Result

# In[ ]:


last_df.isnull().sum()


# In[ ]:


def justify(a, invalid_val=0, axis=1, side='left'):    
 """
 Justifies a 2D array

 Parameters
 ----------
 A : ndarray
     Input array to be justified
 axis : int
     Axis along which justification is to be made
 side : str
     Direction of justification. It could be 'left', 'right', 'up', 'down'
     It should be 'left' or 'right' for axis=1 and 'up' or 'down' for axis=0.

 """

 if invalid_val is np.nan:
     #change to notnull
     mask = pd.notnull(a)
 else:
     mask = a!=invalid_val
 justified_mask = np.sort(mask,axis=axis)
 if (side=='up') | (side=='left'):
     justified_mask = np.flip(justified_mask,axis=axis)
 #change dtype to object
 out = np.full(a.shape, invalid_val, dtype=object)  
 if axis==1:
     out[justified_mask] = a[mask]
 else:
     out.T[justified_mask.T] = a.T[mask.T]
 return out                    


# In[ ]:


new_df = prod_for_basket_whole_df.iloc[:, 0:12]


df = pd.DataFrame(justify(new_df.values, invalid_val=np.nan, side='left', axis=1), 
                  columns=new_df.columns)


df


# In[ ]:


df.dropna(axis=1 , inplace=True) 

df = df.astype('int')
df


# In[ ]:


#df.info()
prod_for_basket_whole_df.target


# In[ ]:


from sklearn import preprocessing




X = df.values
y = prod_for_basket_whole_df.target.values





# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42   )
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]
print(number_of_train, number_of_test)

type(y)


# ## Classification

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score




# In[ ]:


from sklearn.model_selection import validation_curve 
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc





scores = []
neighbors= []

for n in range(1 , 5000, 50):
    knn = KNeighborsClassifier(n_neighbors=n)
    neighbors.append(n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score = accuracy_score(y_test,y_pred)
    scores.append(score)
    print("Accuracy in neighbor {0} : ".format(n) ,score)

    
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(neighbors, scores, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("Accuracy score")
plt.xlabel('n_neighbor')
plt.show()
    


# In[ ]:


import catboost as cb
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import accuracy_score


# In[ ]:


scores = []
params= []

for n in range(1 , 4000, 50):
    model = DecisionTreeClassifier(max_depth=n)
    params.append(n)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test,y_pred)
    scores.append(score)
    print("Accuracy in max_depth {0} : ".format(n) ,score)

    
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(params, scores, 'r', label="Test accuracy")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("Accuracy score")
plt.xlabel('n_neighbor')
plt.show()


# In[ ]:


#CatBoost 

model = CatBoostClassifier(iterations=1500, learning_rate=0.01,random_seed=42)
model.fit(X_train,y_train)
print('Model params:\n{}'.format(model.get_params()))
print('Predicted values')
train_predictions = model.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print("Accuracy: {:.4%}".format(acc))


# In[ ]:


items_by_basket = df.groupby("user_session")[["product_id"]].count()
desc = items_by_basket.describe()
desc


#mean = 3.60
#product_count  = mean + std  = 13
#at least 3


# In[ ]:


# labels = ['view', 'cart','purchase']
# size = df['event_type'].value_counts()
# explode = [0.25, 0.1,0.1] 

# plt.rcParams['figure.figsize'] = (15, 12)
# plt.pie(size, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
# plt.title('Percentages of event_types', fontsize = 24)
# plt.axis('off')
# plt.legend()
# plt.show()


# ## Data Mugging
# 
# As there are different interactions types, we associate them with a weight or strength, assuming that, for example, purchase of an item  indicates a higher interest of the user on the item than adding to cart, or than a simple view.

# In[ ]:


# #add ranking to event types 

# event_ranking   = {
#     'view' : 1.5,
#     'cart' : 3,
#     'purchase': 5
    
# }

# df['event_ranking'] = df['event_type'].apply(lambda x: event_ranking[x])

# df_targets = df.loc[df["event_type"].isin(["purchase"])].drop_duplicates(subset=['event_type', 'product_id',
#                                                                                          'user_id',
#                                                                                         'user_session'])

# print(df_targets.shape)
# df_targets.tail()
# df


# In[ ]:




# new_df = df.drop(columns=['brand','price','event_time','category_code','category_id','event_type'])[:100]
# # basket = new_df.groupby(['user_id','user_session','product_id'])[['event_ranking']].sum()
# basket = new_df.groupby(['user_session']).filter(lambda x: True)
# basket.duplicated(['user_session', 'product_id']).any()
# basket


# In[ ]:


# i, r = pd.factorize(basket['user_session'].values)
# # get integer factorization `j` and unique values `c`            
# # for column `'col'`
# j, c = pd.factorize(basket['product_id'].values)
# # `n` will be the number of rows
# # `m` will be the number of columns
# n, m = r.size, c.size
# # `i * m + j` is a clever way of counting the 
# # factorization bins assuming a flat array of length
# # `n * m`.  Which is why we subsequently reshape as `(n, m)`
# b = np.bincount(i * m + j, minlength=n * m).reshape(n, m)

# basket = pd.DataFrame(b, r, c)


# In[ ]:


pd.get_dummies(new_df['user_session']).T.dot(pd.get_dummies(new_df['product_id']))


# In[ ]:


df_targets.drop_duplicates(subset=['event_type', 'product_id', 'user_id', 'user_session']).shape[0]

df_targets["purchased"] = np.where(df_targets["event_type"]=="purchase",1,0)
print(df_targets.shape)
df_targets["purchased"] = df_targets.groupby(["user_session","product_id"])["purchased"].transform("max")
df_targets


# 

# ## Cold start problem
# 
# Recommender systems have a problem known as user cold-start, in which is hard do provide personalized recommendations for users with none or a very few number of consumed items, due to the lack of information to model their preferences.
# For this reason, we are keeping in the dataset only users with at leas 2 interactions.

# ## Interactions from users with  at least 5 interactions

# ## Smoothing the distrubition
# 
#  Aggregating all the interactions the user has performed in an item by a weighted sum of interaction type strength and apply a log transformation to smooth the distribution.

# In[ ]:


def smooth_user_preference(x):
    return math.log(1+x, 2)


interactions_full_df = interactions_from_selected_users_df                     .groupby(['user_id', 'product_id'])['event_ranking'].sum()                     .apply(smooth_user_preference).reset_index()

print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_full_df.head(10)


# ## Evaluation 
# 
# Using here a simple **cross-validation** approach named **holdout**, in which a random data sample (20% in this case) are kept aside in the training process, and exclusively used for evaluation. All evaluation metrics reported here are computed using the **test set.**

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                   stratify=interactions_full_df['user_id'],                       
                                   test_size=0.20,
                                   random_state=42)

print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))


# In Recommender Systems, there are a set metrics commonly used for evaluation. We chose to work with Top-N accuracy metrics, which evaluates the accuracy of the top recommendations provided to a user, comparing to the items the user has actually interacted in test set.
# This evaluation method works as follows:
# 
# * For each user
#     * For each item the user has interacted in test set
#     
#     1. Sample 100 other items the user has never interacted.
#        Ps. Here we naively assume those non interacted items are not relevant to the user, which        might not be true, as the user may simply not be aware of those not interacted items.
#   
#     2.   Ask the recommender model to produce a ranked list of recommended items, from a set              composed one interacted item and the 100 non-interacted ("non-relevant!) items
#     
#     3.   Compute the Top-N accuracy metrics for this user and interacted item from the                    recommendations ranked list
# * Aggregate the global Top-N accuracy metrics
# 
# 
# The Top-N accuracy metric choosen was **Recall@N** which evaluates whether the interacted item is among the top N items (hit) in the ranked list of 101 recommendations for a user.
# Ps. Other popular ranking metrics are** NDCG@N** and** MAP@N**, whose score calculation takes into account the position of the relevant item in the ranked list (max. value if relevant item is in the first position).

# In[ ]:


#Indexing by user_id to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index('user_id')
interactions_train_indexed_df = interactions_train_df.set_index('user_id')
interactions_test_indexed_df = interactions_test_df.set_index('user_id')

interactions_train_indexed_df.head(10)
interactions_test_indexed_df.head(10)


# In[ ]:


def get_items_interacted(person_id,df):
    interacted_items = df.loc[person_id]['product_id']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])


# In[ ]:


# #Top-N accuracy metrics consts
# EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

# class ModelEvaluator:


#     def get_not_interacted_items_sample(self, user_id, sample_size, seed=42):
#         interacted_items = get_items_interacted(user_id, interactions_full_indexed_df)
#         all_items = set(df[:30000]['product_id'])
#         non_interacted_items = all_items - interacted_items

#         random.seed(seed)
#         non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
#         return set(non_interacted_items_sample)

#     def _verify_hit_top_n(self, item_id, recommended_items, topn):        
#             try:
#                 index = next(i for i, c in enumerate(recommended_items) if c == item_id)
#             except:
#                 index = -1
#             hit = int(index in range(0, topn))
#             return hit, index

#     def evaluate_model_for_user(self, model, user_id):
#         #Getting the items in test set
#         interacted_values_testset = interactions_test_indexed_df.loc[user_id]
        
#         if type(interacted_values_testset['product_id']) == pd.Series:
#             person_interacted_items_testset = set(interacted_values_testset['product_id'])
#         else:
#             person_interacted_items_testset = set([int(interacted_values_testset['product_id'])])  
#         interacted_items_count_testset = len(person_interacted_items_testset) 

#         #Getting a ranked recommendation list from a model for a given user
#         person_recs_df = model.recommend_items(user_id, 
#                                                items_to_ignore=get_items_interacted(user_id, 
#                                                                                     interactions_train_indexed_df), 
#                                                topn=10000000000)

#         hits_at_5_count = 0
#         hits_at_10_count = 0
#         #For each item the user has interacted in test set
#         for item_id in person_interacted_items_testset:
#             #Getting a random sample (100) items the user has not interacted 
#             #(to represent items that are assumed to be no relevant to the user)
#             non_interacted_items_sample = self.get_not_interacted_items_sample(user_id, 
#                                                                           sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
#                                                                           seed=item_id%(2**32))

#             #Combining the current interacted item with the 100 random items
#             items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

#             #Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
#             valid_recs_df = person_recs_df[person_recs_df['product_id'].isin(items_to_filter_recs)]                    
#             valid_recs = valid_recs_df['product_id'].values
#             #Verifying if the current interacted item is among the Top-N recommended items
#             hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
#             hits_at_5_count += hit_at_5
#             hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
#             hits_at_10_count += hit_at_10

#         #Recall is the rate of the interacted items that are ranked among the Top-N recommended items, 
#         #when mixed with a set of non-relevant items
#         recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
#         recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

#         person_metrics = {'hits@5_count':hits_at_5_count, 
#                           'hits@10_count':hits_at_10_count, 
#                           'interacted_count': interacted_items_count_testset,
#                           'recall@5': recall_at_5,
#                           'recall@10': recall_at_10}
#         return person_metrics

#     def evaluate_model(self, model):
#         #print('Running evaluation for users')
#         people_metrics = []
#         for idx, user_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
#             #if idx % 100 == 0 and idx > 0:
#             #    print('%d users processed' % idx)
#             person_metrics = self.evaluate_model_for_user(model, user_id)  
#             person_metrics['_user_id'] = user_id
#             people_metrics.append(person_metrics)
#         print('%d users processed' % idx)

#         detailed_results_df = pd.DataFrame(people_metrics) \
#                             .sort_values('interacted_count', ascending=False)
        
#         global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
#         global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
#         global_metrics = {'modelName': model.get_model_name(),
#                           'recall@5': global_recall_at_5,
#                           'recall@10': global_recall_at_10}    
#         return global_metrics, detailed_results_df
    
# model_evaluator = ModelEvaluator()


# ## Popularity model
# 
# A common (and usually hard-to-beat) baseline approach is the Popularity model. This model is not actually personalized - it simply recommends to a user the most popular items that the user has not previously consumed. As the popularity accounts for the "wisdom of the crowds", it usually provides good recommendations, generally interesting for most people.

# In[ ]:


# item_popularity_df = interactions_full_df.groupby('product_id')['event_ranking'].sum().sort_values(ascending=False).reset_index()
# item_popularity_df.head(10)


# In[ ]:


# class PopularityRecommender:
    
#     MODEL_NAME = 'Popularity'
    
#     def __init__(self, popularity_df, items_df=None):
#         self.popularity_df = popularity_df[]
#         self.items_df = items_df
        
#     def get_model_name(self):
#         return self.MODEL_NAME
        
#     def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
#         # Recommend the more popular items that the user hasn't seen yet.
#         recommendations_df = self.popularity_df[~self.popularity_df['product_id'].isin(items_to_ignore)] \
#                                .sort_values('event_ranking', ascending = False) \
#                                .head(topn)

#         if verbose:
#             if self.items_df is None:
#                 raise Exception('"items_df" is required in verbose mode')

#             recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
#                                                           left_on = 'product_id', 
#                                                           right_on = 'product_id')[['event_ranking', 'product_id']]


#         return recommendations_df
    
# popularity_model = PopularityRecommender(item_popularity_df, df)


# In[ ]:


# print('Evaluating Popularity recommendation model...')
# pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
# print('\nGlobal metrics:\n%s' % pop_global_metrics)
# pop_detailed_results_df.head(10)


# In[ ]:



# pivot ratings into movie features
df_features = interactions_train_df[:30000].pivot(
    index='product_id',
    columns='user_id',
    values='event_ranking'
).fillna(0)
# convert dataframe features to scipy sparse matrix


# In[ ]:


df_features.transpose()


# In[ ]:



df_features_matrix = df_features.as_matrix()
df_features_matrix[:10]


# In[ ]:


user_ids  = list(df_features.index)
user_ids[:10]


# In[ ]:


users_item_pivot_sparse_matrix = csr_matrix(df_features_matrix)
users_item_pivot_sparse_matrix


# In[ ]:


#The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 15
#Performs matrix factorization of the original user item matrix
#U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
U, sigma, Vt = svds(users_item_pivot_sparse_matrix, k = NUMBER_OF_FACTORS_MF)


# In[ ]:


U.shape


# In[ ]:


sigma = np.diag(sigma)
sigma.shape


# In[ ]:


Vt.shape


# In[ ]:


# training_dataset = (
#     tf.data.Dataset.from_tensor_slices(
#         (
#             tf.cast(df_features.values, tf.float32),
#         )
#     )
# )

# for x in training_dataset:
#     print (x)


# In[ ]:


all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
all_user_predicted_ratings


# In[ ]:


all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())


# In[ ]:


#Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns = df_features.columns, index=user_ids).transpose()


# In[ ]:


# class CFRecommender:
    
#     MODEL_NAME = 'Collaborative Filtering'
    
#     def __init__(self, cf_predictions_df, items_df=df[:30000]):
#         self.cf_predictions_df = cf_predictions_df
#         self.items_df = items_df
        
#     def get_model_name(self):
#         return self.MODEL_NAME
        
#     def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
#         # Get and sort the user's predictions
#         sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
#                                     .reset_index().rename(columns={user_id: 'recStrength'})

#         # Recommend the highest predicted rating movies that the user hasn't seen yet.
#         recommendations_df = sorted_user_predictions[ sorted_user_predictions['product_id'].isin(items_to_ignore)] \
#                                .sort_values('recStrength', ascending = False) \
#                                .head(topn)

#         if verbose:
#             if self.items_df is None:
#                 raise Exception('"items_df" is required in verbose mode')

#             recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
#                                                           left_on = 'product_id', 
#                                                           right_on = 'product_id')[['recStrength','product_id']]


#         return recommendations_df
    
# cf_recommender_model = CFRecommender(cf_preds_df, df[:10000])


# In[ ]:


len(cf_preds_df.columns)


# In[ ]:


# print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
# cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)
# print('\nGlobal metrics:\n%s' % cf_global_metrics)
# cf_detailed_results_df.head(10)


# In[ ]:


tensor = tl.tensor([df_features.values])


# In[ ]:


from tensorly.decomposition import tucker
from tensorly import unfold
unfolded_tensor = unfold(tensor, 1)
unfolded_tensor[:5]


# In[ ]:


core , factors = tucker(unfolded_tensor,ranks=[2,3]) 
core.shape


# In[ ]:


len(factors)


# In[ ]:


core


# In[ ]:


[f  for f in factors]


# In[ ]:


from surprise import SVD
from surprise import Dataset
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split
from surprise import Reader
from surprise import accuracy
from surprise import NormalPredictor
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold

interactions_full_df = interactions_full_df[:1000]

reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(interactions_full_df[['user_id', 'product_id', 'event_ranking']], reader)

cross_validate(NormalPredictor(), data, cv=3)


# In[ ]:


kf = KFold(n_splits=60)

algo = SVD()

for trainset, testset in kf.split(data):

    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)


# In[ ]:


param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])


# In[ ]:


interactions_full_df[interactions_full_df['user_id']  == 10280338]


# In[ ]:


from collections import defaultdict


def get_top_n(predictions, n=5):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# First train an SVD algorithm on the  dataset.
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(interactions_full_df[['user_id', 'product_id', 'event_ranking']], reader)

trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Then predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
# for i in testset:
#     if i == (10280338,5875289,2.084533813998394):
#         testset.remove(i)
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=10)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print( uid, [iid  for (iid, _) in user_ratings])

