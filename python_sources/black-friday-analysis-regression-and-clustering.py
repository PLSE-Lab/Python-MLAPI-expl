#!/usr/bin/env python
# coding: utf-8

# ## Description
# We are going to analyse data of black friday transactions of a retail store. Dataset is provided by [Analytics Vidhya](http://analyticsvidhya.com).   
# 
# Key questions we will be answering are:
# * How are transactions distributed over different age groups, occupations and cities.
# * Which groups of people have a higher transaction number and which have higher purchase amount?
# * How does marital status and years of living in the city affect number and amount of purchases?
# 
# and in the end, we will create a model to predict purchase amount based on other features.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


# In[ ]:


data: pd.DataFrame = pd.read_csv('../input/BlackFriday.csv')
describe = data.describe()
describe.loc['#unique'] = data.nunique()
display(describe)


# ## Data Overview
# Dataset has 537577 rows (transactions) and 12 columns (features) as described below:
# * `User_ID`: Unique ID of the user. There are a total of 5891 users in the dataset.
# * `Product_ID`: Unique ID of the product. There are a total of 3623 products in the dataset.
# * `Gender`: indicates the gender of the person making the transaction.
# * `Age`: indicates the age group of the person making the transaction.
# * `Occupation`: shows the occupation of the user, already labeled with numbers 0 to 20.
# * `City_Category`: User's living city category. Cities are categorized into 3 different categories 'A', 'B' and 'C'.
# * `Stay_In_Current_City_Years`: Indicates how long the users has lived in this city.
# * `Marital_Status`: is 0 if the user is not married and 1 otherwise.
# * `Product_Category_1` to `_3`: Category of the product. All 3 are already labaled with numbers.
# * `Purchase`: Purchase amount.

# Now, before getting to our questions, let's get some insights about the dataset and transactions.

# In[ ]:


purchase_desc = data['Purchase'].describe()
purchase_desc.drop(['count', 'std'], inplace=True)
purchase_desc.loc['sum'] = data['Purchase'].sum()
purchase_desc.loc['mean_by_user'] = data['Purchase'].sum() / data['User_ID'].nunique()
display(pd.DataFrame(purchase_desc).T)


# Mean purchase amount by transaction is 9333 and mean amount by each user is about 850,000. Values are probably not in USD.

# In[ ]:


null_percent = (data.isnull().sum() / len(data))*100
display(pd.DataFrame(null_percent[null_percent > 0].apply(lambda x: "{:.2f}%".format(x)),columns=['Null %']))


# Only `Product_Category_2` and `Product_Category_3` have null values which is good news. However `Product_Category_3` is null for nearly 70% of transactions so it can't give us much information.

# In[ ]:


cat_describe = data[['Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category', 'Marital_Status', 'Product_Category_1']].astype('object').describe()
cat_describe.loc['percent'] = 100*cat_describe.loc['freq'] / cat_describe.loc['count']
display(cat_describe)


# A basic observation is that:
# * Product `P00265242` is the most popular product.
# * Most of the transactions were made by men.
# * Age group with most transactions was `26-35`.  
# 
# but we will cover each of these in more depth later.

# ## Feature Analysis

# #### Gender
# So we begin with `Gender`. Let's see how much men and women have purchased and how many transactions they have done.

# In[ ]:


plt.figure(figsize=(13, 6))
gender_gb = data[['Gender', 'Purchase']].groupby('Gender').agg(['count', 'sum'])
params = {
#     'colors': [(255/255, 102/255, 102/255, 1), (102/255, 179/255, 1, 1)],
    'labels': gender_gb.index.map({'M': 'Male', 'F': 'Female'}),
    'autopct': '%1.1f%%',
    'startangle': -30, 
    'textprops': {'fontsize': 15},
    'explode': (0.05, 0),
    'shadow': True
}
plt.subplot(121)
plt.pie(gender_gb['Purchase']['count'], **params)
plt.title('Number of transactions', size=17)
plt.subplot(122)
plt.pie(gender_gb['Purchase']['sum'], **params)
plt.title('Sum of purchases', size=17)
plt.show()


# Men have had transactions about 3 times higher than women in black friday. They've also had proportionaly higher purchase amount, which leads to the assumption that there is no meaningful difference between mean purchase amounts of men and women. Let's check that.

# In[ ]:


gender_gb = data[['Gender', 'Purchase']].groupby('Gender', as_index=False).agg('mean')
sns.barplot(x='Gender', y='Purchase', data=gender_gb)
plt.ylabel('')
plt.xlabel('')
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.title('Mean purchase amount by gender', size=14)
plt.show()


# Our assumption was correct and we can't say either men have purchased more expensive products than women or other way around.

# #### Age
# Let's see what we can observe from `Age` groups provided in dataset

# In[ ]:


plt.figure(figsize=(16, 8))
plt.subplot(121)
sns.countplot(y='Age', data=data, order=sorted(data.Age.unique()))
plt.title('Number of transactions by age group', size=14)
plt.xlabel('')
plt.ylabel('Age Group', size=13)
plt.subplot(122)
age_gb = data[['Age', 'Purchase']].groupby('Age', as_index=False).agg('mean')
sns.barplot(y='Age', x='Purchase', data=age_gb, order=sorted(data.Age.unique()))
plt.title('Mean purchase amount by age group', size=14)
plt.xlabel('')
plt.ylabel('')
plt.show()


# People within the ages of 26 to 35 have purchased the most (in number and amount), and as we saw about gender, people in different ages have nearly same mean purchase amount, too.
# 
# Let's check what products were most popular in each age group.

# In[ ]:


age_product_gb = data[['Age', 'Product_ID', 'Purchase']].groupby(['Age', 'Product_ID']).agg('count').rename(columns={'Purchase': 'count'})
age_product_gb.sort_values('count', inplace=True, ascending=False)
ages = sorted(data.Age.unique())
result = pd.DataFrame({
    x: list(age_product_gb.loc[x].index)[:5] for x in ages
}, index=['#{}'.format(x) for x in range(1,6)])
display(result)


# #### Occupation

# In[ ]:


men = data[data.Gender == 'M']['Occupation'].value_counts(sort=False)
women = data[data.Gender == 'F']['Occupation'].value_counts(sort=False)
pd.DataFrame({'M': men, 'F': women}, index=range(0,21)).plot.bar(stacked=True)
plt.gcf().set_size_inches(10, 4)
plt.title("Count of different occupations in dataset (Separated by gender)", size=14)
plt.legend(loc="upper right")
plt.xlabel('Occupation label', size=13)
plt.ylabel('Count', size=13)
plt.show()


# Observation is that people occupied in job labels 0, 4 and 7 have purchased the most in black friday.
# 
# Let's check what products people from different occupations were most interested in:

# In[ ]:


import random
color_mapping = {}
def random_color(val):    
    if val in color_mapping.keys():
        color = color_mapping[val]
    else:
        r = lambda: random.randint(0,255)
        color = 'rgba({}, {}, {}, 0.4)'.format(r(), r(), r())
        color_mapping[val] = color
    return 'background-color: %s' % color

occ_product_gb = data[['Occupation', 'Product_ID', 'Purchase']].groupby(['Occupation', 'Product_ID']).agg('count').rename(columns={'Purchase': 'count'})
occ_product_gb.sort_values('count', inplace=True, ascending=False)
result = pd.DataFrame({
    x: list(occ_product_gb.loc[x].index)[:5] for x in range(21)
}, index=['#{}'.format(x) for x in range(1,6)])
display(result.style.applymap(random_color))


# Table above represents top 5 seller products categorized by the user occupation (same products have the same background color).
# 1. First thing you notice is that `P00265242` is the most-purchased product for 15 out of 21 occupations and an interesting fact is that this product is not even present in top-5 products of occupations 8, 10 and 17. I wonder what this product is and what these occupations are.
# 2. Second interesting thing about this illustration is how similar the first 4 occupations' top-5 are.
# 3. Third and last interesting fact from these charts: from top 5 products of occupation 9, one of them is `P00265242` and present in most of other top 5s, one of them is only present in occupation 16's list and the rest are not repeated in any other lists. Adding to account the fact that we saw from previous chart, this was the only occupation with more women than men (even though the totall number of men in dataset was higher), makes occupation 9 a unique occupation among the list.

# #### City Category and City Stability
# Cities are categorized in 3 different categories A, B and C. We have living city of each user in the time of transaction and also we know for how long had the user been in that city, by that time.  
# 
# Let's explore number of transactions in each city.

# In[ ]:


stay_years = [data[data.Stay_In_Current_City_Years == x]['City_Category'].value_counts(sort=False).iloc[::-1] for x in sorted(data.Stay_In_Current_City_Years.unique())]

f, (ax1, ax2) = plt.subplots(1,2, gridspec_kw = {'width_ratios':[1, 2]}, sharey=True)

years = sorted(data.Stay_In_Current_City_Years.unique())
pd.DataFrame(stay_years, index=years).T.plot.bar(stacked=True, width=0.3, ax=ax1, rot=0, fontsize=11)
ax1.set_xlabel('City Category', size=13)
ax1.set_ylabel('# Transactions', size=14)
ax1.set_title('# Transactions by city (separated by stability)', size=14)

sns.countplot(x='Stay_In_Current_City_Years', data=data, ax=ax2, order=years)
ax2.set_title('# Transactions by stability', size=14)
ax2.set_ylabel('')
ax2.set_xlabel('Years in current city', size=13)

plt.gcf().set_size_inches(15, 6)
plt.show()


# People living in city category of `B` have had most transactions to this store, following by categories `C` and `B` with relatively close values.
# Those who have been in their living city for 1 year had double the number of transactions than any other stay durations, and then comes the people living in their city for 2 years, 3 years, 4+ years and 0 years (<1 year). The pattern is the same within each city category as well.  
# Seems like people living their second year in a city tend to shop more than others.

# #### Marital Status

# In[ ]:


out_vals = data.Marital_Status.value_counts()
in_vals = np.array([data[data.Marital_Status==x]['Gender'].value_counts() for x in [0,1]]).flatten()

fig, ax = plt.subplots(figsize=(7, 7))

size = 0.3
cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(2)*4)
inner_colors = cmap(np.array([1, 2, 5, 6]))

ax.pie(out_vals, radius=1, colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'), labels=['Single', 'Married'],
       textprops={'fontsize': 15}, startangle=50)

ax.pie(in_vals, radius=1-size, colors=inner_colors,
       wedgeprops=dict(width=size, edgecolor='w'), labels=['M', 'F', 'M', 'F'],
       labeldistance=0.75, textprops={'fontsize': 12, 'weight': 'bold'}, startangle=50)

ax.set(aspect="equal")
plt.title('Marital Status / Gender', fontsize=16)
plt.show()


# Single people have purchased more than married people and in both categories men, following the general pattern of dataset, have purchased more than women.

# #### Best sellers
# Which products sold the most and which categories contain most-sold products? We will only use `Product_Category_1` since the other two have alot of null values.
# Also, let's see which users have purchased the most.

# In[ ]:


col_names = ['Product_ID', 'Product_Category_1', 'User_ID']
renames = ['Product', 'Category', 'User']
results = []
for col_name, new_name in zip(col_names, renames):
    group = data[[col_name, 'Purchase']].groupby(col_name, as_index=False).agg('count')
    result = group.sort_values('Purchase', ascending=False)[:10]
    result.index = ['#{}'.format(x) for x in range(1,11)]
    results.append(result.rename(columns={col_name: new_name}))    
    
from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline; padding-right: 3em !important;"'),raw=True)
display_side_by_side(*results)


# 
# ## Building models
# 
# Now, let's build some models to predict purchase amount based on other features. 
# 
# #### Feature engineering
# All of our features are categorical features.  
# All variables are already encoded except `Product_ID` and `User_ID` which we will encode using LabelEncoder.  
# We will also remove `Product_Category_2` and `_3` since they have a high rate of null values, and rename some of our remaining features to easier names.   
# Other categorical features have a few number of unique values so we will encode them using OneHotEncoder.
# 

# In[ ]:


train = data.drop(['Product_Category_2', 'Product_Category_3'], axis=1)            .rename(columns={
                'Product_ID': 'Product',
                'User_ID': 'User',
                'Product_Category_1': 'Category',
                'City_Category': 'City',
                'Stay_In_Current_City_Years': 'City_Stay'
})
y = train.pop('Purchase')

train.loc[:, 'Gender'] = np.where(train['Gender'] == 'M', 1, 0)


# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV

# Label encoding Product ID and User ID
for col in ['Product', 'User']:    
    train.loc[:, col] = LabelEncoder().fit_transform(train[col])
        
# One hot encoding other features
categoricals = ['Occupation', 'Age', 'City', 'Gender', 'Category', 'City_Stay']
encoder = OneHotEncoder().fit(train[categoricals])
train = pd.concat([train, pd.DataFrame(encoder.transform(train[categoricals]).toarray(), index=train.index, columns=encoder.get_feature_names(categoricals))], axis=1)
train.drop(categoricals, axis=1, inplace=True)


# Next, we will split our train and test data and will standardize the data using StandardScaler

# In[ ]:


# Splitting train and test sets
X_train, X_test, y_train, y_test = train_test_split(train, y)

# Standardizing
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)


# Let's build and fit our models.
# 
# #### RandomForestRegressor
# What we are trying to do here is to build a model to predict purchase amount (`Purchase` column in dataset) from other features, namely `User_ID`, `City_Category`, `Age`, etc.  
# This is a regression problem and we'll use random forest. We'll choose number of trees (`n_estimators`) in our forest and `max_depth` for each tree by calculating scores for each combination and choosing the best one. Scoring metric we'll use is RMSE.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

params = {
    'n_estimators': [10, 30, 100, 300],
    'max_depth': [3, 5, 7]
}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid=params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
preds = grid_search.predict(X_test)
print("Best params found: {}".format(grid_search.best_params_))
print("RMSE score: {}".format(mean_squared_error(y_test, preds) ** 0.5))


# Let's plot learning curve to address possible under/over fitting

# In[ ]:


sizes, train_scores, test_scores = learning_curve(RandomForestRegressor(**grid_search.best_params_), X_train, y_train, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
train_scores = np.mean((-1*train_scores)**0.5, axis=1)
test_scores = np.mean((-1*test_scores)**0.5, axis=1)
sns.lineplot(sizes, train_scores, label="Train")
sns.lineplot(sizes, test_scores, label="Test")
plt.xlabel("Size of training set", size=13)
plt.ylabel("Round Mean Squared Error", size=13)
plt.show()


# What features are more important to our model?

# In[ ]:


model = grid_search.best_estimator_
impo = pd.Series(model.feature_importances_[:10], index=train.columns[:10]).sort_values()
impo_plot = sns.barplot(x=impo.index, y=impo.values)
for item in impo_plot.get_xticklabels():
    item.set_rotation(50)
plt.gcf().set_size_inches(8, 4)
plt.title("Top 10 most important features", size=14)
plt.show()


# `Product_ID` is having a significant impact on our model and we have `User_ID` in second place with sharp decrease.  
# We could predict purchase amount with a relatively small error (about half of the mean purchase value) but our model is highly relied on `Product_ID` and `User_ID` which means probably it won't perform as well for new products/users. Thus, it may not be a good model for predicting future sales for future customers and products but it can be used for other purposes like describing which products tend to be better options to advertise or to give vouchers for.

# #### Clustering (K-Means)
# Now, let's try and see if we can cluster our customers (`User_ID`s) based on products they have purchased and other features.   
# We need to change our features a little bit to be more suitable for clustering. Firstly, we are going to cluster users and not the transactions so we should group our transactions by `User_ID` and create our features based on that.   
# Now, what we know about each user, is his/her gender, living city, stay in current city years, age, marital status, occupation and products (s)he has bought. Let's exclude products he has bought and their categories for now, we will get to them later.   

# In[ ]:


train = data.drop(['Product_ID', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'Purchase'], axis=1).groupby('User_ID')


# In terms of other features (gender, city, stay in city years, age, marital status and occupation) their values should all be the same in all rows for a particular `User_ID` but we assume there might be small noises and some of them may have wrong values in different rows, so we will take the mode value (one which is repeated the most) for each of them.   
# On the other hand, they are all nominal features so we will one hot encode them since there are not that much unique values.

# In[ ]:


train = train.agg(lambda x: x.value_counts().index[-1])
feaures_list = list(train.columns.values)
encoder = OneHotEncoder().fit(train[feaures_list])
train = pd.concat([train, pd.DataFrame(encoder.transform(train[feaures_list]).toarray(), index=train.index, columns=encoder.get_feature_names(feaures_list))], axis=1)
train.drop(feaures_list, axis=1, inplace=True)


# Now in regard to products, we can't add a column for each product and make it 1 if the user has bought it or 0 otherwise (due to huge number of products) so we'll do it for only top 100 products (by number of transactions).  
# Note that not only we know which user has bought which product, but we can also use purchase amounts as a metric for how much does this user likes/needs this product. We'll do the same thing for product categories but with all of them as features since there are no more than 18 categories.
# 
# So, in conclusion, we are going to find 100 most selling products and 18 categories (by number of transactions) and for each user, put purchase amount of this product/product category as a new feature for him, adding totally 118 new features to our data.  

# In[ ]:


columns = ['Product_ID', 'Product_Category_1']
for column in columns:
    top_100 = data[column].value_counts().index[:100]    
    user_purchase = pd.pivot_table(
        data[['User_ID', column, 'Purchase']],
        values='Purchase',
        index='User_ID',
        columns=column,
        aggfunc=np.sum
    ).fillna(0)[top_100]  
    train = train.join(user_purchase)


# Note that we filled null values with 0 since they might be some NaN values (not all users have bought all top 100 products).   
# Let's also add total purchase amount for each user as a new feature:

# In[ ]:


train = train.join(data[['User_ID', 'Purchase']].groupby('User_ID').agg('sum'))


# We standardize data in columns so features will have same scales in order to get clustered.

# In[ ]:


train_scaled = StandardScaler().fit_transform(train)


# Now we should choose number of clusters (K). We will use elbow method to choose one. So let's plot different distance sums for different number of clusters:

# In[ ]:


from sklearn.cluster import KMeans

k_values = np.arange(1, 11)
models = []
dists = []
for k in k_values:
    model = KMeans(k).fit(train_scaled)
    models.append(model)
    dists.append(model.inertia_)

plt.figure(figsize=(9, 6))
plt.plot(k_values, dists, 'o-')
plt.ylabel('Sum of squared distances', size=13)
plt.xlabel('K', size=13)
plt.xticks(k_values)
plt.show()


# So we'll go with 3 clusters.

# In[ ]:


from sklearn.metrics import silhouette_score

model = models[2]
print("Silhouette score: {:.2f}".format(silhouette_score(train_scaled, model.predict(train_scaled))))


# 
# ## Conclusion
# We described the dataset and features, did some exploratory data analysis and got some facts and points about dataset. Then we built a Random Forest Regressor to predict purchase amount based on user id, product id and other features available in dataset. Our model had RMSE around mean of the target value which might be good but it was strongly based on product_ids which limits our usage of model as described above.  
# Finally we clustered our customers using K-Means algorithm with 3 clusters and a silhouette score of 0.09 which shows maybe we can't group users confidently since most of them take place near the decision boundaries rather than our centroids. 
# 
