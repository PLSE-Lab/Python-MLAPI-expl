#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


apps_df = pd.read_csv('../input/googleplaystore.csv')
reviews = pd.read_csv('../input/googleplaystore_user_reviews.csv')


# # Initial data exploration

# In[ ]:


apps_df.head(10)


# In[ ]:


reviews.head()


# In[ ]:


apps_df['Rating'].describe()


# In[ ]:


extraordinary_app = apps_df[apps_df['Rating'] > 5.]
extraordinary_app.head()


# In[ ]:


extraordinary_app_reviews = reviews[reviews['App'] == extraordinary_app.iloc[0]['App']]


# In[ ]:


extraordinary_app_reviews.head()


# It seems that the row have some missplaced values. Let's remove the row.

# In[ ]:


apps_df = apps_df.drop(apps_df[apps_df['Rating'] > 5].index, axis=0)


# In[ ]:


apps_df['Rating'].isnull().sum()


# Rating is one of the essential stats about the app. We're going to drop all rows without ratings.

# In[ ]:


apps_df = apps_df.drop(apps_df[apps_df['Rating'].isnull()].index, axis=0)


# In[ ]:


apps_df['Rating'].describe()


# In[ ]:


apps_df['Rating'].hist(bins=50)


# In[ ]:


plt.figure(figsize=(18,10))
apps_df['Category'].value_counts().plot(kind='bar')


# In[ ]:


apps_df['Content Rating'].value_counts().plot(kind='bar')


# In[ ]:


apps_df['Type'].value_counts().plot(kind='bar')


# In[ ]:


apps_df['Price'] = apps_df['Price'].apply(lambda x: float(x.replace('$', ''))).astype('float64')


# In[ ]:


plt.figure(figsize=(18,10))
apps_df[apps_df['Price'] > 0.]['Price'].hist(bins=100)


# In[ ]:


apps_df['Genres'].value_counts()


# It seems that there are multiple genres listed in Genres column. We need to split it it.

# In[ ]:


apps_df['Genres'].apply(lambda x: len(x.split(';'))).value_counts()


# In[ ]:


apps_df['Main Genre'] = apps_df['Genres'].apply(lambda x: x.split(';')[0])
apps_df['Sub Genre'] = apps_df['Genres'].apply(lambda x: x.split(';')[1] if len(x.split(';')) > 1 else 'no sub genre')


# In[ ]:


plt.figure(figsize=(18,10))
apps_df['Main Genre'].value_counts().plot(kind='bar')


# In[ ]:


apps_df[apps_df['Sub Genre'] != 'no sub genre']['Sub Genre'].value_counts().plot(kind='bar')


# In[ ]:


plt.figure(figsize=(18,10))
apps_df['Installs'].value_counts().plot(kind='bar')


# While Google does not give away exact information about how many installs an app have, we can access to the amount of reviews, and based on that we can roughly (but better than Google-given information) estimate the installs.

# In[ ]:


apps_df['Reviews'] = apps_df['Reviews'].astype('int64')


# In[ ]:


plt.figure(figsize=(18,10))
apps_df['Reviews'].hist(bins=100)


# The plot turned out to be highly uninformative because of very large values at X scale (1 step is 1e7=10,000,000 reviews), and most of apps on that scale go very close to 0. Let's make several subplots for different categories.

# First, let's take a look at reviews distribution for apps where there are 0-1000, 0-100,000, 0-1,000,000 reviews, and then look at more higher spans.

# In[ ]:


spans = [1000, 10000, 100000, 1000000, 10000000, 100000000]

plt.figure(figsize=(18, 4 * len(spans)))
prev=0
for i, span in enumerate(spans):
    plt.subplot(len(spans), 1, i+1)
    subset = apps_df[(apps_df['Reviews'] > prev) & (apps_df['Reviews'] < span)]
    subset['Reviews'].hist(bins=100)
    plt.title("{:,}".format(prev) + ' - ' + "{:,}".format(span))
    prev=span


# Now let's look at reviews distribution for each category in 'Installs'

# In[ ]:


installs_categories = apps_df['Installs'].value_counts().index
installs_categories_list = [(x, int(x.replace(',', '').replace('+', ''))) for x in installs_categories]
sorted_installs = sorted(installs_categories_list, key= lambda x: x[1])

plt.figure(figsize=(18, 5 * len(sorted_installs)))
for i, installs in enumerate(sorted_installs):
    plt.subplot(len(sorted_installs), 1, i+1)
    subset = apps_df[apps_df['Installs'] == installs[0]]
    subset['Reviews'].hist(bins=100)
    plt.title("Installs: "+installs[0])


# In[ ]:


apps_df['Installs (int)'] = apps_df['Installs'].apply(lambda x: int(x.replace(',', '').replace('+', '')))


# In[ ]:


sns.lmplot("Installs (int)", "Reviews", data=apps_df, aspect=2)
ax = plt.gca()
_ = ax.set_title('Overall correlation between installs and reviews')


# In[ ]:


sns.lmplot("Installs (int)", "Reviews", data=apps_df, aspect=2, hue='Type')


# It seems that there differences in correlation between installs and the amount of reviews for free apps.

# In[ ]:


apps_df.corr()['Reviews']


# In[ ]:


apps_df[apps_df['Type'] == 'Free'].corr()['Reviews']


# In[ ]:


apps_df[apps_df['Type'] == 'Paid'].corr()['Reviews']


# In[ ]:


sns.lmplot("Installs (int)", "Reviews", data=apps_df[apps_df['Type'] == 'Paid'], aspect=2)


# In[ ]:


sns.lmplot("Installs (int)", "Reviews", data=apps_df[(apps_df['Type'] == 'Paid') & (apps_df['Installs (int)'] < 1e7)], aspect=2)


# In[ ]:


apps_df[(apps_df['Type'] == 'Paid') & (apps_df['Installs (int)'] < 1e7)].corr()['Reviews']


# Even though correlation between Installs and Reviews for Paid application was stronger with whole dataset rather than with dataset, limited on Installs by 10mil, the limited dataset in that matter seems to have a better distribution and less outstanding values.

# In[ ]:


apps_under10mil_installs_df = apps_df[apps_df['Installs (int)'] < 1e7]


# # Target Installs, Rating, Reviews
# 
# The most important things about an app are Installs, Rating and Reviews in that order. As a publisher, we are interested in our product to reach bigger audience, with better ratings and optionally with more feedbacks (as reviews).
# 
# Let's explore correlations about those 3 columns.

# If we are to take all possible correlations into the picture, we have to one-hot-encode all rlevant categorical data, such as Category, Content Rating, Main Genre, Sub Genre.

# In[ ]:


columns_to_encode = ['Category', 'Main Genre', 'Content Rating']

for col in columns_to_encode:
    def get_prefix(col):
        if col == 'Main Genre':
            return 'Genre'
        else:
            return col
    
    col_labels_ctg = apps_under10mil_installs_df[col].astype('category')
    col_dummies = pd.get_dummies(col_labels_ctg, prefix=get_prefix(col))
    
    apps_under10mil_installs_df = pd.concat([apps_under10mil_installs_df, col_dummies], axis=1)
    del apps_under10mil_installs_df[col]


# In[ ]:


subgenres = set(list(apps_df[apps_df['Sub Genre'] != 'no sub genre']['Sub Genre']))

for subgenre in subgenres:
    col_name = 'Genre_' + subgenre
    apps_under10mil_installs_df[col_name] = apps_under10mil_installs_df['Sub Genre'].apply(
        lambda x: 1 if x == subgenre else 0)
    
del apps_under10mil_installs_df['Sub Genre']


# In[ ]:


def get_strongest_correlations(col, num):
    corrs = apps_under10mil_installs_df.corr()[col]
    max_num = len(list(corrs))
    if num > max_num:
        num = max_num
        print ('Features limit exceeded. Max number of features: ', max_num)
        
    corrs = corrs.drop(col)
    idx = list(corrs.abs().sort_values(ascending=False).iloc[:num].index)
    return corrs[idx], idx


# In[ ]:


installs_corrs, _ = get_strongest_correlations('Installs (int)', 20)
plt.figure(figsize=(18,10))
installs_corrs.plot(kind='bar')


# In[ ]:


reviews_corrs, _ = get_strongest_correlations('Reviews', 20)
plt.figure(figsize=(18,10))
reviews_corrs.plot(kind='bar')


# Installs correlate strongly with reviews amount, and that is already explored correlation. There are some genres and categories that got some correlations with installs, but nothing strong (<0.1).

# In[ ]:


rating_corrs, _ = get_strongest_correlations('Rating', 20)
plt.figure(figsize=(18,10))
rating_corrs.plot(kind='bar')


# Ratings positively correlate with reviews. It seems natural that on average people would like to send review for a better app, and this is the strongest correlation.

# Overall, there is no strong correlation that could indicate that a particular app will have better rating or more install. The overall quality of the app is seems to be the most obvious and strongest predictor. Another good predictor is a budget that was set for adds campaign.

# # Predicting rating of the app.
# Let's try to predict rating of the app. We'll use random forest regressor to try to predict it.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[ ]:


train = apps_under10mil_installs_df.sample(frac=0.8)
test_and_validation = apps_under10mil_installs_df.loc[~apps_under10mil_installs_df.index.isin(train.index)]
validation = test_and_validation.sample(frac=0.5)
test = test_and_validation.loc[~test_and_validation.index.isin(validation.index)]

print(train.shape, validation.shape, test.shape)


# In[ ]:


def get_features(num_features):
    col_to_predict = 'Rating'
    rating_corrs, idx = get_strongest_correlations(col_to_predict, num_features)
    if col_to_predict in idx:
        idx.remove(col_to_predict)
    return idx

def compare_predictions(predicted, test_df, target_col):
    check_df = pd.DataFrame(data=predicted, index=test_df.index, columns=["Predicted "+target_col])
    check_df = pd.concat([check_df, test_df[[target_col]]], axis=1)
    check_df["Error, %"] = np.abs(check_df["Predicted "+target_col]*100/check_df[target_col] - 100)
    check_df['Error, val'] = check_df["Predicted "+target_col] - check_df[target_col]
    return (check_df.sort_index(), check_df["Error, %"].mean())

def evaluate_predictions(model, train_df, test_df, features, target_col):
    train_pred = model.predict(train_df[features])
    train_rmse = mean_squared_error(train_pred, train_df[target_col]) ** 0.5

    test_pred = model.predict(test_df[features])
    test_rmse = mean_squared_error(test_pred, test_df[target_col]) ** 0.5

    print("RMSEs:")
    print(train_rmse, test_rmse)
    
    return test_pred


# In[ ]:


def rfr_model_evaluation(num_features=30, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_leaf_nodes=None, use_test=False):
    rfr = RandomForestRegressor(random_state=42, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_leaf_nodes=max_leaf_nodes)
    features = get_features(num_features)
    rfr.fit(train[features], train['Rating'])
    if use_test:
        rfr_test_predictions = evaluate_predictions(rfr, train, test, features, 'Rating')
        check_df, avg_error = compare_predictions(rfr_test_predictions, test, 'Rating')
        print("Average test error:", avg_error)
    else:
        rfr_validation_predictions = evaluate_predictions(rfr, train, validation, features, 'Rating')
        check_df, avg_error = compare_predictions(rfr_validation_predictions, validation, 'Rating')
        print("Average validation error:", avg_error)
    return check_df, avg_error


# In[ ]:


check, error = rfr_model_evaluation()


# Outstanding performance! Algorithm managed to generalize on 30 features with max correlation of about 0.15! Let's try to improve it further.

# In[ ]:


num_features_list = [1] + [x for x in range(5, 91, 5)] + [96]
max_depth_list = [None] + [x for x in range(3, 11)] + [x for x in range(15, 36, 5)]
min_samples_split_list = [x for x in range(2, 11)] + [x for x in range(15, 101, 5)]
min_samples_leaf_list = [x for x in range(1, 15)] + [0.001] + list(np.linspace(0.005,0.1,20).round(3))
min_weight_fraction_leaf_list = [0., 0.001] + list(np.linspace(0.005, 0.3, 30).round(3))
max_leaf_nodes_list = [None] + [x for x in range(5, 101, 5)]
n_estimators = [x for x in range(10, 220, 20)]

hyperparams = {
    'num_features': num_features_list,
    'max_depth': max_depth_list,
    'min_samples_split': min_samples_split_list,
    'min_samples_leaf': min_samples_leaf_list,
    'min_weight_fraction_leaf': min_weight_fraction_leaf_list,
    'max_leaf_nodes': max_leaf_nodes_list,
    'n_estimators': n_estimators
}

validation_results = []
for hp_name, hp_list in hyperparams.items():
    errors = []
    for hp_val in hp_list:
        if hp_name == 'num_features':
            _, error = rfr_model_evaluation(num_features=hp_val)
        elif hp_name == 'max_depth':
            _, error = rfr_model_evaluation(max_depth=hp_val)
        elif hp_name == 'min_samples_split':
            _, error = rfr_model_evaluation(min_samples_split=hp_val)
        elif hp_name == 'min_samples_leaf':
            _, error = rfr_model_evaluation(min_samples_leaf=hp_val)
        elif hp_name == 'min_weight_fraction_leaf':
            _, error = rfr_model_evaluation(min_weight_fraction_leaf=hp_val)
        elif hp_name == 'max_leaf_nodes':
            _, error = rfr_model_evaluation(max_leaf_nodes=hp_val)
        elif hp_name == 'n_estimators':
            _, error = rfr_model_evaluation(n_estimators=hp_val)
            
        errors.append(error)
    validation_results.append((hp_name, errors))


# In[ ]:


fig = plt.figure(figsize=(18, 30))

for i, result in enumerate(validation_results):
    ax = fig.add_subplot(len(validation_results), 1, i+1)
    hp_name = result[0]
    hp_errors = result[1]
    
    ax.set_title(hp_name)
    ax.plot(range(0, len(hp_errors)), hp_errors)
    plt.sca(ax)
    x_labels = hyperparams[hp_name]
    plt.xticks(range(0, len(hp_errors)), x_labels)
    
fig.tight_layout()
plt.show()


# In[ ]:


check, error = rfr_model_evaluation(num_features=96, n_estimators=110, max_depth=10, min_samples_split=45,
                                    min_samples_leaf=14, min_weight_fraction_leaf=0.005, max_leaf_nodes=75)


# Now we have ~10% error rate on validation set. Final assesment against test set:

# In[ ]:


check, error = rfr_model_evaluation(num_features=96, n_estimators=110, max_depth=10, min_samples_split=45,
    min_samples_leaf=14, min_weight_fraction_leaf=0.005, max_leaf_nodes=75, use_test=True)


# In[ ]:




