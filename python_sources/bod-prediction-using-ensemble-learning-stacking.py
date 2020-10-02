#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


get_ipython().system('ls ../input/bod-results/')


# # Part 0 : Description of the used approaches and corresponding scores.

# In[ ]:


results = pd.read_csv('../input/bod-results/BOD Kaggle_results.csv',usecols=['Methodology','RMSE','Change log'])


# In[ ]:


pd.set_option('display.max_colwidth', -1)


# In[ ]:


results['Methodology'] = results['Methodology'].str.replace('\n','')


# In[ ]:


results


# # Part 1 : EDA

# # Reading of data and basic statistics

# In[ ]:


df = pd.read_csv('../input/prediction-bod-in-river-water/train.csv')


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


corr =df[df.columns.to_list()[1:]].corr()


# In[ ]:


corr


# From the plot below it's obvious that the biggest correlation is between first two features and our target value.

# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# In[ ]:


len(df)


# We won't need an Id column for our predictions.

# In[ ]:


df.drop(columns=['Id'],inplace=True)


# We will also analyse the distribution of our dataset w.r.t target value.

# In[ ]:


sns.set()
plt.figure(figsize=(14,10))
sns.distplot(df['target'])
plt.title('Distribution of data points in the dataset');


# # Imputation of missing values

# Let's check our dataset for missing values. It's obvious that the feature columns from 3 to 7 
# consist of mostly NaN values. Thus it's better to drop them.

# In[ ]:


df.isna().sum()


# In[ ]:


df.columns.to_list()[:3]


# In[ ]:


df = df[df.columns.to_list()[:3]]


# In[ ]:


df.isna().sum()


# We still have some NaN values in first two features, thus we will impute them using KNN.

# In[ ]:


nan_cols = df.isna().sum()[df.isna().sum()>0].index.to_list()


# In[ ]:


from sklearn.impute import KNNImputer


# In[ ]:


imputer = KNNImputer(n_neighbors=5)


# In[ ]:


df[nan_cols] = imputer.fit_transform(df[nan_cols])


# In[ ]:


df


# In[ ]:


df.max()


# In[ ]:


df.min()


# In[ ]:


corr = df[df.columns.to_list()].corr()


# In[ ]:


corr


# # Feature engineering

# Let's also create a new feature which is just an average of the availbale ones.

# In[ ]:


df.columns.to_list()[1:3]


# In[ ]:


df['combined'] = df[df.columns.to_list()[1:]].mean(axis=1)


# In[ ]:


df.head()


# We see that the correlation between new feature and target value is also big enough, thus we will use it for models training.

# In[ ]:


corr = df[df.columns.to_list()].corr()
display(corr)
plt.figure(figsize=(10,8))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# # Part 2 : first level models training

# # Making gbd model and training 

# First of all we will train a gbd model.

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


# In[ ]:


from sklearn.metrics import mean_squared_error,make_scorer


# In[ ]:


from sklearn.model_selection import cross_val_score, train_test_split


# In[ ]:


from skopt import forest_minimize


# As the score is validated using rmse, we will implement it and use as a scorer for cross validation.

# In[ ]:


def rmse(y,y_pred):
    return np.sqrt(mean_squared_error(y_true=y,y_pred=y_pred))


# In[ ]:


scorer = make_scorer(rmse,greater_is_better=False)


# In[ ]:


feature_columns = df.columns.to_list()
target_column = feature_columns.pop(0)
(feature_columns,target_column)


# In[ ]:


y = df[target_column].values


# In[ ]:


X = df[feature_columns].values


# Function <b>optimize_gbd</b> - finds the best hyperparameters for GradientBoostingRegressor in a defined space with respect to averaged by cross validation with kfold of 5 rmse score. 

# In[ ]:


def optimize_gbd(space):
    alpha,learning_rate, max_depth,max_features,max_leaf_nodes,n_estimators= space
    gbd = GradientBoostingRegressor(alpha=alpha,learning_rate=learning_rate, max_depth=max_depth,
                                    max_features=max_features,max_leaf_nodes=max_leaf_nodes,
                                    n_estimators=n_estimators,random_state=5, criterion='mse')
    score =  -1*cross_val_score(gbd,X,y,cv=5,scoring=scorer).mean()
    print('Error : {}'.format(score))
    return score


# Given best parameters, <b>function train_best_gbd</b> trains the best model.

# In[ ]:


def train_best_gbd(space):
    alpha,learning_rate, max_depth,max_features,max_leaf_nodes,n_estimators= space
    max_depth,max_features,max_leaf_nodes,n_estimators = int(max_depth), int(max_features), int(max_leaf_nodes), int(n_estimators)
    gbd = GradientBoostingRegressor(alpha=alpha,learning_rate=learning_rate, max_depth=max_depth,
                                    max_features=max_features,max_leaf_nodes=max_leaf_nodes,
                                    n_estimators=n_estimators,random_state=5,
                                   criterion='mse')
    gbd.fit(X=X,y=y)
    error = np.sqrt(mean_squared_error(y_pred=gbd.predict(X),y_true=y))
    print('Overall error : {}'.format(error))
    return gbd


# space - the space of parameters we will seek for best ones in

# In[ ]:


space = [(0.1,0.9),#alpha
         (1e-3,0.8),#learning_rate
         (2,20),#max_depth
         (1,3),#max_features
         (2,100),#max_leaf_nodes
         (100,1000)#n_estimators
    
]


# As there are lot's of parameters that we tune, we will make 200 calls to forest minimize optimizer to find the best ones.

# In[ ]:


best_params = forest_minimize(optimize_gbd,dimensions=space,n_calls=200,n_jobs=6,random_state=5)


# Function <b>to_df</b> casts our experiments to data frame instance.

# In[ ]:


def to_df(best_params,cols=['alpha','learning_rate','max_depth',
                            'max_features','max_leaf_nodes','n_estimators']):
    params =  np.array(best_params['x_iters'])
    df = pd.DataFrame(columns=cols,data=params)
    df['scores'] = best_params['func_vals']
    return df
    


# In[ ]:


df_scores = to_df(best_params)


# In[ ]:


df_scores.head()


# In[ ]:


df_scores['scores'].median()


# In[ ]:


df_scores['scores'].quantile(0.25)


# The best model is not the right choice as it's too complex. The better choice is to use the middle accurate model.
# For this purprose we will choose a the subarray of experiments, which consists of middle accurate models.

# In[ ]:


by_percentile = df_scores[(df_scores['scores']>df_scores['scores'].quantile(0.45)) & (df_scores['scores']<df_scores['scores'].quantile(0.55))]


# In[ ]:


by_percentile


# We then can choose the random experiment. We will set a constant seed for reproducability.

# In[ ]:


np.random.seed(14)
choice = np.random.choice(len(by_percentile))


# We now will choose the best hyperparameters with respect to our current choice.

# In[ ]:


params = by_percentile.iloc[choice][['alpha','learning_rate','max_depth',
                            'max_features','max_leaf_nodes','n_estimators']].values


# In[ ]:


params


# Now we are ready to train our first model in an ensemble.

# In[ ]:


gbd_trained = train_best_gbd(params)


# We will store the predictions of the gbd in ensemble_df as the first feature for the second lvl model.

# In[ ]:


ensemble_df = pd.DataFrame()
ensemble_df['gbd'] = gbd_trained.predict(X)


# In[ ]:


ensemble_df.head()


# Let's also visualize the predictions of our model and real y values with respect to timestamp.

# In[ ]:


sns.set()
plt.figure(figsize=(14,10))
plt.scatter(range(len(df)),df['target'],color='black')
plt.plot(range(len(df)),ensemble_df['gbd'],color='blue')
plt.legend(['Values predicted with GBT','Real values'])
plt.xlabel('Timestamps')
plt.ylabel('Target values')
plt.title('Results of GBT model');


# # ExtraTree train

# The second and last model we will use in ensemble is ExtraTreesRegressor. We will also tune this model with respect to average cross val score with kfold of 5 using rmse as the scoring function.

# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor


# In[ ]:


def optimize_extratree(space):
    n_estimators,max_depth,min_samples_split,min_samples_leaf,max_features = space
    extra_tree = ExtraTreesRegressor(min_samples_split=min_samples_split,max_features=max_features,random_state=5,
                            min_samples_leaf=min_samples_leaf,max_depth=max_depth,n_estimators=n_estimators)
    score =  -1*cross_val_score(extra_tree,X,y,cv=5,scoring=scorer).mean()
    print('Error : {}'.format(score))
    return score


# In[ ]:


def train_best_extratree(space):
    n_estimators,max_depth,min_samples_split,min_samples_leaf,max_features = list(map(int,space))
    extra_tree = ExtraTreesRegressor(min_samples_split=min_samples_split,max_features=max_features,random_state=5,
                            min_samples_leaf=min_samples_leaf,max_depth=max_depth,n_estimators=n_estimators)
    extra_tree.fit(X,y)
    error = np.sqrt(mean_squared_error(y_true=extra_tree.predict(X),y_pred=y))
    print('Overall error : {}'.format(error))
    return extra_tree


# In[ ]:


space = [(100,1000),#n_estimators
        (8,20),#max_depth
        (2,4),#min_samples_split
         (2,5),#min_samples_leaf
        (1,3)#max_features
        ]


# In[ ]:


best_params= forest_minimize(optimize_extratree,random_state=5,dimensions=space,n_calls=30)


# In[ ]:


df_scores = to_df(best_params,cols=["n_estimators","max_depth","min_samples_split",
                                    "min_samples_leaf",
                                    "max_features"])


# In[ ]:


df_scores.head()


# In[ ]:


df_scores['scores'].mean()


# We can already choose the middle accurate model, that has 1.458386 cross val score.

# In[ ]:


params = df_scores.iloc[2][["n_estimators","max_depth","min_samples_split",
                                    "min_samples_leaf",
                                    "max_features"]].values


# In[ ]:


ex_tree_reg = train_best_extratree(params)


# In[ ]:


ensemble_df['extra_tree'] = ex_tree_reg.predict(X)


# We will also visualize results of Extra Trees.

# In[ ]:


sns.set()
plt.figure(figsize=(14,10))
plt.scatter(range(len(df)),df['target'],color='black')
plt.plot(range(len(df)),ensemble_df['extra_tree'],color='green')
plt.legend(['Values predicted with ETR','Real values'])
plt.xlabel('Timestamps')
plt.ylabel('Target values')
plt.title('Results of ETR model');


# # Part 3 : Second level model training

# # Gathering everything togather

# As we now have two features from both gbd and extra tree, we can train a linear regression using y as target. We use a linear regession here because the relationship is simple enough, thus we don't need to use a more complex model.

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


ensemble_df['y'] = y


# In[ ]:


ensemble_df.head()


# We tend to have a different correlation between gbd and extra_tree predictions with target value, which is  good for overall generalization.

# In[ ]:


ensemble_df.corr()


# In[ ]:


first_lvl_features = ensemble_df[['gbd','extra_tree']].values


# In[ ]:


labels = ensemble_df['y'].values


# In[ ]:


lr_second_lvl = LinearRegression()


# In[ ]:


-1*cross_val_score(lr_second_lvl,first_lvl_features,labels,scoring=scorer).mean()


# In[ ]:


lr_second_lvl.fit(first_lvl_features,labels)


# In[ ]:


np.sqrt(mean_squared_error(y_pred=lr_second_lvl.predict(first_lvl_features),y_true=labels))


# Finally, we can show the linear regression predictions over "stacked" features.  

# In[ ]:


sns.set()
plt.figure(figsize=(14,10))
plt.scatter(range(len(df)),df['target'],color='black')
plt.plot(range(len(df)),lr_second_lvl.predict(first_lvl_features),color='red')
plt.legend(['Final predictions with second level model','Real values'])
plt.xlabel('Timestamps')
plt.ylabel('Target values')
plt.title('Results of predictions using second level model');


# In[ ]:


sns.set()
plt.figure(figsize=(14,10))
plt.scatter(range(len(df)),df['target'],color='black')
plt.plot(range(len(df)),ensemble_df['extra_tree'],color='green')
plt.plot(range(len(df)),ensemble_df['gbd'],color='blue')
plt.plot(range(len(df)),lr_second_lvl.predict(first_lvl_features),color='red')
plt.legend(['Values predicted with ETR','Values predicted with GBT','Final predictions with second level model','Real values'])
plt.xlabel('Timestamps')
plt.ylabel('Target values')
plt.title('Results of predictions using second level model');


# Let's show the outliers which weren't covered by my model.

# In[ ]:


def detect_outliers(df,constant=2):
    outliers = df[(df['target']<=df['target'].mean()-constant*df['target'].std()) | (df['target']>=df['target'].mean()+constant*df['target'].std())]
    return outliers


# In[ ]:


outliers = detect_outliers(df,1)


# In[ ]:


sns.set()
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='red', lw=4, label='Final predictions with second level model'),
                   Line2D([0], [0], marker='o', label='Outliers in real values',
                          markerfacecolor='orange',color='#999999', markersize=15),
                   Line2D([0], [0], marker='o',color='#999999', label='Real values',
                          markerfacecolor='black', markersize=15),
                   ]
plt.figure(figsize=(14,10))
for c,i in enumerate(df['target'].values):
    if i in outliers['target'].values:
        plt.scatter(c,i,color='orange',label='Outliers in real values')
    else:
        plt.scatter(c,i,color='black',label='Real values')
plt.plot(range(len(df)),lr_second_lvl.predict(first_lvl_features),color='red')
plt.legend(handles=legend_elements)
plt.xlabel('Timestamps')
plt.ylabel('Target values')
plt.title('Results of predictions using second level model');


# In[ ]:


sns.set()
plt.figure(figsize=(14,10))
plt.scatter(range(len(df))[-24:],df['target'][-24:],color='black')
plt.plot(range(len(df))[-24:],lr_second_lvl.predict(first_lvl_features)[-24:],color='red')
plt.legend(['Final predictions with second level model','Real values'])
plt.xlabel('Monthes')
plt.ylabel('Target values')
plt.title('Results of predictions using second level model (showing data only for last two years)')
plt.yticks(np.arange(min(df['target'][-24:]),max(df['target'][-24:])+0.5,0.5))
plt.xticks(range(len(df))[-24:],range(1,25));


# # Making submission

# We are now ready to make the final submission using our model stacking technique.

# In[ ]:


sub_df = pd.read_csv("../input/prediction-bod-in-river-water/test.csv",usecols=['Id','1','2'])


# In[ ]:


sub_df.head()


# In[ ]:


sub_df.isna().sum()


# In[ ]:


sub_df.head()


# We still need to create the additional feature.

# In[ ]:


sub_df['combined'] = sub_df[sub_df.columns.to_list()[1:]].mean(axis=1)


# In[ ]:


sub_df.head()


# In[ ]:


feature_columns = sub_df.columns.to_list()[1:]


# In[ ]:


ensemble_df_test = pd.DataFrame()
ensemble_df_test['gbd'] = gbd_trained.predict(sub_df[feature_columns])
ensemble_df_test['extra_tree'] = ex_tree_reg.predict(sub_df[feature_columns])


# In[ ]:


ensemble_df_test.head()


# Finnaly we can predict the final value.

# In[ ]:


sub_df['Predicted'] = lr_second_lvl.predict(ensemble_df_test.values)


# In[ ]:


sub_df = sub_df[['Id','Predicted']]


# In[ ]:


sub_df.to_csv('submission.csv',index=False)


# In[ ]:


sub_df['Predicted'].values


# We hope, that this notebook was useful for you.

# In[ ]:




