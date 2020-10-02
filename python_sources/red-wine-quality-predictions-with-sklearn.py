#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/winequality-red.csv')


# # Initial data exploration

# In[ ]:


df.head()


# In[ ]:


df.describe()


# No categorical data, pure regression problem. Let's explore correlations.

# In[ ]:


correlations = df.corr()['quality'].drop('quality')
print(correlations)


# In[ ]:


_ = correlations.plot(kind='bar')


# Let's have a quick look on the whole correlation matrix to understand how different features correlate with each other.

# In[ ]:


import seaborn as sns
sns.heatmap(df.corr())


# As suspected, acidity somewhat correlate with each other and pH.

# Let's split data into train (80%), validation (10%) and test (10%) sets.

# In[ ]:


train = df.sample(frac=0.8)
test_and_validation = df.loc[~df.index.isin(train.index)]
validation = test_and_validation.sample(frac=0.5)
test = test_and_validation.loc[~test_and_validation.index.isin(validation.index)]

print(train.shape, validation.shape, test.shape)


# In[ ]:


def get_features(correlation_threshold):
    abs_corrs = correlations.abs()
    high_correlations = abs_corrs[abs_corrs > correlation_threshold].index.values.tolist()
    return high_correlations


# In[ ]:


def compare_predictions(predicted, test_df, target_col):
    # Since we have to predict integer values, and the regressor will return float, let's round predicted dataframe
    predicted = predicted.round(0)
    check_df = pd.DataFrame(data=predicted, index=test_df.index, columns=["Predicted "+target_col])
    check_df = pd.concat([check_df, test_df[[target_col]]], axis=1)
    check_df["Error, %"] = np.abs(check_df["Predicted "+target_col]*100/check_df[target_col] - 100)
    check_df['Error, val'] = check_df["Predicted "+target_col] - check_df[target_col]
    return (check_df.sort_index(), check_df["Error, %"].mean())


# In[ ]:


def evaluate_predictions(model, train_df, test_df, features, target_col):
    train_pred = model.predict(train_df[features])
    train_rmse = mean_squared_error(train_pred, train_df[target_col]) ** 0.5

    test_pred = model.predict(test_df[features])
    test_rmse = mean_squared_error(test_pred, test_df[target_col]) ** 0.5

    print("RMSEs:")
    print(train_rmse, test_rmse)
    
    return test_pred


# # Linear regression approach

# In[ ]:


def lr_model_evaluation(feature_correlation_threshold=0):
    lr = LinearRegression()
    features = get_features(feature_correlation_threshold)
    lr.fit(train[features], train['quality'])
    lr_validation_predictions = evaluate_predictions(lr, train, validation, features, 'quality')
    check_df, avg_error = compare_predictions(lr_validation_predictions, validation, 'quality')
    print("Average validation error:", avg_error)
    return check_df


# In[ ]:


check = lr_model_evaluation()


# Let's try different feature selection thresholds in hopes for better results

# In[ ]:


thresholds = [x * 0.05 for x in range(1, 8)] #threshold will scale up to 0.4

for thr in thresholds:
    print('For threshold =', thr)
    _ = lr_model_evaluation(thr)
    print()


# The best result so far was achieved with feature selection threshold of 0.15, but improvement was not too impressive.

# In[ ]:


print(get_features(0.15))


# # Decision tree approach

# In[ ]:


def dtr_model_evaluation(feature_correlation_threshold=0, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_leaf_nodes=None):
    dtr = DecisionTreeRegressor(random_state=42, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_leaf_nodes=max_leaf_nodes)
    features = get_features(feature_correlation_threshold)
    dtr.fit(train[features], train['quality'])
    dtr_validation_predictions = evaluate_predictions(dtr, train, validation, features, 'quality')
    check_df, avg_error = compare_predictions(dtr_validation_predictions, validation, 'quality')
    print("Average validation error:", avg_error)
    return check_df, avg_error


# In[ ]:


check, error = dtr_model_evaluation()


# Average error dropped, but there is still room for improvements! As RMSEs suggest, an overfitting takes place. Let's start by trying different combinations for hyperparams of decision tree and then move on to random forests.
thresholds = [x * 0.05 for x in range(1, 8)]
max_depth_list = [None, 3, 4, 5, 7, 10, 15, 20]
min_samples_split_list = [2, 3, 4, 5, 7, 10, 15, 20]
min_samples_leaf_list = [1, 2, 3, 5, 7, 10, 0.01, 0.03, 0.05, 0.07, 0.1]
min_weight_fraction_leaf_list = [0., 0.01, 0.02, 0.03, 0.05, 0.07]
max_leaf_nodes_list = [None, 5, 10, 15, 20, 25, 30]

results = []
for max_depth in max_depth_list:
    for min_samples_split in min_samples_split_list:
        for min_samples_leaf in min_samples_leaf_list:
            for min_weight_fraction_leaf in min_weight_fraction_leaf_list:
                for max_leaf_nodes in max_leaf_nodes_list:
                    for thr in thresholds: 
                        hyperparameters = {
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'min_weight_fraction_leaf': min_weight_fraction_leaf,
                            'max_leaf_nodes': max_leaf_nodes,
                            'threshold': thr
                        }

                        check, error = dtr_model_evaluation(thr, max_depth=max_depth, min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
                            max_leaf_nodes=max_leaf_nodes)
                        
                        results.append((hyperparameters, error))
# The above method is computationally impractical. Better approach would to use default parameters and change only one at the time to understand how it affects the results, and only then, with narrowed lists of hyperparameters, try every plausible combination.

# In[ ]:


thresholds = [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4]
max_depth_list = [None, 3, 4, 5, 7, 10, 15, 20]
min_samples_split_list = [2, 3, 4, 5, 7, 10, 15, 20]
min_samples_leaf_list = [1, 2, 3, 5, 7, 10, 0.01, 0.03, 0.05, 0.07, 0.1]
min_weight_fraction_leaf_list = [0., 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.18, 0.2, 0.23, 0.25, 0.3]
max_leaf_nodes_list = [None, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

hyperparams = {
    'threshold': thresholds,
    'max_depth': max_depth_list,
    'min_samples_split': min_samples_split_list,
    'min_samples_leaf': min_samples_leaf_list,
    'min_weight_fraction_leaf': min_weight_fraction_leaf_list,
    'max_leaf_nodes': max_leaf_nodes_list
}

validation_results = []
for hp_name, hp_list in hyperparams.items():
    errors = []
    for hp_val in hp_list:
        if hp_name == 'threshold':
            _, error = dtr_model_evaluation(feature_correlation_threshold=hp_val)
        elif hp_name == 'max_depth':
            _, error = dtr_model_evaluation(max_depth=hp_val)
        elif hp_name == 'min_samples_split':
            _, error = dtr_model_evaluation(min_samples_split=hp_val)
        elif hp_name == 'min_samples_leaf':
            _, error = dtr_model_evaluation(min_samples_leaf=hp_val)
        elif hp_name == 'min_weight_fraction_leaf':
            _, error = dtr_model_evaluation(min_weight_fraction_leaf=hp_val)
        elif hp_name == 'max_leaf_nodes':
            _, error = dtr_model_evaluation(max_leaf_nodes=hp_val)
            
        errors.append(error)
    validation_results.append((hp_name, errors))


# In[ ]:


fig = plt.figure(figsize=(6, 18))

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


# Now, based on graphics above, it is easy to select best hyperparameters. For hyperparameters where it is still not crystally clear what is the best hyperparameter let's use several and search among them.

# In[ ]:


thresholds = [0.2]
max_depth_list = [5, 6]
min_samples_split_list = [3]
min_samples_leaf_list = [0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095]
min_weight_fraction_leaf_list = [0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095]
max_leaf_nodes_list = [None]

results = []
for max_depth in max_depth_list:
    for min_samples_split in min_samples_split_list:
        for min_samples_leaf in min_samples_leaf_list:
            for min_weight_fraction_leaf in min_weight_fraction_leaf_list:
                for max_leaf_nodes in max_leaf_nodes_list:
                    for thr in thresholds: 
                        hyperparameters = {
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'min_weight_fraction_leaf': min_weight_fraction_leaf,
                            'max_leaf_nodes': max_leaf_nodes,
                            'threshold': thr
                        }

                        check, error = dtr_model_evaluation(thr, max_depth=max_depth, min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
                            max_leaf_nodes=max_leaf_nodes)
                        
                        results.append((hyperparameters, error))


# In[ ]:


min(results, key = lambda x: x[1])


# So far the best result for validation set with Decision Tree Regressor is ~8% of errors, with the following hyperparameters (apart from default ones): Feature Selection Correlation Threshold = 0.2, max_depth=5, min_samples_split=3,         min_samples_leaf=0.055, min_weight_fraction_leaf=0.07

# In[ ]:


check, error = dtr_model_evaluation(0.2, max_depth=5, min_samples_split=3, min_samples_leaf=0.055, min_weight_fraction_leaf=0.07)
check.head(10)


# RMSEs for train and validation sets are quite close, it's unlikely there is a lot of overfitting.

# ### Random Forest approach

# Let's use random forest with newly found best hyperparameters for decision tree.

# In[ ]:


def rfr_model_evaluation(n_estimators=100):
    rfr = RandomForestRegressor(random_state=42, n_estimators=n_estimators, max_depth=5, min_samples_split=3, min_samples_leaf=0.055, min_weight_fraction_leaf=0.07)
    features = get_features(0.2)
    rfr.fit(train[features], train['quality'])
    rfr_validation_predictions = evaluate_predictions(rfr, train, validation, features, 'quality')
    check_df, avg_error = compare_predictions(rfr_validation_predictions, validation, 'quality')
    print("Average validation error:", avg_error)
    return check_df, avg_error


# In[ ]:


check, error = rfr_model_evaluation()


# Random forest didn't showed any improvements in comparison with tuned decision tree. Perhaps using the same tuning for random forest as for decision tree is not as effective. Let's run random forest regressor with all default hyperparameters.

# In[ ]:


rfr = RandomForestRegressor(random_state=42, n_estimators=100)
features = get_features(0.2)
rfr.fit(train[features], train['quality'])
rfr_validation_predictions = evaluate_predictions(rfr, train, validation, features, 'quality')
check_df, avg_error = compare_predictions(rfr_validation_predictions, validation, 'quality')
print("Average validation error:", avg_error)


# We got a significant improvement. Let's try to tune hyperparameters as we did it with a single decision tree.

# In[ ]:


def rfr_model_evaluation(feature_correlation_threshold=0, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_leaf_nodes=None):
    rfr = RandomForestRegressor(random_state=42, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_leaf_nodes=max_leaf_nodes)
    features = get_features(feature_correlation_threshold)
    rfr.fit(train[features], train['quality'])
    rfr_validation_predictions = evaluate_predictions(rfr, train, validation, features, 'quality')
    check_df, avg_error = compare_predictions(rfr_validation_predictions, validation, 'quality')
    print("Average validation error:", avg_error)
    return check_df, avg_error


# In[ ]:


thresholds = [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4]
max_depth_list = [None, 3, 4, 5, 7, 10, 15, 20, 25, 30, 35]
min_samples_split_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
min_samples_leaf_list = [1, 2, 3, 5, 7, 10, 0.01, 0.03, 0.05, 0.07, 0.1]
min_weight_fraction_leaf_list = [0., 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.18, 0.2, 0.23, 0.25, 0.3]
max_leaf_nodes_list = [None, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
n_estimators = [10, 20, 40, 50, 75, 100, 125, 150, 175, 200, 225]

hyperparams = {
    'threshold': thresholds,
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
        if hp_name == 'threshold':
            _, error = rfr_model_evaluation(feature_correlation_threshold=hp_val)
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


fig = plt.figure(figsize=(7, 21))

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


# As with decision tree, let's narrow down best hyperparameters with additional parameter search. We should expect that best results should have less than 5.5% of errors.

# In[ ]:


thresholds = [0.05, 0.15]
max_depth_list = [16, 17, 18, 19, 20, 21, 22, 23, 24]
min_samples_split_list = [7]
min_samples_leaf_list = [1, 2]
min_weight_fraction_leaf_list = [0.0]
max_leaf_nodes_list = [None]
n_estimators = [130, 140, 150, 160, 170]

results = []
for n_estimator in n_estimators:
    for max_depth in max_depth_list:
        for min_samples_split in min_samples_split_list:
            for min_samples_leaf in min_samples_leaf_list:
                for min_weight_fraction_leaf in min_weight_fraction_leaf_list:
                    for max_leaf_nodes in max_leaf_nodes_list:
                        for thr in thresholds:
                            hyperparameters = {
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf,
                                'min_weight_fraction_leaf': min_weight_fraction_leaf,
                                'max_leaf_nodes': max_leaf_nodes,
                                'threshold': thr,
                                'n_estimators': n_estimator
                            }

                            check, error = rfr_model_evaluation(thr, n_estimators=n_estimator, max_depth=max_depth, min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
                                max_leaf_nodes=max_leaf_nodes)

                            results.append((hyperparameters, error))


# In[ ]:


min(results, key = lambda x: x[1])


# The best results on validation set found with random forest regressor have ~5.13% of errors with the hyperparameters shown above.

# In[ ]:


check, error = rfr_model_evaluation(0.05, n_estimators=130, max_depth=19, min_samples_split=7,
                                min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                max_leaf_nodes=None)


# In[ ]:


check.head(10)


# # Gradient boosted tree approach

# In[ ]:


import xgboost as xgb


# In[ ]:


def xgbr_model_evaluation(feature_correlation_threshold=0, learning_rate=0.3, gamma=0, max_depth=3,
        min_child_weight=1, max_delta_step=0, subsample=1., reg_lambda=1, reg_alpha=0, n_estimators=100):

    xgbr = xgb.XGBRegressor(max_depth=max_depth, min_child_weight=min_child_weight, max_delta_step=max_delta_step,
                           learning_rate=learning_rate, reg_lambda=reg_lambda, reg_alpha=reg_alpha, gamma=gamma,
                           n_estimators=n_estimators, subsample=subsample)
    features = get_features(feature_correlation_threshold)
    xgbr.fit(train[features], train['quality'])
    xgbr_validation_predictions = evaluate_predictions(xgbr, train, validation, features, 'quality')
    check_df, avg_error = compare_predictions(xgbr_validation_predictions, validation, 'quality')
    print("Average validation error:", avg_error)
    return check_df, avg_error


# In[ ]:


check_df, avg_error = xgbr_model_evaluation()


# ~8.2% errors with default parameters. Let's try to tune the model.

# In[ ]:


thresholds = [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4]
max_depth_list = [3, 4, 5, 7, 10, 15, 20, 25, 30, 35]
max_delta_steps = [x for x in range(0,11)]
min_child_weights = [0, 1, 3, 10, 30, 100, 300, 1000, 3000]
learning_rates = [0.03, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1]
gammas = [0, 0.01, 0.03, 0.1, 0.3, 0.5, 1, 3, 10, 30]
subsamples = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.6]
reg_lambdas = [0.1, 0.3, 1, 1.3, 1.5, 2, 3, 3.5, 4, 4.5, 5, 5.5, 10]
reg_alphas = [0.1, 0.3, 1, 1.3, 1.5, 1.75, 2, 2.25, 2.5, 3]
n_estimators = [10, 20, 40, 50, 75, 100, 125, 150, 175, 200, 225]

hyperparams = {
    'threshold': thresholds,
    'max_depth': max_depth_list,
    'max_delta_step': max_delta_steps,
    'min_child_weight': min_child_weights,
    'learning_rate': learning_rates,
    'gamma': gammas,
    'subsample': subsamples,
    'reg_lambda': reg_lambdas,
    'reg_alpha': reg_alphas,
    'n_estimators': n_estimators,
}

validation_results = []
for hp_name, hp_list in hyperparams.items():
    errors = []
    for hp_val in hp_list:
        if hp_name == 'threshold':
            _, error = xgbr_model_evaluation(feature_correlation_threshold=hp_val)
        elif hp_name == 'max_depth':
            _, error = xgbr_model_evaluation(max_depth=hp_val)
        elif hp_name == 'max_delta_step':
            _, error = xgbr_model_evaluation(max_delta_step=hp_val)
        elif hp_name == 'min_child_weight':
            _, error = xgbr_model_evaluation(min_child_weight=hp_val)
        elif hp_name == 'learning_rate':
            _, error = xgbr_model_evaluation(learning_rate=hp_val)
        elif hp_name == 'gamma':
            _, error = xgbr_model_evaluation(gamma=hp_val)
        elif hp_name == 'subsample':
            _, error = xgbr_model_evaluation(subsample=hp_val)
        elif hp_name == 'reg_lambda':
            _, error = xgbr_model_evaluation(reg_lambda=hp_val)
        elif hp_name == 'reg_alpha':
            _, error = xgbr_model_evaluation(reg_alpha=hp_val)
        elif hp_name == 'n_estimators':
            _, error = xgbr_model_evaluation(n_estimators=hp_val)
            
        errors.append(error)
    validation_results.append((hp_name, errors))


# In[ ]:


fig = plt.figure(figsize=(7, 30))

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


check_df, avg_error = xgbr_model_evaluation(feature_correlation_threshold=0.1, max_depth=10,
    min_child_weight=0, learning_rate=0.9, gamma=0.1, subsample=0.5, reg_lambda=5, reg_alpha=2, n_estimators=125)


# # Final check of best models against the TEST set.

# We have tuned 3 models against validation sets. Now it is time to assess models against data that they haven't seen at all before.

# ### Linear regression

# In[ ]:


lr = LinearRegression()
features = get_features(0.15)
lr.fit(train[features], train['quality'])
lr_test_predictions = evaluate_predictions(lr, train, test, features, 'quality')
check_df, avg_error = compare_predictions(lr_test_predictions, test, 'quality')
print("Average test error:", avg_error)


# ### Decision tree

# In[ ]:


dtr = DecisionTreeRegressor(random_state=42, max_depth=5, min_samples_split=3, min_samples_leaf=0.055, min_weight_fraction_leaf=0.07, max_leaf_nodes=None)
features = get_features(0.2)
dtr.fit(train[features], train['quality'])
dtr_test_predictions = evaluate_predictions(dtr, train, test, features, 'quality')
check_df, avg_error = compare_predictions(dtr_test_predictions, test, 'quality')
print("Average test error:", avg_error)


# ### Random forest

# In[ ]:


rfr = RandomForestRegressor(random_state=42, n_estimators=130, max_depth=19, min_samples_split=7, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_leaf_nodes=None)
features = get_features(0.05)
rfr.fit(train[features], train['quality'])
rfr_test_predictions = evaluate_predictions(rfr, train, test, features, 'quality')
check_df, avg_error = compare_predictions(rfr_test_predictions, test, 'quality')
print("Average validation error:", avg_error)


# ### Gradient boosted tree

# In[ ]:


xgbr = xgb.XGBRegressor(max_depth=10, min_child_weight=0, learning_rate=0.9, reg_lambda=5, reg_alpha=2, gamma=0.1,
                           n_estimators=125, subsample=0.5)
features = get_features(0.1)
xgbr.fit(train[features], train['quality'])
xgbr_test_predictions = evaluate_predictions(xgbr, train, test, features, 'quality')
check_df, avg_error = compare_predictions(xgbr_test_predictions, test, 'quality')
print("Average test error:", avg_error)


# # Conclusion
# We have found an approach that showed about 92.8% of accuracy (~7.2% of errors).
# 
# The model above could be improved in the following manner:
# 
# * Quality of wine could be regarded as categorical data. In predictions we rounded the values, and that added additional error to that, about which model had no idea. 
# 
# * We could ran more precise search for best hyperparameters. Some hyperparameters tend to work better or worse in combination with others. We didn't test that possibility.

# In[ ]:




