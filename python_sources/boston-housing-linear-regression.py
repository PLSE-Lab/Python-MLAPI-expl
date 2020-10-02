#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[ ]:


boston = load_boston()
type(boston)


# In[ ]:


boston.keys()


# In[ ]:


boston.DESCR


# In[ ]:


boston.feature_names


# In[ ]:


data = boston.data
type(data)


# In[ ]:


data.shape


# In[ ]:


#load data from load_boston
data = pd.DataFrame(data = data, columns = boston.feature_names)
data.head()


# In[ ]:


#add price column
data['Price'] = boston.target
data.head()


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


#check for null values
data.isnull().sum()


# In[ ]:


#for a closer look at where the null values are remove .sum()
data.isnull().head()


# In[ ]:


#visualisation of pair-wise relationships and correlations between different features
sns.pairplot(data)


# In[ ]:


#distribution plot
rows = 2
cols = 7

fig, ax = plt.subplots(nrows = rows, ncols = cols, figsize=(16,4))

col = data.columns
index = 0

for i in range(rows):
    for j in range(cols):
        sns.distplot(data[col[index]], ax = ax[i][j])
        index = index + 1

plt.tight_layout() #removes overlap between subplots


# In[ ]:


#look for features with distribution as its more likely theres a correleatin with the target variable. 
#highly skewed features wont help predict the target price
#this is feature selection


# In[ ]:


#correlation matrix
corrmat = data.corr()
corrmat


# In[ ]:


#plot corrmat into heatmap
fig, ax = plt.subplots(figsize=(18,10))
sns.heatmap(corrmat, annot = True, annot_kws={'size' : 12})


# In[ ]:


corrmat.index.values


# In[ ]:


#select features correlated with price (target variable)


# In[ ]:


def getCorrelatedFeature(corrdata, threshold):
    feature = []
    value = []
    
    for i, index in enumerate(corrdata.index):
        if abs(corrdata[index])> threshold:
            feature.append(index)
            value.append(corrdata[index])
            
    df = pd.DataFrame(data = value, index = feature, columns=['corr value'])
    return df


# In[ ]:


threshold = 0.5
corr_value = getCorrelatedFeature(corrmat['Price'], threshold)
corr_value


# In[ ]:


corr_value.index.values


# In[ ]:


correlated_data = data[corr_value.index]
correlated_data.head()


# In[ ]:


#pair plot and correlated matrix of correlated data


# In[ ]:


sns.pairplot(correlated_data)
plt.tight_layout()


# In[ ]:


sns.heatmap(correlated_data.corr(), annot=True, annot_kws={'size': 12})


# In[ ]:


#shuffle and split data


# In[ ]:


X = correlated_data.drop(labels=['Price'], axis = 1) #x = feature vector
y = correlated_data['Price'] #y = target vector
X.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


X_train.shape, X_test.shape


# In[ ]:


#train model


# In[ ]:


model = LinearRegression()
model.fit (X_train, y_train)


# In[ ]:


y_predict = model.predict(X_test)


# In[ ]:


df = pd.DataFrame(data = [y_predict, y_test])
df.T #df.T transposes df


# In[ ]:


#evaluate the performance of our model using R**2 (0-1, 1 = perfect model)


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


score = r2_score(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
print('r2_score: ', score)
print('mae: ', mae)
print('mse: ', mse)


# In[ ]:


#store feature performance


# In[ ]:


total_features = []
total_features_name = []
selected_correlation_value = []
r2_scores = []
mae_value = []
mse_value = []


# In[ ]:


#performance matrix method using two parameters to give all matrix parameters automatically

def performance_metrics(features, th, y_true, y_pred):
    score = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_predict)
    mse = mean_squared_error(y_true, y_predict)
    
    total_features.append(len(features)-1)
    total_features_name.append(str(features))
    selected_correlation_value.append(th)
    r2_scores.append(score)
    mae_value.append(mae)
    mse_value.append(mse)
    
    metrics_dataframe = pd.DataFrame(data = [total_features_name, total_features, selected_correlation_value, r2_scores, mae_value, mse_value],
                                    index = ['features name', '#feature', 'corr_value', 'r2_score', 'MAE', 'MSE'])
    return metrics_dataframe.T


# In[ ]:


performance_metrics(correlated_data.columns.values, threshold, y_test, y_predict)


# In[ ]:


#regression plot of features correlated with the house price using sns


# In[ ]:


rows = 2
cols = 2
fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(16,4))
col = correlated_data.columns
index = 0

#for loop for 2x2 matrix
for i in range(rows):
    for j in range(cols):
        sns.regplot(x = correlated_data[col[index]], y = correlated_data['Price'], ax = ax[i][j])
        index = index + 1
fig.tight_layout()


# In[ ]:


#features with >60%


# In[ ]:


threshold = 0.60
corr_value = getCorrelatedFeature(corrmat['Price'], threshold)
corr_value


# In[ ]:


correlated_data = data[corr_value.index]
correlated_data.head()


# In[ ]:


def get_y_predict(corr_data):
    X = corr_data.drop(labels = ['Price'], axis = 1)
    y = corr_data['Price']
    X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return y_predict


# In[ ]:


y_predict = get_y_predict(correlated_data)


# In[ ]:


performance_metrics(correlated_data.columns.values, threshold, y_test, y_predict)


# In[ ]:


# train 70% accuracy model


# In[ ]:


threshold = 0.70
corr_value = getCorrelatedFeature(corrmat['Price'], threshold)
corr_value


# In[ ]:


correlated_data = data[corr_value.index]
correlated_data.head()


# In[ ]:


y_predict = get_y_predict(correlated_data)


# In[ ]:


performance_metrics(correlated_data.columns.values, threshold, y_test, y_predict)


# In[ ]:


#>40% 


# In[ ]:


threshold = 0.40
corr_value = getCorrelatedFeature(corrmat['Price'], threshold)
corr_value


# In[ ]:


correlated_data = data[corr_value.index]
correlated_data.head()


# In[ ]:


y_predict = get_y_predict(correlated_data)
performance_metrics(correlated_data.columns.values, threshold, y_test, y_predict)


# In[ ]:


#normalised model


# In[ ]:


model = LinearRegression(normalize=True)
model.fit(X_train, y_train)


# In[ ]:


y_predict = model.predict(X_test)
r2_score(y_test, y_predict)


# In[ ]:


#plot learning curve to see how our model learns


# In[ ]:


from sklearn.model_selection import learning_curve, ShuffleSplit


# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

X = correlated_data.drop(labels = ['Price'], axis = 1)
y = correlated_data['Price']

title = "Learning Curves (Linear Regression) " + str(X.columns.values)

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = LinearRegression()
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=-1)

plt.show()

