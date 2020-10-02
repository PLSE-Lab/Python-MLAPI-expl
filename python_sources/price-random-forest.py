# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from subprocess import check_output

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

# Training head content
print (train.head())
print (test.head())

SalePrice = train['SalePrice']
ids = test['Id']

train = train.drop(['Id'], axis=1)
test = test.drop(['Id'], axis=1)

a = train.columns.values

# Print data skewness

print(train[a].skew())

# Change saleprice skew

plt.rcParams['figure.figsize'] = (20, 10)
prices = pd.DataFrame({"log of price":np.log1p(train["SalePrice"]),"price":train["SalePrice"]})
prices.hist(color='orange')

plt.savefig("Fig0.png")

a = np.delete(a, len(a)-1)

# Change skewness of the train numerical data

num_ft = train.dtypes[train.dtypes != "object"].index

sk_ft = train[num_ft].apply(lambda x: skew(x.dropna())) 
sk_ft = sk_ft[sk_ft > 0.75]
sk_ft = sk_ft.index

train[sk_ft] = np.log1p(train[sk_ft])

#--------------------------

# Change skewness of the test numerical features

num_ft_test = test.dtypes[test.dtypes != "object"].index

sk_ft_test = test[num_ft_test].apply(lambda x: skew(x.dropna())) 
sk_ft_test = sk_ft_test[sk_ft_test > 0.75]
sk_ft_test = sk_ft_test.index

test[sk_ft_test] = np.log1p(test[sk_ft_test])

#--------------------------

# Plot numerical features

train.hist(column=a, bins=10, figsize=(20,20), xlabelsize = 7, color='green', log=True)

plt.savefig("Fig1.png")

#--------------------------

# Plot categorical features

fig_dims = (9, 5)

cols = train.columns
num_cols = train._get_numeric_data().columns
cat_cols=list(set(cols) - set(num_cols))

n_cols = 5
n_rows = 9

for i in range(n_rows):
    for j in range(n_cols):
        if (i*n_cols+j) > 42: 
            break
        plt.subplot2grid(fig_dims, (i, j))
        train[cat_cols[i*n_cols+j]].value_counts().plot(kind='bar',title=cat_cols[i*n_cols+j],color='red')
        
plt.savefig("Fig2.png")
#--------------------------

# Correlations

cor = train[num_cols].corr()

threshold = 0.7

corlist = []

for i in range(0,len(num_cols)):
    for j in range(i+1,len(num_cols)):
        if (j != i and cor.iloc[i,j] >= threshold and cor.iloc[i,j] < 1) or (cor.iloc[i,j] < 0 and cor.iloc[i,j] <= -threshold):
            corlist.append([cor.iloc[i,j],i,j]) 

#Sort higher correlations first            
sort_corlist = sorted(corlist,key=lambda x: -abs(x[0]))

#Print correlations and column names
for x,i,j in sort_corlist:
    print (num_cols[i],num_cols[j],x)

#--------------------------

# Hot encoding for categorical data

train = pd.get_dummies(train)
train = train.fillna(train.median())

test = pd.get_dummies(test)
test = test.fillna(test.median())

#--------------------------

# Plot the learning curves

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy'):
    plt.figure(figsize=(10,6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring,
                                                            n_jobs=n_jobs, train_sizes=train_sizes)
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

X = train.drop(['SalePrice'], axis=1)
y = train.SalePrice

X=X.astype(int)
y=y.astype(int)

same_cols = [col for col in X.columns if col in test.columns]

#--------------------------

# Apply random forest clasifier to plot the learning curves

X_train, X_test, y_train, y_test = train_test_split(X[same_cols], y, random_state=42)

rfc = RandomForestClassifier(random_state=42, criterion='entropy', min_samples_split=5, oob_score=True)
parameters = {'n_estimators':[500], 'min_samples_leaf':[12]}

scoring = make_scorer(accuracy_score, greater_is_better=True)

cl_rand_fr = GridSearchCV(rfc, param_grid=parameters, scoring=scoring)
cl_rand_fr.fit(X_train, y_train)
cl_rand_fr = cl_rand_fr.best_estimator_

# Show prediction accuracy score

print (accuracy_score(y_test, cl_rand_fr.predict(X_test)))
print (cl_rand_fr)
plot_learning_curve(cl_rand_fr, 'Random Forest', X[same_cols], y, cv=4);
plt.savefig("Fig3.png")

#--------------------------

# Show feature importance coefficients. 
coef = pd.Series(cl_rand_fr.feature_importances_, index = X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 10))
coef.head(10).plot(kind='bar')
plt.title('Feature Significance')

plt.savefig("Fig4.png")

#--------------------------

# Show actual vs predicted saleprice result

X=train
X=train.drop(['SalePrice'], axis=1)
y=train.SalePrice

Xt=test

X_train, X_test, y_train, y_test = train_test_split(X[same_cols], y)

reg_rand_fr = RandomForestRegressor(n_estimators=500, n_jobs=-1,random_state=42)
reg_rand_fr.fit(X_train, y_train)
y_pred = reg_rand_fr.predict(X_test)

y_pred_test = reg_rand_fr.predict(Xt)

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred,s=30,color='black')
plt.title('Predicted vs. Actual')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
plt.tight_layout()

plt.savefig("Fig5.png")

#--------------------------

# Submit result

Result = [x - 1 for x in np.exp(y_pred_test)]

submission = pd.DataFrame({
        "Id": ids,
        "SalePrice": Result
    })
submission.to_csv('Test-Price.csv', index=False)
