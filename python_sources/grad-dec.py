import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')





def gradient_descent(X, y, alpha, iterations):
    
    m = X.shape[0]
    t0 = 0.0
    t1 = [0.0] * X.shape[1]
    y = np.matrix(y).T
    for i in range(0, iterations):
        h = t0 + np.dot(X, np.matrix(t1).T)
       # print(t0)
       # print(h)
        loss = h - y
        cost = np.sum(np.power(loss, 2)) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        
        t0 = t0 - alpha*(1/m)*np.sum(h - y)
        
        for j in range(0, X.shape[1]):
            #print(alpha*(1/m)*np.sum((h - y)))
            t1[j] = t1[j] - alpha*(1/m)*np.sum(np.array((h - y).tolist()).flatten() * np.array(X[X.columns[j]].tolist()).flatten())
            
    
    return t0, t1


df_train = pd.read_csv('../input/train.csv')

print(df_train.head())

headers = df_train.columns

print(headers)

target = 'SalePrice'

print(df_train[target].describe())

sns.distplot(df_train[target])

fig = plt.figure()
res = stats.probplot(df_train[target], plot=plt)
plt.show()




percentiles = np.array([2.5, 25, 50, 75, 97.5])
# Compute percentiles: ptiles_vers
ptiles_prices = np.percentile(df_train[target], percentiles)

_ = plt.plot(ptiles_prices ,percentiles/100, marker='D', color='red', linestyle='none')

plt.show()




df_train[target] = np.log1p(df_train[target])
sns.distplot(df_train[target] , fit=norm)
(mu, sigma) = norm.fit(df_train[target])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(df_train[target], plot=plt)
plt.show()





#heatmap for better observing of correlation

corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()



#see missing values

missings = df_train.isnull().sum().sort_values(ascending = False)
missing_percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([missings, missing_percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))



#dropping columns with missing data

df_train = df_train.drop(missing_data[missing_data['Total'] > 1].index, axis=1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)


df_train.drop('Id', axis=1).duplicated()

'''
sns.set()
sns.pairplot(df_train[cols[::-1]], size = 2.5)
plt.show()
'''

#handling outliers
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice'] < 13)].index)
df_train = df_train.drop(df_train[(df_train['TotalBsmtSF']>5000)].index)
df_train = df_train.drop(df_train[(df_train['1stFlrSF'] > 4000)].index)

#preparing data
data = df_train[cols]
data = data.drop(target, axis=1)
data = pd.concat([data, df_train['TotRmsAbvGrd']], axis=1)
print(data.columns)

df_train[target] = np.expm1(df_train[target])

df_train[target] = (df_train[target] - np.mean(df_train[target])) / np.std(df_train[target])



output_val = df_train[target]

learning_rate = 0.003

num_of_iter = 500


for col in data.columns:
    data[col] = (data[col] - np.mean(data[col])) / np.std(data[col])



print(data.shape)

t0, t1 = gradient_descent(data, output_val, learning_rate, num_of_iter)

#train_data, test_data, train_out, test_out = train_test_split(data, output_val, train_size=0.8, random_state= 42)


#t0, t1 = gradient_descent(train_data, train_out, learning_rate, num_of_iter)

#predicts = t0 + np.dot(test_data, np.matrix(t1).T)

#print(mean_absolute_error(test_out, predicts))


test = pd.read_csv('../input/test.csv')

test_data = test[data.columns]

for col in data.columns:
    test_data[col] = (test_data[col] - np.mean(test_data[col])) / np.std(test_data[col])


predicts = t0 + np.dot(test_data, np.matrix(t1).T)


my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': np.array(predicts.tolist()).flatten()})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)




