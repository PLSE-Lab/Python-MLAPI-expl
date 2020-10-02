#!/usr/bin/env python
# coding: utf-8

# # Housing Prices
# 
# Hi guys, I made an attempt to analyze the dataset and built 2 machine learning models to predict the Sale Price of houses. Enjoy! Working on this, I have studied many notebooks & other websites. So yes, it is not 100% original, but I'll be grateful to get any pointers from you guys...

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os

house = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
print('house shape:',house.shape)
print('test shape:',test.shape)
original_test = test


# # 1 - Quick glance at data

# In[ ]:


pd.set_option('display.max_columns', None)
house.head(6)


# In[ ]:


display(house.select_dtypes('object').head(6))


# # Column types

# In[ ]:


house.dtypes.value_counts()
categorical_columns = house.select_dtypes('object').columns
print(len(house.select_dtypes('object').columns),'categorical columns:')
print(list(house.select_dtypes('object').columns),'\n')
print(len(house.columns)-len(house.select_dtypes('object').columns),'numerical columns (including sales price):')
print([i for i in list(house.columns) if i not in list(house.select_dtypes('object').columns)])


# # Statistics of columns

# In[ ]:


display(house._get_numeric_data().describe().round(decimals=1))


# In[ ]:


display(house.select_dtypes('object').describe())
# display(house._get_numeric_data().columns.str.contains('Yr|Year'))


# # Visualization (43 categorical, 38 numerical columns) 

# In[ ]:


print('univariate analysis on distribution of categorical variables...')
columns = list(house.select_dtypes('object').columns)
plt.figure(figsize=[16,30])
for i in range(len(house.select_dtypes('object').columns)):
    ax = plt.subplot(11,4,i+1)
    group = house[columns[i]].value_counts()
    plt.barh(group.index,(group.values/len(house)),)
    plt.title(columns[i])
    ax.set_xlim([0,1])
    box = ax.get_position()
    box.y1 = box.y1 - 0.0075 
    ax.set_position(box)


# In[ ]:


print('bivariate analysis on categorical variables to dependent variables')
columns = list(house.select_dtypes('object').columns)
plt.figure(figsize=[16,30])
for i in range(len(house.select_dtypes('object').columns)):
    ax = plt.subplot(11,4,i+1)
    sns.boxplot(data=house,y=columns[i],x='SalePrice')
    plt.title(columns[i])
#     ax.set_xlim([0,1])


# * Outliers in numerical data relation

# # More Visualization on Numerical variables...

# In[ ]:


print('Analyze:\n','1: outliers\n','2: skewed variables\n','The outliers and the skewed variables must be addressed accordingly...')
columns = list(house.select_dtypes(exclude='object').columns)
plt.figure(figsize=[16,30])
for i in range(len(house.select_dtypes(exclude='object').columns)):
    try:
        ax = plt.subplot(10,4,i+1)
        plt.scatter(house[columns[i]],house.SalePrice,alpha=0.15)
        plt.title(columns[i])
        box = ax.get_position()
        box.y1 = box.y1 - 0.01 
        ax.set_position(box)
    except:
        pass
plt.show()


# # Data Processing (Dependant Variables - Filter Outliers)

# In[ ]:


original_house = house
print('filtering outliers...')

old = house.LotArea
house = house.drop(house['LotFrontage'][house['LotFrontage']>280].index)
print('rows deleted so far:',str(len(original_house)-len(house)))
house = house.drop(house['LotArea'][house['LotArea']>50000].index)
print('rows deleted so far:',str(len(original_house)-len(house)))
house = house.drop(house['BsmtFinSF1'][house['BsmtFinSF1']>4000].index)
print('rows deleted so far:',str(len(original_house)-len(house)))
house = house.drop(house['BsmtFinSF2'][house['BsmtFinSF2']>1400].index)
print('rows deleted so far:',str(len(original_house)-len(house)))
house = house.drop(house['TotalBsmtSF'][house['TotalBsmtSF']>5800].index)
print('rows deleted so far:',str(len(original_house)-len(house)))
house = house.drop(house['1stFlrSF'][house['1stFlrSF']>4000].index)
print('rows deleted so far:',str(len(original_house)-len(house)))
house = house.drop(house['GrLivArea'][house['GrLivArea']>4000].index)
print('rows deleted so far:',str(len(original_house)-len(house)))
house = house.drop(house['PoolArea'][house['PoolArea']>400].index)
print('rows deleted so far:',str(len(original_house)-len(house)))
house = house.drop(house['MiscVal'][house['MiscVal']>3000].index)
print('rows deleted so far:',str(len(original_house)-len(house)))


# # Data Processing (Reduce Skew on y)

# * There are negatively/positively skewed columns.
# * for negatively skewed, we apply log function
# * for positively skewed, we apply exp function
# 

# In[ ]:


plt.figure(figsize=[11,4])
plt.subplot(1,2,1)
plt.title('Sale Price')
plt.hist(house.SalePrice,bins=40)

plt.subplot(1,2,2)
plt.title('Log of Sale Price')
log = house.SalePrice.apply(lambda x: np.log(x))
plt.hist(log,bins=40)
house.SalePrice = log


# # Data Processing (Reduce Skew on x)

# In[ ]:


print('Transformed skewed variables (seen on bivariate plot) to be more symmetric')

plt.figure(figsize=[11,4])
plt.subplot(1,2,1)
plt.title('Before')
plt.scatter(house.LotArea,house.SalePrice,alpha=0.25)
plt.ylabel('log(SalePrice)')
plt.xlabel('LotArea')

log_columns = ['LotFrontage','LotArea','BsmtFinSF1','BsmtFinSF2','MasVnrArea','BsmtUnfSF','TotalBsmtSF','1stFlrSF','GrLivArea','WoodDeckSF','OpenPorchSF']
for i in log_columns:
    try:
        house[i] = house[i].apply(lambda x: np.log(x) if x!=0 else x)
        test[i] = test[i].apply(lambda x: np.log(x) if x!=0 else x)
    except:
        print('failed log on',house[i])

plt.figure(figsize=[11,4])
plt.subplot(1,2,2)
plt.title('After')
plt.scatter(house.LotArea,house.SalePrice,alpha=0.25)
plt.xlabel('LotArea')


# # Data Processing (Feature Engineer)

# In[ ]:


print('Transformed year columns to age-based (relative to year sold)')

plt.figure(figsize=[11,4])
plt.subplot(1,2,1)
plt.title('Before')
plt.scatter(house.YearBuilt,house.SalePrice,alpha=0.25)
plt.ylabel('log(SalePrice)')
plt.xlabel('YearBuilt')

year = ['YearBuilt','YearRemodAdd','GarageYrBlt']
for i in year:
    house[i] = house['YrSold'] - house[i]
    test[i] = test['YrSold'] - test[i]

plt.subplot(1,2,2)
plt.title('After')
plt.scatter(house.YearBuilt,house.SalePrice,alpha=0.25)
plt.xlabel('YearBuilt (now age)')


# In[ ]:


print('here to doublecheck for any skewed distribution in variables...')
columns = list(house.select_dtypes(exclude='object').columns)
plt.figure(figsize=[16,30])
for i in range(len(house.select_dtypes(exclude='object').columns)):
    try:
        ax = plt.subplot(10,4,i+1)
        plt.title(columns[i])
        box = ax.get_position()
        box.y1 = box.y1 - 0.01 
        ax.set_position(box)
        sns.distplot(house[columns[i]])
    except:
        pass
plt.show()


# # Correlation Study (Numerical Data)

# In[ ]:


correlation = house.corr()

plt.subplots(figsize=(14,12))
plt.title('Correlation of numerical attributes', size=16)
sns.heatmap(correlation)


# In[ ]:


decision_point = 0.1
print('selecting variables with correlation of more than',decision_point)
correlation[correlation.SalePrice>decision_point].SalePrice.sort_values(ascending=False)

columns = correlation[correlation.SalePrice > decision_point].SalePrice.sort_values(ascending=False).index
columns_below_correlation_threshold = correlation[correlation.SalePrice<=decision_point].SalePrice.sort_values(ascending=False).index


# Looking at the correlation chart, we ought to drop columns that is below a correlation value

# In[ ]:


print('plotting the qualified correlated columns...')
plt.figure(figsize=[16,30])
for i in range(len(columns)):
    try:
        ax = plt.subplot(10,4,i+1)
        plt.scatter(house[columns[i]],house.SalePrice,alpha=0.15)
        plt.title(columns[i]+' (cor: '+str(round(correlation[correlation.index==columns[i]]['SalePrice'].values[0],2))+')')
        box = ax.get_position()
        box.y1 = box.y1 - 0.01 
        ax.set_position(box)
    except:
        pass
plt.show()


# Among the well-correlated variables. Check cross-correlation with other features. If found, check if they correlate well to SalePrice, if they do, you have to choose the best correlated one.

# # Multi-colinearity (Numerical)

# In[ ]:


threshold = 0.7
print('checking for multi-colinearity above',threshold,'if found, choose only the more highly correlated column')
cross_correlation = house[columns[1:]].corr()
plt.subplots(figsize=(10,8))
plt.title('Correlation of numerical attributes (high correlation to SalePrice)', size=16)
sns.heatmap(pd.DataFrame(np.where(np.array(cross_correlation)>threshold,1,0),columns=cross_correlation.columns,index=cross_correlation.columns))


# Looking at the heat plot for cross correlation. We have keep the former, drop the latter as the former is better correlated to the SalePrice:
# * GrLivArea / TotRmsAbvGrd
# * GarageCars / GarageArea
# * LotArea / LotFrontage

# In[ ]:


co_correlation = ['TotRmsAbvGrd','GarageArea','LotFrontage']
print('dropping multicolinear columns for good:\n',['TotRmsAbvGrd','GarageArea','LotFrontage'])
house = house.drop(columns=co_correlation)

num_columns = list(columns)
for i in num_columns: 
    if i in co_correlation: num_columns.remove(i)
    if i == 'SalePrice': num_columns.remove(i)
print('the numerical columns to keep are:\n',num_columns)


# In[ ]:


print('one last glance at numerical variables to be good to go...')
house[num_columns].describe()


# # Addressing Missing data (Numerical)

# In[ ]:


def missing_values_table(df):
    mis_val=df.isnull().sum()    
    mis_val_perc=100*df.isnull().sum()/len(df)
    mis_val_table=pd.concat([mis_val, mis_val_perc], axis=1) 
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
    print ("Your selected data frame has " + str(df.shape[1]) + " columns.\n"+"There are " + str(mis_val_table_ren_columns.shape[0]) +
 " columns that have missing values.")
    return mis_val_table_ren_columns

miss = missing_values_table(house.select_dtypes(exclude='object'))
print(miss,'\n')

miss = missing_values_table(test.select_dtypes(exclude='object'))
print('Test:')
print(miss,'\n')
print('Missing data is few in the dataset. We will just impute with ''median'' values later in modelling')


# In[ ]:


def missing_values_table(df):
    mis_val=df.isnull().sum()    
    mis_val_perc=100*df.isnull().sum()/len(df)
    mis_val_table=pd.concat([mis_val, mis_val_perc], axis=1) 
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
    print ("Your selected data frame has " + str(df.shape[1]) + " columns.\n"+"There are " + str(mis_val_table_ren_columns.shape[0]) +
 " columns that have missing values.")
    return mis_val_table_ren_columns

miss = missing_values_table(house.select_dtypes('object'))
print(miss[miss['% of Total Values']>15],'\n')

miss = missing_values_table(test.select_dtypes('object'))
print('Test:')
print(miss[miss['% of Total Values']>15],'\n')
print('Many missing values in the categorical columns')


# In[ ]:


cat_cols_fill_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                     'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType',
                     'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond',
                     'MasVnrType']
for cat in cat_cols_fill_none:
    house[cat] = house[cat].fillna("None")
    test[cat] = test[cat].fillna('None')
print('Replaced columns with ''None''as many of the columns are nan because they don''t exist in those houses')


# In[ ]:


def missing_values_table(df):
    mis_val=df.isnull().sum()    
    mis_val_perc=100*df.isnull().sum()/len(df)
    mis_val_table=pd.concat([mis_val, mis_val_perc], axis=1) 
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
    print ("Your selected data frame has " + str(df.shape[1]) + " columns.\n"+"There are " + str(mis_val_table_ren_columns.shape[0]) +
 " columns that have missing values.")
    return mis_val_table_ren_columns

miss = missing_values_table(house.select_dtypes('object'))
print('Train:')
print(miss[miss['% of Total Values']>15],'\n')

miss = missing_values_table(test.select_dtypes('object'))
print('Test:')
print(miss[miss['% of Total Values']>15],'\n')


# Below we add 'columns_miss' for columns with a lot of nan, the fact that they are nan could be useful for model.

# # Label Encode (Categorical)

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le_count = 0

# house.Alley = house.Alley.apply(lambda x: str(x))
for col in house:
    if house[col].dtype == 'object' and house[col].nunique() <= 2:
        le.fit(house[col])
        house[col] = le.transform(house[col])
        le_count += 1
print('%d columns encoded.' % le_count)

print('house shape:',house.shape)
print('test shape:',test.shape)


# # One Hot Encode (Categorical)

# In[ ]:


house = pd.get_dummies(house,drop_first=True)
test = pd.get_dummies(test,drop_first=True)


# In[ ]:


print('house shape:',house.shape)
print('test shape:',test.shape)

train = house
train_label = train.SalePrice
train, test = house.align(test,join='inner', axis = 1)
train['SalePrice'] = train_label

print('training features shape:',train.shape)
print('testing features shape:',test.shape)


# 1. Save dataframe in case we use another model which don't require the steps necessary for linear model

# In[ ]:


train_knn = train


# # Multi-colinearity (Categorical)

# In[ ]:


from scipy.stats import ttest_ind
p_stack = []
col_stack = []
p_stack_qualified = []
col_stack_qualified = []
k=1
for j in categorical_columns:
    variable = j
    cat_columns = [i for i in train.columns if variable in i]
    for i in cat_columns:
        zstat, pval = ttest_ind(train.SalePrice[train[i]==0],train.SalePrice[train[i]==1])
        if pval < 0.05:
            p_stack_qualified.append(pval)
            col_stack_qualified.append(i)
        k+=1
plt.figure(figsize=[8,25])
plt.plot(p_stack_qualified,col_stack_qualified)
plt.title('p-value of all the one-hot-encoded columns')


# look for multicollinearity within

# In[ ]:


correlation = house[col_stack_qualified].corr()
plt.figure(figsize=[18,18])
dt_heatmap = pd.DataFrame(np.where(np.logical_or(np.array(correlation)>0.7,np.array(correlation)<-0.7),1,0),columns=correlation.columns,index=correlation.columns)
sns.heatmap(dt_heatmap)
print('Notice highly correlated categorical variables...')


# In[ ]:


col_dic =dict(list(zip(col_stack_qualified,p_stack_qualified)))
temp_dict = {}
cat_column = []
list_=dt_heatmap.index
p=0
for i in dt_heatmap.index:
    col_compare = dt_heatmap[i][dt_heatmap[i]==1].index
    if dt_heatmap[i][dt_heatmap[i]==1].index.nunique()==1: cat_column.append(i)
    else:
        for k in range(dt_heatmap[i][dt_heatmap[i]==1].index.nunique()): #2
            temp_dict[col_compare[k]]=col_dic[col_compare[k]]
        if min(temp_dict, key=temp_dict.get) == i: cat_column.append(i)
    temp_dict = {}
    p+=1
print('deleted',len(dt_heatmap.index)-len(cat_column),'columns that show multi-collinearity and the other colinear variable had more significant p-value')


# In[ ]:


correlation = house[cat_column].corr()
plt.figure(figsize=[18,18])
dt_heatmap = pd.DataFrame(np.where(np.logical_or(np.array(correlation)>0.7,np.array(correlation)<-0.7),1,0),columns=correlation.columns,index=correlation.columns)
sns.heatmap(dt_heatmap)
print('No more multi-colinearity...')


# # Linear Regression

# In[ ]:


from sklearn.model_selection import train_test_split
y = train['SalePrice']
x = train[num_columns + cat_column]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 6)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'median')
imputer.fit(x_train)
x_train = imputer.transform(x_train)
x_test = imputer.transform (x_test)

from sklearn.linear_model import LinearRegression
from sklearn import metrics
mlr = LinearRegression()
mlr.fit(x_train,y_train)
print('AdjR2 for train',mlr.score(x_train, y_train))
print('AdjR2 for test',mlr.score(x_test, y_test))
y_pred = mlr.predict(x_test)
mean = np.exp(house.SalePrice.mean())
print('Mean:',mean)
y_test = np.exp(y_test)
y_pred = np.exp(y_pred)
print('RMS Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('+- of',str(round(100*np.sqrt(metrics.mean_squared_error(y_test, y_pred))/mean,0))+'%')


# In[ ]:


hi = pd.DataFrame(list(zip(num_columns + col_stack_qualified,list(mlr.coef_))),columns=['columns','coefficient'])
# hi['log_coefficient']=hi.coefficient.apply(lambda x: np.log(x))
pd.set_option('display.max_columns', None)
plt.figure(figsize=[8,29])
plt.plot(hi['coefficient'],hi['columns'])
plt.xlabel('coefficients')


# In[ ]:


df = pd.DataFrame({'Actual': y_test, 'Predicted':y_pred})
df1 = df.head(20)
df1.plot(kind='bar',figsize=(16,10))


# In[ ]:


y_train = train['SalePrice']
x_train = train[num_columns + cat_column]
x_test = test[num_columns + cat_column]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'median')
imputer.fit(x_train)
x_train = imputer.transform(x_train)
x_test = imputer.transform (x_test)

from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(x_train,y_train)
y_pred = mlr.predict(x_test)
y_pred = np.exp(y_pred)

print('AdjR2 for train',mlr.score(x_train, y_train))
pd.DataFrame(list(zip(list(original_test['Id']),list(y_pred))),columns=['Id','SalePrice']).to_csv('results_linear.csv',index=False)


# In[ ]:


hi = pd.DataFrame(list(zip(num_columns + col_stack_qualified,list(mlr.coef_))),columns=['columns','coefficient'])
# hi['log_coefficient']=hi.coefficient.apply(lambda x: np.log(x))
plt.figure(figsize=[8,20])
pd.set_option('display.max_columns', None)
plt.plot(hi['coefficient'],hi['columns'])


# * Linear model is unimpressive. With values going to negative and values too big in magnitude. We can explore other models like KNN
# * But lets run the same steps on the whole train dataset and apply it on test data for submission.

# # K Nearest Neighbours

# * We try to fit our data with KNN and check for the best number of neighbors
# * KNN works on distance metrics so it is advised to perform normalization on y too, which is not done in linear regression

# In[ ]:


from sklearn.model_selection import train_test_split
y = train['SalePrice'] #undo the ln function used above
x = train[num_columns + cat_column]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 6)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
scalery = StandardScaler()
y_train = scalery.fit_transform(y_train.values.reshape(-1,1))
y_test = scalery.transform(y_test.values.reshape(-1,1))

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'median')
imputer.fit(x_train)
x_train = imputer.transform(x_train)
x_test = imputer.transform (x_test)


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics

score = []
for i in range(2,40):
    classifier = KNeighborsRegressor(i, weights = 'distance')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    score.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

plt.figure(figsize=[16,7])
plt.subplot(1,2,1)
plt.title('Distance')
plt.plot(range(2,40),score)

uniform_score = []
for i in range(2,40):
    classifier = KNeighborsRegressor(i, weights = 'uniform')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    uniform_score.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

plt.subplot(1,2,2)
plt.title('Uniform')
plt.plot(range(2,40),uniform_score)


# * Now we finalized our values with lowest Root Mean Square Error

# In[ ]:


neighbors = score.index(min(score))+2
classifier =KNeighborsRegressor(neighbors, weights = 'distance')
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print('Best number of neighbors',neighbors)

y_pred = scalery.inverse_transform(y_pred)
y_test = scalery.inverse_transform(y_test)
mean = np.exp(house.SalePrice.mean())
y_test = np.exp(y_test)
y_pred = np.exp(y_pred)

print('Mean:',mean)
print('RMS Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('+- of',str(round(100*np.sqrt(metrics.mean_squared_error(y_test, y_pred))/mean,0))+'%')


# In[ ]:


print('some preperation to look at our prediction errors...')
new_y_test = []
for i in y_test:
    new_y_test.append(i[0])
y_test = new_y_test   

new_y_pred = []
for i in list(y_pred):
    new_y_pred.append(i[0])
y_pred = list(new_y_pred)    


# In[ ]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(20)
df1.plot(kind='bar',figsize=(16,10))


# * Now we fit the 'test.csv' data into our model for submission!

# In[ ]:


#now we fit all the features
y = train['SalePrice'] #undo the ln function used above
x = train[num_columns + cat_column]
competition_data = test[num_columns + cat_column]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
competition_data = scaler.transform(competition_data)
scalery = StandardScaler()
y = scalery.fit_transform(y.values.reshape(-1,1))

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='median')
x = imputer.fit_transform(x)
competition_data = imputer.transform(competition_data)

classifier = KNeighborsRegressor(neighbors,weights = 'distance')
classifier.fit(x,y)
y_pred = classifier.predict(competition_data)
y_pred = scalery.inverse_transform(y_pred)
y_pred = np.exp(y_pred)
answer = []
for i in y_pred:
    answer.append(i[0])
pd.DataFrame(list(zip(list(original_test['Id']),list(answer))),columns=['Id','SalePrice']).to_csv('results_knn.csv',index=False)

