#!/usr/bin/env python
# coding: utf-8

# In this kernel I have tried to find out first which feature is catergorical and which feature is numerical. Then in categorical , using EDA I have divided it into ordinal and nominal. For ordinal I have mapped it seperately and for nominal I have used one hot encoding.
# 
# Moving forward checking the skewness of the dependent variable, I have used log of the dependent variable. Taking the log reduces the skewness.
# 
# Fruther I have introduced a new feature called volume, using feature x,y,z and removed this 3 features. This new feature introduced has helped to reach better accuracy.
# 
# Finally I have fed the data into model, and getting a accuracy of around 98%.

# In[ ]:





# In[ ]:


# Ignore warnings :
import warnings
warnings.filterwarnings('ignore')


# Handle table-like data and matrices :
import numpy as np
import pandas as pd
import math 



# Modelling Algorithms :

# Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis

# Regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor




# Modelling Helpers :
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV , KFold , cross_val_score



#preprocessing :
from sklearn.preprocessing import MinMaxScaler , StandardScaler, Imputer, LabelEncoder

#evaluation metrics :

# Regression
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 

# Classification
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  



# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
#import missingno as msno



# Configure visualisations
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use( 'ggplot' )
plt.style.use('fivethirtyeight')
sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)
params = { 
    'axes.labelsize': "large",
    'xtick.labelsize': 'x-large',
    'legend.fontsize': 20,
    'figure.dpi': 150,
    'figure.figsize': [25, 7]
    
}
plt.rcParams.update(params)    
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import r2_score


# In[ ]:


data = pd.read_csv("/kaggle/input/diamonds/diamonds.csv")
data.head()


# In[ ]:


data.shape


# In[ ]:


idsUnique = len(set(data['Unnamed: 0']))
idsTotal = data.shape[0]
idsdupe = idsTotal - idsUnique
print(idsdupe)
#drop id col


# In[ ]:


data=data.drop(['Unnamed: 0'],axis=1)


# In[ ]:


data.info()


# In[ ]:


data_nas = data.isnull().sum()
data_nas = data_nas[data_nas>0]
data_nas.sort_values(ascending = False)


# In[ ]:


data.describe()


# In[ ]:


print("Number of rows with x == 0: {} ".format((data.x==0).sum()))
print("Number of rows with y == 0: {} ".format((data.y==0).sum()))
print("Number of rows with z == 0: {} ".format((data.z==0).sum()))
print("Number of rows with depth == 0: {} ".format((data.depth==0).sum()))


# In[ ]:


data[['x','y','z']] = data[['x','y','z']].replace(0,np.NaN)


# In[ ]:


data.isnull().sum()


# In[ ]:


data.dropna(inplace=True)


# In[ ]:


data.shape


# In[ ]:


data.isnull().sum()


# In[ ]:


categorical_features = data.select_dtypes(include=['object']).columns
categorical_features


# In[ ]:


numerical_features = data.select_dtypes(exclude = ["object"]).columns
numerical_features


# In[ ]:


data_num = data[numerical_features]
data_cat = data[categorical_features]


# In[ ]:


data_num.head()


# In[ ]:


data_cat.head()


# In[ ]:


data.describe()


# In[ ]:


data['cut'].unique()


# In[ ]:


sns.countplot(x='cut',data=data)


# In[ ]:


g=sns.catplot(x='cut',y='price',data=data,kind='bar')


# In[ ]:


data['cut'].value_counts()


# In[ ]:


data_cat['color'].unique()


# In[ ]:


sns.countplot(x='color',data=data)


# In[ ]:


p=sns.catplot(x='color',y='price',data=data,kind='bar')
p


# In[ ]:


data['color'].value_counts()


# In[ ]:


data_cat['clarity'].unique()


# In[ ]:


sns.countplot(x='clarity',data=data)


# In[ ]:


sns.catplot(x='clarity',y='price',data=data,kind='bar')


# In[ ]:


data['clarity'].value_counts()


# In[ ]:


data_cat.head()


# In[ ]:


diamond_onehot=data_cat.copy()


# In[ ]:


for i in range(data_cat.shape[1]):
    diamond_onehot=pd.get_dummies(diamond_onehot,columns=[data_cat.columns[i]],prefix=[data_cat.columns[i]])
diamond_onehot


# In[ ]:


from scipy import stats
from scipy.stats import norm, skew #for some statistics

sns.distplot(data['price'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(data['price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Price distribution')

fig = plt.figure()
res = stats.probplot(data['price'], plot=plt)
plt.show()


# In[ ]:


data['price'] = np.log1p(data['price'] )
y=data['price']

sns.distplot(data['price'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(data['price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Price distribution')

fig = plt.figure()
res = stats.probplot(data['price'], plot=plt)
plt.show()


# In[ ]:


data['price'] = np.log1p(data['price'] )
y=data['price']

sns.distplot(data['price'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(data['price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Price distribution')

fig = plt.figure()
res = stats.probplot(data['price'], plot=plt)
plt.show()


# In[ ]:


data_num=data_num.drop(['price'],axis=1)


# In[ ]:


from scipy.stats import skew 
skewness = data_num.apply(lambda x: skew(x))
skewness.sort_values(ascending=False)


# In[ ]:


type(skewness)


# In[ ]:


skewness = skewness[abs(skewness)>0.1]
skewness.index


# In[ ]:


data_skew = data_num[skewness.index]
data_skew .columns


# In[ ]:


data_skew = np.log1p(data_skew )


# In[ ]:


data_skew 


# In[ ]:


data_num


# In[ ]:


data_num1=data_num.drop(['carat', 'table', 'x', 'y', 'z'],axis=1)
data_numerical=pd.concat([data_skew ,data_num1],axis=1)
data_numerical


# In[ ]:


data_final=pd.concat([data_numerical,diamond_onehot],axis=1)


# In[ ]:


data_normal=data_final


# In[ ]:


data_normal.describe()
data_final=pd.concat([data_normal,y],axis=1)


# In[ ]:


data_final


# In[ ]:


print("Find most important features relative to target")
corr = data_final.corr()
corr.sort_values(["price"], ascending = False, inplace = True)
corr
print(corr['price'])


# In[ ]:


abc = corr['price'][abs(corr['price'])>0.05]
abc.index


# In[ ]:


type(corr)


# In[ ]:


data_model=data_final[abc.index]
data_model


# In[ ]:


data_model['volume'] = data_model['x']*data_model['y']*data_model['z']


# In[ ]:


sns.jointplot(x='volume', y='price' , data=data_model, size=5)


# In[ ]:


data_model.drop(['x','y','z'], axis=1, inplace= True)


# In[ ]:


data_model


# In[ ]:


Data_out=data_model['price']
input_data=data_model.drop(['price'],axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(input_data,Data_out,test_size=0.2, random_state=66)


# In[ ]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


X_train


# In[ ]:


# Collect all R2 Scores.
R2_Scores = []
models = ['Linear Regression' , 'AdaBoost Regression' , 'Ridge Regression' , 'GradientBoosting Regression',
          'RandomForest Regression' ,
         'KNeighbours Regression']


# In[ ]:


clf_lr = LinearRegression()
clf_lr.fit(X_train , y_train)
accuracies = cross_val_score(estimator = clf_lr, X = X_train, y = y_train, cv = 5,verbose = 1)
y_pred = clf_lr.predict(X_test)
print('')
print('####### Linear Regression #######')
print('Score : %.4f' % clf_lr.score(X_test, y_test))
print(accuracies)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

R2_Scores.append(r2)


# In[ ]:


clf_ar = AdaBoostRegressor(n_estimators=1000)
clf_ar.fit(X_train , y_train)
accuracies = cross_val_score(estimator = clf_ar, X = X_train, y = y_train, cv = 5,verbose = 1)
y_pred = clf_ar.predict(X_test)
print('')
print('###### AdaBoost Regression ######')
print('Score : %.4f' % clf_ar.score(X_test, y_test))
print(accuracies)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

R2_Scores.append(r2)


# In[ ]:


clf_rr = Ridge(normalize=True)
clf_rr.fit(X_train , y_train)
accuracies = cross_val_score(estimator = clf_rr, X = X_train, y = y_train, cv = 5,verbose = 1)
y_pred = clf_rr.predict(X_test)
print('')
print('###### Ridge Regression ######')
print('Score : %.4f' % clf_rr.score(X_test, y_test))
print(accuracies)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

R2_Scores.append(r2)


# In[ ]:


clf_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls',verbose = 1)
clf_gbr.fit(X_train , y_train)
accuracies = cross_val_score(estimator = clf_gbr, X = X_train, y = y_train, cv = 5,verbose = 1)
y_pred = clf_gbr.predict(X_test)
print('')
print('###### Gradient Boosting Regression #######')
print('Score : %.4f' % clf_gbr.score(X_test, y_test))
print(accuracies)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

R2_Scores.append(r2)


# In[ ]:


clf_rf = RandomForestRegressor()
clf_rf.fit(X_train , y_train)
accuracies = cross_val_score(estimator = clf_rf, X = X_train, y = y_train, cv = 5,verbose = 1)
y_pred = clf_rf.predict(X_test)
print('')
print('###### Random Forest ######')
print('Score : %.4f' % clf_rf.score(X_test, y_test))
print(accuracies)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)


# In[ ]:


no_of_test=[100]
params_dict={'n_estimators':no_of_test,'n_jobs':[-1],'max_features':["auto",'sqrt','log2']}
clf_rf=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='r2')
clf_rf.fit(X_train,y_train)
print('Score : %.4f' % clf_rf.score(X_test, y_test))
pred=clf_rf.predict(X_test)
r2 = r2_score(y_test, pred)
print('R2     : %0.2f ' % r2)
R2_Scores.append(r2)


# In[ ]:


clf_knn = KNeighborsRegressor()
clf_knn.fit(X_train , y_train)
accuracies = cross_val_score(estimator = clf_knn, X = X_train, y = y_train, cv = 5,verbose = 1)
y_pred = clf_knn.predict(X_test)
print('')
print('###### KNeighbours Regression ######')
print('Score : %.4f' % clf_knn.score(X_test, y_test))
print(accuracies)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)


# In[ ]:


n_neighbors=[]
for i in range (0,50,5):
    if(i!=0):
        n_neighbors.append(i)
params_dict={'n_neighbors':n_neighbors,'n_jobs':[-1]}
clf_knn=GridSearchCV(estimator=KNeighborsRegressor(),param_grid=params_dict,scoring='r2')
clf_knn.fit(X_train,y_train)
print('Score : %.4f' % clf_knn.score(X_test, y_test))
pred=clf_knn.predict(X_test)
r2 = r2_score(y_test, pred)
print('R2     : %0.2f ' % r2)
R2_Scores.append(r2)


# In[ ]:


compare = pd.DataFrame({'Algorithms' : models , 'R2-Scores' : R2_Scores})
compare.sort_values(by='R2-Scores' ,ascending=False)


# In[ ]:


sns.barplot(x='R2-Scores' , y='Algorithms' , data=compare)


# In[ ]:


sns.factorplot(x='Algorithms', y='R2-Scores' , data=compare, size=6 , aspect=4)


# In[ ]:




