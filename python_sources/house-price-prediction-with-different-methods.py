#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### importing libraries
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
sns.set(font_scale=1.5)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#uploading the data and check its head
data= pd.read_csv('../input/kc-housesales-data/kc_house_data.csv')
data.head(3)


# In[ ]:


data.shape


# In[ ]:


#checking the data types of the columns and the number of missing values
data.info()


# In[ ]:


#check if the zipcode has rapition which lead to it will have effect on the result
data['zipcode'].value_counts().head()


# In[ ]:


#after looking at the data info, the lat and long and id should be droped as the location is not defined with refrence to city or any other refrence
# the date should be converted as a month and year columns
#zipcode should be converted as catagory as so as the condition column


# In[ ]:


#deleting columns
del data['id']; del data['lat'];del data['long']


# In[ ]:


#extracting the year and month of the selling dates
data['date']= pd.to_datetime(data['date']) 
data['year']= (pd.DatetimeIndex(data['date']).year)
data['month']= (pd.DatetimeIndex(data['date']).month)
del data['date']


# In[ ]:


#fixing the data types
data['year']=data['year'].astype(int)
data['month']= data['month'].astype(str)
data['zipcode']=data['zipcode'].astype(str)
data['condition']=data['condition'].astype(str)


# In[ ]:


#convert the date of renoved as int type 1 for it has and 0 for nune
# data['yr_renovated']=(data['yr_renovated']>0).astype(int) 
#or 
data['yr_renovated'] = data['yr_renovated'].apply(lambda x : 1 if x>0 else 0)
 #this command will convert the numbers to boolen based on the condition and the results are True or False and back to integer


# In[ ]:


#make sure the data is ready 
data.info()


# In[ ]:


data.head(3)


# In[ ]:


#feature Engineering: it is about creating new columns that may help to find direct relation between the goal and the data
data['age']=data['year']-data['yr_built']
data['sqft_with_basembent']=data['sqft_above']+data['sqft_basement']
data['sqft']=data['sqft_living']+data['sqft_lot']
data['sqft15']=data['sqft_living15']+data['sqft_lot15']


# In[ ]:


#convert the data to X for data given and Y for the price column which is our Target
y = data.pop('price')
#convert objects columns to binary columns by getting dummy values for them
X = pd.get_dummies(data, drop_first=True)


# In[ ]:


#apply spiliting for the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,  random_state=42)


# In[ ]:


#apply standard scaling. and import cross validation function
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


#import linear regretion models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV


# In[ ]:


#applying Linear Regression
model = LinearRegression()

scores = cross_val_score(model, X_train, y_train, cv=20)

print("Mean cross-validated training score:", scores.mean())

# fit and evaluate the data on the whole training set
model.fit( X_train, y_train)
print("Training Score:", model.score( X_train, y_train))
print("Test Score:", model.score( X_test, y_test))
L_R_S=model.score( X_test, y_test)


# In[ ]:


#applying Ridge
model = Ridge()

scores = cross_val_score(model, X_train, y_train, cv=20)

print("Mean cross-validated training score:", scores.mean())

# fit and evaluate the data on the whole training set
model.fit( X_train, y_train)
print("Training Score:", model.score( X_train, y_train))
print("Test Score:", model.score( X_test, y_test))
Ridge_S=model.score( X_test, y_test)


# In[ ]:


#applying RidgeCV as the model
model = RidgeCV( normalize=True, cv=20)

scores = cross_val_score(model, X_train, y_train, cv=20)

print("Mean cross-validated training score:", scores.mean())

# fit and evaluate the data on the whole training set
model.fit( X_train, y_train)
print("Training Score:", model.score( X_train, y_train))
print("Test Score:", model.score( X_test, y_test))
Ridge_CV_S=model.score( X_test, y_test)


# In[ ]:


##Another lasso try 
model = LassoCV(alphas=np.logspace(-50,100, 5), cv=5) 
model.fit(X_train, y_train)
print('Best alpha:', model.alpha_)
print('Training score:', model.score(X_train, y_train))
print("Test Score:", model.score( X_test, y_test))


# In[ ]:


#applying lasso as the model
model = Lasso()

scores = cross_val_score(model, X_train, y_train, cv=20)

print("Mean cross-validated training score:", scores.mean())

# fit and evaluate the data on the whole training set
model.fit( X_train, y_train)
print("Training Score:", model.score( X_train, y_train))
print("Test Score:", model.score( X_test, y_test))
Lasso_S=model.score( X_test, y_test)


# In[ ]:


#applying lassoCV as the model
model = LassoCV(normalize=True,cv=20)

scores = cross_val_score(model, X_train, y_train, cv=20)

print("Mean cross-validated training score:", scores.mean())

# fit and evaluate the data on the whole training set
model.fit( X_train, y_train)
print("Training Score:", model.score( X_train, y_train))
print("Test Score:", model.score( X_test, y_test))
Lasso_CV_S=model.score( X_test, y_test)


# In[ ]:


Models=['Linear Regression','Ridge','RidgeCV','Lasso','LassoCV']
scores=[L_R_S,Ridge_S,Ridge_CV_S,Lasso_S,Lasso_CV_S]
fig = plt.figure()
ax = fig.add_axes([0,0,3,1])
ax.bar(Models,scores)
plt.show()


# the accurecy of all models almost the same, however Ridge and Lasso are slightly better

# In[ ]:


#apply Decision Tree Regressor,
from sklearn.tree import DecisionTreeRegressor
dtr1 = DecisionTreeRegressor(max_depth=1) # change depth to 5 and see the difference
dtr2 =DecisionTreeRegressor(max_depth=2) 
dtr3 = DecisionTreeRegressor(max_depth=3)
dtr4 = DecisionTreeRegressor(max_depth=None) 

# fit the 4 models
dtr1.fit(X_train, y_train)
dtr2.fit(X_train, y_train)
dtr3.fit(X_train, y_train)
dtr4.fit(X_train, y_train)

dtr1_scores = cross_val_score(dtr1,X_train, y_train, cv=20)
dtr2_scores = cross_val_score(dtr2, X_train, y_train, cv=20)
dtr3_scores =cross_val_score(dtr3,X_train, y_train, cv=20)
dtrN_scores =cross_val_score(dtr4, X_train, y_train, cv=20)
print('the score of deciton tree for the depth of one to three and None')
print (dtr1_scores.mean() ,dtr2_scores.mean(), dtr3_scores.mean() ,dtrN_scores.mean())


# all the results are not as good as the models above

# In[ ]:


#apply KNN modelto the train
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=7)
knr.fit(X_train, y_train)
knr_scores = cross_val_score(knr,X_train, y_train, cv=20)
print(knr_scores.mean())


# In[ ]:


# KNeighborsRegressor + evaluation
from sklearn import metrics
Ks =20
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    knn_model = KNeighborsRegressor(n_neighbors = n).fit(X_train,y_train)
    mean_acc[n-1] = knn_model.score(X_test, y_test)
    yhat=knn_model.predict(X_test)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc 

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()
print( "The best test accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)


# In[ ]:


knn_model = KNeighborsRegressor(n_neighbors = 6).fit(X_train,y_train)
print("Training Score:", knn_model.score( X_train, y_train))
print("Test Score:", knn_model.score( X_test, y_test))


# In[ ]:


knn_model = KNeighborsRegressor(n_neighbors = n).fit(X_train,y_train)
yhat=knn_model.predict(X_test)
print(len(yhat))


# In[ ]:


print(type(yhat))


# In[ ]:


print(type(yhat))


# In[ ]:


print(len(np.array(y_test)))


# In[ ]:


X_train.shape


# In[ ]:


#RandomForest without grid search
from sklearn.ensemble import RandomForestRegressor
forest_cv = RandomForestRegressor(n_jobs=-1)
forest_cv.fit(X_train,y_train)
print("Training Score:", forest_cv.score( X_train, y_train))
print("Test Score:", forest_cv.score( X_test, y_test))


# In[ ]:


#Random forest with grid search
from sklearn.model_selection import GridSearchCV
def warn(*args, **kwargs): #disable warnings
    pass
import warnings
warnings.warn = warn
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4],
    'min_samples_split': [8, 10],
    'n_estimators': [100, 1000]
}
rf = RandomForestRegressor()
clf = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
clf.fit(X_train, y_train)
rf = clf.best_estimator_
rf = rf.fit(X_train, y_train) 
rf.score(X_train,y_train)


# In[ ]:


# AdaBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
regr = AdaBoostRegressor(random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
predicted_regr = regr.predict(X_train)
print("Test Score:", model.score( X_test, y_test))
print(regr.feature_importances_)


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=100)
pca.fit(X_train)
pca_transformed = pca.fit_transform(X_train)
original_pca = pd.DataFrame(data = pca.components_)
original_pca


# In[ ]:


pca.explained_variance_ratio_.sum() 


# In[ ]:


def plot_pca(pca):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 4))
    size = len(pca.explained_variance_ratio_)
    ax1.bar(range(size), pca.explained_variance_ratio_)
    ax2.plot(range(size), np.cumsum(pca.explained_variance_ratio_), '--')
    plt.show()
plot_pca(pca)

