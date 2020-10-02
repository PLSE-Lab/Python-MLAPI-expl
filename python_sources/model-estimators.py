#!/usr/bin/env python
# coding: utf-8

# # Preprocessing 
# 

# In[ ]:


# loading requred packages for setting up the python Environment  
import numpy as np
import pandas as pd
import math # For mathematical calculations
import warnings       # To ignore any warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing  #requred packages for preprocessing


# In[ ]:


# loading requred packages for Ploting
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import plotly.offline as pyoff
import plotly.figure_factory as ff


# In[ ]:


dftrain = pd.read_csv("../input/Train.csv") #loading the data sets 
dftest = pd.read_csv("../input/Test.csv")


# ## Birds eye view 

# In[ ]:


dftest.shape


# In[ ]:


dftrain.shape


# In[ ]:


dftest.head()


# In[ ]:


dftrain.head()


# In[ ]:


dftrain.describe()


# In[ ]:


dftest.describe()


# In[ ]:


dftrain.info() 


# In[ ]:


dftest.info()


# ## Creating more attributes  
# If new attributes can or help define the variables in the target attributes and the model will 
# be able to understand difference between the Yes No and indeterminate and predict better

# ### oneunitSalesValue:
# Price for a unit value 

# In[ ]:


oneunitSalesValue = dftrain['TotalSalesValue']/dftrain['Quantity']
dftrain["OneUnitSalesValue"]=oneunitSalesValue
dftrain.head()


# In[ ]:


oneunitSalesValue = dftest['TotalSalesValue']/dftrain['Quantity']
dftest["OneUnitSalesValue"]=oneunitSalesValue
dftest.head()


# ### looking at where its contributing more 

# In[ ]:


dftrain.groupby(['Suspicious'])[['OneUnitSalesValue']].mean().plot.bar(color = "#b53838")


# In[ ]:


#dftrain.drop(dftrain['AvgUnitSales'],axis = 1)


# ### AvgSelingPrice :
# Average selling price for a given product, I got iy by taking the average of TotalSalesValue with respect by product ID

# In[ ]:


df1=dftrain.groupby(['SalesPersonID','ProductID']).mean()['TotalSalesValue']
df1 = (pd.DataFrame(df1))
df1 = df1.reset_index()
df1.columns


# In[ ]:


dftrain=pd.merge(dftrain,df1,on=['SalesPersonID','ProductID'],how='left')
dftrain = dftrain.rename(index = str,columns ={'TotalSalesValue_x':'TotalSalesValue','TotalSalesValue_y':'AvgSelingPrice'})
dftrain.head()


# In[ ]:


tdf1=dftest.groupby(['SalesPersonID','ProductID']).mean()['TotalSalesValue']
tdf1 = (pd.DataFrame(tdf1))
tdf1 = tdf1.reset_index()
tdf1.columns


# In[ ]:


dftest=pd.merge(dftest,tdf1,on=['SalesPersonID','ProductID'],how='left')
dftest = dftest.rename(index = str,columns ={'TotalSalesValue_x':'TotalSalesValue','TotalSalesValue_y':'AvgSelingPrice'})
dftest.head()


# ### Lets see the what insites does this attribute give

# In[ ]:


dftrain.groupby(['Suspicious'])[['AvgSelingPrice']].mean().plot.bar(color = "#b53838")


# ### AvgQuantity:
# Average Quantity sold for a given product

# In[ ]:


df1=dftrain.groupby(['SalesPersonID','ProductID']).mean()['Quantity']
df1 = (pd.DataFrame(df1))
df1 = df1.reset_index()
df1.columns


# In[ ]:


dftrain=pd.merge(dftrain,df1,on=['SalesPersonID','ProductID'],how='left')
dftrain = dftrain.rename(index = str,columns ={'TotalSalesValue_x':'TotalSalesValue','Quantity_x':'Quantity','Quantity_y':'AvgQuantity'})
#dftrain.drop(dftrain['TotalSalesValue_y'],axis = 1)
dftrain.head()


# In[ ]:


tdf1=dftest.groupby(['SalesPersonID','ProductID']).mean()['Quantity']
tdf1 = (pd.DataFrame(tdf1))
tdf1 = tdf1.reset_index()
tdf1.columns


# In[ ]:


dftest=pd.merge(dftest,tdf1,on=['SalesPersonID','ProductID'],how='left')
dftest = dftest.rename(index = str,columns ={'TotalSalesValue_x':'TotalSalesValue','Quantity_x':'Quantity','Quantity_y':'AvgQuantity'})
#dftrain.drop(dftrain['AvgQuantity'],axis = 1)
dftest.head()


# ### Lets see the what insites does this attribute give

# In[ ]:


dftrain.groupby(['Suspicious'])[['AvgQuantity']].mean().plot.bar(color = "#b53838")


# ### AvgQuantityperguy:
# Average Quantity sold for a given SalesPerson

# In[ ]:


AvgQuantityperguy=dftrain.groupby(['SalesPersonID']).mean()['AvgQuantity']
AvgQuantityperguy=pd.DataFrame(AvgQuantityperguy)
#AvgQuantityperguy.head()


# In[ ]:


dftrain=pd.merge(dftrain,AvgQuantityperguy,on=['SalesPersonID'],how='left')
dftrain=dftrain.rename(index=str, columns={"AvgQuantity_y": "AvgQuantityperguy"})
dftrain.head()


# In[ ]:


AvgQuantityperguy=dftest.groupby(['SalesPersonID']).mean()['AvgQuantity']
AvgQuantityperguy=pd.DataFrame(AvgQuantityperguy)
#AvgQuantityperguy.head()


# In[ ]:


dftest=pd.merge(dftest,AvgQuantityperguy,on=['SalesPersonID'],how='left')
dftest=dftest.rename(index=str, columns={"AvgQuantity_y": "AvgQuantityperguy"})

dftest.head()


# In[ ]:


AvgQuantityperProduct=dftrain.groupby(['ProductID']).mean()['Quantity']
AvgQuantityperProduct=pd.DataFrame(AvgQuantityperProduct)
#AvgQuantityperguy.head()


# ### AvgQuantityperProduct:
# Average Quantity sold for a given ProductID

# In[ ]:


dftrain=pd.merge(dftrain,AvgQuantityperProduct,on=['ProductID'],how='left')
dftrain=dftrain.rename(index=str, columns={"Quantity_y": "AvgQuantityperProduct"})
dftrain.head()


# ### Lets see the what insites does this attribute give

# In[ ]:


dftest=pd.merge(dftest,AvgQuantityperProduct,on=['ProductID'],how='left')
dftest=dftest.rename(index=str, columns={"Quantity_y": "AvgQuantityperProduct"})

dftest.head()


# In[ ]:


dftrain.groupby(['Suspicious'])[['AvgQuantityperProduct']].mean().plot.bar(color = "#b53838")


# ### AvgSalesValueperProduct:
# Average TotalSalesValue sold for a given ProductID

# In[ ]:


AvgSalesValueperProduct=dftrain.groupby(['ProductID']).mean()['TotalSalesValue']
AvgSalesValueperProduct=pd.DataFrame(AvgSalesValueperProduct)
#AvgSalesValueperProduct.head()


# In[ ]:


dftrain=pd.merge(dftrain,AvgSalesValueperProduct,on=['ProductID'],how='left')
dftrain=dftrain.rename(index=str, columns={"TotalSalesValue_y": "AvgSalesValueperProduct",'TotalSalesValue_x':'TotalSalesValue','Quantity_x':'Quantity','AvgQuantity_x':'AvgQuantity'})
dftrain.head()


# In[ ]:


AvgSalesValueperProduct=dftest.groupby(['ProductID']).mean()['TotalSalesValue']
AvgSalesValueperProduct=pd.DataFrame(AvgSalesValueperProduct)
#AvgSalesValueperProduct.head()


# In[ ]:


dftest=pd.merge(dftest,AvgSalesValueperProduct,on=['ProductID'],how='left')
dftest=dftest.rename(index=str, columns={"TotalSalesValue_y": "AvgSalesValueperProduct",'TotalSalesValue_x':'TotalSalesValue','Quantity_x':'Quantity','AvgQuantity_x':'AvgQuantity'})
dftest.head()


# In[ ]:


dftrain.groupby(['Suspicious'])[['AvgSalesValueperProduct']].mean().plot.bar(color = "#b53838")


# ### AvgTransactionsSalesPrice:
# Average TotalSalesValue sold for a given SalesPersonID

# In[ ]:


AvgTransactionsSalesPrice = dftrain.groupby(['SalesPersonID'])[['TotalSalesValue']].mean()
AvgTransactionsSalesPrice = AvgTransactionsSalesPrice.rename(columns={'TotalSalesValue': 'AvgTransactionsSalesPrice'})
dftrain = dftrain.join(AvgTransactionsSalesPrice,on = ['SalesPersonID'])
dftrain.head()


# In[ ]:


AvgTransactionsSalesPrice = dftest.groupby(['SalesPersonID'])[['TotalSalesValue']].mean()
AvgTransactionsSalesPrice = AvgTransactionsSalesPrice.rename(columns={'TotalSalesValue': 'AvgTransactionsSalesPrice'})
dftest = dftest.join(AvgTransactionsSalesPrice,on = ['SalesPersonID'])
dftest.head()


# In[ ]:


dftrain.groupby(['Suspicious'])[['AvgTransactionsSalesPrice']].mean().plot.bar(color = "#b53838")


# In[ ]:


dftrain['RatioOfQuantitySold'] = dftrain['Quantity']/dftrain['AvgQuantityperProduct']


# In[ ]:


dftest['RatioOfQuantitySold'] = dftest['Quantity']/dftest['AvgQuantityperProduct']


# In[ ]:


dftrain.groupby(['Suspicious'])[['RatioOfQuantitySold']].mean().plot.bar(color = "#b53838")


# ### TotalProductperSalesPerson
# Average TotalSalesValue sold for a given SalesPersonID

# In[ ]:


df7 = dftrain.groupby(['SalesPersonID','ProductID'])[['Quantity']].sum()
df7 = df7.rename(columns={'Quantity': 'TotalProductperSalesPerson'})
dftrain = dftrain.join(df7,on=['SalesPersonID','ProductID'])
dftrain.head()


# In[ ]:


df7 = dftest.groupby(['SalesPersonID','ProductID'])[['Quantity']].sum()
df7 = df7.rename(columns={'Quantity': 'TotalProductperSalesPerson'})
dftest = dftest.join(df7,on=['SalesPersonID','ProductID'])
dftest.head()


# ### RatioOfQuantiy:
# Quantity by Average Quantity sold for a given product gives the ratio of the Quantity sold

# In[ ]:


dftrain['RatioOfQuantiy'] = dftrain['Quantity']/dftrain['AvgQuantity']


# In[ ]:


dftest['RatioOfQuantiy'] = dftest['Quantity']/dftest['AvgQuantity']


# ### TransactionsPerProduct
# Number of Transactions per product per SalesPerson

# In[ ]:


TransactionsPerProduct = dftrain.groupby(['ProductID'])[['SalesPersonID']].count()
TransactionsPerProduct = TransactionsPerProduct.rename(columns={'SalesPersonID': 'TransactionsPerProduct'})
dftrain = dftrain.join(TransactionsPerProduct,on=['ProductID'])


# In[ ]:


TransactionsPerProduct = dftest.groupby(['ProductID'])[['SalesPersonID']].count()
TransactionsPerProduct = TransactionsPerProduct.rename(columns={'SalesPersonID': 'TransactionsPerProduct'})
dftest = dftest.join(TransactionsPerProduct,on=['ProductID'])


# In[ ]:


TransactionsPerPersom = dftrain.groupby(['SalesPersonID'])[['ProductID']].count()
TransactionsPerPersom = TransactionsPerPersom.rename(columns={'ProductID': 'TransactionsPerPerson'})
dftrain = dftrain.join(TransactionsPerPersom,on=['SalesPersonID'])


# In[ ]:


TransactionsPerPersom = dftest.groupby(['SalesPersonID'])[['ProductID']].count()
TransactionsPerPersom = TransactionsPerPersom.rename(columns={'ProductID': 'TransactionsPerPerson'})
dftest = dftest.join(TransactionsPerPersom,on=['SalesPersonID'])


# In[ ]:


dftrain.groupby(['Suspicious'])[['TransactionsPerProduct']].mean().plot.bar(color = "#b53838")


# ### TotalValueOfProductPerPerson:
# Total value of product sold per SalesPerson

# In[ ]:


TotalValueOfProductPerPerson = dftrain.groupby(['SalesPersonID','ProductID'])[['TotalSalesValue']].sum()
TotalValueOfProductPerPerson=TotalValueOfProductPerPerson.rename(columns={'TotalSalesValue': 'TotalValueOfProductPerPerson'})
dftrain = dftrain.join(TotalValueOfProductPerPerson,on=['SalesPersonID','ProductID'])


# In[ ]:


TotalValueOfProductPerPerson = dftest.groupby(['SalesPersonID','ProductID'])[['TotalSalesValue']].sum()
TotalValueOfProductPerPerson=TotalValueOfProductPerPerson.rename(columns={'TotalSalesValue': 'TotalValueOfProductPerPerson'})
dftest = dftest.join(TotalValueOfProductPerPerson,on=['SalesPersonID','ProductID'])


# ### AvgPricePerProduct:
# Average UnitSalesValue sold for a given ProductID

# In[ ]:


AvgPricePerProduct = dftrain.groupby(['ProductID'])[['OneUnitSalesValue']].mean()
AvgPricePerProduct = AvgPricePerProduct.rename(columns={'OneUnitSalesValue': 'AvgPricePerProduct'})
dftrain = dftrain.join(AvgPricePerProduct,on ='ProductID')
dftrain.head()


# In[ ]:


AvgPricePerProduct = dftest.groupby(['ProductID'])[['OneUnitSalesValue']].mean()
AvgPricePerProduct = AvgPricePerProduct.rename(columns={'OneUnitSalesValue': 'AvgPricePerProduct'})
dftest = dftest.join(AvgPricePerProduct,on ='ProductID')


# ### AvgSellingQuantitySalesPerson:
# Average Quantity sold for a given SalesPerson

# In[ ]:


AvgSellingQuantitySalesPerson = dftrain.groupby(['SalesPersonID'])[['Quantity']].mean()
AvgSellingQuantitySalesPerson = AvgSellingQuantitySalesPerson.rename(columns={'Quantity': 'AvgSellingQuantitySalesPerson'})
dftrain = dftrain.join(AvgSellingQuantitySalesPerson,on=['SalesPersonID'])


# In[ ]:


AvgSellingQuantitySalesPerson = dftest.groupby(['SalesPersonID'])[['Quantity']].mean()
AvgSellingQuantitySalesPerson = AvgSellingQuantitySalesPerson.rename(columns={'Quantity': 'AvgSellingQuantitySalesPerson'})
dftest = dftest.join(AvgSellingQuantitySalesPerson,on=['SalesPersonID'])


# In[ ]:


dftrain.groupby(['Suspicious'])[['AvgPricePerProduct']].mean().plot.bar(color = "#b53838")


# In[ ]:


#just having a copy 
data = dftrain 


# In[ ]:


# replacing the vaule in the traget attribute
Suspicious_dict = {'Yes':1, 'No':2, 'indeterminate':3}

data['Suspicious'] = data['Suspicious'].replace(Suspicious_dict, regex=True)


# In[ ]:


#just having a copy
testdata = dftest


# In[ ]:


testdata.shape


# ### droping some attributes (Id's)

# In[ ]:


data = data.drop(['ReportID'],axis=1)
data=data.drop(axis=1,columns=['ProductID','SalesPersonID'])


# In[ ]:


testdata = testdata.drop(['ReportID'],axis=1)
testdata=testdata.drop(axis=1,columns=['ProductID','SalesPersonID'])


# In[ ]:


data.info()


# In[ ]:


testdata.info()


# ### conveting attribute to integers 

# In[ ]:


data['OneUnitSalesValue']=data['OneUnitSalesValue'].astype('int64')
data['AvgSelingPrice']=data['AvgSelingPrice'].astype('int64')
data['AvgQuantity']=data['AvgQuantity'].astype('int64')
data['RatioOfQuantiy']=data['RatioOfQuantiy'].astype('int64')
data['TotalProductperSalesPerson']=data['TotalProductperSalesPerson'].astype('int64')
data['AvgQuantityperguy']=data['AvgQuantityperguy'].astype('int64')
data['AvgQuantityperProduct']=data['AvgQuantityperProduct'].astype('int64')
data['AvgSalesValueperProduct']=data['AvgSalesValueperProduct'].astype('int64')
data['RatioOfQuantitySold']=data['RatioOfQuantitySold'].astype('int64')
data['TransactionsPerProduct']=data['TransactionsPerProduct'].astype('int64')
data['TransactionsPerPerson']=data['TransactionsPerPerson'].astype('int64')
data['TotalValueOfProductPerPerson']=data['TotalValueOfProductPerPerson'].astype('int64')
data['AvgPricePerProduct']=data['AvgPricePerProduct'].astype('int64')
data['AvgSellingQuantitySalesPerson']=data['AvgSellingQuantitySalesPerson'].astype('int64')


# In[ ]:


testdata['OneUnitSalesValue']=testdata['OneUnitSalesValue'].astype('int64')
testdata['AvgSelingPrice']=testdata['AvgSelingPrice'].astype('int64')
testdata['AvgQuantity']=testdata['AvgQuantity'].astype('int64')
testdata['RatioOfQuantiy']=testdata['RatioOfQuantiy'].astype('int64')
testdata['AvgQuantityperguy']=testdata['AvgQuantityperguy'].astype('int64')
testdata['AvgQuantityperProduct']=testdata['AvgQuantityperProduct'].astype('int64')
testdata['AvgSalesValueperProduct']=testdata['AvgSalesValueperProduct'].astype('int64')
testdata['RatioOfQuantitySold']=testdata['RatioOfQuantitySold'].astype('int64')
testdata['TransactionsPerProduct']=testdata['TransactionsPerProduct'].astype('int64')
testdata['TransactionsPerPerson']=testdata['TransactionsPerPerson'].astype('int64')
testdata['TotalValueOfProductPerPerson']=testdata['TotalValueOfProductPerPerson'].astype('int64')
testdata['AvgPricePerProduct']=testdata['AvgPricePerProduct'].astype('int64')
testdata['AvgSellingQuantitySalesPerson']=testdata['AvgSellingQuantitySalesPerson'].astype('int64')


# In[ ]:


data.columns


# In[ ]:


testdata.columns


# In[ ]:


# loading the attributes to a feat_columns
feat_col = ['Quantity', 'TotalSalesValue',
       'OneUnitSalesValue','AvgSelingPrice','RatioOfQuantitySold','AvgQuantity', 'AvgQuantityperguy', 'AvgQuantityperProduct',
       'AvgSalesValueperProduct','AvgTransactionsSalesPrice','TotalProductperSalesPerson','RatioOfQuantiy', 'TransactionsPerProduct','TransactionsPerPerson','TotalValueOfProductPerPerson',
           'AvgPricePerProduct','AvgSellingQuantitySalesPerson']


# In[ ]:


# numbers of rows and columns for the given 
data.shape


# In[ ]:


testdata.shape


# In[ ]:


data[feat_col].shape


# ## Split to Train and Validation

# In[ ]:


from sklearn.model_selection import train_test_split #loading the requred packages for the split 
X=data[feat_col]
y=data['Suspicious']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0,stratify=y)


# In[ ]:


# Standardizing the data
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)


# In[ ]:


# numbers of rows and columns for the given 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


# # Model building 
# ## Knn

# In[ ]:


# loading the package for the KNNClassifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train) #fitting the model for the given data set(train set) this where the model learns 


# In[ ]:


#from the learnt model we are predicting for the unknowns
knn_train_predictions_train=knn.predict(X_train)
knn_test_predictions_test=knn.predict(X_test)


# In[ ]:


# from sklearn.metrics with classification_report we can validate the model by recall in this multiclass classification
from sklearn.metrics import classification_report
print(classification_report(y_train, knn_train_predictions_train))
print(classification_report(y_test, knn_test_predictions_test))


# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[ ]:


plot_learning_curve(knn, title=None,X=X_train, y=y_train)


# ## RandomForestClassifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 400,max_depth = 5,criterion = 'entropy',class_weight = 'balanced',max_features = 3,random_state = 123)
RandomForestClassifier(random_state= 42)


# In[ ]:


rf.fit(X_train,y_train)#fitting the model for the given data set(train set) this where the model learns 


# In[ ]:


#from the learnt model we are predicting for the unknowns 
rf_train_predictions_train=rf.predict(X_train)
rf_test_predictions_test=rf.predict(X_test)


# In[ ]:


# from sklearn metrics we create the confusion matrix to validate the given model
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test,rf_test_predictions_test)
cnf_matrix


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_train, rf_train_predictions_train))
print("Precision:",metrics.precision_score(y_train, rf_train_predictions_train,average='macro'))
print("Recall:",metrics.recall_score(y_train, rf_train_predictions_train,average='macro'))
print("F1 score:",metrics.f1_score(y_train,rf_train_predictions_train,average='macro'))


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, rf_test_predictions_test))
print("Precision:",metrics.precision_score(y_test, rf_test_predictions_test,average='macro'))
print("Recall:",metrics.recall_score(y_test, rf_test_predictions_test,average='macro'))
print("F1 score:",metrics.f1_score(y_test,rf_test_predictions_test,average='macro'))


# In[ ]:


# from sklearn.metrics with classification_report we can validate the model by recall in this multiclass classification
from sklearn.metrics import classification_report
print(classification_report(y_test, rf_test_predictions_test))
print(classification_report(y_train, rf_train_predictions_train))


# In[ ]:


# plotting the attribute contributing more form the given model(Random Forest)
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
features = feat_col
importances = rf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='y', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ### Plotting Learning Curves

# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[ ]:


plot_learning_curve(rf, title=None,X=X_train, y=y_train)


# ## DecisionTreeClassifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz,DecisionTreeRegressor
estimator = DecisionTreeClassifier(max_depth=5,random_state=123)
estimator.fit(X_train, y_train) #fitting the model for the given data set(train set) this where the model learns 


# In[ ]:


#from the learnt model we are predicting for the unknowns 
estimator_train_predictions_train=estimator.predict(X_train)
estimator_test_predictions_test=estimator.predict(X_test)


# In[ ]:


# from sklearn.metrics with classification_report we can validate the model by recall in this multiclass classification
from sklearn.metrics import classification_report
print(classification_report(y_test, estimator_test_predictions_test))
print(classification_report(y_train, estimator_train_predictions_train))


# ### Plotting Learning Curves

# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[ ]:


plot_learning_curve(estimator, title=None,X=X_train, y=y_train)


# ## RandomForestClassifier gave best results compared to the other models that I tryed
# looking at the learing rate RandomForestClassifier is the closet to the test scores with given a better recall than the other models 
