#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

trainingData = pd.read_csv('../input/Train_BigMart.csv')
testingData = pd.read_csv('../input/Test_BigMart.csv')
combined = pd.concat([trainingData, testingData],ignore_index=True, sort=False)
print (combined.shape)
print (combined.columns)
Y = trainingData['Item_Outlet_Sales']
testingItemIdentifier =  testingData['Item_Identifier'].values
testingOutletIdentifier =  testingData['Outlet_Identifier'].values


# In[ ]:


trainingData.describe()


# In[ ]:


print (combined.apply(lambda x: len(x.unique())))


# In[ ]:


print ("Displyaing the Number of values present in different columns = ")
requiredColumnNames = [ 'Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type','Outlet_Size','Outlet_Type']
for col in requiredColumnNames:
    print ("Number of distinct values present for - ", col)
    print (combined[col].value_counts())
    print ("_________________________")


# In[ ]:


requiredLFIndexVals = combined[(combined['Item_Fat_Content'] == "LF") | (combined['Item_Fat_Content'] == "low fat")].index.values
combined['Item_Fat_Content'].iloc[requiredLFIndexVals] = "Low Fat"

requiredRegIndexVals = combined[(combined['Item_Fat_Content'] == "reg")].index.values
combined['Item_Fat_Content'].iloc[requiredRegIndexVals] = "Regular"
print (combined.Item_Fat_Content.value_counts())


# In[ ]:


#Determine the average weight per item:
ItemAverageWeight = combined.pivot_table(values='Item_Weight', index='Item_Identifier')

#Get a boolean variable specifying missing Item_Weight values
getBooleanData = combined['Item_Weight'].isnull() 

#Impute data and check #missing values before and after imputation to confirm
print ('Orignal #missing: %d'% sum(getBooleanData))
for index,value in getBooleanData.iteritems():
    if(value == True):
        #print (combined.loc[[index]]['Item'])
        requiredItemIdentifier = combined.loc[index]['Item_Identifier']
        #print (requiredItemIdentifier)
        #print (ItemAverageWeight.loc[requiredItemIdentifier]['Item_Weight'])
        combined.loc[index,'Item_Weight'] = ItemAverageWeight.loc[requiredItemIdentifier]['Item_Weight']
print ('Final #missing: %d'% sum(combined['Item_Weight'].isnull()))


# In[ ]:


#Import mode function:
from scipy.stats import mode

#Determing the mode for each
outlet_size_mode = combined.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x.astype(str)).mode[0]) )
print ('Mode for each Outlet_Type:')
#Get a boolean variable specifying missing Item_Weight values
miss_bool = combined['Outlet_Size'].isnull() 

#Impute data and check #missing values before and after imputation to confirm
print ('\nOrignal #missing: %d'% sum(miss_bool))

for index,value in miss_bool.iteritems():
    if(value == True):
        requiredOutletType = combined.loc[index]['Outlet_Type']
        combined.loc[index,'Outlet_Size'] = outlet_size_mode[requiredOutletType]['Outlet_Size']

print ('\n Final #missing: %d'%sum(combined['Outlet_Size'].isnull()))


# In[ ]:


requiredIndexVals = combined[combined['Outlet_Size'] == "nan"].index.values
combined['Outlet_Size'].iloc[requiredIndexVals] = "Medium"
combined.Outlet_Size.value_counts()


# In[ ]:


combined.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
cols = ("Item_Fat_Content", "Outlet_Identifier","Outlet_Location_Type","Outlet_Size", "Outlet_Type", "Item_Type"  )
for c in cols:
    label = LabelEncoder()
    label.fit(list(combined[c].values))
    combined[c] = label.transform(list(combined[c].values))
combined.head()


# In[ ]:


trainData = combined[:trainingData.shape[0]]
plt.figure(figsize=(10,10))
sns.heatmap(trainData.iloc[:, 2:].corr(), annot=True, square=True, cmap='Greens')
plt.show()


# In[ ]:


sns.countplot(x = "Outlet_Size", data = trainData )


# In[ ]:


sns.barplot( x = "Item_Fat_Content", y = "Item_Weight" , data = trainData)


# In[ ]:


import matplotlib.pyplot as plt
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot( ax=ax,x = "Outlet_Identifier", y = "Item_Outlet_Sales" , data = trainingData)


# In[ ]:


plt.figure(figsize=(20,20))
requiredColumns = trainingData.Item_Type.unique()
counter = 1
for col in requiredColumns:
    plt.subplot(4, 4, counter)
    sns.kdeplot(trainingData[trainingData['Item_Type'] == col]['Item_Outlet_Sales'], color='r', label=col, shade=True)
    plt.legend(loc='upper right')
    plt.title(col)
    counter = counter + 1


# In[ ]:


trainingData.Outlet_Size.value_counts()


# In[ ]:


plt.figure(figsize=(20,20))
requiredColumns = trainData.Outlet_Identifier.unique()
counter = 1
for col in requiredColumns:
    plt.subplot(4, 4, counter)
    sns.kdeplot(trainData[trainData['Outlet_Identifier'] == col]['Item_Outlet_Sales'], color='purple', label=col, shade=True)
    plt.legend(loc='upper right')
    plt.title(col)
    counter = counter + 1


# In[ ]:


combined["Item_Outlet_Sales"].fillna(combined["Item_Outlet_Sales"].mean(),inplace = True)
combined = combined.drop("Item_Identifier",axis =1)
combined = combined.drop("Outlet_Establishment_Year",axis =1)
combined = combined.drop("Item_Outlet_Sales",axis =1)
Y = trainingData["Item_Outlet_Sales"]

nrow_train = trainingData.shape[0]
print (nrow_train)
print (combined.shape)
print (combined.columns)
print ("___________________________")
combined = pd.get_dummies(combined, columns=['Outlet_Type','Outlet_Identifier','Item_Fat_Content'])
X_train = combined[:nrow_train]
X_test = combined[nrow_train:]
print (X_train.shape)
print (X_train.columns)
print ("___________________________")
print (X_test.shape)
print (X_test.columns)
print ("___________________________")
#X_train = combined[:nrow_train][['Outlet_Type','Outlet_Identifier','Item_Fat_Content','Item_MRP']]
#X_train = combined[:nrow_train]
#X_train = pd.get_dummies(X_train, columns=['Outlet_Type','Outlet_Identifier','Item_Fat_Content'])
#X_train = pd.get_dummies(X_train , columns=X_train.columns)
#X_train.dtypes


# In[ ]:


from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split

scalar = RobustScaler()
X_train_Scaled = scalar.fit(X_train).transform(X_train)
X_test_Scaled = scalar.fit(X_train).transform(X_test)
y_log = np.log(Y)
t_X, val_X, t_y, val_y = train_test_split(X_train_Scaled,y_log, test_size=0.2)
print (t_X.shape , val_X.shape , t_y.shape, val_y.shape)


# In[ ]:


from xgboost import XGBRegressor

print ("XGBOOST REGRESSOR")
xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.05)
xgb_model.fit(t_X,t_y)
predictions = xgb_model.predict(val_X)

from sklearn.metrics import mean_absolute_error, mean_squared_error
print("Mean Absolute Error : " + str(mean_absolute_error(val_y,predictions)))
print("Mean Squared Error : " + str(mean_squared_error(val_y,predictions)))
print("Root Mean Squared Error : " + str(np.sqrt(mean_squared_error(val_y,predictions))))


# In[ ]:


from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(max_depth=15, random_state=0,n_estimators=500)
rf_model.fit(t_X,t_y)
predictions = rf_model.predict(val_X)

from sklearn.metrics import mean_absolute_error, mean_squared_error
print("Mean Absolute Error : " + str(mean_absolute_error(val_y,predictions)))
print("Mean Squared Error : " + str(mean_squared_error(val_y,predictions)))
print("Root Mean Squared Error : " + str(np.sqrt(mean_squared_error(val_y,predictions))))

cv_score = cross_val_score(rf_model, t_X,t_y, cv=10, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print ("\n Scores ")
print (cv_score)
#Print model report:
print ("\nModel Report")
#print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(val_y,predictions)))
print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),
                                                                         np.min(cv_score),np.max(cv_score)))


# In[ ]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso

print ("RIDGE REGRESSION")
ridge_model = Ridge(alpha=0.05,normalize=True)
ridge_model.fit(t_X,t_y)
predictions = ridge_model.predict(val_X)

from sklearn.metrics import mean_absolute_error, mean_squared_error
print("Mean Absolute Error : " + str(mean_absolute_error(val_y,predictions)))
print("Mean Squared Error : " + str(mean_squared_error(val_y,predictions)))
print("Root Mean Squared Error : " + str(np.sqrt(mean_squared_error(val_y,predictions))))

cv_score = cross_val_score(ridge_model, t_X,t_y, cv=10, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print ("\n Scores ")
print (cv_score)
#Print model report:
print ("\nModel Report")
#print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(val_y,predictions)))
print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),
                                                                         np.min(cv_score),np.max(cv_score)))


# In[ ]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso

print ("LINEAR REGRESSION")
linear_model = LinearRegression(normalize=True)
linear_model.fit(t_X,t_y)
predictions = linear_model.predict(val_X)

from sklearn.metrics import mean_absolute_error, mean_squared_error
print("Mean Absolute Error : " + str(mean_absolute_error(val_y,predictions)))
print("Mean Squared Error : " + str(mean_squared_error(val_y,predictions)))
print("Root Mean Squared Error : " + str(np.sqrt(mean_squared_error(val_y,predictions))))

cv_score = cross_val_score(linear_model, t_X,t_y, cv=10, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print ("\n Scores ")
print (cv_score)
#Print model report:
print ("\nModel Report")
#print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(val_y,predictions)))
print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),
                                                                         np.min(cv_score),np.max(cv_score)))


# In[ ]:


X_test_Scaled = scalar.fit(X_train).transform(X_test)
testingPredictions = xgb_model.predict(X_test_Scaled)


# In[ ]:


testingPredictions
testingPredictions_org = np.exp(testingPredictions)
testingPredictions_org


# In[ ]:


sub = pd.DataFrame({'Item_Identifier' : testingItemIdentifier, 'Outlet_Identifier' : testingOutletIdentifier,
                    'Item_Outlet_Sales' : testingPredictions_org})
sub.to_csv('submission3.csv', index=False)


# In[ ]:





# In[ ]:




