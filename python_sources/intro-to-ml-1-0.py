#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **This will be your workspace for Kaggle's Machine Learning education track.**
# 
# You will build and continually improve a model to predict housing prices as you work through each tutorial.  Fork this notebook and write your code in it.
# 
# The data from the tutorial, the Melbourne data, is not available in this workspace.  You will need to translate the concepts to work with the data in this notebook, the Iowa data.
# 
# Come to the [Learn Discussion](https://www.kaggle.com/learn-forum) forum for any questions or comments. 
# 
# # Write Your Code Below
# 
# 

# In[ ]:


import pandas as pd

main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
print('hello world')


# In[ ]:


print(data.describe())


# In[ ]:


print(data.columns)


# In[ ]:


data.SalePrice.head()


# In[ ]:


data[['SalePrice', 'LotArea']].describe()


# In[ ]:


y = data.SalePrice


# In[ ]:


X = data[['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]


# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


iowaModel = DecisionTreeRegressor()


# In[ ]:


iowaModel.fit(X,y)


# In[ ]:


X.head()


# In[ ]:


price_predictions = iowaModel.predict(X)


# In[ ]:


from sklearn.metrics import mean_absolute_error


# In[ ]:


mean_absolute_error(y,price_predictions)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# In[ ]:


iowaModelV2 = DecisionTreeRegressor()
iowaModelV2.fit(train_X, train_y)
iowaPredictionsV2 = iowaModelV2.predict(val_X)
mean_absolute_error(val_y, iowaPredictionsV2)


# In[ ]:


def getMae(maxLeafNodes, predVals, predTrain, targetVals, targetTrain):
    model = DecisionTreeRegressor(max_leaf_nodes=maxLeafNodes, random_state=0)
    model.fit(predTrain, targetTrain)
    predictions = model.predict(predVals)
    mae = mean_absolute_error(targetVals, predictions)
    return(mae)


# In[ ]:


for maxNodes in [5,50,100,250,500,5000]:
    mae = getMae(maxNodes, val_X, train_X, val_y, train_y)
    print("Max Leaf Nodes: %d \t\t Mean Abs Error: %d" %(maxNodes, mae))


# In[ ]:


"Random Forest"
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

rfModel = RandomForestRegressor()
rfModel.fit(train_X, train_y)
rfPredictions = rfModel.predict(val_X)
rfmae = mean_absolute_error(val_y ,rfPredictions)
print(rfmae)


# In[ ]:


trainData = data.drop(['SalePrice'], axis=1)
trainData = trainData[['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]
trainTarget = data['SalePrice']

rfModel.fit(trainData, trainTarget)

testFilePath = '../input/test.csv'
testData = pd.read_csv(testFilePath)
testDataModel = testData[['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]


rfPredictionsFinal = rfModel.predict(testDataModel)
print(rfPredictionsFinal)


# In[ ]:


firstSubmission = pd.DataFrame({'Id': testData.Id, 'SalePrice': rfPredictionsFinal})


# In[ ]:


firstSubmission.to_csv('submission.csv', index=False)

