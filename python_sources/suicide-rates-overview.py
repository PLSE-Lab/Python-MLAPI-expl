#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy import array


# In[ ]:


#df1 = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv', delimiter=',', nrows = nRowsRead)
df1 = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv', delimiter=',')
df1.dataframeName = 'master.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)


# In[ ]:


print(df1.describe())
print("----------------------------")
print(df1.dtypes)
print(df1.groupby('HDI for year').size())
print("----------------------------")


# In[ ]:


#Deletion of unnecessary columns
print(df1.columns.values)
df1 = df1.drop("suicides_no", axis=1)
df1 = df1.drop("country-year", axis=1)
df1 = df1.drop(" gdp_for_year ($) ", axis=1)
print("df1.shape after column deletion= ",df1.shape)
print(df1.dtypes)

#Check missing data
print("Missing data----------------------")
missing_val_count_by_column = (df1.isnull().sum())
print(missing_val_count_by_column)
print("HDI for year= ")
print(df1['HDI for year'].head(4))
print("HDI for year mean= ",df1['HDI for year'].mean())
print("HDI for year min= ",df1['HDI for year'].min())
print("HDI for year max= ",df1['HDI for year'].max())

print("Data after filling NaN----------------------") #filling NaN with mean value

df1['HDI for year']=df1['HDI for year'].fillna(df1['HDI for year'].mean())
print("HDI for year= ")
print(df1['HDI for year'].head(4))
missing_val_count_by_column2 = (df1.isnull().sum())
print(missing_val_count_by_column2)


# In[ ]:


# Categorical encoding 
df1["country"] = df1["country"].astype('category')  #Change type from object to category
df1["sex"] = df1["sex"].astype('category')
df1["age"] = df1["age"].astype('category')
df1["generation"] = df1["generation"].astype('category')
print(df1.dtypes)

df1["country_cat"] = df1["country"].cat.codes #Append new column to 
df1["sex_cat"] = df1["sex"].cat.codes
df1["age_cat"] = df1["age"].cat.codes
df1["generation_cat"] = df1["generation"].cat.codes

print("df1.shape after adding categorical column= ",df1.shape)
print(df1.head(5))
print("unique values country_cat = ",df1["country_cat"].nunique())
print("unique values sex_cat = ",df1["sex_cat"].nunique())
print("unique values age_cat = ",df1["age_cat"].nunique())
print("unique values generation_cat = ",df1["generation_cat"].nunique())



# In[ ]:


#Check relationship of each features

df1.plot(kind="scatter", x="population", y="suicides/100k pop")
df1.plot(kind="scatter", x="country_cat", y="suicides/100k pop")
df1.plot(kind="scatter", x="age_cat", y="suicides/100k pop")
df1.plot(kind="scatter", x="generation_cat", y="suicides/100k pop")
df1.plot(kind="scatter", x="year", y="suicides/100k pop")
df1.plot(kind="scatter", x="HDI for year", y="suicides/100k pop")
df1.plot(kind="scatter", x="gdp_per_capita ($)", y="suicides/100k pop")
df1.plot(kind="scatter", x="sex_cat", y="suicides/100k pop")

'''
#fails to create automatic loops
print(df1.columns[1])
print(df1.columns.dtype)

a=df1.columns.to_numpy()
print("a[0:3]====================")
print(a[0:3])

#str_obj=repr(a[0])
#str_obj=str(a[0])
#astr="country"
b = df1.columns.map(str)
#b = df1.columns.astype(str)
print(b)
print(b[0]=="country")


for i in range (df1.shape[1]):
    #print(df1.columns[i])
    #str=df1.columns[i]
    str="population"
    #str=str(a[i])
    df1.plot(kind="scatter", x=str, y="suicides/100k pop")


    
'''


# In[ ]:


import seaborn as sns

correlation_matrix = df1.corr()
print(correlation_matrix)

sns.heatmap(correlation_matrix) #most features are unrelated to suicides rate except sex_cat


# In[ ]:


#X & y separation
X=df1.drop("suicides/100k pop", axis=1)
X=X.drop("country", axis=1)
X=X.drop("sex", axis=1)
X=X.drop("age", axis=1)
X=X.drop("generation", axis=1)

y=df1["suicides/100k pop"]

print("X.shape= ",X.shape)
print(X.head(3))
print(X.dtypes)
print("------------------------------------------")
print("y.shape= ",y.shape)
print(y.head(3))
print(y.dtypes)


# In[ ]:


# Split training & test
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
print("X_train.shape= ",X_train.shape)
print("X_test.shape= ",X_test.shape)
print("y_train.shape= ",y_train.shape)
print("y_test.shape= ",y_test.shape)


# In[ ]:


from sklearn.preprocessing import StandardScaler

print(X_train.head(3))

scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)   # train only on X_train not the whole set
print("X after StandardScaler-------------------------")
print(X_train[:3,:])

X_test = scaler.transform (X_test)  # only transform


# In[ ]:


# Training & Evaluation

# LinearRegression (vanilla)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

classifier_linreg =  LinearRegression()
classifier_linreg.fit(X_train, y_train)

y_pred = classifier_linreg.predict(X_test)

print ("Linreg (vanilla)-----------------------------")
print("Training error: " + str(mean_squared_error(y_train, classifier_linreg.predict(X_train))))
print("Test error: " + str(mean_squared_error(y_test, classifier_linreg.predict(X_test))))
print ("Use CV, score= -----------------------------")
CV_score=cross_val_score(classifier_linreg, X_train, y_train,scoring="neg_mean_squared_error", cv=4)
print(CV_score)


# LinearRegression (poly)
trafo = PolynomialFeatures(4)      #set degree of polynomials = 4
X_train_poly = trafo.fit_transform(X_train)
X_test_poly = trafo.fit_transform(X_test)
print ("")
classifier_linreg_poly = LinearRegression()
classifier_linreg_poly.fit(X_train_poly, y_train)
print ("Linreg (polynomials)-----------------------------")
print("Training error Poly: " + str(mean_squared_error(y_train, classifier_linreg_poly.predict(X_train_poly))))
print("Test error Poly: " + str(mean_squared_error(y_test, classifier_linreg_poly.predict(X_test_poly))))
print ("Use CV, score= -----------------------------")
CV_score=cross_val_score(classifier_linreg_poly, X_train_poly, y_train,scoring="neg_mean_squared_error", cv=4)
print(CV_score)
#Test Error is getting worse if we increase the degree of polynomials


# In[ ]:


# LinearRegression (vanilla) for only 1 feature

print ("Linreg (vanilla) for only 1 feature----------------------------")

feature_idx=[]

for i in range(X_train.shape[1]):    
    feature_idx.append(i)

for i in feature_idx:
    X_train_01 =X_train[:,i]
    X_test_01 =X_test[:,i]

    classifier_linreg_01 =  LinearRegression()
    classifier_linreg_01.fit(X_train_01.reshape(-1, 1), y_train)

    y_pred = classifier_linreg_01.predict(X_test.reshape(-1, 1))

    print("Use feature no",i)
    print("   Training error: " + str(mean_squared_error(y_train, classifier_linreg_01.predict(X_train_01.reshape(-1, 1)))))
    print("   Test error: " + str(mean_squared_error(y_test, classifier_linreg_01.predict(X_test_01.reshape(-1, 1)))))


# In[ ]:


# DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

tree_reg = DecisionTreeRegressor()
tree_reg. fit(X_train, y_train)

y_pred = tree_reg. predict(X_test)
tree_mse = mean_squared_error(y_test, y_pred)
tree_rmse = np.sqrt(tree_mse)
print("tree_rmse= ",tree_rmse)
print("y_train.mean= ",y_train.mean())

print("----------Use CV-------------------")
scores = cross_val_score(tree_reg, X_train, y_train,scoring="neg_mean_squared_error", cv=4)
rmse_scores = np.sqrt(-scores)
print("Scores: ", scores)
print("Mean: ", scores.mean())
print("Standard deviation: ", scores.std())


# In[ ]:


# RandomForestRegressor
from sklearn. ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)

y_pred = forest_reg. predict(X_test)
tree_mse = mean_squared_error(y_test, y_pred)
tree_rmse = np.sqrt(tree_mse)
print("tree_rmse= ",tree_rmse)
print("y_train.mean= ",y_train.mean())

print("----------Use CV-------------------")
scores = cross_val_score(tree_reg, X_train, y_train,scoring="neg_mean_squared_error", cv=4)
rmse_scores = np.sqrt(-scores)
print("Scores: ", scores)
print("Mean: ", scores.mean())
print("Standard deviation: ", scores.std())


# #Shallow NN
# from keras import layers, models, optimizers
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.metrics import categorical_accuracy
# from keras.utils.vis_utils import plot_model
# 
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# 
# input_size=X_train.shape[1]
# '''
# net = Sequential()
# net.add(Dense(100, activation='relu' , input_shape=(input_size,)))
# net.add(Dense(1, activation='sigmoid' ))
# net.compile(loss='binary_crossentropy' , optimizer=optimizers.Adam(), metrics=['accuracy'] )
# net.summary()
# 
# plot_model(net, show_shapes=True, show_layer_names=True)
# 
# net.fit(X_train, y_train, epochs=10, verbose=0)
# print("training error: " + str(net.evaluate(X_train, y_train, verbose=0)))
# print("test error: " + str(net.evaluate(X_test, y_test, verbose=0)))
# '''
# 
# def baseline_model():
#     net = Sequential()
#     net.add(Dense(100, activation='relu' , input_shape=(input_size,)))
#     net.add(Dense(1, activation='sigmoid' ))
#     net.compile(loss='binary_crossentropy' , optimizer=optimizers.Adam(), metrics=['accuracy'] )
#     return net
# 
# estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
# kfold = KFold(n_splits=2)
# results = cross_val_score(estimator, X_train, y_train, cv=kfold)
# print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# 
# # NN performs the worse

# # Deep NN
# 
# from keras.layers import Conv1D, MaxPooling1D
# 
# input_size=X_train.shape[1]
# '''
# input_layer = layers.Input(shape=(input_size, ))
# 
# # Add the convolutional Layer
# #conv_layer = layers.Convolution1D(100, 3, activation="relu")(input_layer)
# conv_layer = layers.Convolution1D(100, 3, activation="relu")(input_layer)
# 
# model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
# 
# # Add the pooling Layer
# pooling_layer = layers.GlobalMaxPool1D()(conv_layer)
# 
# # Add the output Layers
# output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
# output_layer1 = layers.Dropout(0.25)(output_layer1)
# output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)
# # Compile the model
# model = models.Model(inputs=input_layer, outputs=output_layer2)
# model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
# plot_model(model, show_shapes=True, show_layer_names=True)
# '''
# #-------------------------------------------------------------------------
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# 
# #-------------------------------------------------------------------------
# 
# 

# In[ ]:


'''
model.fit(X_train, y_train)
print("training error: " + str(model.evaluate(X_train, y_train, verbose=0)))
print("test error: " + str(model.evaluate(X_test, y_test, verbose=0)))

#accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, valid_y, is_neural_net=True)
print ("CNN accuracy",  accuracy)
'''

