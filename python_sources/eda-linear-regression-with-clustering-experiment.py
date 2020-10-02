#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('figure', figsize=(10,10))


# In[ ]:


# Importing the data
data = pd.read_csv('/kaggle/input/iris/Iris.csv')


# In[ ]:


# Understanding the properties of the columns
data.describe()


# In[ ]:


data.info()


# In[ ]:


# Splitting the data in 3:1 Ratio for Training and Testing
from sklearn.model_selection import train_test_split
x_data = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y_data = data[['Species']]
x_train,x_test, y_train, y_test = train_test_split(x_data,y_data,test_size = 0.33, random_state = 0)


# <h> <b>Working on the training data </h1>

# In[ ]:


# Correlation of various values


# In[ ]:


data_tr = pd.concat([x_train,y_train], axis = 1)
sns.heatmap(data_tr.corr())


# In[ ]:


# The corr function does not consider the Species column as its non-numeric


# In[ ]:


# Understanding the categorical values

figure , ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
sns.violinplot('Species','SepalLengthCm', data = data_tr
            ,ax = ax1
            ,fliersize = 7)
sns.violinplot('Species','SepalWidthCm', data = data_tr
            ,ax = ax2
           ,fliersize = 7)
sns.violinplot('Species','PetalLengthCm', data = data_tr
            ,ax = ax3
           ,fliersize = 7)
sns.violinplot('Species','PetalWidthCm', data = data_tr
            ,ax = ax4
           ,fliersize = 7)


# In[ ]:


# Label Encoding
from sklearn.preprocessing import LabelEncoder
data_tr_speciesCat = data_tr.copy()
lb_species = LabelEncoder()
data_tr_speciesCat['Species'] = lb_species.fit_transform(data_tr_speciesCat['Species'])
data_tr_speciesCat.head()


# In[ ]:


plt.rc('figure'
       , figsize=(20,10))
sns.heatmap(data_tr_speciesCat.corr()
            ,cmap = 'Greens')


# In[ ]:


# But this label encoding will provide weightd to different categories


# In[ ]:


#One Hot Encoding
# Trying out One Hot Encoding to see how the corr gets affected


# In[ ]:


data_tr_OHE = data_tr.copy()
data_tr_OHE = pd.get_dummies(data_tr_OHE
                             ,columns = ['Species']
                             ,prefix = ['Species'])
data_tr_OHE.head()


# In[ ]:


corr_df = data_tr_OHE.corr()
corr_df.drop(['Species_Iris-setosa','Species_Iris-versicolor','Species_Iris-virginica']
             ,axis = 1
             ,inplace = True)
corr_df.drop(['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
             ,axis = 0
             ,inplace = True)
corr_df


# In[ ]:


sns.heatmap(corr_df
            ,vmin = -1
            ,annot = True
            ,cmap = 'BuPu')


# 1. Iris-setosa : Correlation to all variables
# 2. Iris-versicolor : Correlation is not so strong with other variables except for Sepal Width
# 3. Iris-virginica : Correlation with Sepal Length, Petal Length and Petal Width

# <h1> Training a Model </h1>
# <br>
# <h2>Multiple Linear Regression</h2>

# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
X = data_tr_speciesCat[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
Y = data_tr_speciesCat[['Species']]
lm.fit(X,Y)
print('Co-effecient: ',lm.coef_)
print('Intercept: ',lm.intercept_)


# In[ ]:


# Predicting values from this model
yhat = lm.predict(X)


# <h3>Distibution plots between the Actual Values and the Predicted Values

# In[ ]:


ax1 = sns.distplot(data_tr_speciesCat[['Species']], hist = False, color = 'r', label = 'Actual Value')
sns.distplot(yhat, hist = False, color = 'b', label = 'Fitted Value' , ax = ax1)


# Fitted value behaves a lot differently from the actual values
# <br>
# The Actual Values display a distribution of degree 3

# In[ ]:


from sklearn.metrics import mean_squared_error
print('MSE :',mean_squared_error(Y,yhat))
print('R-Squared :',lm.score(X,Y))


# The MSE and R-Squared above are pretty good
# <br>
# But we can do better

# <h2>Polynomial Regression</h2>

# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree = 3)
x_poly = poly_features.fit_transform(X)
poly_lm = LinearRegression()
poly_lm.fit(x_poly,Y)


# In[ ]:


print('Co-effecient: ',poly_lm.coef_)
print('Intercept: ',poly_lm.intercept_)


# In[ ]:


# Predicting values uising this model
y_poly_predict = poly_lm.predict(x_poly)


# <h3>Distibution plots between the Actual Values and the Predicted Values

# In[ ]:


ax1 = sns.distplot(data_tr_speciesCat[['Species']], hist = False, color = 'r', label = 'Actual Value')
sns.distplot(y_poly_predict, hist = False, color = 'b', label = 'Fitted Value' , ax = ax1)


# Above distibution graphs display an almost perfect model for trained data
# <br>
# Let's see how the predicted data looks like

# In[ ]:


y_poly_predict[:5]


# Looks like the values are fractions since we used Polynomial Regression

# In[ ]:


# Plotting the Predicted values against the actual values
polyReg_df = pd.DataFrame({})
polyReg_df['Actual Values'] = data_tr_speciesCat['Species']
polyReg_df['Predicted Values'] = y_poly_predict
plt.scatter(polyReg_df['Actual Values'],polyReg_df.index,label = 'Actual Values')
plt.scatter(polyReg_df['Predicted Values'],polyReg_df.index,label = 'Predicted Values')
plt.legend()


# Predicted Values are scattered around the Actual Values due to the former being fractions
# <br>The Predicted Values remain within the boundaries of 1 Unit so they might do good when rounded off

# In[ ]:


polyReg_round = polyReg_df.copy()
polyReg_round['Predicted Values'] = np.absolute(np.round(polyReg_df['Predicted Values'],0))
polyReg_round.head()


# Plotting the distribution plots again

# In[ ]:


ax1 = sns.distplot(polyReg_round['Actual Values'], hist = False, color = 'r', label = 'Actual Value')
sns.distplot(polyReg_round['Predicted Values'], hist = False, color = 'b', label = 'Fitted Value' , ax = ax1)


# <h3>Both the plots now are in perfect sync

# In[ ]:


print('MSE (non-rounded):',mean_squared_error(Y,y_poly_predict))
print('MSE (rounded):',mean_squared_error(Y,polyReg_round['Predicted Values']))
print('R-Squared :',poly_lm.score(x_poly,Y))


# <b>Implications:</b>
# <br>
# When the predicted values are rounded off, we get 0 error
# <br>
# R-Squared for this model is better then the previous

# In[ ]:


# Instead of using the Rounding Off, let's try clustering for the same


# In[ ]:


from sklearn.cluster import KMeans 
from sklearn import metrics 
from scipy.spatial.distance import cdist 


# In[ ]:


# Getting the optimum cluster value
X = polyReg_df
distortions = [] 
inertias = [] 
mapping1 = {} 
mapping2 = {} 
K = range(1,10) 
  
for k in K: 
    #Building and fitting the model 
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)     
      
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                      'euclidean'),axis=1)) / X.shape[0]) 
    inertias.append(kmeanModel.inertia_) 
  
    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                 'euclidean'),axis=1)) / X.shape[0] 
    mapping2[k] = kmeanModel.inertia_ 


# In[ ]:


plt.plot(K, distortions, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Distortion') 
plt.title('The Elbow Method using Distortion') 
plt.show() 


# In[ ]:


# The elbow is at 3
knn_mod = KMeans(n_clusters = 3)
knn_mod.fit(polyReg_df)


# In[ ]:


knn_df = polyReg_df.copy()
knn_df['Clusters'] = knn_mod.labels_
knn_df['Clusters'].replace({2:1,1:2},inplace = True)


# In[ ]:


knn_df.head()


# In[ ]:


plt.scatter(knn_df['Predicted Values'],knn_df.index,c = knn_df['Clusters'],label = 'Predicted Values')
plt.scatter(knn_df['Actual Values'],knn_df.index,c = 'black', label = 'Actual Values')
# plt.scatter(df_temp['y_rounded'],df_temp.index,marker = 'x')
plt.legend()


# In[ ]:


print('MSE :',mean_squared_error(knn_df['Actual Values'],knn_df['Clusters']))


# Clusters define the section better
# <br>
# Let's try the same 2 methods on the Test data

# <h1> Working on Test Data

# In[ ]:


print('---------------------------------------------------------------------------------------------------------')
print(x_test.head())
print('---------------------------------------------------------------------------------------------------------')
print(y_test.head())
print('---------------------------------------------------------------------------------------------------------')


# In[ ]:


# Encoding the Test Target Data
y_test_cat = y_test.copy()
lb_species = LabelEncoder()
y_test_cat['Species'] = lb_species.fit_transform(y_test_cat['Species'])
y_test_cat.head()


# <h2> Polynomial Regression + Multiple Linear Regression

# In[ ]:


# Fitting Polynomial Features for the Test X data
x_poly_test = poly_features.fit_transform(x_test)


# In[ ]:


y_poly_test = poly_lm.predict(x_poly_test)


# <h3> Checking the distribution plot for the predicted and actual values

# In[ ]:


y_val_df = y_test_cat.copy()
y_val_df['y_predicted'] = y_poly_test


# In[ ]:


ax1 = sns.distplot(y_test_cat, hist = False, color = 'r', label = 'Actual Value')
sns.distplot(y_poly_test, hist = False, color = 'b', label = 'Fitted Value' , ax = ax1)


# Plotting the scatter plots for the Actual Values and the Predicted Values

# In[ ]:


plt.scatter(y_val_df['Species'],y_val_df.index,label = 'Predicted Values')
plt.scatter(y_val_df['y_predicted'],y_val_df.index, label = 'Actual Values')
# plt.scatter(df_temp['y_rounded'],df_temp.index,marker = 'x')
plt.legend()


# Applying the clustering for this data

# In[ ]:


knn_test_mod = KMeans(n_clusters=3)
knn_test_mod.fit(y_val_df[['y_predicted']])
knn_pred_df = y_val_df.copy()
knn_pred_df['Clusters'] = knn_test_mod.labels_


# In[ ]:


plt.scatter(knn_pred_df['Species'],knn_pred_df.index,label = 'Predicted Values')
plt.scatter(knn_pred_df['y_predicted'],knn_pred_df.index,c = knn_pred_df['Clusters'], label = 'Actual Values')
# plt.scatter(df_temp['y_rounded'],df_temp.index,marker = 'x')
plt.legend()


# In[ ]:


sns.scatterplot(data=knn_pred_df, x='Species', y='y_predicted', hue='Clusters')


# **Defining the mapping for the cluster values based on the above Scatter plot to be consistent to the Species label encoding**

# In[ ]:


group_data = knn_pred_df.groupby(['Species','Clusters'])
# for key in gb.groups.keys():
#     print(key,':',gb.get_group(key).count().values[0])


# In[ ]:


# Getting unique values
species_vals = knn_pred_df['Species'].unique()
clusters_vals = knn_pred_df['Clusters'].unique()


# In[ ]:


max_vals = {}
for key in group_data.groups.keys():
    max_vals[key[0]] = 0

map_dict = max_vals.copy()

for key in group_data.groups.keys():
#     print(group_data.get_group(key).count().values[0])
    if max_vals[key[0]] < group_data.get_group(key).count().values[0]:
        max_vals[key[0]] = group_data.get_group(key).count().values[0]
        map_dict[key[0]] = key


# In[ ]:


mapping = {}
for tup in list(map_dict.values()):
    mapping[tup[1]] = tup[0]
mapping


# In[ ]:


knn_pred_df['Clusters'].replace(mapping,inplace = True)


# In[ ]:


sns.scatterplot(data=knn_pred_df, x='Species', y='y_predicted', hue='Clusters')


# In[ ]:


ax1 = sns.distplot(knn_pred_df['Clusters'], hist = False, color = 'b', label = 'Fitted Value' ,kde_kws=dict(linewidth=5,shade = True,alpha = 0.5))
sns.distplot(knn_pred_df['Species'], hist = False, color = 'r', label = 'Actual Value', ax = ax1,kde_kws=dict(linewidth=2))


# In[ ]:


print('MSE :',mean_squared_error(knn_pred_df['Species'],knn_pred_df['Clusters']))


# In[ ]:


# Checking the MSE if we used the Rounding Off logic
print('MSE :',mean_squared_error(knn_pred_df['Species'],np.absolute(np.round(knn_pred_df['Clusters'],0))))


# In[ ]:


ax1 = sns.distplot(np.absolute(np.round(knn_pred_df['Clusters'],0)), hist = False, color = 'b', label = 'Fitted Value' ,kde_kws=dict(linewidth=5,shade = True,alpha = 0.5))
sns.distplot(knn_pred_df['Species'], hist = False, color = 'r', label = 'Actual Value', ax = ax1,kde_kws=dict(linewidth=2))


# Both Clustering and Rounding Off the value provides same results

# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


acc_score = accuracy_score(knn_pred_df['Species'],knn_pred_df['Clusters'])


# In[ ]:


acc_score


# <h1><center> ACCURACY SCORE : 96%

# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_mat = pd.DataFrame(confusion_matrix(knn_pred_df['Species'],knn_pred_df['Clusters']),index = [0,1,2], columns = [0,1,2])


# In[ ]:


confusion_mat


# ---
