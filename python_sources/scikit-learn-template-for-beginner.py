#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import plotly as py
import plotly.graph_objs as go
py.offline.init_notebook_mode(connected = True)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff


# # REGRESSION

# In[ ]:


diamonds = pd.read_csv("/kaggle/input/diamonds/diamonds.csv")


# In[ ]:


diamonds.info()


# In[ ]:


sns.lmplot(y="carat", x="price", hue="clarity", data= diamonds, fit_reg= False)


# In[ ]:


sns.lmplot(y="carat", x="price", hue="cut", data= diamonds, fit_reg= False)


# In[ ]:


sns.set(style = "whitegrid", font_scale = 1.5)

f, axes = plt.subplots(3, figsize = (8,16))

sns.countplot(y = "clarity", data = diamonds, ax = axes[0])

sns.countplot(y = "color", data = diamonds, ax = axes[1])

sns.countplot(y = "cut", data = diamonds, ax = axes[2])

plt.tight_layout()


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
cut = diamonds.iloc[:,1:2]
color = diamonds.iloc[:,2:3]
clarity = diamonds.iloc[:,3:4]
cut = ohe.fit_transform(cut).toarray()
color = ohe.fit_transform(color).toarray()
clarity = ohe.fit_transform(clarity).toarray()

diamonds.drop(columns = ['cut', 'color', 'clarity'], inplace = True)

cut = pd.DataFrame(cut)
color = pd.DataFrame(color)
clarity = pd.DataFrame(clarity)

diamonds = pd.concat([diamonds, cut, color, clarity], axis = 1)

X = diamonds.drop(columns = 'price').values
Y = diamonds.iloc[:,3:4].values

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X, Y,test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


# # OLS Regression Results

# In[ ]:


import statsmodels.api as sm 

X_l = diamonds.iloc[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
].values
r_ols = sm.OLS(endog = diamonds.iloc[:,-1:], exog =X_l).fit()

print(r_ols.summary())


# **Significance Level: 0.05 we do not screen because there is no column exceeding this value.**

# ## Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)
y_pred_linReg = lin_reg.predict(x_test)

print(r2_score(y_test, y_pred_linReg))

print('coef', lin_reg.coef_,'\n\n')

print('intercept', lin_reg.intercept_)


# ## Polynomial Regression

# In[ ]:



from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 1) 
x_poly = poly_reg.fit_transform(x_train) 
x_poly2 = poly_reg.fit_transform(x_test) 
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y_train)
y_pred_poly = lin_reg.predict(x_poly2)

print(r2_score(y_test, y_pred_poly))


# ## Support Vector Regression

# In[ ]:


from sklearn.svm import SVR
svr_reg = SVR(kernel = 'linear')
svr_reg.fit(X_train, y_train)
y_pred_svr = svr_reg.predict(X_test)

print(r2_score(y_test, y_pred_svr))


# Kernel trick: 'rbf', 'poly', 'linear', 'sigmoid', 'precomputed'

# ## Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state = 0)
dt_reg.fit(X_train,y_train)
y_pred_dt = dt_reg.predict(X_test)

print(r2_score(y_test, y_pred_dt))


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(random_state = 0, n_estimators = 100)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

print(r2_score(y_test, y_pred_rf))


# ## XGBRegressor

# In[ ]:


from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators = 100)
xgb.fit(x_train, y_train)
y_pred_xgb = xgb.predict(x_test)

print(r2_score(y_test, y_pred_xgb))


# In[ ]:





# # CLASSIFICATION

# In[ ]:


iris = sns.load_dataset('iris')


# In[ ]:


iris.head()


# In[ ]:


iris.info()


# In[ ]:


for n in range(0,150):
    if iris['species'][n] == 'setosa':
        plt.scatter(iris['sepal_length'][n], iris['sepal_width'][n], color = 'red')
        plt.xlabel('sepal_length')
        plt.ylabel('sepal_width')
    elif iris['species'][n] == 'versicolor':
        plt.scatter(iris['sepal_length'][n], iris['sepal_width'][n], color = 'blue')
        plt.xlabel('sepal_length')
        plt.ylabel('sepal_width')
    elif iris['species'][n] == 'virginica':
        plt.scatter(iris['sepal_length'][n], iris['sepal_width'][n], color = 'green')
        plt.xlabel('sepal_length')
        plt.ylabel('sepal_width')


# In[ ]:


sns.lmplot(x = 'sepal_length', y = 'sepal_width', data = iris, hue = 'species', col = 'species')


# ## Preprocessing

# In[ ]:


X = iris.iloc[:,:4].values
y = iris.iloc[:,4:5].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


# ## Modelling

# In[ ]:


def model_evaluate(model, test):
    y_pred = model.predict(test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    categories = ['Setosa', 'Versicolor', 'Virginica']
    
    sns.heatmap(cm, cmap = 'Blues', fmt = '', annot = True,
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

model_evaluate(model, X_test)


# ## Support Vector Classifier

# In[ ]:


from sklearn.svm import SVC
model = SVC(kernel = 'linear') #kernel = poly, rbf, precomputed
model.fit(X_train, y_train)

model_evaluate(model, X_test)


# ## Naive Bayes
# 
# * **Gaussian Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)

model_evaluate(model, X_test)


# * **Multinomial Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)

model_evaluate(model, x_test)


# * **Complement Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import ComplementNB
model = ComplementNB()
model.fit(x_train, y_train)

model_evaluate(model, x_test)


# * **Bernoulli Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
model.fit(X_train, y_train)

model_evaluate(model, X_test)


# * **Categorical Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import CategoricalNB
model = CategoricalNB()
model.fit(x_train, y_train)

model_evaluate(model, x_test)


# ## KNeighbors Classifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski')
model.fit(X_train, y_train)

model_evaluate(model, X_test)


# ## Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state = 0)
model.fit(X_train, y_train)

model_evaluate(model, X_test)


# ## Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
model.fit(X_train, y_train)

model_evaluate(model, X_test)


# ## AdaBoost

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators = 50)
model.fit(X_train, y_train)

model_evaluate(model, X_test)


# ## Other
# 
# * **XGBClassifier**

# In[ ]:


from xgboost import XGBClassifier
model = XGBClassifier(n_estimators = 100)
model.fit(X_train, y_train)

model_evaluate(model, X_test)


# * **ExplainableBoostingClassifier**

# In[ ]:


get_ipython().system('pip install interpret')


# In[ ]:


from interpret.glassbox import ExplainableBoostingClassifier
ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)

model_evaluate(model, X_test)


# In[ ]:





# # CLUSTERING

# In[ ]:


mall_customers = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')


# In[ ]:


mall_customers.head()


# In[ ]:


mall_customers.info()


# In[ ]:


mall_customers.describe()


# In[ ]:


lab = mall_customers["Gender"].value_counts().keys().tolist()
val = mall_customers["Gender"].value_counts().values.tolist()

trace = go.Pie(labels = lab ,
               values = val ,
               marker = dict(colors =  [ 'royalblue' ,'lime'],
                             line = dict(color = "white",
                                         width =  1.3)
                            ),
               rotation = 20,
               hoverinfo = "label+value+text",
               hole = .5
              )
layout = go.Layout(dict(title = "Customer attrition in data",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                       )
                  )

data = [trace]
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# In[ ]:


sns.set(style="darkgrid",font_scale=1.5)
f, axes = plt.subplots(1,3,figsize=(20,8))
sns.distplot(mall_customers["Age"], ax = axes[0], color = 'y')     
sns.distplot(mall_customers["Annual Income (k$)"], ax = axes[1], color = 'g')
sns.distplot(mall_customers["Spending Score (1-100)"],ax = axes[2], color = 'r')
plt.tight_layout()


# In[ ]:


dz=ff.create_table(mall_customers.groupby('Gender').mean())
py.iplot(dz)


# In[ ]:


plt.figure(figsize=(8,4))
sns.heatmap(mall_customers.corr(),annot=True,cmap=sns.cubehelix_palette(light=1, as_cmap=True),fmt='.2f',linewidths=2)
plt.show()


# In[ ]:


x = mall_customers.iloc[:,2:]
print(x.head())
x = x.values


# In[ ]:


kMeans = KMeans(n_clusters = 3, init = 'k-means++')
y_pred = kMeans.fit_predict(x)
print('Pred:\n', y_pred)
print('\n\ninertia: ', kMeans.inertia_, '\n\nclusters centers:\n', kMeans.cluster_centers_)


# In[ ]:


result = []
for i in range(1, 12):
    kMeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 123)
    kMeans.fit(x)        
    result.append(kMeans.inertia_)


plt.plot(range(1,12), result)
plt.title('WCSS')
plt.show()


# In[ ]:


kMeans = KMeans(n_clusters = 6, init = 'k-means++') 
y_pred_kMeans = kMeans.fit_predict(x)
print('Pred:\n', y_pred_kMeans)
print('\n\ninertia: ', kMeans.inertia_, '\n\nclusters centers:\n', kMeans.cluster_centers_)


# ## Hierarchical Clustering

# In[ ]:


agglomerative = AgglomerativeClustering(n_clusters = 6, affinity = 'euclidean', linkage = 'ward')
y_pred_agg = agglomerative.fit_predict(x)
print('Pred:\n', y_pred_agg)


# In[ ]:


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.show()


# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, sharey='col', num = 10, figsize = (15,5))

ax1.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = mall_customers , c = y_pred_kMeans,s = 100)
ax1.title.set_text('KMeans')

ax2.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = mall_customers , c = y_pred_agg,s = 100)
ax2.title.set_text('Agglomerative')
f.show()


# **throwing the age column**

# In[ ]:


x = mall_customers.iloc[:,3:].values


# In[ ]:


kMeans = KMeans(n_clusters = 6, init = 'k-means++') 
y_pred_kMeans = kMeans.fit_predict(x)
print('Pred:\n', y_pred_kMeans)
print('\n\ninertia: ', kMeans.inertia_, '\n\nclusters centers:\n', kMeans.cluster_centers_)

result = []
for i in range(1, 14):
    kMeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 123)
    kMeans.fit(x)        
    result.append(kMeans.inertia_)


plt.plot(range(1,14), result)
plt.title('WCSS')
plt.show()


# In[ ]:


print('K-Means')
kMeans = KMeans(n_clusters = 5, init = 'k-means++') 
y_pred_kMeans = kMeans.fit_predict(x)
print('Pred:\n', y_pred_kMeans)
print('\n\ninertia: ', kMeans.inertia_, '\n\nclusters centers:\n', kMeans.cluster_centers_)

print('\n\nAgglomerative')
agglomerative = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_pred_agg = agglomerative.fit_predict(x)
print('Pred:\n', y_pred_agg)


# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, sharey='col', num = 10, figsize = (15,5))

ax1.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = mall_customers , c = y_pred_kMeans,s = 100)
ax1.title.set_text('K-Means')
ax2.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = mall_customers , c = y_pred_agg,s = 100)
ax2.title.set_text('Agglomerative')
f.show()

