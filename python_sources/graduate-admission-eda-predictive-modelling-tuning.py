#!/usr/bin/env python
# coding: utf-8

# Welcome to my new kernal, in this kernal we are going to perform a lot of interesting things and exploration on the graduation-admission dataset. So,let's dive into it --->

# ![image.png](attachment:image.png)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
test = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')


# # Helper Functions

# Func: show_the_data

# In[ ]:


def show_the_data(data):
    data.info()
    print("\n\nThe columns are {}".format(data.columns))
    print("\n\nActual Data\n",data.head(2))


# Func: categorize

# In[ ]:


def categorize(data,col):
    numerical,category=[],[]
    for i in col:
        if data[i].dtype ==object:
            category.append(i)
        else:
            numerical.append(i)
    print("The numerical features {}:".format(numerical))
    print("The categorical features {}:".format(category))
    return category,numerical


# Func: get_correlated

# In[ ]:


def get_correlated(cor):
    correlated =set()
    for i in cor.columns:
        for j in cor.columns:
            if cor[i][j]>0.8  and i!=j:
                correlated.add(i)
                correlated.add(j)
    print("The Correlated columns: {}".format(list(correlated)))
    return correlated


# In[ ]:


show_the_data(train)


# In[ ]:


show_the_data(test)


# In[ ]:


cat,num = categorize(train,train.columns)


# In[ ]:


print("There are {} categorical features in train dataset".format(len(cat)))
print("There are {} numerical features in train dataset".format(len(num)))


# In[ ]:


cat1,num1 = categorize(test,test.columns)


# In[ ]:


print("There are {} categorical features in test dataset".format(len(cat1)))
print("There are {} numerical features in test dataset".format(len(num1)))


# One thing about this dataset is that it doesn't have any categorical values therefore there is no need to perform categorical to numerical conversion which is one of the crucial steps in Feature Engineering

# # <center>LET'S CHECK FOR MISSING VALUES</center>

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# The good news is that we don't have any missing values so now we can exclude handling missing values step which i one another important step in feature engineering.

# # <center>Visualizing the Target Feature's Distribution</center>

# In[ ]:


plt.figure(figsize = (10,10))
plt.subplot(2,2,1)
#train['Chance of Admit '].hist()
sns.distplot(train['Chance of Admit '],bins=50,color='Violet',  kde_kws={"color": "g", "lw": 5, "label": "KDE"},hist_kws={"linewidth": 5,"alpha": 0.8 })
plt.subplot(2,2,2)
sns.boxplot(train['Chance of Admit '])


# <b>As we are having a lot of numerical features it's our main job to find out the highly correlated features so that we can perform further analysis</b>

# In[ ]:


corr = get_correlated(train.corr())


# In[ ]:


if len(corr) == len(train.columns):
    print("ALL THE FEATURES ARE HIGHELY CORRELATED!!!")
else:
    print("THERE ARE SOME FEATURES WITH LOW CORRELATION...")


# In[ ]:


plt.figure(figsize =(10,10))
sns.heatmap(train.corr(),annot= True,cmap = 'rocket')


# NOW OUR JOB IS VERY SIMPLE
# * ONLY TAKE THE HIGHLY CORRELATED FEATURES
# * EXAMINE THEIR DISTRIBUTION AND SPREAD
# * PERFORM SCALING AND NORMALIZATION
# * CREATE MODELS
# * PERFORM CROSS VALIDATION
# * TUNE THEM

# In[ ]:


data = train[corr]
test_data = test[corr]


# In[ ]:


sns.pairplot(data)


# In[ ]:



data.describe().T


# # OUTLIER DETECTION USING Z-SCORE METHOD

# In[ ]:



def outlier(data):
    out1=[]
    for col in data.columns:
        outliers =[]
        mean = data[col].mean()
        std = data[col].std()
        for i in data[col]:
                z = (i - mean)/std
                if z>2:
                    outliers.append(i)
        out1.append(list(outliers))
        print("There are {} outliers in {} feature".format(len(outliers),col))
    return out1


# In[ ]:


out = outlier(data)


# **To Delete the ouliers I will first replace all the outliers values with np.nan and will delete them using dropna method**

# In[ ]:


j =0
columns =data.columns
for i in out:
    for val in data[columns[j]]:
        if val in i:
            data[columns[j]]= data[columns[j]].replace(val,np.nan)
    j =j+1


# In[ ]:


data.isnull().sum()


# In[ ]:


data.dropna(axis = 0,inplace =True)


# In[ ]:


data.info()


# # Visualizing the data

# In[ ]:


fig = px.density_contour(data, x="CGPA", y="Chance of Admit ")
fig.show()


# In[ ]:


plt.figure(figsize = (10,10))
sns.jointplot(data=data, x="TOEFL Score", y="Chance of Admit ",color='Indigo', marker="*", s=100)


# In[ ]:


#px.area(data, x="GRE Score", y="Chance of Admit ")
fig = px.scatter(data, x="GRE Score", y="Chance of Admit ", marginal_y="box", marginal_x="histogram")
fig.show()


# # <center>3-D Visualization of CGPA, GRE Score and TOEFL Score</center>

# In[ ]:


fig = px.scatter_3d(data, x="CGPA", y="GRE Score", z="TOEFL Score", hover_name="Chance of Admit ",)
fig.show()


# # Lets Seperately Analyse Academic, TOEFL and GRE Toppers

# In[ ]:


toppers=data[data['CGPA']>=9.5].sort_values(by=['CGPA'],ascending=False)
print('There are {} university toppers'.format(len(toppers)))
sns.barplot(x='CGPA',y='Chance of Admit ',data=toppers, linewidth=1.5,edgecolor="0.1")


# In[ ]:


GREtoppers=data[data['GRE Score']>=330].sort_values(by=['GRE Score'],ascending=False)
print('There are {} GRE toppers'.format(len(GREtoppers)))
sns.barplot(x='GRE Score',y='Chance of Admit ',data=GREtoppers, linewidth=1.5,edgecolor="0.1")


# In[ ]:


Toefltoppers=data[data['TOEFL Score']>=115].sort_values(by=['TOEFL Score'],ascending=False)
print('There are {} TOEFL toppers'.format(len(Toefltoppers)))
sns.barplot(x='TOEFL Score',y='Chance of Admit ',data=Toefltoppers, linewidth=1.5,edgecolor="0.1")


# # Dividing the data into dependent and independent features

# In[ ]:


y  = data['Chance of Admit ']
t_test = test_data['Chance of Admit ']
data.drop(['Chance of Admit '],axis =1,inplace =True)
test_data.drop(['Chance of Admit '],axis = 1,inplace = True)


# # Performing Standardization

# In[ ]:


from sklearn.preprocessing import StandardScaler as SS
ss = SS()
data_ss = ss.fit_transform(data)
ss1 = SS()
test_ss = ss.fit_transform(test_data)


# # Let's try some machine learning models

# In[ ]:


from sklearn.linear_model import LinearRegression as LR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.svm import SVR 
from sklearn.linear_model import Ridge as RR
from sklearn.metrics import r2_score


# In[ ]:


key = ['Linear Regression','Decision Tree Regression','Random Forest Regression','Gradient Boosting Regression','Ada Boosting Regression','K-Neighbors Regression','Support Vector Regression','Lasso Regression']
value = [LR(),DTR(),RFR(),GBR(),ABR(),KNR(),SVR(),RR()]
pred=[]
models = dict(zip(key,value))
print(models)


# In[ ]:


for name,algo in models.items():
    model=algo
    model.fit(data_ss,y)
    predictions = model.predict(test_ss)
    acc=r2_score(t_test, predictions)
    pred.append(acc)
    print(name,acc)


# # Plotting the accuracy of each model

# In[ ]:


sns.barplot(y=key,x=pred,linewidth=1.5,orient ='h',edgecolor="0.1")


# # Without any hyper parameter tuning we found that KNN Algorithm is good in terms of accuracy. Let's Perform Hyper Parameter Tuning on it

# In[ ]:


n_neighbors = list(np.arange(1,6))
weights = ['uniform','distance']
algorithm = ['auto','ball_tree','kd_tree','brute']
metric =['euclidean','manhattan','chebyshev','minkowski']
p =[1,2]
leaf_size = list(np.arange(20,200,40))
random_grid = {'n_neighbors':n_neighbors,'weights':weights,'p':p,'leaf_size':leaf_size,'algorithm':algorithm,'metric':metric}
print(random_grid)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
rf = KNR()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(data_ss, y)


# In[ ]:


rf_random.best_estimator_


# In[ ]:


final = KNR(algorithm='kd_tree', leaf_size=60, metric='chebyshev',
                    metric_params=None, n_jobs=None, n_neighbors=5, p=1,
                    weights='distance')
final.fit(data_ss,y)


# # Evaluating the effect of hyperparmeter optimization

# In[ ]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


# In[ ]:


base_model = KNR()
base_model.fit(data_ss, y)
base_accuracy = evaluate(base_model, test_ss, t_test)


# In[ ]:


best_model = final
best_accuracy = evaluate(best_model, test_ss, t_test)


# In[ ]:


print('Improvement of {:0.2f}%.'.format( 100 * (best_accuracy - base_accuracy) / base_accuracy))


# As there is a improvement we will use our best model  as  our final model

# In[ ]:


final = best_model
pred = final.predict(test_ss)
pred[:20]


# In[ ]:


sns.distplot(t_test,hist=False,label = 'Actual')
sns.distplot(pred,hist=False, label ='Predicted')
plt.legend(loc="upper left")
plt.xlabel('Prediction level')


# ![image.png](attachment:image.png)

# **Thanks for viewing my work. Please upvote this if you like it and share this to your friends for whom this will be helpfull!!!!.**
