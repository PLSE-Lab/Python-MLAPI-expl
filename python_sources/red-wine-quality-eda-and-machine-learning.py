#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# 
# ## 1-Data Visualization Part:
# * [Histograms](#1)
# * [Countplot](#2)
# * [Correlation Map](#3)
# * [Pairplot](#4)
# * [Catplots](#5)
# * [Pointplot](#6)
# * [3D Scatterplot](#7)
# 
# ## 2-Machine Learning Part:
# * [Feature Selection](#8)
# * [Standardizing](#9)
# * [Cross Validation](#10)
# 
# 

# # Descriptions
# 
# * fixed acidity: most acids involved with wine or fixed or nonvolatile (do not evaporate readily)
# * volatile acidity:the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste
# * citric acid:found in small quantities, citric acid can add 'freshness' and flavor to wines
# * residual sugar:the amount of sugar remaining after fermentation stops, it's rare to find wines with less than 1 gram/liter and wines with greater than 45 grams/liter are considered sweet
# * chlorides:the amount of salt in the wine
# * free sulfur dioxide:the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine
# * total sulfur dioxide:amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine
# * densitythe density of water is close to that of water depending on the percent alcohol and sugar content
# * pH:describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale
# * sulphates: a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and antioxidant
# * alcohol:the percent alcohol content of the wine
# * quality: output variable (based on sensory data, score between 0 and 10)

# In[ ]:


import numpy as np # Numerical Python
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline import iplot
import plotly.offline as py
py.init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings('ignore') 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
sns.set()


# In[ ]:


dataset=pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


dataset.describe().T


# # 1- Data Visualization Part

# <a id="1"></a><br>
# ## Histograms

# In[ ]:


plt.figure(figsize=(15,9))
sns.distplot(dataset['fixed acidity'])
plt.show()


# In[ ]:


plt.figure(figsize=(15,9))
sns.distplot(dataset['volatile acidity'])
plt.show()


# In[ ]:


plt.figure(figsize=(15,9))
sns.distplot(dataset['citric acid'])
plt.show()


# In[ ]:


plt.figure(figsize=(15,9))
sns.distplot(dataset['residual sugar'])
plt.show()


# In[ ]:


plt.figure(figsize=(15,9))
sns.distplot(dataset['chlorides'])
plt.show()


# In[ ]:


plt.figure(figsize=(15,9))
sns.distplot(dataset['free sulfur dioxide'])
plt.show()


# In[ ]:


plt.figure(figsize=(15,9))
sns.distplot(dataset['total sulfur dioxide'])
plt.show()


# In[ ]:


plt.figure(figsize=(15,9))
sns.distplot(dataset['density'])
plt.show()


# In[ ]:


plt.figure(figsize=(15,9))
sns.distplot(dataset['pH'])
plt.show()


# In[ ]:


plt.figure(figsize=(15,9))
sns.distplot(dataset['sulphates'])
plt.show()


# In[ ]:


plt.figure(figsize=(15,9))
sns.distplot(dataset['alcohol'])
plt.show()


# <a id="2"></a><br>
# ## Countplot

# In[ ]:


plt.figure(figsize=(15,9))
sns.countplot(dataset['quality'])
plt.show()


# <a id="3"></a><br>
# ## Correlation Map

# In[ ]:


#Correlation Heatmap
corelation_matrix=dataset.corr()
fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corelation_matrix, annot=True, linewidths=0.5,linecolor="red", fmt= '.2f',ax=ax,cmap='inferno')
plt.show()


# <a id="4"></a><br>
# ## Pairplot

# In[ ]:


plt.figure(figsize=(15,9))
sns.pairplot(dataset,hue="quality",palette=sns.color_palette("RdBu_r", 7))
plt.legend()
plt.show()


# <a id="5"></a><br>
# ## Catplots

# In[ ]:


dataset.info()


# In[ ]:


plt.figure(figsize=(15,9))
sns.catplot(x="quality", y="fixed acidity", data=dataset,kind='violin')
plt.show()


# In[ ]:


plt.figure(figsize=(15,9))
sns.catplot(x="quality", y="volatile acidity", data=dataset,kind='violin')
plt.show()


# In[ ]:


plt.figure(figsize=(15,9))
sns.catplot(x="quality", y="citric acid", data=dataset,kind='violin')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.catplot(x="quality", y="residual sugar", data=dataset,kind='violin')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.catplot(x="quality", y="chlorides", data=dataset,kind='violin')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.catplot(x="quality", y="free sulfur dioxide", data=dataset,kind='violin')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.catplot(x="quality", y="total sulfur dioxide", data=dataset,kind='violin')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.catplot(x="quality", y="density", data=dataset,kind='violin')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.catplot(x="quality", y="pH", data=dataset,kind='violin')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.catplot(x="quality", y="sulphates", data=dataset,kind='violin')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.catplot(x="quality", y="alcohol", data=dataset,kind='violin')
plt.show()


# <a id="6"></a><br>
# ## Pointplot

# In[ ]:


#Normalizing the data
normalized_data=dataset.copy()
for column in normalized_data.columns:
    normalized_data[column]=normalized_data[column]/normalized_data[column].max()
    
normalized_data=normalized_data.round(3)
normalized_data.head()


# In[ ]:


fig,ax1 = plt.subplots(figsize =(15,9))
sns.pointplot(x=normalized_data['volatile acidity'],y=normalized_data['quality'],data=normalized_data,color='sandybrown',alpha=0.7)
sns.pointplot(x=normalized_data['citric acid'],y=normalized_data['quality'],data=normalized_data,color='seagreen',alpha=0.6)
sns.pointplot(x=normalized_data['alcohol'],y=normalized_data['quality'],data=normalized_data,color='red',alpha=0.6)
plt.xticks(rotation=90)
plt.text(5.5,1,'Volatile Acidity-Quality',color='sandybrown',fontsize = 18,style = 'italic')
plt.text(5.4,0.96,'Citric Acid-Quality',color='seagreen',fontsize = 18,style = 'italic')
plt.text(5.3,0.92,'Alcohol-Quality',color='red',fontsize = 18,style = 'italic')
plt.xlabel('X - Axis',fontsize = 15,color='black')
plt.ylabel('Y - Axis',fontsize = 15,color='black')
plt.title('Volatile Acidity-Quality vs Citric Acid-Quality vs Alcohol-Quality',fontsize = 20,color='blue')
plt.grid()


# <a id="7"></a><br>
# ## 3D Scatterplot

# In[ ]:


#In this part i changed dependent variables as 1,2 and 3 to get better results
a=0
for i in dataset['quality'].values:
    if i==8 or i==7:
        dataset['quality'][a]=3
    elif i==6 or i==5:
        dataset['quality'][a]=2
    elif i==4 or i==3:
        dataset['quality'][a]=1
    a=a+1


# In[ ]:


dataset['quality'].value_counts()


# In[ ]:


import plotly.express as px
fig = px.scatter_3d(dataset, x='alcohol',
                    y='volatile acidity', 
                    z='sulphates', 
                   color='quality', 
       color_continuous_scale='solar'
       )
iplot(fig)


# # 2-Machine Learning Part

# In[ ]:


X = dataset.iloc[:,0:-1].copy()
Y = dataset.iloc[:,-1].copy()
Y=Y.values


# <a id="8"></a><br>
# ## Feature Selection

# In[ ]:


import statsmodels.api as sm
x=sm.add_constant(X)
y=Y.copy()
results=sm.OLS(Y,x).fit()
print(results.summary())


# In[ ]:


X.drop(['density','fixed acidity'],axis=1,inplace=True) #These features have big noise. So i drop them.
x=sm.add_constant(X) 
y=Y.copy()
results=sm.OLS(Y,x).fit()
print(results.summary())


# As you see that values of F-statistics and Adj-R have increased. Which means model is better now.

# <a id="9"></a><br>
# ## Standardizing

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)


# <a id="10"></a><br>
# ## Cross Validation

# In[ ]:


from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA,KernelPCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from xgboost import XGBClassifier


# In[ ]:


models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('K-NN', KNeighborsClassifier()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Decision Tree (CART)',DecisionTreeClassifier())) 
models.append(('AdaBoostClassifier', AdaBoostClassifier()))
models.append(('SVC', SVC(kernel = 'rbf', random_state = 42)))
models.append(('BaggingClassifier', BaggingClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier())) 
models.append(('XGBoost', XGBClassifier(n_estimators=200)))


# In[ ]:


from sklearn.metrics import classification_report
np.random.seed(123) #To get the same results

for name, model in models:
    model = model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    from sklearn import metrics
    print(name," --> Accuracy: ",(metrics.accuracy_score(Y_test, Y_pred)*100),"%")
    report = classification_report(Y_test, Y_pred)
    print(report)


# Results of Bagging Classifier and Random Forest Classifier are so close. 
# 
# Let's try K-Fold Cross Validation with 10 iterations to make final decision.

# In[ ]:


from sklearn.model_selection import cross_val_score
Bagging_Classifier=BaggingClassifier()
Bagging_Accuracies=cross_val_score(estimator=Bagging_Classifier,X=X,y=Y,cv=10,n_jobs=-1)

RandomForest_Classifier=RandomForestClassifier()
RandomForest_Accuricies=cross_val_score(estimator=RandomForest_Classifier,X=X,y=Y,cv=10,n_jobs=-1)

final_results=pd.DataFrame(index=['Results'],columns=['Bagging Classifier','Random Forest Classifier'],data=[[Bagging_Accuracies.mean(),RandomForest_Accuricies.mean()]])
final_results


# According to these results, RandomForestClassifier is the best model for this dataset.
