#!/usr/bin/env python
# coding: utf-8

# **How can we prevent customers from leaving the company?**
# 
# In order to find out we are going to build a logistic regression model and identify important factors.
# We are going to perform the following steps:
# * Load the data
# * Preprocess the data
# * Create customer clusters using KMeans to get an overview
# * Perform a backward elemination of the features to filter out only the significant features
# * Identify reasons why customers are leaving
# * Propose solutions

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plot
import seaborn as sns

sns.set()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
sns.set()
# Any results you write to the current directory are saved as output.


# **Reading the data**
# 
# For reading the data we are using pandas and its read_csv method.

# In[ ]:


data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[ ]:


data.head()


# We can now investigate all available features Our target feature is "Churn". Let us convert all features in numeric ones:

# In[ ]:


for col in data.columns:
    if not np.issubdtype(data[col].dtype, np.number):
        if len(data[col].unique()) < 11:
            _dat = pd.get_dummies(data[col], prefix=col).iloc[:,1:]
            data = pd.concat([data, _dat], 1)
            data = data.drop(col, 1)
        else:
            if "Charges" in col:
                data[col] = pd.to_numeric(data[col].replace(" ", 0))

Y = data["Churn_Yes"]
X = data.drop(["Churn_Yes", "customerID"], 1)
X.head()


# **Clustering**
# 
# We can now cluster our customers. In order to find a good number of clusters we are using the Elbow method.

# In[ ]:


from sklearn.cluster import KMeans

wcss = []
number = range(2,10)

for n in number:
    kmeans = KMeans(n)
    kmeans.fit(X)
    
    wcss.append(kmeans.inertia_)
    
plot.plot(number, wcss, "-o")
plot.show()


# Our ideal number of clusters seems to be either 4 our 5. I am chosing 4 here.

# In[ ]:


kmeans = KMeans(4)
kmeans.fit(X)

clusters = kmeans.predict(X)


# In[ ]:


def PlotClusters(X, v1, v2, clusters):
    Xc = pd.concat([X, pd.Series(clusters).rename("cluster")],1)

    Xc = Xc[[v1, v2, "cluster"]]

    for i in range(np.max(clusters)+1):
        _Xc = Xc[Xc["cluster"]==i]
        plot.scatter(_Xc[v1], _Xc[v2])

    plot.xlabel(v1)
    plot.ylabel(v2)


# In[ ]:


plot.figure(figsize=(15,15))
plot.subplot(221)
PlotClusters(X, "tenure", "TotalCharges", clusters)
plot.subplot(222)
PlotClusters(X, "tenure", "MonthlyCharges", clusters)
plot.subplot(223)
PlotClusters(X, "MonthlyCharges", "TotalCharges", clusters)

plot.show()


# We can see that the two main features that distiguish the generated groups are tenure and the total charges.
# 
# **Backward Elimination**
# 
# Fro backward elimination we are using an automated algorithm. We calculate the pvalues and drop the largest one if it is larger than our treshold. Otherwise we are finished.

# In[ ]:


import statsmodels.api as sm

treshold=0.05
X2 = sm.add_constant(X.drop("TotalCharges", 1))
pdroplist=["TotalCharges"]

while True:
    ols = sm.OLS(Y, X2).fit()
    
    if ols.pvalues.max() > treshold:
        col = ols.pvalues.argmax()
        print("Dropping "+str(col))
        X2 = X2.drop(col,1)
        pdroplist.append(col)
    else:
        break


# Let us look at our statistical summary.

# In[ ]:


ols = sm.OLS(Y, X2).fit()
print(ols.summary())


# Looks good so far. The remaining features all seem to be quite significant.
# 
# **Preprocessing**
# 
# Next we are preprocessing our data. Therefore we are replacing NaNs with the respective mean value and we are standardizing our data.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer, StandardScaler

X3=X2.drop("const",1)

imputer = Imputer()
X3 = imputer.fit_transform(X3)

scaler = StandardScaler()
X3 = scaler.fit_transform(X3)


# In order to be able to assess the accuracy of our model we are using a train-test split.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X3, Y, test_size=0.2)


# In[ ]:


from sklearn.metrics import accuracy_score

reg = LogisticRegression()
reg.fit(X_train, Y_train)

acc = accuracy_score(Y_val, reg.predict(X_val))
print("Accuracy: "+str(acc))


# That is a quite good accuracy for such a simple model. The advantage of using such a simple model is that we easily can extract valuable insights. Let us look at the coefficients of our logistic regression. We are using the following transformation in order to make them comparable and more readable.
# 
# $x \rightarrow (e^x-1)$

# In[ ]:


xvals = X2.columns[1:]
coeffs =np.exp(reg.coef_[0])-1.0

plot.figure(figsize=(10,5))
plot.bar(xvals, coeffs)
plot.xticks(rotation=90)
plot.ylabel("Coefficients [a.u.]")
plot.show()


# We can now see that the main factors are
# * tenure
#     * Long time customers are less likely to leave the company.
#     * Loyality
# * TotalCharges
#     * More paying customers are more likely to leave the company.
#     * Are there cheaper offers from competitors?
# * InternetService_Fiber optic
#     * People having Fiber optic are more likely to leave.
#     * Is there something wrong with the companies fiber optics product?
# * OnlineSecurity Yes
#     * People having this product are more loyal.
#     * How can this product attract a wider audience?
# * TechSupport Yes
#     * Same like OnlineSecurity
# * Streaming Movie/TV
#     * Are customers unhappy with these products?
#     * How can they be improved?
# * Contact One/Two year
#     * This one is very obvious.
#     * Can the company make this kind of contract more attractive?
# * PaperlessBilling and Electronic Payment
#     * People using these services are more likely to leave.
#     * More technology affine people ready to change frequently?
#     * Easier to change when anything can be done online.
#    
# 

# **Closer look at the Fiber Optic product**
# 
# In order to find out whether there is really something wrong with the fibre optic product, we can perform a chi-squared test.

# In[ ]:


inet_customers = data[data["InternetService_No"]==0]
fo_data = inet_customers[["InternetService_Fiber optic", "Churn_Yes"]]


# Calculating the contingency table:

# In[ ]:


c00 = fo_data[(fo_data["InternetService_Fiber optic"]==0) & (fo_data["Churn_Yes"]==0)].shape[0]
c01 = fo_data[(fo_data["InternetService_Fiber optic"]==0) & (fo_data["Churn_Yes"]==1)].shape[0]
c10 = fo_data[(fo_data["InternetService_Fiber optic"]==1) & (fo_data["Churn_Yes"]==0)].shape[0]
c11 = fo_data[(fo_data["InternetService_Fiber optic"]==1) & (fo_data["Churn_Yes"]==1)].shape[0]


# In[ ]:


from scipy.stats import chi2_contingency

c_table = np.array([[c00, c01], [c10, c11]])
chi2, p, dof, expected = chi2_contingency(c_table)

print("P-Value: "+str(p))


# The p-value is very small. It is therefore reasonable to say that the fiber optic product has a significant impact on the customer churn.

# **Predicting improvements**
# 
# A main issue seems to be the fiber optics product as well as the total charge. While reducing the price of the products may not be possible easily for the company, an improvement of the fiber optics product should be enforced. How many customers  less would leave the company if fiber optics was neutral?

# In[ ]:


leaving = reg.predict(X_val)
leaving = np.sum(leaving)/len(leaving)

print("Customers currently leaving:\t\t\t"+str(np.round(leaving*100,2))+"%")

reg2 = LogisticRegression()

reg2.coef_=np.copy(reg.coef_)
reg2.coef_[0,7]=0
#reg2.coef_[0,2]=reg2.coef_[0,2]*np.exp(0.25)
reg2.intercept_ = reg.intercept_
reg2.classes_ = reg.classes_

leaving_new = reg2.predict(X_val)
leaving_new = np.sum(leaving_new)/len(leaving_new)


print("Customers leaving with better fiber optics:\t"+str(np.round(leaving_new*100,2))+"%")

