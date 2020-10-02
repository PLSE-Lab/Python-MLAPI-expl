#!/usr/bin/env python
# coding: utf-8

# ### Import packages

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Import data

# In[2]:


data = pd.read_csv("../input/winequality-red.csv")
data.head()


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


from sklearn.preprocessing import LabelEncoder

bins = (2, 5, 8)
group_names = ['bad', 'good']

data['quality'] = pd.cut(data["quality"], bins = bins, labels = group_names)

label_quality = LabelEncoder()

data['quality'] = label_quality.fit_transform(data['quality'].astype(str))
data['quality'].value_counts()


# In[6]:


sns.countplot(data['quality'])
plt.show()


#  ### Feature Selection (filter method)

# * The model is built after selecting the features.
# * The filtering here is done using correlation matrix and it is most commonly done using Pearson correlation.

# In[7]:


# Pearson correlation
plt.subplots(figsize=(15, 9))
cor = data.corr()
sns.heatmap(cor, annot=True, linewidths=.5)
plt.show()


# In[8]:


#correlated features with quality
cor_feat = abs(cor["quality"])

#relevant features
rel_feat = cor_feat[cor_feat>0.2]
rel_feat


# In[9]:


# correlation btw relevant features
data[["volatile acidity","citric acid"]].corr()


# In[10]:


data[["volatile acidity","sulphates"]].corr()


# In[11]:


data[["volatile acidity","alcohol"]].corr()


# Final relevant features are:
# volatile acidity, sulphates, alcohol            

# ### Feature Selection (wrapper method)

# * A wrapper method needs one machine learning algorithm and uses its performance as evaluation criteria.
# * There are different wrapper methods such as Backward Elimination, Forward Selection, Bidirectional Elimination and RFE.

# **Backward Elimination : **
# * we feed all the possible features to the model at first
# * The performance metric used here to evaluate feature performance is pvalue. 

# In[12]:


X = data.drop("quality", axis=1)
y = data["quality"]


# In[13]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=10)


# In[14]:


#Backward Elimination
import statsmodels.api as sm
cols = list(X_train.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X_train[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y_train,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break

selected_features_BE = cols
selected_features_BE


# **Recursive feature elimination (RFE) : **
# 
# * The goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. 
# * First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. 
# * Then, the least important features are pruned from current set of features.

# In[15]:


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

model = LinearRegression()

rfe = RFE(model, 8)
X_rfe = rfe.fit_transform(X_train, y_train)

model.fit(X_rfe, y_train)
print(rfe.support_)
print(rfe.ranking_)


# We do that by using loop starting with 1 feature and going up to 11. We then take the one for which the accuracy is highest.

# In[16]:


#no of features
nof_list=np.arange(1,11)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]

print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


# As the optimum number of features is 7. We now feed 10 as number of features to RFE and get the final set of features given by RFE method

# In[17]:


cols = list(X_train.columns)
model = LinearRegression()

#Initializing RFE model
rfe = RFE(model, 7)             

#Transforming data using RFE
X_rfe = rfe.fit_transform(X_train,y_train)  

#Fitting the data to model
model.fit(X_rfe,y_train)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


# ###  Embedded Method:
# 

# * Embedded methods are iterative in a sense that takes care of each iteration of the model training process and carefully extract those features which contribute the most to the training for a particular iteration.
# * Regularization methods are the most commonly used embedded methods which penalize a feature given a coefficient threshold.

# In[18]:


from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

reg = LassoCV()
reg.fit(X_train, y_train)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X_test,y_test))
coef = pd.Series(reg.coef_, index = X.columns)


# In[19]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[20]:


imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()


# **For Classification :**
# [Follow this link](https://www.kaggle.com/alokevil/red-wine-classification)

# In[ ]:




