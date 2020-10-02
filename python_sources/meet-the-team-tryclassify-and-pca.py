#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# 
# In this kernel we try to predict the machine breakdown based on its indicators. There are more than 50 of them and all are unnamed. Therefore, we also need to understand how much we can reduce the number of available dimensions to simplify calculations.
# 
# So we have two questions on the way:
# 1. Which model do we need to accurately predict machine breakdown?
# 2. What is the intrinsic dimension of the data provided?
# 
# Because of the provided marked data and output as classes, we have a classification problem. 
# Thus we meet the team of most popular classfiying models:
# * Logistic Regression
# * Decision Tree
# * Random Forest Classifier
# * k-Nearest Neighbors 
# * Support Vector Machines
# 
# However this list is far to be full, it is enough to see the difference between algorithms performance.<br>
# And as for the second question we will use the principal component analysis to reveal how many dimensions contribute to the output the most. So let us open the veil and figure out which indicators help us predict the future of the machine.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, cross_val_score

import statsmodels.discrete.discrete_model as sm
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

import time


# ### Looking at the data
# 
# Luckily we are provided a lot of data. However, we will use only 5 000 observations in this kernel to simplify calculations and to compare models performance. But nevertheless,  we need to import all rows to make random selection from all provided dataset.

# In[ ]:


chunksize = 10**5
chunks = pd.read_csv('../input/xtrain.csv',chunksize=chunksize, iterator=True)

X = pd.concat(chunks)


# In[ ]:


X.head()


# It is already easy to see a bunch of NaN values. And these have to be filled in because simply omitting NaNs dramatically reduces the size of our data. Just look:

# In[ ]:


print('Initial size: {}'.format(X.shape))
print('After NaN omit size: {}'.format(X.dropna().shape))


# Therefore, we have to fill missing data to avoid unwanted surprises during the models calculation.

# In[ ]:


X = X.fillna(method='bfill').fillna(method='ffill')
X.head()


# Here we import our desired labels, our classes which we will predict later.

# In[ ]:


chunks = pd.read_csv('../input/ytrain.csv',chunksize=chunksize, iterator=True)

y = pd.concat(chunks)


# In[ ]:


y.describe()


# And convert loaded y into array to work with **train_test_split** function.

# In[ ]:


y = np.array(y).ravel()


# ### Defining the models to use
# 
# As it was mentioned before - we solve the classification problem. Fortunately, there are a lot of powerfull classifiers but we will use only some of them: (only because of my curiosity to these models)
# * Logistic Regression
# * Decision Tree
# * Random Forest Classifier
# * k-Nearest Neighbors 
# * Support Vector Machines
# 

# #### Train and test dataset splitting
# 
# For models tests we have to split training data into train and test parts.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# #### Logistic Regression Test Statistics
# 
# Before the actual modelling and prediction let us see the how the variables (features) impact on outcome and their significance. To see that the model estimation statistics will help us.
# 
# There will be a lot of indicators, but basically we are interested in:
# * Pseudo R squared as one of the important indicators of model quality
# * P-value of each variable as the indicator of variable significance
# 
# Of course, we should also check other parameters, check for heteroscedasticity to see if our calculated coefficients are not biased. But for now we just compare models with and without limitations on dimensions.

# In[ ]:


model = sm.Logit(y_train, X_train)
result = model.fit()
print(result.summary())


# The pseudo R squared is super low. That means our model is not much better than predicting average value. Let's take a look if there will be an improvement if we choose variables with p-value equal or lower 10%. 

# In[ ]:


sig_columns = [i for i,x in enumerate(result.pvalues.ravel()) if x<=0.1]

X_train.iloc[:,sig_columns].head()


# In[ ]:


model = sm.Logit(y_train, X_train.iloc[:,sig_columns])
result = model.fit()
print(result.summary())


# Pffffff.... So we have got that our reduction from 58 features to 11 led us to negative R squared. That means our model with limitations on dimensions predicts worser than if we simply predict every time average value. Interesting, that the same story happens, when we reduce the significance level from 10% to 1% - the pseudo R squared become lower and lower.
# 
# Okay. We get that throwing away the features based only on their p-value is not a good idea. However, we should not stop here and conduct further modelling with all indicators. We should provide one more approach for dimension reduction to see if there is still a hope to make life easier for a computer.

# #### The PCA incoming
# 
# By now let us draw conclusions about our dimensions with another interesting method - the Principal Component Analysis.
# 
# With this analysis we may understand how much we can reduce the number of our dimensions to exclude the noise in the data. And we may also transorm the data in a way to get components which determine the highest variance of the data and thus the greatest impact.
# 
# But before the data transormation we need to determine the number of components we want to use. And here the histogram of explained variances by the number of PCA features may help us. 

# In[ ]:


pca = PCA()

pca.fit(X_train)
plt.figure(figsize=(15, 8))
features = range(pca.n_components_)
plt.plot(features, pca.explained_variance_ratio_.cumsum(),'--o', label='cumulative explained variance ratio')
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.legend()
plt.show()


# As can be clearly seen from the chart with a single component we can explain **~95%** of the variance!
# Looks like we have a dominator. 
# 
# If we want to be closer to normal distribution or we find that there are problems with feature scale then we may normalize our data. And with normalization we might see that other components will contribute as well. However, this should be calculated before claiming that some features specific scale prevents to do the right predictions.

# Therefore, let us look at the original features variance:

# In[ ]:


plt.figure(figsize=(15,8))
plt.plot(range(0,58),X_train.var().cumsum(),'--o', label="cumulative feature variance")
plt.legend()
plt.show()


# It looks like the 36th variable has outstanding variance. And as we supposed before this domination should be prevented via rescaling:

# In[ ]:


scaler = StandardScaler()

pca = PCA()

pipeline = make_pipeline(scaler, pca)

pipeline.fit(X_train)

plt.figure(figsize=(15, 8))
features = range(pca.n_components_)
plt.plot(features, pca.explained_variance_ratio_.cumsum(),'--o', label='normalized cumulative explained variance ratio')
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.legend()
plt.show()


# Oh... And now we got that all components have equal contribution. That is super strange. However, let's give it a chance.

# In[ ]:


pca_X_train = pipeline.transform(X_train)
pca_X_test = pipeline.transform(X_test)

print('pca_X_train shape: {}'.format(pca_X_train.shape))
print('pca_X_test shape: {}'.format(pca_X_test.shape))


# #### Modelling without PCA features
# 
# From now we will implement imported models to see the difference in accuracy, log loss and time usage. And no PCA will be applied.
# 
# With such decision we get the opportunity to compare the performance not only between models but also between used features.

# In[ ]:


def compute_models(X_train, y_train, X_test, y_test):
    results={}
    def test_model(model):
        start_time = time.time()
        
        model.fit(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        model_probs = model.predict_proba(X_test)
        test_log_loss = log_loss(y_test, model_probs)
        cv_acc = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5)
        
        scores= [cv_acc.mean(), test_accuracy, test_log_loss, (time.time() - start_time)]
        return scores
    m = LogisticRegression()
    results['Logistic Regression'] = test_model(m)
    
    for i in range(6, 15, 4):
        m = DecisionTreeClassifier(max_depth=3, min_samples_leaf=i)
        results['Decision tree {}'.format(i)] = test_model(m)
    
    for i in range(60,150,40):
        m = RandomForestClassifier(n_estimators=i)
        results['Random forest {}'.format(i)] = test_model(m)
    
    for i in range(6,15,4):
        m = KNeighborsClassifier(n_neighbors = i)
        results['KNN {}'.format(i)] = test_model(m)
        
    m = SVC(probability=True)
    results['SVM'] = test_model(m)
    
    results = pd.DataFrame.from_dict(results,orient='index')
    results.columns=["Train mean accuracy", "Test accuracy", "Test log loss", "Calculation time (sec)"] 
    results=results.sort_values(by=["Test accuracy","Test log loss"],ascending=[False,True])

    return results


# In[ ]:


limit = 10**3*5
no_pca_results = compute_models(X_train[:limit], y_train[:limit], X_test[:limit], y_test[:limit])
no_pca_results


# #### Modelling with PCA
# 
# As we have got from the table above, there might be high test accuracy, but high log loss score.<br>
# Moreover, we have got a Runtime Warning, which may mean that the data need to be normalized.
# So now the time for the PCA feature has come. And therefore we have a chance to check whether the specific scale in our data plays a role.

# Computing the models with all PCA features as we think normalizing will help us get better results:

# In[ ]:


pca_results = compute_models(pca_X_train[:limit], y_train[:limit], pca_X_test[:limit], y_test[:limit])
pca_results


# Great! We resolved the issue with runtime warning.

# Computing the PCA without rescaling data and with only one component. As we have observed from one of the charts above one PCA feature may explain about 95% of the variance. 

# In[ ]:


pca = PCA(n_components=1)

pca.fit(X_train)

pca1_X_train = pca.transform(X_train)
pca1_X_test = pca.transform(X_test)

print('pca_X_train shape: {}'.format(pca1_X_train.shape))
print('pca_X_test shape: {}'.format(pca1_X_test.shape))


# After calculation let us see the difference in results with only one, not normalized PCA component:

# In[ ]:


pca_results = compute_models(pca1_X_train[:limit], y_train[:limit], pca1_X_test[:limit], y_test[:limit])
pca_results


# Wait what. We have got almost the same results using only one PCA feature? And look how we improved the calculation time!

# ### Choosing the most interesting model
# 
# As we have observed, there is no big difference in test accuracy between models and features that we used. Of course, we have taken only 5 000 rows from splitted datasets. But it is enough to understand the time complexity and quality of algorithms.
# 
# And as we mentioned time complexity, look at the variance in calculation time. There are some models which have almost the same accuracy and log loss, but there are much faster:
# * Decision Tree
# * KNN
# 
# And with them we will provide model tests with 100 000 observations to see the difference not only between model parameters but also between results collected from testing on 5 000 data sample.

# In[ ]:


def draw_models(model_str,X_train, y_train, X_test, y_test, range_):
    test_accuracy = [0]*len(range_)
    test_log_loss = [0]*len(range_)
    def test_model(model):
        model.fit(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        model_probs = model.predict_proba(X_test)
        test_log_loss = log_loss(y_test, model_probs)
        return test_accuracy, test_log_loss
    if (model_str=='DT'):
        for i,x in enumerate(range_):
            m = DecisionTreeClassifier(max_depth=3, min_samples_leaf=x)
            test_accuracy[i], test_log_loss[i] = test_model(m)
    else:
        for i,x in enumerate(range_):
            m = KNeighborsClassifier(n_neighbors = x)
            test_accuracy[i], test_log_loss[i] = test_model(m)
    
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.plot(range_, test_accuracy,'--o', label='Test accuracy')
    plt.legend()
    plt.subplot(122)
    plt.plot(range_, test_log_loss,'--o', label='Test log loss')
    plt.legend()
    plt.show()


# #### Decision Tree
# 
# As the decision trees work better than KNN on 5 000 sample, we start our experiment with trees adjusting.

# In[ ]:


limit = 10**5
draw_models('DT',pca1_X_train[:limit], y_train[:limit], pca1_X_test[:limit], y_test[:limit],range(2,20,4))


# Interesting that with 14 as minimum number of samples at a leaf node we get a compromise between test accuracy and log loss.

# #### KNN
# 
# We have seen that the best log loss of KNN come from 10 as number of neighbors. Thus let us draw a chart with increasing number of neighbors from 10 to see the impact on performance

# In[ ]:


draw_models('KNN',pca1_X_train[:limit], y_train[:limit], pca1_X_test[:limit], y_test[:limit],range(10,19))


# So the KNN shows almost the same results. However, it has a little lower accuracy and higher calculation time.

# ### Temporary Conclusion
# 
# We have observed that Decision Tree algorithm leads in the group of classifiers we have implemented. However, with another models tuning we might get different results. Nevertheless, we observed how well decision trees work with the lowest calculation time.
# 
# The second interesting thing is that we got the same models performance with only one PCA component applied. Which means we highly reduced the computational time and also kept the results.
# 
# On the other hand, we have got only ~70% test accuracy and ~0.61 log loss. Therefore, there is still a need to find better approach than what was used above.

# In[ ]:




