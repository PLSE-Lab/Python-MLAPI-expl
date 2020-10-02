#!/usr/bin/env python
# coding: utf-8

# # Heart Diesease Predictor
# The goal of the model is to predict wether the patient has a heart diesease based on numeral description of his physical and psychological state. 
# 
# Goals:
#     - Properly describe the code
#     - Use ensemble learning
#     - Use heavy cross-validation
#     
#     
# Feature description:
# ### age
# age in years
# 
# ### sex
# (1 = male; 0 = female)
# 
# ### cp
# chest pain type
# 
# ### trestbps
# resting blood pressure (in mm Hg on admission to the hospital)
# 
# ### chol
# serum cholestoral in mg/dl
# 
# ### fbs
# (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 
# ### restecg
# resting electrocardiographic results
# 
# ### thalach
# maximum heart rate achieved
# 
# ### exang
# exercise induced angina (1 = yes; 0 = no)
# 
# ### oldpeak
# ST depression induced by exercise relative to rest
# 
# ### slope
# the slope of the peak exercise ST segment
# 
# ### ca
# number of major vessels (0-3) colored by flourosopy
# 
# ### thal
# 3 = normal; 6 = fixed defect; 7 = reversable defect
# 
# ### target
# 1 or 0 

# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 5, 10
plt.style.use('seaborn-dark')

import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

data   = pd.read_csv('../input/heart.csv')
labels = data['target']


# In[21]:


def quartile_probs(frame, metric, target_metric):
    first  = frame[frame[metric] <= frame.describe()[metric][4]]
    second = frame[(frame[metric] > frame.describe()[metric][4]) & (frame[metric] <= frame.describe()[metric][5])]
    third  = frame[(frame[metric] > frame.describe()[metric][5]) & (frame[metric] <= frame.describe()[metric][6])]
    fourth = frame[(frame[metric] > frame.describe()[metric][6]) & (frame[metric] <= frame.describe()[metric][7])]
    
    one_q_prob = first[first[target_metric] == 1].count()[target_metric] / first[target_metric].count()
    two_q_prob = second[second[target_metric] == 1].count()[target_metric] / second[target_metric].count()
    three_q_prob = third[third[target_metric] == 1].count()[target_metric] / third[target_metric].count()
    four_q_prob = fourth[fourth[target_metric] == 1].count()[target_metric] / fourth[target_metric].count()
    
    labels   = ['1Q \n n = '+str(first[target_metric].count()),
                '2Q \n n = '+str(second[target_metric].count()), 
                '3Q \n n = '+str(third[target_metric].count()), 
                '4Q \n n = '+str(fourth[target_metric].count())]
    problist = [one_q_prob, two_q_prob, three_q_prob, four_q_prob]
    
    print('1Q {} prob: {:.2f}% (n = {})'.format(metric, one_q_prob*100, first[target_metric].count()))
    print('2Q {} prob: {:.2f}% (n = {})'.format(metric, two_q_prob*100, second[target_metric].count()))
    print('3Q {} prob: {:.2f}% (n = {})'.format(metric, three_q_prob*100, third[target_metric].count()))
    print('4Q {} prob: {:.2f}% (n = {})'.format(metric, four_q_prob*100, fourth[target_metric].count()))
    
    sns.barplot(labels, problist)
    
def class_probs(frame, metric, target_metric):
    results = []
    for i in frame[metric].unique():
        subset = data[data[metric] == i]
        pct = subset[target_metric][subset[target_metric] == 1].count()/subset[target_metric].count()
        results.append([str(i), pct, subset[target_metric].count()])
        
    for i in results:
        print("Probability for {}: {:.2f}% (n = {})".format(i[0], i[1]*100, i[2]))       
    
    sns.barplot([x[0] for x in results], [x[1] for x in results]);


# ## EDA

# In[22]:


data.head()


# In[24]:


data.describe()


# ### Age Analysis

# In[25]:


quartile_probs(data, 'age', 'target')


# First conclussions are interesting: younger people in the sample are generally less likely to have a heart diesease. The lowest probability in sample is however for the third quartile (55-61).

# # Sex Analysis

# In[26]:


class_probs(data, 'sex', 'target')


# It also appears that women are twice as likely to have a heart diesease.

# # Chest Pain Analysis

# In[27]:


class_probs(data, 'cp', 'target')


# Whenever Atypical Angina or non-anginal pain occurs, there is a very high risk of heart diesease. The same goes for asymptomatic cases, however the sample for those is quite low, so our estimation should be most troublesome for asymptomatic cases and for Typical Angina cases, where the sample is relatively high.  

# # Resting BPM Analysis

# In[28]:


quartile_probs(data, 'trestbps', 'target')


# Resting BPM does not seem very relevant as a sole determiner. Lower probability for 4Q is likely a random walk.

# # Choresterol Analysis

# In[29]:


quartile_probs(data, 'chol', 'target')


# Probability of heart diesease appears to be falling in a stable manner as serum choresteral leves increase. What is also worth noticing is amazingly even distributon of n's among the quartiles.

# # Blood Sugar Analysis

# In[30]:


class_probs(data, 'fbs', 'target')


# Blood sugar does not seem to be relevant as a sole determiner.

# # Resting electrocardiographic results Analysis

# In[31]:


class_probs(data, 'restecg', 'target')


# Samples with type 1 score seem to have higher risk of diesease than those with type 0. No conclussions should be made with regard to type 2, as there were only 4 samples in this category.

# In[32]:


quartile_probs(data, 'thalach', 'target')


# Highly significant and useful feature. The probability seems to be rising in an almost ideally linear manner, and the distribution of samples is almost perfectly even.

# # Exercise Induced Angina Analysis

# In[33]:


class_probs(data, 'exang', 'target')


# Those who experienced EIA are much less likely to actually have a heart diesease.

# # ST Depression Analysis

# In[34]:


quartile_probs(data, 'oldpeak', 'target')


# The higher the ST depression value, the less likely it is to have a heart diesease

# # ST Slope Analysis

# In[35]:


class_probs(data, 'slope', 'target')


# It seems that only slope 2 has significant influence over the diesease prob.

# # Fluoroscopy Analysis

# In[36]:


class_probs(data, 'ca', 'target')


# Type 4 should be disregarded, as n=5, therefore, we can conclude that the coorelation is inverse.

# # Thal Analysis

# In[37]:


class_probs(data, 'thal', 'target')


# It appears that only thal 2 has a significant influence over the diesease.

# # Dropping Irrelevant Data

# In[38]:


data = data.drop(['fbs', 'trestbps'], axis = 1)


# # Machine Learning Model

# In[41]:


from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split


# In[42]:


x = data
y = labels


# In[53]:


d = {'Gaussian Process': [], 
     'Random Forest': [], 
     'K Neighbours': [], 
     'SVC': [], 
     'Logit': []}

scores = pd.DataFrame(data = d)


# In[54]:


def validate(rs):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = rs)
    
    GPC = GaussianProcessClassifier()
    GPC.fit(x_train, y_train)
    GPC.score(x_test, y_test)

    RFC = RandomForestClassifier(n_estimators = 10)
    RFC.fit(x_train, y_train)
    RFC.score(x_test, y_test)

    KNC = KNeighborsClassifier()
    KNC.fit(x_train, y_train)
    KNC.score(x_test, y_test)
    
    SVC1 = SVC()
    SVC1.fit(x_train, y_train)
    SVC1.score(x_test, y_test)
    
    LOGIT = LogisticRegression()
    LOGIT.fit(x_train, y_train)
    LOGIT.score(x_test, y_test)
    
    tempr = pd.DataFrame([[GPC.score(x_test, y_test), 
                        RFC.score(x_test, y_test),
                        KNC.score(x_test, y_test),
                        SVC1.score(x_test, y_test),
                        LOGIT.score(x_test, y_test)]], columns=['Gaussian Process',
                                                                  'Random Forest',
                                                                  'K Neighbours',
                                                                  'SVC',
                                                                  'Logit'])
                         
    return tempr


# In[55]:


for rs in np.random.randint(100_000, size = 1000):
    scores = scores.append(validate(rs))


# In[56]:


scores.mean()


# In[57]:


logit_success_rate  = scores[scores['Logit'] == 1.0]['Logit'].count() / scores['Logit'].count()
print(' {:.2f} %'.format(logit_success_rate*100))


# ### In 1000 randomly generated 0.2 splits, the logistic regression classifier has achieved 100% success rate. Not much else to be done here....
