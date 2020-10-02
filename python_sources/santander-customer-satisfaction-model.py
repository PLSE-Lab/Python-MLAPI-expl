#!/usr/bin/env python
# coding: utf-8

# <h1 style="font-size:30px;">Overview</h1>
# <section style="background-color:light-blue;">
# <span><u>This notebook is structured as follows:</u></span>
# <ol>
# <li>Data collection</li>
# <li>Data preparation</li>
# <li>EDA-Exploratory Data Analysis</li>
# <li>Feature Engineering-Part 1</li>
# <li>Modelling</li>
# <li>Model optimiation</li>
# </ol>
# </section>

# In[ ]:


# Data manipulation libraries
import numpy as np
import pandas as pd

# Visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Default visualisation settings
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='darkgrid')


# <h1>1. Data collection</h1>
# * We are going to use the Santander Customer satisfaction dataset provided by kaggle.
# * The dataset is anonimised, i.e. The real column names are not provided
# * The **target** variable is used to indicate  **satisfied = 1** and **unsatisfied=0** customers
# 

# In[ ]:


santander = pd.read_csv('../input/train141414.csv')
santander.head()


# In[ ]:


test_sample = pd.read_csv('../input/test.csv')
test_sample.head()


# <h1> 2. Data preparation</h1>
# <ul>
# <li>At this stage, we should evaluate our data for any missing values, class imbalance, data format etc.</li>
# </ul>

# In[ ]:


santander.info()


# * It turns out that there are no missing values in this case
# * All data formats appear to be correct and machine readable for the purpose of this activity

# **Statistical properties of the data**

# In[ ]:


santander.describe()


# * It appears that the training data contains more unsatisfied (Target=0) customers than satisfied customers
# * Let's confirm this below*** ( You can also see this by checking the mean of the TARGET column above )***

# In[ ]:


(len(santander['TARGET']) - len(santander[ santander['TARGET']==0 ]['TARGET']) )/len(santander['TARGET']) * 100


# <span style="color:blue;"><b>The problem of class imbalance</b></span>
# <ul>
# <li>We cannot use error rate or model accuracy for tuning and model evaluation here.</li>
# <li>This would lead to our model choosing the majority class almost always - Biased predictions which lead to misleading accuracy metric</li>
# <li>Proposed solutions such as undersampling or oversampling can cause loss of valuable information or over-fiiting, respectively</li><br>
# </ul>
# <span style="color:puple;"><b>Solution: SMOTE</b></span>
# * This is a Synthetic Minority Oversampling Technique
# * Creates new **synthetic - artificial** obeservations
# * Read more about [SMOTE](http://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwiHkO_vgLjaAhVICcAKHVwpDC8QFghBMAE&url=https%3A%2F%2Fwww.jair.org%2Fmedia%2F953%2Flive-953-2037-jair.pdf&usg=AOvVaw2ro1NjNTwDUZsFHw_4re3c)
# 

# In[ ]:


import IPython.display as im

art = im.Image('https://ars.els-cdn.com/content/image/1-s2.0-S0020025517310083-gr3.jpg')
art


# <span style="color:purple;"><b>FOR THE PURPISE OF THIS EXERCISE WE'LL BE DEALING WITH THIS PROBLEM LATER IN THE MODEL <i>OPTIMIZATION SECTION</i>.</b><span>

# <h1> 3. EDA </h1>

# In[ ]:


plt.figure(figsize=(10,6))
sns.pairplot(santander, hue='TARGET', palette='bwr')


# In[ ]:


plt.figure(figsize=(16,8))
sns.heatmap(santander.corr(), annot=True, cmap='viridis')


# <span style="color:green;"><b>Final comments:</b></span>
# * It appears that none of the fields are strongly correlated with each other
# * **This is great!! **Move on to the next sections to find out why.

# <h1> 4. Feature engineering - Part 1 </h1>

# <h2>4.1. Constant features</h2>
# * **Aim**: Identify featues that dont change, i.e. have no effect on the target
#     * Drop these features from the dataframe

# In[ ]:


santander['TARGET'].unique()


# In[ ]:


def const_identifier(df):
    unique_values_per_field = df.apply( lambda t: len( t.unique() ) ) 
    constants = unique_values_per_field[ unique_values_per_field == 1 ].index.tolist()
    return constants


# In[ ]:


const_identifier(santander)


# <h2>4.2. Identical features</h2>
# * These can either be identified manually e.g. lenght in cm and length in inches colum
# * For anonimous datasets we want to automate this processs
# 
# ** Ultimately, we want a dataset with linearly independent features to maximise the perfomance of our models**

# In[ ]:


from itertools import combinations


# In[ ]:


def identify_eq_feat(df):
    df_columns = list( combinations( df.columns.tolist(), 2 ) ) # Return 2 length subsequences of elements from the idf columns
    identified_feat = []
    for col in df_columns:
        is_eq = np.array_equal( df[col[0]], df[col[1]] )
        if is_eq:
            identified_feat.append(list(col))
    return identified_feat      


# In[ ]:


identify_eq_feat(santander)


# In[ ]:


# Drop these features: IF NECESSARY !!
# santander.drop( np.array( identify_eq_feat(santader) )[:,1] ,axis =1, inplace=True)


# <h1>5. Modelling</h1>
# <span style="color:green;">Note that in practise, <span style="color:black;"><b>the first step in this section is model selection</b></span>,which is covered in section 5.4. below</span>

# In[ ]:


# Module imports
# Model training libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Model evaluation libraries
from sklearn.metrics import classification_report, con


# <h2>5.1 Train test split</h2>

# In[ ]:


santander.columns


# In[ ]:


X = santander[['var15', 'var38', 'var36', 'num_var30_0', 'num_var42_0', 'var3',
       'num_var5_0']]
y = santander['TARGET']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)


# <h2>5.2. Model Fitting</h2>

# In[ ]:


log_classifer = LogisticRegression()


# In[ ]:


log_classifer.fit( X_train, y_train)


# <h2>5.3. Model Evaluation</h2>

# In[ ]:


predictions = log_classifer.predict(X_test)


# In[ ]:


predictions


# In[ ]:


print( classification_report(y_test, predictions) )


# In[ ]:





# <h2>5.4. Alternative models</h2>
# * In this section, we'll evaluate the perfomance of othe classifiers according to computation time required to train each model
# * We particularly use **Stratified K-fold** cross validation to evaluate the **ROCAUC** scores for each potential binary classifier 

# In[ ]:


from sklearn import cross_validation as cv
from sklearn import tree, metrics, ensemble, linear_model, naive_bayes

import xgboost as xgb


# In[ ]:


score_metric = 'roc_auc'
scores = {}
stratified_iterator = cv.StratifiedKFold( y, n_folds=3, shuffle=True )


# In[ ]:


def score_model(model):
    return cv.cross_val_score( model, X, y, cv=stratified_iterator, scoring=score_metric )


# In[ ]:


alt_models = 'tree extra_tree forest ada_boost bagging grad_boost ridge passive sgd gaussian xgboost'.split()


# In[ ]:


import warnings
warnings.filterwarnings('ignore')

scores['tree'] = score_model( tree.DecisionTreeClassifier() )
scores['extra_tree'] = score_model( ensemble.ExtraTreesClassifier() )
scores['forest'] = score_model ( ensemble.RandomForestClassifier() )

scores['ada_boost'] = score_model(ensemble.AdaBoostClassifier())
scores['bagging'] = score_model( ensemble.BaggingClassifier() )
scores['grad_boost'] = score_model( ensemble.GradientBoostingClassifier() )
scores['ridge'] = score_model( linear_model.RidgeClassifier())

scores['PASSIVE'] = score_model( linear_model.PassiveAggressiveClassifier())
scores['SGD'] = score_model( linear_model.SGDClassifier() )
scores['GAUSSIAN'] = score_model( naive_bayes.GaussianNB() )
scores['XGBOOST'] = score_model( xgb.XGBClassifier() )
score_metric


# In[ ]:


model_scores = pd.DataFrame(scores).mean().sort_values(ascending=False)
print( 'Model scores\n {}'.format(model_scores) )


# In[ ]:




