#!/usr/bin/env python
# coding: utf-8

# # My first public Kernel...comments are welcome !
# # My main aim is to quickly: 
# 1) Run some fast stats through the data
# 
# 2) Plot charts to see how data looks like 
# 
# 3) Run a fast simulation without any feature preprocessing --> Tree algo like Random Forest, xgboost comes to mind 

# # Import Packages

# In[ ]:


import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.special import boxcox1p, boxcox
from scipy.stats import boxcox_normmax

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb

# custom class to preprocess features
from sklearn.base import BaseEstimator, TransformerMixin  

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport

import shap
# load JS visualization code to notebook
shap.initjs()  


# # Import Data

# In[ ]:


train = pd.read_csv("../input/learn-together/train.csv", index_col='Id')
test = pd.read_csv("../input/learn-together/test.csv", index_col='Id')
Target = 'Cover_Type'


# In[ ]:


# quick check for missing values
train.isnull().sum()


# # Check if data has hi-cardinality (ie many categories)

# In[ ]:


#Ref : https://www.kaggle.com/vprokopev/mean-likelihood-encodings-a-comprehensive-study

def describe_dataset(data):
    datatype = []  # added by Sid
    ncats = []
    ncats10 = []
    ncats100 = []
    nsamples_median = []
    X_col_names = list(data.columns)
    #X_col_names.remove(target_col)
    print('Number of samples: ', data.shape[0])
    for col in X_col_names:
        datatype.append(data.dtypes[col])
        counts = data.groupby([col])[col].count()
        ncats.append(len(counts))
        ncats10.append(len(counts[counts<10]))
        ncats100.append(len(counts[counts<100]))
        nsamples_median.append(counts.median())
    data_review_df = pd.DataFrame({'Column':X_col_names, 'DType':datatype, 'Number of categories':ncats, 
                                   'Categories with < 10 samples':ncats10,
                                   'Categories with < 100 samples':ncats100,
                                   'Median samples in category':nsamples_median})
    data_review_df = data_review_df.loc[:, ['Column', 'DType', 'Number of categories',
                                             'Median samples in category',
                                             'Categories with < 10 samples',
                                             'Categories with < 100 samples']]
    return data_review_df.sort_values(by=['Number of categories'], ascending=False)


# In[ ]:


## check for hi cardinality feat
describe_dataset(train)


# In[ ]:


#Check Target classes 
train[Target].value_counts()


# Define numerical & categorical features --> since all columns are type 'int64'

# In[ ]:


all_cols = [train.columns.values]
all_cols


# In[ ]:


num_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
              'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points' ]

cat_cols = [col for col in train.columns if col not in num_cols]
cat_cols.remove("Cover_Type")
cat_cols


# In[ ]:


len(num_cols), len(cat_cols)


# Apply Boxcox transformation to skewed numerical features
# 
# Note: This is not necessary for Tree classifiers, but useful for linear classifiers like Logistic Regression or Support Vector Machine

# In[ ]:


# Feat Engr

class MyBoxCox(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_to_skew = num_cols
        self.skew_features = []
        self.lam = []
        self.feat_min = np.nan
        
    def fit(self, X, y=None):
        self.skew_features = X[self.cols_to_skew].apply(lambda x: skew(x)).sort_values(ascending=False)
        self.high_skew = self.skew_features[self.skew_features > 0.5]  #t/f large -ve skew gets strange results
        self.skew_index = self.high_skew.index
        
        for i in self.skew_index:
            self.feat_min = X[i].min()
            if self.feat_min > 0 :
                self.lam.append( boxcox_normmax(X[i]) )
            else :
                self.lam.append( boxcox_normmax(X[i] + 1.0 - self.feat_min) ) #min is 1, can use boxcox
     
        self.lam = np.asarray(self.lam)
                
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        if self.skew_index is None:
            raise ValueError('ERROR !! skew index is None !!')
        
        for idx, i in enumerate(self.skew_index) :
            self.feat_min = df[i].min()
            if self.feat_min > 0 :
                df[i] = boxcox(df[i], self.lam[idx])
            else :
                df[i] = boxcox(df[i] + 1.0 - self.feat_min, self.lam[idx])   #min is 1, can use boxcox
        
        return df
        



# In[ ]:


bc_tf = MyBoxCox()
bc_tf.fit(train)
z1=bc_tf.transform(train)
#z3=bc_tf.transform(test)


# # Histograms for Target 

# In[ ]:


plt.figure(figsize=(15,6))
sns.distplot(a=train[Target], label="Cover Type", kde=False) 
plt.title("Distribution of Trees Cover Type")
plt.legend()  # force legend


# # Quick & fast way to see all data

# In[ ]:


train.hist(figsize=(20,30));


# # Train data before Boxcox transform

# In[ ]:


f = pd.melt(train, value_vars=num_cols)
g = sns.FacetGrid(f, col="variable",  col_wrap=5, sharex=False, sharey=False)
g = g.map(sns.distplot, "value", kde=False)


# # Train data after Boxcox transform (numerical features with skew > 0.5 )

# In[ ]:


f = pd.melt(z1, value_vars=num_cols)
g = sns.FacetGrid(f, col="variable",  col_wrap=5, sharex=False, sharey=False)
g = g.map(sns.distplot, "value", kde=False)


# # Correlation Matrix

# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(50,35))
plt.title('Pearson Correlation of Features', y=1.05, size=50)
sns.heatmap(train.corr(),linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# # Run xgboost for pipe cleaning --> get a baseline

# In[ ]:


X = pd.read_csv("../input/learn-together/train.csv", index_col='Id')
X_test = pd.read_csv("../input/learn-together/test.csv", index_col='Id')

y = X.Cover_Type
X = X.drop([Target], axis=1)


# In[ ]:


# Keep a small valid set for SHAP plot
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9 )  #no random seed

xgb_model = XGBClassifier(n_estimators=3000, learning_rate=0.05,  random_state=0, )
xgb_model.fit(X_train, y_train, early_stopping_rounds=50, eval_set=[(X_valid, y_valid)])


# # Use xgboost internal feature importance plot

# In[ ]:


from xgboost import plot_importance
from matplotlib import pyplot
plot_importance(xgb_model)
plt.rcParams["figure.figsize"] = (24, 20)
pyplot.show()


# # Confusion Matrix
# From plots below, it is clear some Target Classes can be separated easily, while other Classes need more work
# 
# This is where SHAP plots come in

# In[ ]:


fig, ax = plt.subplots()
cm = ConfusionMatrix(xgb_model,  ax=ax)

#Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
#cm.fit(X_train, y_train)

cm.score(X_valid, y_valid)
cm.poof()


# In[ ]:


fig, ax = plt.subplots()
visualizer = ClassificationReport(xgb_model,  ax=ax)
ax.grid(False)

#visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_valid, y_valid) 
g = visualizer.poof() 


# # What is SHAP plot ?
# Ref: 
# 
# 1) https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27
# 
# 2) https://github.com/slundberg/shap
# 
# Basically, SHAP values can explain the output of any machine learning model by using a simpler model to approximate the exact model.
# Any model means it is not just restricted to tree classifiers, but non-tree models like SVM. 
# However, computation time is fastest for Tree based models

# # Note: '0' refers to Class 1, '1' refers to Class 2, and so on...

# In[ ]:


# explain the model's predictions using SHAP values
# For non tree models, use shap.KernelExplainer() instead of shap.TreeExplainer(), but computation time may be LONG
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_valid)
shap.summary_plot(shap_values, X_valid)


# # Note: Expected value near 0.5 means Class is difficult to separate

# In[ ]:


# shap_values[0 to 6][0 to n_rows] --> 7 Tgt classes, n rows

explainer.expected_value # mean value of each class


# # SHAP summary plot above is similar to xgboost internal "feature importance" plot
# # However, we can use force_plot or dependence plot to drill down influence of each feature, or each data sample (row) on the Target 

# # Explain entire dataset
# You can use pulldown menu on chart to filter & select patterns !
# 
# Blue means negative contribution to Target, while red is positive contribution

# In[ ]:


shap.force_plot(explainer.expected_value[1], shap_values[1], X_valid) # index [0 to 6] refers to Target Class, select '1' here  


# # Check how feature "Elevation" affects Target

# In[ ]:


shap.dependence_plot("Elevation", shap_values[0], X_valid)


# # Can also check how "Aspect" & "Hillshade_3pm" affect Target

# In[ ]:


shap.dependence_plot("Aspect", shap_values[0], X_valid, interaction_index="Hillshade_3pm")

