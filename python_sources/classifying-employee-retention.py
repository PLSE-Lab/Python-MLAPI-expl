#!/usr/bin/env python
# coding: utf-8

# # This is my first mostly unguided attempt at a full kernel on Kaggle.
# ### Its quite messy as my primary goal is getting the various methods under my fingers. After my first attempt of tackling the dataset on my own, I begin using methods other members use to: 
# ### a) Learn new methods of analysis as well as their rationale
# ### b) See how other people structure their code
# ### c) Make sure I didn't miss anything
# ### When I do use other people's methods in this kernel I credit them (from the first one that I see that uses it; often many of them will use the same methods). This is a *learning* kernel for me, and likely the next few kernels I make public.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.preprocessing import scale, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.utils import shuffle


# # Data Acquisition and initial observations

# In[ ]:


df = pd.read_csv("../input/HR_comma_sep.csv")
print( df.isnull().any() )
df.head()


# In[ ]:


print( df.left.value_counts() )
print( df.sales.value_counts() )
print( df.salary.value_counts() )
df.describe()


# ### The response is imbalanced (~24% left). There are many ways to correct for this situation. In this sort of situation, some sort of resampling would be helpful. Since the two classes are not incredibly unbalanced, oversampling is likely to be the best approach. SMOTE is something worth considering although there are categorical  predictors, making the KNN section of that procedure problematic.

# # Data Preprocessing

# ### I recognize I'm playing pretty fast and loose with the information storage on this notebook. Since the dataset is relatively small I have no problems storing all these various "versions" of the same dataset in python's memory for the sake of my convenience. I'm also making all of these instances with various names to keep myself organized as I get used to the muscle memory of writing kernels.

# In[ ]:


X = df.drop('left', axis=1)
y = df['left']
print( X.shape )
print( y.shape )


# In[ ]:


X = X.rename(columns={'sales': 'sales1'})
X_sales = pd.get_dummies(X['sales1'])
X_salary = pd.get_dummies( X['salary'] )
Xd = pd.concat( [X, X_sales, X_salary], axis=1 )
Xd = Xd.drop( ['sales1','salary'], axis=1)
print( Xd.shape )
Xd.head()


# In[ ]:


sc = StandardScaler().fit(Xd)
Xs = sc.transform(Xd)
Xs, y = shuffle( Xs, y, random_state = 10 )


# ### Time to try a bunch of resampling methods

# In[ ]:


from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN


# In[ ]:


ROS = RandomOverSampler()
Xo1, yo1 = ROS.fit_sample(Xs, y)
Xo1, yo1 = shuffle( Xo1, yo1 )
print( Xo1.shape )
print( yo1.shape )
print( yo1.mean() )


# In[ ]:


SMO = SMOTE()
Xo2, yo2 = SMO.fit_sample(Xs, y)
Xo2, yo2 = shuffle( Xo2, yo2 )
print( Xo2.shape )
print( yo2.shape )
print( yo2.mean() )


# In[ ]:


ADA = ADASYN()
Xo3, yo3 = ADA.fit_sample(Xs, y)
Xo3, yo3 = shuffle( Xo3, yo3 )
print( Xo3.shape )
print( yo3.shape )
print( yo3.mean() )


# # Model Fitting

# ### While it's going to be pretty gross at first, I will be making heavy use of cross_validation here. I haven't quite gotten the knack of efficiently or elegantly handling it, and this is merely another iteration of me trying it. I mostly go through all this rigor as I'm mostly getting a feel for how models perform and I feel more confident with cross validation when possible as opposed to fitting one model.

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
import xgboost as xgb


# ### As is typical with all of my analysis, I start with a Linear/Logistic Regression

# In[ ]:


# Xo1_train, Xo1_test, yo1_train, yo1_test = train_test_split( Xo1, yo1, test_size = 0.3 )
kf = KFold( n_splits = 5 )
ac = np.zeros( 5 ); re = np.zeros( 5 ); pr = np.zeros( 5 )
Xo1, yo1 = shuffle( Xo1, yo1, random_state = 0 )
i = 0
for train_index, test_index in kf.split(Xo1):
    Xo1_train, Xo1_test = Xo1[train_index], Xo1[test_index]
    yo1_train, yo1_test = yo1[train_index], yo1[test_index]
    LgR = LogisticRegression()
    LgR.fit( Xo1_train, yo1_train )
    yo1_pred = LgR.predict( Xo1_test )
    ac[i] = accuracy_score( yo1_test, yo1_pred )
    re[i] = recall_score( yo1_test, yo1_pred )
    pr[i] = precision_score( yo1_test, yo1_pred )
    i = i+1


# In[ ]:


def d_method( X, y, model, random_state = 0 ):
    kf = KFold( n_splits = 5 )
    ac = np.zeros( 5 ); re = np.zeros( 5 ); pr = np.zeros( 5 )
    X, y = shuffle( X, y, random_state = random_state )
    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit( X_train, y_train )
        y_pred = model.predict( X_test )
        ac[i] = accuracy_score( y_test, y_pred )
        re[i] = recall_score( y_test, y_pred )
        pr[i] = precision_score( y_test, y_pred )
        i = i+1
    return( ac.mean(), re.mean(), pr.mean(), model )


# In[ ]:


ac, re, pr, model = d_method( Xo1, yo1, LogisticRegression())
print( "Accuracy Score =  ", ac )
print( "Recall Score =    ", re )
print( "Precision Score = ", pr )


# In[ ]:


from astropy.table import Table, Column


# In[ ]:


t = Table( names = ('C','Accuracy','Recall','Precision') )
reg_strength = [0.01, 0.1, 1, 10, 100]
for c in reg_strength:
    ac, re, pr, model = d_method( Xo1, yo1, LogisticRegression(penalty = 'l1', C=c) )
    t.add_row( (c, ac, re, pr) )
t


# In[ ]:


t = Table( names = ('C','Accuracy','Recall','Precision') )
reg_strength = [0.01, 0.1, 1, 10, 100]
for c in reg_strength:
    ac, re, pr, m = d_method( Xo2, yo2, LogisticRegression(penalty = 'l1', C=c) )
    t.add_row( (c, ac, re, pr) )
t


# In[ ]:


t = Table( names = ('C','Accuracy','Recall','Precision') )
reg_strength = [0.01, 0.1, 1, 10, 100]
for c in reg_strength:
    ac, re, pr, model = d_method( Xo3, yo3, LogisticRegression(penalty = 'l1', C=c) )
    t.add_row( (c, ac, re, pr) )
t


# In[ ]:


t = Table( names = ('Accuracy','Recall','Precision') )
ac, re, pr, model = d_method( Xo1, yo1, DecisionTreeClassifier() )
t.add_row( (ac, re, pr) )
t


# This classification seems a bit *too* on the nose. To get a sligtly better idea of what's going on here, I'll find feature importance from a random forest algorithm. From there I can determine the useful features and investigate them visually to notice any trends.

# In[ ]:


t = Table( names = ('Accuracy','Recall','Precision') )
ac, re, pr, model = d_method( Xo1, yo1,RandomForestClassifier() )
t.add_row( (ac, re, pr) )
t


# In[ ]:


fi = pd.DataFrame()
fi['features'] = Xd.columns.values
fi['importance'] = model.feature_importances_
fi


# It looks like there are a clean 5 features to investigate:
# Satisfaction Level
# Last Evaulation
# Number Project,
# Average Monthly Hours
# and Time Spent with Company.
# Since these likely have some correlation between them, I will run a quick correlation chart
# 

# ## Note from looking at randy lao's kernel: I did not need to use dummy variables for my categorical predictors when using decision tree methods. While it likely does not affect the *fit* of the results, the feature importances were not properly weighted.
# 
# ## Decision tree methods are also more resistant to imbalanced classes than most classification methods, so it may not be necessary to oversample here. Same goes with scaling the predictors.
# 

# In[ ]:


Xc = df.drop('left', axis=1)
yc = df['left']
Xc['sales'] = Xc['sales'].astype('category').cat.codes
Xc['salary'] = Xc['salary'].astype('category').cat.codes
Xc = np.array(Xc)
yc = np.array(yc)
print(X.shape)
print(y.shape)


# In[ ]:


t = Table( names = ('Accuracy','Recall','Precision') )
ac, re, pr, model = d_method( Xc, yc, DecisionTreeClassifier(class_weight = "balanced") )
t.add_row( (ac, re, pr) )
t


# In[ ]:


fi = pd.DataFrame()
fi['features'] = X.columns.values
fi['importance'] = model.feature_importances_
fi


# The importances didnt change significantly with the preprocessing difference. I did however realize that preprocessing this late into the kernel is a logistical pain, so I will be more careful with my data nomenclature in the future. Anyways, onto the heatmap of correlation (with methodology borrowed from randy lao's kernel)

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.plotly as py
import plotly.graph_objs as go


# Ahh, the actual data visualization, which when it comes to python I am at this point completely new to. I'm primarily used to R visualization so I apologise in advance for the mess.

# In[ ]:


df = pd.read_csv("../input/HR_comma_sep.csv")
df.head()


# In[ ]:


X = df.drop('left',axis = 1 )
y = df['left']
print(X.shape)
print(y.shape)


# In[ ]:


def d_hist(X,y,look,n_bins):
    X0 = X.loc[ y == 0 ]
    X1 = X.loc[ y == 1 ]
    d0 = X0[look]
    d1 = X1[look]
    n1, bins1, patches1 = plt.hist(d0, n_bins, facecolor='orange', alpha = 0.5)
    n0, bins0, patches0 = plt.hist(d1, n_bins, facecolor='blue', alpha = 0.5)
    plt.xlabel(look)
    plt.ylabel('Employees')
    plt.title(look + ' vs Retention')
    plt.legend(labels = ['retained','left'])
    plt.show()


# In[ ]:


d_hist( X,y, 'satisfaction_level',50)
d_hist( X,y, 'last_evaluation', 50)
d_hist( X,y, 'average_montly_hours', 50)


# In[ ]:


def d_bar(X,y,look):
    X0 = X.loc[ y == 0 ]
    X1 = X.loc[ y == 1 ]
    c0 = X0[look].value_counts(sort = False)
    c1 = X1[look].value_counts(sort = False)
    f0 = pd.DataFrame(c0)
    f1 = pd.DataFrame(c1)
    f = pd.concat([f0,f1],axis=1)
    f = f.fillna(0).astype(int)
    l = np.arange(f.shape[0])
    plt.bar(l,f.iloc[:,0]/sum(f.iloc[:,0]),facecolor = 'orange', alpha = 0.5)
    plt.bar(l,f.iloc[:,1]/sum(f.iloc[:,1]),facecolor = 'blue', alpha = 0.5)
    if isinstance(f.index.values[0],str):
        plt.xticks(l,f.index.values, rotation = 70)
    else:
        plt.xticks(l,f.index.values)
    plt.xlabel(look)
    plt.ylabel('Percent Employees')
    plt.title(look + ' vs Retention')
    plt.legend(labels = ['retained','left'])
    plt.show()


# In[ ]:


d_bar( X, y, 'number_project')
d_bar( X, y, 'time_spend_company')
d_bar( X, y, 'Work_accident')
d_bar( X, y, 'promotion_last_5years')
d_bar( X, y, 'sales')
d_bar( X, y, 'salary')


# In[ ]:


def d_dot( X, y, look1, look2):
    x1 = X[look1]; x2 = X[look2]
    x10 = x1[y==0]; x20 = x2[y==0]
    x11 = x1[y==1]; x21 = x2[y==1]
    plt.plot(x10,x20, marker = '.', linestyle = 'None', 
             color = 'orange', alpha = 0.5, label = 'Retained')
    plt.plot(x11,x21, marker = '.', linestyle = 'None',
             color = 'blue',alpha = 0.5, label = 'Left')
    plt.xlabel(look1)
    plt.ylabel(look2)
    plt.title(look1 + ' vs ' + look2)
    plt.legend()
    plt.show()


# In[ ]:


d_dot( X, y, 'satisfaction_level', 'last_evaluation')
d_dot( X, y, 'satisfaction_level', 'average_montly_hours')
d_dot( X, y, 'last_evaluation','average_montly_hours')


# Here's where we're getting such good results from decicion trees. This is a weakness with practicing on simulated data.

# In[ ]:




