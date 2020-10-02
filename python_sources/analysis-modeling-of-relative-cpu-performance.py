#!/usr/bin/env python
# coding: utf-8

# **Analysis of CPU Performance - Balaji**

# **About the dataset:**
# 
# <br>
# <br>
# 1. Title: Relative CPU Performance Data 
# <br>
# 2. Number of Instances: 209 
# <br>
# 3. Number of Attributes: 10 (6 predictive attributes, 2 non-predictive, 
#                              1 goal field, 1 guess field)
# <br>
# 4. Attribute Information:
#    1. vendor name: 30 
#       (adviser, amdahl,apollo, basf, bti, burroughs, c.r.d, cambex, cdc, dec, 
#        dg, formation, four-phase, gould, honeywell, hp, ibm, ipl, magnuson, 
#        microdata, nas, ncr, nixdorf, perkin-elmer, prime, siemens, sperry, 
#        sratus, wang)
#   <br>
#  2. Model Name: many unique symbols
# <br>
#    3. MYCT: machine cycle time in nanoseconds (integer)
#   
# <br> 4. MMIN: minimum main memory in kilobytes (integer)
#    <br>5. MMAX: maximum main memory in kilobytes (integer)
#    <br>6. CACH: cache memory in kilobytes (integer)
#    <br>7. CHMIN: minimum channels in units (integer)
#    <br>8. CHMAX: maximum channels in units (integer)
#    <br>9. PRP: published relative performance (integer)
#   <br>10. ERP: estimated relative performance from the original article (integer)
# 
# 5. Missing Attribute Values: None
# 
# 6. Class field: PRP
# 
# 7. Guess field: ERP
# 

# **Problem Statement:** Predict teh PRP (Published Relative Performance) based on the otehr independent features

# *Importing all required libraries*

# In[ ]:


#Ignore warnings
import warnings
warnings.filterwarnings('ignore')
import os


# *Importing libraries for handling table-like data and matrices*

# In[ ]:


import numpy as np
import pandas as pd


# *Importing Modelling Algorithm libraries*

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.model_selection import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV


# *Importing Visualisation libraries*

# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns


# Creating different functions for easy Visulizations & less coding

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

#Setup helper functions

def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

def plot_correlation_map(df):
    corr = data.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr,
        cmap = cmap,
        square=True,
        cbar_kws={ 'shrink' : .9 },
        ax=ax,
        annot = True,
        annot_kws = { 'fontsize' : 12 }
    )


def describe_more(df):
    var = [];
    l = [];
    t = []
    for x in df:
        var.append(x)
        l.append(len(pd.value_counts(df[x])))
        t.append(df[x].dtypes)
    levels = pd.DataFrame({'Variable': var, 'Levels': l, 'Datatype': t})
    levels.sort_values(by='Levels', inplace=True)
    return levels


def plot_variable_importance(X, y):
    tree = DecisionTreeClassifier(random_state=99)
    tree.fit(X, y)
    plot_model_var_imp(tree, X, y)


def plot_model_var_imp(model, X, y):
    imp = pd.DataFrame(
        model.feature_importances_,
        columns=['Importance'],
        index=X.columns
    )
    imp = imp.sort_values(['Importance'], ascending=True)
    imp[: 10].plot(kind='barh')
    print(model.score(X, y))


# In[ ]:


print(os.listdir('../input'))
os.curdir #gives us the current directory
os.listdir() #Lists all the files in teh current directory
names = ['VENDOR','MODEL_NAME','MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP'] #Since our data has no lables we are creating a df with lables and will be later added to the Data
data = pd.read_csv('../input/machine.data', names= names)
data = pd.DataFrame(data)


# In[ ]:


pd.set_option('display.expand_frame_repr', False) # to display max cols & rows


# In[ ]:


data.head() #shows the first five rows


# In[ ]:


data.tail() #shows last five rows


# In[ ]:


data.describe()


# In[ ]:


data.info()


# From the above we can see that there are 10 columns where 8 are int & 2 are Categorical

# From the data description we can see having both ERP & PRP do not make much, and the reason is 'PRP: published relative performance (integer)'
# 'ERP: estimated relative performance from the original article (integer)' so we can remove 'ERP' & keep only 'PRP' for our prediction purpose.

# In[ ]:


#Copying ERP colum into a seperate dataframe
data_ERP =  data[data.columns[-1]]
data_ERP.head()
#Dropping ERP column
data=data.drop('ERP',axis=1)
data.info()


# Now we can see that we only have 9 columns since ERP is dropped.

# In[ ]:


describe_more(data)


# In[ ]:


data.corr() #Correlation matrix to see the tentative dependencies between columns


# In[ ]:


data['VENDOR'].value_counts().sort_values(ascending=False)
data.groupby('VENDOR')['MODEL_NAME'].nunique().sort_values(ascending=False)
(data.groupby('VENDOR')['PRP'].mean()).sort_values(ascending=False)


# **Checking for Skewness.**
# 
# **What is Skewness?**
# 
# Skewness. It is the degree of distortion from the symmetrical bell curve or the normal distribution. It measures the lack of symmetry in data distribution. It differentiates extreme values in one versus the other tail.
# 
# **Why it is important?**
# **Reason: **
# 
# Skewed data leads to Bias.
# 
# **Reference link: **https://whatis.techtarget.com/definition/skewness

# In[ ]:


data[['MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP']].skew()


# In[ ]:


data[['MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP']].kurtosis()


# We can see that the dataset is highly Skewed, an optimal value for skewness should be -0.5 to 0.5 for a normalised data.<br> Inorder to unskew the data, there are various methods followed and one of the popular method is taking natural log.

# Before we take log tarnsaformation to treat the skewness we need to check if we have any negative or zero values, if we have then we can add a constant to it then we can take log transaformation.

# In[ ]:


(data.loc[:,'MYCT':'PRP']).apply(lambda column: (column<= 0).sum()) 
#Checking for '0' values


# Here we can see CACH,CHMIN,CHMAX columns has <=0 values so we can add constant to it then take log i.e log(x+1)

# Before that, exploring Rows where CACH = 0, Because 69 rows containg zero is very high since our total number of rows = 209, that is here we can see 33% of the rows are '0' for CACH.

# In[ ]:


Cach_zero_rows = data.loc[data['CACH']== 0]


# In[ ]:


#Confirming the CACH == 0
Cach_zero_rows.head()


# In[ ]:


#Cach_zero_rows['CACH'].value_counts().sort_values(ascending=False)
(Cach_zero_rows.groupby('VENDOR')['MODEL_NAME'].nunique().sort_values(ascending=False) ,
data.groupby('VENDOR')['MODEL_NAME'].nunique().sort_values(ascending=False))
#(data.groupby('VENDOR')['PRP'].mean()).sort_values(ascending=False)


# In[ ]:


#Corr plot for data to compare with the above
plot_correlation_map(data.loc[data['CACH']!= 0])


# From the above Corr plot we can see the Dependent variable has 61% to 86% of correlation with the independent variables.
# <br>
# Note: 'correlation doesn't imply causation'
# <br>
# Refer: https://towardsdatascience.com/why-correlation-does-not-imply-causation-5b99790df07e

# Now coming back to log transformation, adding constant '1' to the columns where it comtains values equal to '0'

# In[ ]:


data['CACH'] = data['CACH']+1
data['CHMIN'] = data['CHMIN']+1
data['CHMAX'] = data['CHMAX'] +1


# In[ ]:


(data[['MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP']]).apply(lambda column: (column<= 0).sum())


# Here we can see that now there aren't and <= 0 values in any columns, so now we can proceed with teh log transformation.

# In[ ]:


data_log = np.log(data[['MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP']])


# In[ ]:


print('Before Log Transformed')
(data.iloc[:,2:].head(5))


# In[ ]:


print('After Log Transformed')
(data_log.head())


# In[ ]:


data_log[['MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP']].skew() #Checking skewness for the data_log


# In[ ]:


data_transformed = pd.concat([data[['VENDOR','MODEL_NAME']],data_log], axis=1, ignore_index=False) #Combaining data_log dataframe & 'VENDOR','MODEL_NAME' columns from data
data_transformed.head()


# In[ ]:


print("mean : ", np.mean(data_transformed[data_transformed.columns[2:]]))
print("var  : ", np.var(data_transformed[data_transformed.columns[2:]]))
print("skew : ",(data_transformed[data_transformed.columns[2:]]).skew())
print("kurt : ",(data_transformed[data_transformed.columns[2:]]).kurtosis())


# In[ ]:


data_transformed.describe()


# In[ ]:


#From data_transformed creating df & y dataframes for Independent & Dependent variables
df = data_transformed[data_transformed.columns[0:8]]
y = data_transformed[data_transformed.columns[-1:]]


# In[ ]:


plot_variable_importance(data_transformed[data_transformed.columns[2:-1]].astype('int'), y.astype('int'))


# From the above we can see that CACH has the maximum variable importance and all these variables.

# Since we have only 2 categorical variables and the number of unique elements are less we can do the one hot code of converting all categorical elements to numerical

# In[ ]:


#Onehot coding for VENDOR & MODEL
df_V_M = df.iloc[:,0:2]


# In[ ]:


df_V_M.head()
df_V_M = pd.get_dummies(df_V_M) 
df = pd.concat( [df_V_M,df ] , axis=1 ).sort_index()
df = df.drop(columns=['VENDOR', 'MODEL_NAME'])


# In[ ]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(df,y, test_size = 0.15, random_state = 42)


# In[ ]:


print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# **Using Randomforest Algorithm**

# In[ ]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);


# Since the rf output is 'numpy.ndarray' we need to convert test_lables to np array

# In[ ]:


# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
print(type(predictions), type(test_labels))
# Calculate the absolute errors
errors = ((predictions) - np.array(test_labels)) #Here converting the test_lables to 'np.array' to calculate the 'errors'
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), '.')


# In[ ]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * ((errors / np.array(test_labels)))
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# The above seems to be good,but lets try otehr models too

# **Trying linear regression model - #lm method**

# In[ ]:


from sklearn import linear_model
lm = linear_model.LinearRegression()
model = lm.fit(train_features, train_labels)
predictions = lm.predict(test_features)
## The line / model
plt.scatter(test_labels, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

print ('Score:', model.score(test_features, test_labels))


# Here we can see that linear method got us 84% accuracy not bad but comparing to RF its poor.

# **GradientBoosting Regressor**

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
model= GradientBoostingRegressor()
model.fit(train_features, train_labels)
model.score(test_features,test_labels)
print ('Score:', model.score(test_features, test_labels))
predictions = model.predict(test_features)
## The line / model
plt.scatter(test_labels, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()


# **AdaBoostRegressor**

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor()
model.fit(train_features , train_labels)
model.score(test_features,test_labels)
predictions = model.predict(test_features)
## The line / model
plt.scatter(test_labels, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()


# **BaggingRegressor**

# In[ ]:


from sklearn.ensemble import BaggingRegressor
from sklearn import tree
model = BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))
model.fit(train_features , train_labels)
model.score(test_features,test_labels)
predictions = model.predict(test_features)
## The line / model
## The line / model
plt.scatter(test_labels, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()
print ('Score:', model.score(test_features, test_labels))


# As we can see here,
# 
# * Random forest Regressor gives **90.08 %** of Accuracy
# <br>
# * Linear regression gives **84.39%** of Accuracy
# <br>
# * Gradient Boosting Regressor gives **86.41%** of Accuracy
# <br>
# * AdaBoost Regressor gives **84.15%** of Accuracy
# <br>
# * Bagging Regressor gives **84.25%** of Accuracy

# ***Going forward we can create more features, do some data transformation, or even convert the regression into a classification if predicting the class of PRP is useful.***
