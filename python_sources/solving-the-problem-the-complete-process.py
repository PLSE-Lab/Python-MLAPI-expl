#!/usr/bin/env python
# coding: utf-8

# I made this kernel with 3 ideas in mind:
# * Make it simple: This kernel is for beginners, it's purpose is not to get surprising results but to teach.
# * Make it complete: It tries to represent the complete data science task over this problem.
# * Make it short.: It tries to transmit information in a compact way. If you wanted a book, you wouln't be here.
# 
# This kernel is divided in 5 parts:
# 1. **Problem introduction**: This kernel starts by quickly explaining the problem and listing the available features
# 2. **Basic cleaning and Data visualization**: We'll get to know our dataset through different plots and we'll use the information extracted to clean it.
# 3. **Missing values**:  We'll focus on  dealing with the missing data
# 4. **Features transformation and addition**: We'll apply some transformations to our features and we'll add more of them too.
# 5. **Model Selection and Validation**: We'll select a model and adjust its parameters with cross validation. We'll validate our model and then use it to make our submision
# 
# 
# 
# <font size=5>1. **Problem introduction**
# 
# <font size=2> Basically, this is a regression problem where we have to predict the sale price of houses based  on their features
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings('ignore')


# In[ ]:


traindata=pd.read_csv('../input/train.csv')#reading the data
testdata=pd.read_csv('../input/test.csv')
print('Number of rows and columns of the training set: ',traindata.shape)
print('Number of rows and columns of the test set: ',testdata.shape)


# We have one more column (variable) in the training set than in the test set because the test set isn't labelled (Kaggle do this to avoid cheatting).
# 
# We want to know which of these 81 columns are numerical and which are nominal. This is very useful because we will treat these two types in a different way.

# In[ ]:


numeric_cols=traindata.select_dtypes(include=[np.number]).columns#select only numerical
nominal_cols=traindata.select_dtypes(exclude=[np.number]).columns#select only non numerical
print(numeric_cols.shape[0],'numeric columns: ',numeric_cols)
print(nominal_cols.shape[0],'nominal columns: ',nominal_cols)


# For a propper analysis we need to understand the meaning of the variables, so ar this point it is good practice to take a look at data_description.txt.
# 
# <font size=5>2. **Basic cleaning and Data visualization**
# 
# <font size=2>it's ime to visualize the data. Matplotlib and seaborn offer a wide variety of plots that can help us to understand the data and the results. 
# 
# A first approach to the data can be through our target variable SalePrice. For this, we can use different plots like the histogram or the boxplot. In our case, we are using the histogram.

# In[ ]:


fig, ax = plt.subplots()
ax.hist(traindata['SalePrice'],40)
ax.set_xlabel('SalePrice')
plt.show()


# We can see that the variable is positively skewed (the data above the 3rd quartile is very sparse). We don't want skewness, because high skewness implies non normality and that is bad because a lot of clasifiers and statistical tests asume normality of the variables. One technique that is commonly used to correct positive skewness is the logarithm. We plot a second histogram with this transformation.

# In[ ]:


fig, ax = plt.subplots()
ax.hist(np.log(traindata['SalePrice']),40)
ax.set_xlabel('log(SalePrice)')
plt.show()


# We can see how the transformed variable follows a a pretty normal distribution. The base of the logarithm is not trivial and can affect to the results of the transformation, although we can see that here the natural logarithm performs cool.
# 
# Another usual example with this dataset is the scatterplot of GrLivArea and SalePrice. 

# In[ ]:


fig, ax = plt.subplots()
ax.scatter(traindata['SalePrice'],traindata['GrLivArea'])
ax.set_xlabel('SalePrice')
ax.set_ylabel('GrLivArea')
plt.show()


# In the scatter plot, 4 clear outliers can be detected. We are going to remove them by keeping only  the rows with GrLivArea less than 4000. Note that we only can drop outliers of the training set. Never do it on the data set.
# 
# After that, we drop the Id column because it provides no information to the analysis. We keep the test ids in a separate variable because we'll need them for the submision.
# 
# Then, we join the training and test set because future transformations will be done to both of them.

# In[ ]:


traindata=traindata[traindata['GrLivArea']<4000] #drop outliers in train
traindata=traindata.drop('Id',axis=1) #remove the column from train
test_ids = testdata['Id'] #save id column from test
testdata=testdata.drop('Id',axis=1) #remove the column from test
numeric_cols=numeric_cols[numeric_cols!='Id'] #remove the column name from this list as well

data=pd.concat([traindata,testdata],axis=0,ignore_index=True) #concatenate training and test set for future transformations
print(data['SalePrice'].head())#Don't worry about the SalePrice variable that is not in test. 
print(data['SalePrice'].tail()) #It's filled with NAs
data['SalePrice'] = np.log(data['SalePrice']) #apply the logarithm to SalePrice


# Now, we are using one of the most common ones for data analysis, the heatmap. Thanks to it, we can have a graphic representation of the correlation matrix.

# In[ ]:


correlation=traindata[numeric_cols].corr() #obtain the correlation matrix
sns.set()
fig, ax = plt.subplots(figsize=(16,8))
sns.heatmap(correlation,ax=ax)
plt.show() #draw the correlation matrix


# The above image show the correlation matrix (for which only numeric variables can be used). We have some very strong correlations such as GarageCars with GarageArea. May be that some variables are redundant? We can confirm it through the following plot, in which only the variables with a strong correlation with any other variable are showed

# In[ ]:


aux=(abs(correlation)-np.identity(correlation.shape[0])).max() #maximum correlation of each variable
selected_feats=aux[aux>0.5].index#take only variables whose maximum correlation is strong.
sns.set()
fig, ax = plt.subplots(figsize=(16,8))
sns.heatmap(correlation.loc[selected_feats,selected_feats], annot=True,fmt='.2f',ax=ax)
plt.show()


# Here we have some candidates for redundant variables.
# * TotalBsmtSF with 1stFlrSF: They represent moreless the same information because the area of the first flor will always be near the total basement area. For this reason, we can remove one of them, in our case 1stFlrSF
# * GarageCars with GarageArea: Here we have two variables giving the same information (the size of a garage) in two different ways (the maximum number of cars that can fit and the surface). We are going to keep only GarageCars.
# * YearBuilt with GarageYrBlt: Well, this two variables don't say the same, but the garage and the house are usually built in the same year. We remove GarageYrBlt.
# * TotRmsAbvGrd with GrLivArea: These two have a strong correlation too, although I've decided to keep both because I think that we may have some houses with a large living area but a low number of rooms if each room is very big. The contrary may also happen.
# 
# After this cleaning, we'll see our best variables (the most correlated ones with SalePrice)

# In[ ]:


data=data.drop(['GarageArea','1stFlrSF','GarageYrBlt'],axis=1) #remove columns
numeric_cols=numeric_cols[numeric_cols!='GarageArea'] #remove them from our list too
numeric_cols=numeric_cols[numeric_cols!='1stFlrSF']
numeric_cols=numeric_cols[numeric_cols!='GarageYrBlt']

correlation=traindata[numeric_cols].corr() #calculate again the correlation matrix (without the removed columns)
aux=abs(correlation['SalePrice']).sort_values(ascending=False) #sort variables by their correlation with SalePrice
selected_feats=aux[0:19].index #Take the best 19. Why 19? because.
sns.set()
fig, ax = plt.subplots(figsize=(16,8))
sns.heatmap(correlation.loc[selected_feats,selected_feats], annot=True,fmt='.2f',ax=ax)
plt.show()


# Scatter plots are also very useful to see correlations between variables. We are going too see how our best numeric variables affect SalePrice.

# In[ ]:


selected_feats=selected_feats[1:] # don't take SalePrice

fig, axes = plt.subplots(nrows=6,ncols=3,figsize=(16,32),sharey=True)
axes=axes.flatten()
for i in range(len(axes)):
    axes[i].scatter(traindata[selected_feats[i]],traindata['SalePrice'])
    axes[i].set_xlabel(selected_feats[i])
    axes[i].set_ylabel('SalePrice')
plt.show()


# A similar type of plot can be done with the nominal variables. For this example, we are going to randomly choose some of them to visualize

# In[ ]:


selected_nominal_feats= np.random.choice(nominal_cols,18,replace=False)
fig, axes = plt.subplots(nrows=6,ncols=3,figsize=(16,32),sharey=True)
axes=axes.flatten()
for i in range(len(axes)):
    sns.set()
    sns.stripplot(x=selected_nominal_feats[i], y='SalePrice', data=traindata,ax=axes[i],jitter=True)
    axes[i].set_xlabel(selected_nominal_feats[i])
    axes[i].xaxis.set_tick_params(rotation=60)
plt.subplots_adjust(hspace = 0.5)
plt.show()


# With just this litle analysis we have removed some useless data and, more importantly, we have a better understanding of our data.
# 
# <font size=5>3. **Missing values**
# 
# <font size=2>Let's mannage now the missing values: 
# 
# We show the absolute and relative number of missing values for each variable with any missing value.

# In[ ]:


missing_values=data.isnull().sum() #obtain the number of missing values by column
numeric_missing=missing_values[numeric_cols] #separate the numeric variables
numeric_missing['SalePrice']=0 #We don't want to detect SalePrice's NAs because they are not real, they just belong to test set
numeric_missing=numeric_missing[numeric_missing>0] #we only want to see variables with 1 or more missings
numeric_missing_df= pd.DataFrame()
numeric_missing_df[['absolute','relative']]= pd.concat([numeric_missing,numeric_missing/data.shape[0]],axis=1)
numeric_missing_df #table of missing numeric values 


# In[ ]:


#exactly the same for nominal variables
nominal_missing=missing_values[nominal_cols] 
nominal_missing=nominal_missing[nominal_missing>0]
nominal_missing_df= pd.DataFrame()
nominal_missing_df[['absolute','relative']]= pd.concat([nominal_missing,nominal_missing/data.shape[0]],axis=1)
nominal_missing_df


# So many missing values, where do we start from? Well, we can basically do 3 different things.
# 
# 1.   Remove columns: We usually remove columns with a large ammount of missing values because they are useless for the prediction.
# 2.     Replace values: Even if we have a missing , we can try to guess it and replace it by a probable value. Simplifying, we can replace in three ways
#     * By a single value like the median, the mean or the mode. This is dangerous if we are filling a variable with a lot of missings, because we can deform its true distribution.
#     * By randomly generated values from the distrubution of the variable. This is useful if we know (or can deduce) the distribution of the variable and we have to fill a lot of missing values.
#     * By predicting them through regression techniches. This task is more complex but it can achieve better replacements in some cases.
# 3.     Remove rows: THIS CAN ONLY BE DONE ON THE TRAINING SET. For example, we would use this if we had some training rows with a missing value in a column with few missing values in the training set and any in the test set.
# 
# Said this, we are going to start by removing some columns. In more than 93% of the cases we don't have alley. Something similar happens with PoolQC and MiscFeature, so they are not expected to give us much information.

# In[ ]:


data=data.drop(['Alley','PoolQC','MiscFeature'],axis=1) #remove columns
nominal_cols=nominal_cols[nominal_cols!='Alley']
nominal_cols=nominal_cols[nominal_cols!='PoolQC']
nominal_cols=nominal_cols[nominal_cols!='MiscFeature']


# Fortunately, the most numeric columns have none or few missing values. Just one of them LotFrontage has more, but not enough of them to be removed. To avoid the deformation of the distribution of this variable, we are going to substitude their missing values by a random ponderated choice between its quartiles and its median. 
# 
# We  replace the rest numeric variables' missings by their medians. As they are very few, it's not important.
# 
# At this point we only have missing values on the nominal variables. We simply replace them by a new label 'NA'. In this case, a more in depth study of the nominal variables and their missing values would improve the results. I just didn't consider this important enough for this kernel to give it more lines of code.

# In[ ]:


pos_params=data['LotFrontage'].describe()#get position parameters of the variable
pos_params=[pos_params['25%'],pos_params['50%'],pos_params['75%']]
chosen_values=np.random.choice(pos_params,numeric_missing['LotFrontage'],p=[0.25,0.5,0.25]) #randomly choose between 1sQ, median and 3rdQ
data.loc[data['LotFrontage'].isnull(),'LotFrontage']=chosen_values #fill missings
    
for fillvar in numeric_missing.index:
    data[fillvar]=data[fillvar].fillna(data[fillvar].median()) #fill with median

numeric_missing=data[numeric_cols].isnull().sum() 
numeric_missing['SalePrice']=0
print('Remaining numeric missing values: ',numeric_missing.sum())
data=data.fillna('NA') #fill nominal ones with 'NA' label
print('Remaining nominal missing values: ',data[nominal_cols].isnull().sum().sum())


# <font size=5>4. **Features transformation and addition**
# 
# <font size=2>As we have seen at the begining, the logarithmic transformation achieves normality on SalePrice. Then we can expect it to work well on the rest of the positively skewed variables. Note that we will use log1p (that adds one before applying the logarithm) instead of log for preventing posible errors related with log(0).
# 
# The logarithm does not correct negative skewness. For that purpose we have other transformations like the exponential one. Be careful with overflow if you use it. 

# In[ ]:


skewness=data.skew(axis=0,numeric_only=True) #the skewness of the numeric variables (Salesprice is not included)
posskewness = skewness[skewness > 0.5] #We take only the positively skewed vriables
posskewed_features = posskewness.index #The names of that variables        
data[posskewed_features] = np.log1p(data[posskewed_features]) #we apply the log(x+1) to each variable x
print('Corrected features: ',posskewed_features)


# Many models (like the one we're going to use in this kernel) accept only numeric variables, so we need to convert our nominal variables to numeric.This can be done in different ways. Some of them are:
# * Replace the nominal values by integers following a codebook for each variable. 
# * Create dummy variables (binary variables). A dummy variable is created for each nominal value. Note that if we use this approach, we have to get the dummy variables of the trainning and test sets both together because if not we would probably have different dummy variables in both sets.
# 
# In this case, for simplicity, we choose the dummies for converting all the nominal variables. It's not necesary to choose one criterion for all the variables, in fact each one should be studied to decide which transformation to apply on it.

# In[ ]:


data=pd.concat([data[numeric_cols],pd.get_dummies(data[nominal_cols])],axis=1)
print('Number of rows and columns: ',data.shape)


# With the dummies our column number has risen a lot.
# 
# Do you remember our best variables? We found them from their correlation with SalePrice. We are going to create new features from the powers of this variables, adding a total number of 36 new variables (without modifying nor replacing the old ones). This technique can sometimes improve the results, especially if the variables powered have a non linear correlation with the target.

# In[ ]:


data[selected_feats+'2']=np.power(data[selected_feats],2) #create new variables powering to 2
data[selected_feats+'3']=np.power(data[selected_feats],3) #create new variables powering to 3
numeric_cols=np.hstack([numeric_cols,selected_feats+'2',selected_feats+'3']) #add the new features to our list
print('Number of rows and columns: ',data.shape)


# Another transformation that we're going to apply is the scaling. Scikit learn offers different scalers. StandardScaler just substracts the mean and divides by the standard deviation.
# 
# First we separate again our data in our original training data and test data. Do you remember we joined them for the transofrmations?
# 
# Then we separate our train data in a train and a test. This is just because we want to have an estimate of the test error of our submision. We can't do it with our original test data because it's not labelled, so we have to use this trick.
# 
# Then we apply the scaler to the sets separately (we don't want the test mean and standard deviation to affect to the train set, for example).
# 
# At the end we separate the column of SalesPrice for using it as target in our regressor. We don't do it for the test data because it has no SalePrice column.

# In[ ]:


traindata=data.iloc[:traindata.shape[0],:] 
testdata=data.iloc[traindata.shape[0]:,:]
testdata=testdata.drop('SalePrice',axis=1) #We drop the unknown variable in the test. It was just filled with NAs

train, test= train_test_split(traindata, test_size = 0.25, random_state = 0)

stdSc = StandardScaler()

numeric_cols=numeric_cols[numeric_cols!='SalePrice'] #We don't want to scale SalePrice
train.loc[:, numeric_cols] = stdSc.fit_transform(train.loc[:, numeric_cols])#scaling tranformation
test.loc[:, numeric_cols] = stdSc.transform(test.loc[:, numeric_cols])

traindata.loc[:, numeric_cols] = stdSc.fit_transform(traindata.loc[:, numeric_cols])
testdata.loc[:, numeric_cols] = stdSc.transform(testdata.loc[:, numeric_cols])

X_train=train.drop('SalePrice',axis=1)#Separate the target variable form the rest
y_train=train['SalePrice']
X_test=test.drop('SalePrice',axis=1)
y_test=test['SalePrice']

X_traindata=traindata.drop('SalePrice',axis=1)
y_traindata=traindata['SalePrice']


# <font size=5>5. **Model Selection and Validation**
# 
# <font size=2>Now it's time for regression! For me this is most exciting one. Here is where we will get the reward for our hard work. Although this part is not trivial. Sometimes selecting the right clasifier and adjust its parameters propperly can be the true challenge.
# 
# There are a lot of regressors you can try and compare. Listing only some of them you have: linear regression, ridge regression, kernel ridge regression, lasso, elastic net, support vector regressor, random forest, xgboost, neural networks, etc
# 
# We have chosen one regressor called Kernelridge. It is like a linear regression but with ridge regularization and a kernel. The roblem is that it has a lot of hyperparameters and we don't know which combination of hyperparameters will give the best results. For that we use GridSearchCV. This method simply tries all the posible hyperparameter combinations for a hardcoded dictionary. It creates and fits one model for each combination, but it does it with cross validation. This means that its results are robust and reliable but also that It's very slow because it has to fit many models. As our regressor is relatively fast, we can try many combinations in a reasonable time.
# 
# After obtaining our best combination, we use it for predicting the labels of our false test set (extracted from the train data) and having an idea of the error that our submision will have.
# 
# At the end, we fit our true training data to predict our true test data. Then we assemble our predictions in a dataframe and save them as our submision. 
# Note that we are using the function exp over our predictions. We do this because we have previously transformed the SalePrice with the logarithm. We need to make the inverse transformation because we want a prediction of SalePrice, not of its logarithm.

# In[ ]:


def rmse_cv(model,X, y):# function that calculates the root mean squared error of the test set through cross validation.
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring = scorer, cv = 10))
    return(rmse)

scorer = make_scorer(mean_squared_error, greater_is_better = False)
param_keridge={'alpha':[0.05, 0.1, 0.3, 0.6, 1, 1.5, 3, 5, 10, 15, 30, 50, 80],
               'kernel':['linear','poly','rbf'],
               'degree':[2,3],
               'coef0':[0.5,1,1.5,2,2.5,3,3.5]} #parameters for our GridSearchCV to explore
regressor=GridSearchCV(KernelRidge(), param_keridge).fit(X_train,y_train).best_estimator_ #obtain the regressor with the best hyperparameters comnination
print('Best estimator found: ',regressor)
print('Root mean square error on the test partition :',rmse_cv(regressor,X_train,y_train).mean()) #show the estimate of the test error

regressor.fit(X_traindata,y_traindata) #train our regressor with all the train data

result=pd.DataFrame()
result['Id']=test_ids
result['SalePrice']=np.exp(regressor.predict(testdata)) #make the prediction and transform it back by aplying exp
print('The description of the submision:\n',result.describe())
result.to_csv('submision.csv',index=False)


# We have to say goodbye for now. If you have any question or suggestion, leave it as a comment. 
# 
#   I hope you enjoyed this kernel and found it useful. If that's the case, please upvote it. Depending of the acceptance of this tutorial, I will make similar ones for other problems.
