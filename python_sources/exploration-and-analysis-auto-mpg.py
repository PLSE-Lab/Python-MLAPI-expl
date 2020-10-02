#!/usr/bin/env python
# coding: utf-8

# # Welcome to exploration and analysis of the auto mpg data set. 
# Welcome to this ipython notebook created for exploration and analysis of the Auto- MPG data-set from UCI Machine Learning Library.
# The data-set is fairly standard on kaggle but can be accessed separately from the UCI Machine Learning Repository along with many other interesting data-sets. Check http://archive.ics.uci.edu/ml/index.php for more.
# 
# This notebook aims primarily to demonstrate use of pandas and seaborn for exploration and visualization of the data-set along with use of  scikit learn library to build regression models to predict the Miles Per Gallon(MPG) using the factors provided in the data-set
# 
# ## So what is the auto-mpg data set?
#  The following description can be found on the UCI Repository page for the data-set (http://archive.ics.uci.edu/ml/datasets/Auto+MPG)
#  -  This dataset is a slightly modified version of the dataset provided in the StatLib library. In line with the use by Ross Quinlan (1993) in predicting the attribute "mpg", 8 of the original instances were removed because they had unknown values for the "mpg" attribute. The original dataset is available in the file "auto-mpg.data-original".
# 
# "The data concerns city-cycle fuel consumption in miles per gallon, to be predicted in terms of 3 multivalued discrete and 5 continuous attributes." (Quinlan, 1993)
# 
# But for now we will treat this as a expedition to discover unknown knowledge in the unchartered lands of this dataset.
# 
# Let's first ready the equipment we will need for this expedition

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory on kaggle.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# We have imported all the packages and libraries we will be using for the initial exploration of data. This notebook will be split into two major sections, majorly:
# 
#  - Exploration and Visualization using pandas and seaborn packages
#  - Building evaluating and tuning different regression models using scikit learn package
# 
# ## Part 1: So let's begin this exploratory journey into the data-set to reveal its hidden secrets!!
# In order to begin this exciting journey into the unchartered lands of the auto-mpg data set we  first need to know the location of this unexplored land.
# For our dataset, this location is  '../input/auto-mpg.csv'
# So lets tell python to take us to this place. Since we are the first explorers here, we will call this place data..because we like data..

# In[2]:


data = pd.read_csv('../input/auto-mpg.csv',index_col='car name')


# Let's have a look at data

# In[3]:


print(data.head())
print(data.index)
print(data.columns)


# So there it is..lots of numbers. We can see that the dataset has the following columns (with their type):
# 
#  - **mpg**: continuous
#  - **cylinders**: multi-valued discrete
#  - **displacement**: continuous
#  - **horsepower**: continuous
#  - **weight**: continuous
#  - **acceleration**: continuous
#  - **model year**: multi-valued discrete
#  - **origin**: multi-valued discrete
#  - **car name**: string (unique for each instance)

# In[4]:


data.shape


# In[5]:


data.isnull().any()


# Nothing seems to be missing

# In[6]:


data.dtypes


# But then, why is horsepower an object and not a float, the values we saw above were clearly numbers Lets try converting the column using astype()

#     Let's look at the unique elements of horsepower to look for discrepancies 

# In[7]:


data.horsepower.unique()


# When we print out all the unique values in horsepower, we find that there is '?' which was used as a placeholder for missing values. Lest remove these entries.

# In[8]:


data = data[data.horsepower != '?']


# In[9]:


print('?' in data.horsepower)


# In[10]:


data.shape


# In[11]:


data.dtypes


# So we see all entries with '?' as place holder for data are removed. However, we the horsepower data is still an object type and not float. That is because pandas coerced the entire column as object when we imported the data set due to '?', so lest change that data 

# In[12]:


data.horsepower = data.horsepower.astype('float')
data.dtypes


# Now everything looks in order so lets continue, let's describe the dataset

# In[13]:



data.describe()


# ## Step 1: Let's look at mpg

# In[14]:


data.mpg.describe()


# So the minimum value is 9 and maximum is 46, but on average it is 23.44 with a variation of 7.8

# In[15]:


sns.distplot(data['mpg'])


# In[16]:


print("Skewness: %f" % data['mpg'].skew())
print("Kurtosis: %f" % data['mpg'].kurt())


# Using our seaborn tool we can look at mpg:
# 
#  - Slight of 0,.45
#  - Kurtosis of -0.51

# ### Lets visualise some relationships between these data points, but before we do, we need to scale them to same the same range of [0,1]
# In order to do so, lets define a function scale

# In[17]:


def scale(a):
    b = (a-a.min())/(a.max()-a.min())
    return b


# In[18]:


data_scale = data.copy()


# In[19]:


data_scale ['displacement'] = scale(data_scale['displacement'])
data_scale['horsepower'] = scale(data_scale['horsepower'])
data_scale ['acceleration'] = scale(data_scale['acceleration'])
data_scale ['weight'] = scale(data_scale['weight'])
data_scale['mpg'] = scale(data_scale['mpg'])


# In[20]:


data_scale.head()


# All our data is now scaled to the same range of [0,1]. This will help us visualize data better. We used a copy of the original data-set for this as we will use the data-set later when we build regression models.

# In[21]:


data['Country_code'] = data.origin.replace([1,2,3],['USA','Europe','Japan'])
data_scale['Country_code'] = data.origin.replace([1,2,3],['USA','Europe','Japan'])


# In[22]:


data_scale.head()


# Lets look at MPG's relation to categories

# In[23]:


var = 'Country_code'
data_plt = pd.concat([data_scale['mpg'], data_scale[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="mpg", data=data_plt)
fig.axis(ymin=0, ymax=1)
plt.axhline(data_scale.mpg.mean(),color='r',linestyle='dashed',linewidth=2)


# The red line marks the  average of the set. From the above plot we can observe:
# 
#  - Majority of the cars from USA (almost 75%) have MPG below global average.
#  - Majority of the cars from Japan and Europe have MPG above global average.

# ### Let's look at the year wise distribution of MPG

# In[24]:


var = 'model year'
data_plt = pd.concat([data_scale['mpg'], data_scale[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="mpg", data=data_plt)
fig.axis(ymin=0, ymax=1)
plt.axhline(data_scale.mpg.mean(),color='r',linestyle='dashed',linewidth=2)


# ### And MPG distribution for cylinders

# In[25]:


var = 'cylinders'
data_plt = pd.concat([data_scale['mpg'], data_scale[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="mpg", data=data_plt)
fig.axis(ymin=0, ymax=1)
plt.axhline(data_scale.mpg.mean(),color='r',linestyle='dashed',linewidth=2)


# ## Now that we have looked at the distribution of 

# In[26]:


corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, square=True);


# In[27]:


factors = ['cylinders','displacement','horsepower','acceleration','weight','mpg']
corrmat = data[factors].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, square=True);


# In[28]:


#scatterplot
sns.set()
sns.pairplot(data, size = 2.0,hue ='Country_code')
plt.show()


# ## So far we have seen the data
# So far, we have seen the data to get a feel for it, we saw the spread of the desired variable MPG along the various discrete variables, namely, Origin, Year of Manufacturing or Model and Cylinders.  
# Now lets extract an additional discrete variable company name and add it to this data. 
# We will use regular expressions and str.extract() function of pandas data-frame to make this new column

# In[29]:


data.index


# ## As we can see the index of the data frame contains model name along with the company name. Now lets use regular expressions to quickly extract the company names. As we can see the index is in format 'COMPANY_NAME - SPACE -MODEL - SPACE -VARIANT' and so regular expressions will make it an easy task. 

# In[30]:


data[data.index.str.contains('subaru')].index.str.replace('(.*)', 'subaru dl')


# In[31]:


data['Company_Name'] = data.index.str.extract('(^.*?)\s')


# ## That does it, almost, we can see NaN so some text was not extracted, this may be due to difference in formatting. We ca also see that some companies are named differently and also some spelling mistakes, lets correct these.

# In[32]:


data['Company_Name'] = data['Company_Name'].replace(['volkswagen','vokswagen','vw'],'VW')
data['Company_Name'] = data['Company_Name'].replace('maxda','mazda')
data['Company_Name'] = data['Company_Name'].replace('toyouta','toyota')
data['Company_Name'] = data['Company_Name'].replace('mercedes','mercedes-benz')
data['Company_Name'] = data['Company_Name'].replace('nissan','datsun')
data['Company_Name'] = data['Company_Name'].replace('capri','ford')
data['Company_Name'] = data['Company_Name'].replace(['chevroelt','chevy'],'chevrolet')
data['Company_Name'].fillna(value = 'subaru',inplace=True)  ## Strin methords will not work on null values so we use fillna()


# In[33]:


var = 'Company_Name'
data_plt = pd.concat([data_scale['mpg'], data[var]], axis=1)
f, ax = plt.subplots(figsize=(20,10))
fig = sns.boxplot(x=var, y="mpg", data=data_plt)
fig.set_xticklabels(ax.get_xticklabels(),rotation=30)
fig.axis(ymin=0, ymax=1)
plt.axhline(data_scale.mpg.mean(),color='r',linestyle='dashed',linewidth=2)


# In[34]:


data.Company_Name.isnull().any()


# ##Lets look at some extremes

# In[35]:


var='mpg'
data[data[var]== data[var].min()]


# In[36]:


data[data[var]== data[var].max()]


# In[37]:


var='displacement'
data[data[var]== data[var].min()]


# In[38]:


data[data[var]== data[var].max()]


# In[39]:


var='horsepower'
data[data[var]== data[var].min()]


# In[40]:


data[data[var]== data[var].max()]


# In[41]:


var='weight'
data[data[var]== data[var].min()]


# In[42]:


data[data[var]== data[var].max()]


# In[43]:


var='acceleration'
data[data[var]== data[var].min()]


# In[44]:


data[data[var]== data[var].max()]


# Now that we have looked at the distribution of the data along discrete variables and we saw some scatter-plots using the seaborn pairplot. Now let's try to find some logical causation for variations in mpg. We will use the lmplot() function of seaborn with scatter set as true. This will help us in understanding the trends in these relations. We can later verify what we see with ate correlation heat map to find if the conclusions drawn are correct. We prefer lmplot() over regplot() for its ability to plot categorical data better. We will split the regressions for different origin countries.

# In[45]:


var = 'horsepower'
plot = sns.lmplot(var,'mpg',data=data,hue='Country_code')
plot.set(ylim = (0,50))


# In[46]:


var = 'displacement'
plot = sns.lmplot(var,'mpg',data=data,hue='Country_code')
plot.set(ylim = (0,50))


# In[47]:


var = 'weight'
plot = sns.lmplot(var,'mpg',data=data,hue='Country_code')
plot.set(ylim = (0,50))


# In[48]:


var = 'acceleration'
plot = sns.lmplot(var,'mpg',data=data,hue='Country_code')
plot.set(ylim = (0,50))


# In[49]:


data['Power_to_weight'] = ((data.horsepower*0.7457)/data.weight)


# In[50]:


data.sort_values(by='Power_to_weight',ascending=False ).head()


# ## Our journey so far:
# So far, we have a looked at our data using various pandas methods and visualized it using seaborn package. We looked at
# ### MPGs relation with discrete variables
#  - MPG distribution over given years if manufacturing
# - MPG distribution by country of origin
# - MPG distribution by number of cylinders
# 
# ### MPGs relation to other continuous variables:
# 
#  - Pair wise scatter plot of all variables in data.
# ### Correlation
#  - We looked at the correlation heat map of all columns in our data
# 
# ## Lets look at some regression models:
# Now that we know what our data looks like, lets use some machine learning models to predict the value of MPG given the values of the factors. 
#  We will use pythons scikit learn to train test and tune various regression models on our data and compare the results. We shall use the following regression models:-
# 
#  - Linear Regression
#  
#  - GBM Regression
# 
# 

# In[51]:


data.head()


# In[52]:


from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


# In[53]:


factors = ['cylinders','displacement','horsepower','acceleration','weight','origin','model year']
X = pd.DataFrame(data[factors].copy())
y = data['mpg'].copy()


# In[54]:


X = StandardScaler().fit_transform(X)


# In[55]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.33,random_state=324)
X_train.shape[0] == y_train.shape[0]


# In[56]:


regressor = LinearRegression()


# In[ ]:


regressor.get_params()


# In[58]:


regressor.fit(X_train,y_train)


# In[59]:


y_predicted = regressor.predict(X_test)


# In[60]:


rmse = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted))
rmse


# In[61]:


gb_regressor = GradientBoostingRegressor(n_estimators=4000)
gb_regressor.fit(X_train,y_train)


# In[ ]:


gb_regressor.get_params()


# In[63]:


y_predicted_gbr = gb_regressor.predict(X_test)


# In[64]:


rmse_bgr = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted_gbr))
rmse_bgr


# In[65]:


fi= pd.Series(gb_regressor.feature_importances_,index=factors)
fi.plot.barh()


# Good, so our initial models work well, but these metrics were performed on test set and cannot be used for tuning the model, as that will cause bleeding of test data into training data, hence, we will use K-Fold to create Cross Validation sets and use grid search to tune the model. 

# In[66]:


from sklearn.decomposition import PCA


# In[67]:


pca = PCA(n_components=2)


# In[68]:


pca.fit(data[factors])


# In[69]:


pca.explained_variance_ratio_


# In[70]:


pca1 = pca.components_[0]
pca2 = pca.components_[1]


# In[71]:


transformed_data = pca.transform(data[factors])


# In[72]:


pc1 = transformed_data[:,0]
pc2 = transformed_data[:,1]


# In[73]:


plt.scatter(pc1,pc2)


# In[74]:


c = pca.inverse_transform(transformed_data[(transformed_data[:,0]>0 )& (transformed_data[:,1]>250)])


# In[75]:


factors


# In[76]:


c


# In[77]:


data[(data['model year'] == 70 )&( data.displacement>400)]


# ***The exceptionally far away point seems to be the Buick estate wagon. This seems logical as the weight data given in the data set seems to be incorrect. The weight for the vehicle is given to be 3086 lbs, however, on research it can be found that the car weight is 4727-4775 lbs. These values are based on internet search***

# - **Now we use K-fold to create a new K-fold object called 'cv_sets' that contains index values for training and cross validation and use these sets in GridSearchCV to tune our model so that it does not over fit or under fit the data**
# -  **We will also define a dictionary called 'params' with the hyper-parameters we want to tune** 
# - **Lastly we define 'grid' which is a GridSearchCV object which we will provide the parameters to tune and the K folds of data created by using the Kfold in sklearn.model_selection**  

# In[108]:


cv_sets = KFold(n_splits=10, shuffle= True,random_state=100)
params = {'n_estimators' : list(range(40,61)),
         'max_depth' : list(range(1,10)),
         'learning_rate' : [0.1,0.2,0.3] }
grid = GridSearchCV(gb_regressor, params,cv=cv_sets,n_jobs=4)


# In[ ]:


grid = grid.fit(X_train, y_train)


# In[ ]:


grid.best_estimator_


# In[ ]:


gb_regressor_t = grid.best_estimator_


# In[ ]:


gb_regressor_t.fit(X_train,y_train)


# In[ ]:


y_predicted_gbr_t = gb_regressor_t.predict(X_test)


# In[ ]:


rmse = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted_gbr_t))
rmse


# In[ ]:


data.duplicated().any()


# In[ ]:




