#!/usr/bin/env python
# coding: utf-8

# # Attempt to predict game sales in NA

# # Attempt to predict game sales in North America
# ## Overview
# The following code is a predictive analysis of the data in the 'vgsales.csv' file. The file is available on Kaggle.com.
# Using a set of features of interest (namely, 'Rank', 'Genre', 'Platform', 'Year', 'Publisher', 'EU_Sales', and 'JP_Sales', 'Other_Sales') a predictive model has been built to estimate the value of an outcome of interest (namely, 'NA_Sales).
# ## Possible applications of similar models
# Similar models could be used by manufacturers and vendors to estimate the number of games copies to be produced and stocked respectively, and what the profits of sales may be.

# ### The following block of code loads the data

# In[ ]:


import numpy as np 
import pandas as pd
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')

try:
    data = pd.read_csv('../input/vgsales.csv')
    print ('Dataset loaded...')
except:
    print ('Unable to load dataset...') 


# ### Display the data

# In[ ]:


display(data[:10])   


# ### Processing data
# To simplify the analysis process, entries whose 'Year' feature value is missing have been deleted from the DataFrame. 

# In[ ]:


data = data[np.isfinite(data['Year'])]


# ### Setting our Y-value (to-be-predicted) and our X-value (features)
# In the following block of code, I set our 'y-value' column under the variable name 'naSales'. We are interested in predicting this value. Additionally, I set the 'x-value' columns under the variable name 'features'. It is these features that we will use to predict our naSales values. The 'features' variable will store the following columns of the dataframe: 'Rank', 'Genre', 'Platform', 'Year', 'Publisher', 'EU_Sales', and 'JP_Sales', 'Other_Sales'. I am not including the 'Global_Sales' column in 'features' b/c its inclusion would reduce our problem of predicting naSales to a simple subtraction problem. naSales would simply equal 'Global_Sales' - 'EU_Sales' - 'JP_Sales' - 'Other_Sales'

# In[ ]:


naSales = data['NA_Sales']
features = data.drop(['Name', 'Global_Sales', 'NA_Sales'], axis = 1)

# Displaying our features and target columns... 
display(naSales[:5])
display(features[:5])


# ### Principal Component Analysis
# The 'EU_Sales', 'JP_Sales', 'Other_Sales' likely are observables driven by an underlying latent feature. I am herein performing a Principal Component Analysis on these three features to obtain one underlying latent feature.

# In[ ]:


# Firstly, I am dividing the features data set into two as follows. 

salesFeatures = features.drop(['Rank', 'Platform', 'Year', 'Genre', 'Publisher'], 
                              axis = 1)
otherFeatures = features.drop(['EU_Sales', 'JP_Sales', 'Other_Sales', 'Rank'], 
                              axis = 1)

# Secondly, I am obtaining the PCA transformed features...

from sklearn.decomposition import PCA
pca = PCA(n_components = 1)
pca.fit(salesFeatures)
salesFeaturesTransformed = pca.transform(salesFeatures)

# Finally, I am merging the new transfomed salesFeatures 
# (...cont) column back together with the otherFeatures columns...

salesFeaturesTransformed = pd.DataFrame(data = salesFeaturesTransformed, 
                                        index = salesFeatures.index, 
                                        columns = ['Sales'])
rebuiltFeatures = pd.concat([otherFeatures, salesFeaturesTransformed], 
                            axis = 1)

display(rebuiltFeatures[:5])


# ### Processing our data
# Most Machine Learning models expect numeric values. The following block of code converts non-numeric values into numeric values by adding dummy variable columns. For example, the 'Genre' feature with say, 2 values, namely 'a' and 'b' would be divided into 2 features: 'Genre_a' and 'Genre_b', each of which would take binary values.

# In[ ]:


# This code is inspired by udacity project 'student intervention'.
temp = pd.DataFrame(index = rebuiltFeatures.index)

for col, col_data in rebuiltFeatures.iteritems():
    
    if col_data.dtype == object:
        col_data = pd.get_dummies(col_data, prefix = col)
        
    temp = temp.join(col_data)
    
rebuiltFeatures = temp
display(rebuiltFeatures[:5])


# ### Dividing our data into Training and Testing sets. 

# In[ ]:


# Dividing the data into training and testing sets...
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(rebuiltFeatures, 
                                                    naSales, 
                                                    test_size = 0.2, 
                                                    random_state = 2)


# ### Model Selection
# I believe Decision Tree Regression and K-Neighbors Regression will fit the data well. I herein build both these models and analyze the results to ascertain the better of the two. The metric I am using to guage the 'goodness' of the model is the R-squared score. 

# In[ ]:


# Creating & fitting a Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

regDTR = DecisionTreeRegressor(random_state = 4)
regDTR.fit(X_train, y_train)
y_regDTR = regDTR.predict(X_test)

from sklearn.metrics import r2_score
print ('The following is the r2_score on the DTR model...')
print (r2_score(y_test, y_regDTR))

# Creating a K Neighbors Regressor
from sklearn.neighbors import KNeighborsRegressor

regKNR = KNeighborsRegressor()
regKNR.fit(X_train, y_train)
y_regKNR = regKNR.predict(X_test)

print ('The following is the r2_score on the KNR model...')
print (r2_score(y_test, y_regKNR))


# The above results show that the Decision Tree Regression model is the better of the two with a superior R-squared score. 

# ### Optimizing the Decision Tree Regression Model
# The following block of code optimizes the parameters of the DTR model. 

# In[ ]:


# This code is inspired by udacity project 'student intervention'
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
cv_sets = ShuffleSplit(X_train.shape[0], n_iter = 10, 
                       test_size = 0.2, random_state = 2)
regressor = DecisionTreeRegressor(random_state = 4)
params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
          'splitter': ['best', 'random']}
scoring_func = make_scorer(r2_score)
    
grid = GridSearchCV(regressor, params, cv = cv_sets, 
                    scoring = scoring_func)
grid = grid.fit(X_train, y_train)

optimizedReg = grid.best_estimator_
y_optimizedPrediction = optimizedReg.predict(X_test)

print ('The r2_score of the optimal regressor is:')
print (r2_score(y_test, y_optimizedPrediction))


# Strangely enough, the optimization code does not yield better results than the default model. It could be that the model parameters I have selected to optimize are not the right ones.

# ## Conclusion
# 
# In conclusion, we were able to build a model that estimates the target values given theselected features set. Our Decision Tree Regression model performed fairly well with an R squared score of 0.65 ~ 0.7.

# In[ ]:




