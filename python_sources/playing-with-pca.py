#!/usr/bin/env python
# coding: utf-8

# # Playing with PCA

# In this notebook I want to explore a bit how PCA can help with predictions in a regression setting across several models. We'll be using 10 fold cross validation to compare between Linear Regression, Lasso, Ridge and Random Forest.  
#   
# ** Because our main concern is to explore PCA and what it can do, we will not be comparing between individual models but, between the mean error and standard error of errors of different  algorithms' predictions.  **  
# That means that instead of selecting a specific model and saying it is the best one for the problem, we'll be using cross validation to aggregate predictions from each of the algorithms: Linear Regression, LassoCV, RidgeCV and RandomForestRegressor and try and see how PCA affects each of these aggregated predictions.  
#   
# We'll compare between all models before PCA, with all possible numbers of PCA components and also repeat the process after adjusting for multicollinearity to better understand the relationship between PCA and mullticollinearity regarding predictions.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/Financial Distress.csv')


# Take a quick look at our data:

# In[ ]:


df.head()


# In[ ]:


df.describe()


# For the purpose of this notebook we'll assume each line represents a different, independent measurement:

# In[ ]:


cdf=df.drop(labels=['Company','Time'], axis=1)
cdf.head()


# Next, we'll shamelessly remove outliers to get a smaller variation in y and nicer predictions:

# In[ ]:


cdf = cdf[cdf['Financial Distress']>-2.5]
cdf = cdf[cdf['Financial Distress']<4]
cdf.describe()


# Scale the data:

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(cdf.drop('Financial Distress', axis=1)))  
y = cdf['Financial Distress']


# We'll be using cross validation scored by root mean squared error to asses the models:

# In[ ]:


from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

MSE = metrics.make_scorer(metrics.mean_squared_error)


# Now, let's start with some predictions.  
# We'll be trying Linear Regression, Lasso, Ridge and Random Forest and than, all of them again after applying PCA with different components.  
# Let's start naively, without considering multicollinearity.

# In[ ]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lrCV=cross_val_score(lr,X,y,scoring=MSE,cv=10)
print('Mean RMSE is:',np.sqrt(lrCV).mean() )
print('Std is:', np.sqrt(lrCV).std() )


# Remebering that our y's std is ~ 1 this score is not very impressive...  
# Let's see how Lasso Regression handles this:

# In[ ]:


from sklearn.linear_model import LassoCV

las = LassoCV(n_alphas=500, tol=0.001)
lasCV=cross_val_score(las,X,y,scoring=MSE,cv=10)
print('Mean RMSE is:',np.sqrt(lasCV).mean() )
print('Std is:', np.sqrt(lasCV).std() )


# That's a huge improvement over regular regression!  
# Let's see what Ridge Regression can do:

# In[ ]:


from sklearn.linear_model import RidgeCV

ri = RidgeCV(alphas=(0.1,1,10,100,1000,10000))
riCV=cross_val_score(ri,X,y,scoring=MSE,cv=10)
print('Mean RMSE is:',np.sqrt(riCV).mean() )
print('Std is:', np.sqrt(riCV).std() )


# Wuah! Even better!  
# How about Random Forest?

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rfCV=cross_val_score(rf,X,y,scoring=MSE,cv=10)
print('Mean RMSE is:',np.sqrt(rfCV).mean() )
print('Std is:', np.sqrt(rfCV).std() )


# That's actually pretty decent!  
# And such a low std is pretty cool.  
# Great.  
# Now that we got a basic feel for our data we can take a look at what PCA can do for us.  
# I want to get a prediction using each of our models for any number of PCA components and put all that in a nice graph.  

# In[ ]:


from sklearn.decomposition import PCA

# This will be our PCA calculation function.
def calc_pca_models(_X,_y,_cv=10):
    
    # This will be our results DataFrame:
    RMSE_PCA = pd.DataFrame(0, dtype=float, index=range(len(_X.columns)), 
                                columns=['lr','lr_std', 'las', 'las_std', 'ri', 'ri_std',
                                         'rf', 'rf_std','PCA'])
    
    # PCA of the data.
    pca = PCA(n_components=len(_X.columns))
    pca.fit(_X)
    X_pca=pd.DataFrame(pca.transform(_X))

    #Start crunching the numbers!

    for i in range (len(_X.columns)):
        
        X_pca_c = X_pca[X_pca.columns[:i+1]] # Choose how many components to consider.
        
        RMSE_PCA['PCA'][i] = i+1     # Write down how many components we're discussing.

        # Regression
        lrCV=cross_val_score(lr,X_pca_c,y,scoring=MSE,cv=_cv)
        RMSE_PCA['lr'][i] = np.sqrt(lrCV).mean() # Write mean RSME.
        RMSE_PCA['lr_std'][i] = np.sqrt(lrCV).std() # Write down RSME std. 
        
        #Lasso
        lasCV=cross_val_score(las,X_pca_c,y,scoring=MSE,cv=_cv)
        RMSE_PCA['las'][i] = np.sqrt(lasCV).mean() # Write mean RSME.
        RMSE_PCA['las_std'][i] = np.sqrt(lasCV).std() # Write down RSME std.
    
        #Ridge
        riCV=cross_val_score(ri,X_pca_c,y,scoring=MSE,cv=_cv)
        RMSE_PCA['ri'][i] = np.sqrt(riCV).mean() # Write mean RSME.
        RMSE_PCA['ri_std'][i] = np.sqrt(riCV).std() # Write down RSME std.
    
        # Random Forest!
        rfCV=cross_val_score(rf,X_pca_c,y,scoring=MSE,cv=_cv)
        RMSE_PCA['rf'][i] = np.sqrt(rfCV).mean() # Write mean RSME.
        RMSE_PCA['rf_std'][i] = np.sqrt(rfCV).std() # Write down RSME std.
        
    return RMSE_PCA


# In[ ]:


df_precol = calc_pca_models(X,y,)  # Get the numbers.

o_las = np.sqrt(lasCV).mean()*np.ones(df_precol.shape[0]) 
o_ri = np.sqrt(riCV).mean()*np.ones(df_precol.shape[0]) 
o_rf = np.sqrt(rfCV).mean()*np.ones(df_precol.shape[0]) 

# Plot the numbers.
fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
x=df_precol['PCA']
plt.ylim(0.5,1.5)
plt.xlim(0,df_precol.shape[0]+1)

#ax.plot(x, df_precol['lr']-df_precol['lr_std'], '--', 'b')
ax.plot(x, df_precol['lr'], 'b', label='Regression',)  
ax.plot(x, df_precol['lr']+df_precol['lr_std'], 'bv')

#ax.plot(x, df_precol['las']-df_precol['las_std'], '--', color='orange')
ax.plot(x, df_precol['las'],'orange', label='Lasso')
ax.plot(x, df_precol['las']+df_precol['las_std'], 'v', color='orange', label = 'Upper Lasso std')
ax.plot(x, o_las, '.', color='orange', label = 'Original Lasso')


ax.plot(x, df_precol['ri']-df_precol['ri_std'], 'g^')
ax.plot(x, df_precol['ri'], 'g', label='Ridge')
#ax.plot(x, df_precol['ri']+df_precol['ri_std'], 'g--')
ax.plot(x, o_ri, '.', color='g', label = 'Original Ridge')

ax.plot(x, df_precol['rf']-df_precol['rf_std'], 'r^', label = 'Lower RF std')
ax.plot(x, df_precol['rf'], 'r', label='RandomForest')
#ax.plot(x, df_precol['rf']+df_precol['rf_std'], 'r--')
ax.plot(x, o_rf, '.', color='r', label = 'Orginal Random Forest')


ax.set_xlabel('Number of PCA Vectors')
ax.set_ylabel('RMSE')
ax.plot()

ax.legend()


# That's a bit of a messy graph but it's good enough to tell us  a couple of things:  
# 1. Random forest doesn't seem to be affected at all by PCA on this data set.
# 2. Our regressions (specifically their std) seem to improve drastically with less than 10 PCAs.
# 
# Let's remove Random Forest and take a closer look!
# 

# In[ ]:


o_las = np.sqrt(lasCV).mean()*np.ones(df_precol.shape[0]) 
o_las_std = np.sqrt(lasCV).std()*np.ones(df_precol.shape[0]) 
o_ri = np.sqrt(riCV).mean()*np.ones(df_precol.shape[0]) 
o_ri_std = np.sqrt(riCV).std()*np.ones(df_precol.shape[0])

# Plot the numbers.
fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
x=df_precol['PCA']
plt.ylim(0.5,1.5)
plt.xlim(0.8,11)

ax.plot(x, df_precol['lr']-df_precol['lr_std'], 'b^')
ax.plot(x, df_precol['lr'], 'b', label='Regression',)  
ax.plot(x, df_precol['lr']+df_precol['lr_std'], 'bv')

ax.plot(x, df_precol['las']-df_precol['las_std'], '^', color='orange')
ax.plot(x, df_precol['las'],'orange', label='Lasso')
ax.plot(x, df_precol['las']+df_precol['las_std'], 'v', color='orange', label = 'Lasso std')

ax.plot(x, o_las+o_las_std, '--', color='orange', label = 'Original Lasso std')
ax.plot(x, o_las, '.', color='orange', label = 'Original Lasso')
ax.plot(x, o_las-o_las_std, '--', color='orange')

ax.plot(x, df_precol['ri']-df_precol['ri_std'], 'g^', label = 'Ridge std')
ax.plot(x, df_precol['ri'], 'g', label='Ridge')
ax.plot(x, df_precol['ri']+df_precol['ri_std'], 'gv')

ax.plot(x, o_ri+o_ri_std, '--', color='g', label = 'Original Ridge std')
ax.plot(x, o_ri, '.', color='g', label = 'Original Ridge')
ax.plot(x, o_ri-o_ri_std, '--', color='g')


ax.set_xlabel('Number of PCA Vectors')
ax.set_ylabel('RMSE')
ax.plot()

ax.legend()


# So, the triangles mark the std of the PCAs and the dashed lines the std's of the original regressions and we can see that with 4-9 PCA components we get a similar prediction to the one we had with all the data but with much lower std.  
# Cool.  
#   
#   Can we improve this results by adjusting for multicollinearity?  
#    Let's first asses how much multicollinearity we have:

# In[ ]:


corrmat = cdf.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# Quite a bit!  
# Let's deal with that with a little script I adapted from [here](https://stats.stackexchange.com/questions/155028/how-to-systematically-remove-collinear-variables-in-python).  
# It removes variables with VIF over a treshold.
# 

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor    

def remove_multicol(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped=False
        vif = [variance_inflation_factor(X[variables].values, ix) for ix in range(X[variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            del variables[maxloc]
            dropped=True

    return X[variables]

X_colli = remove_multicol(X,5)


# In[ ]:


corrmat = X_colli.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# Looks way better!  
# Let's see how it helped with our models:

# In[ ]:


print('___Before adjusting for multicollinearity___')
lrCV=cross_val_score(lr,X,y,scoring=MSE,cv=10)
print('Regression')
print('Mean RMSE is:',np.sqrt(lrCV).mean() )
print('Std is:', np.sqrt(lrCV).std() )

lasCV=cross_val_score(las,X,y,scoring=MSE,cv=10)
print('Lasso')
print('Mean RMSE is:',np.sqrt(lasCV).mean() )
print('Std is:', np.sqrt(lasCV).std() )

riCV=cross_val_score(ri,X,y,scoring=MSE,cv=10)
print('Ridge')
print('Mean RMSE is:',np.sqrt(riCV).mean() )
print('Std is:', np.sqrt(riCV).std() )

rfCV=cross_val_score(rf,X,y,scoring=MSE,cv=10)
print('Random Forest')
print('Mean RMSE is:',np.sqrt(rfCV).mean() )
print('Std is:', np.sqrt(rfCV).std() )

print('___After adjusting for multicollinearity___')
lrCV=cross_val_score(lr,X_colli,y,scoring=MSE,cv=10)
print('Regression')
print('Mean RMSE is:',np.sqrt(lrCV).mean() )
print('Std is:', np.sqrt(lrCV).std() )

lasCV=cross_val_score(las,X_colli,y,scoring=MSE,cv=10)
print('Lasso')
print('Mean RMSE is:',np.sqrt(lasCV).mean() )
print('Std is:', np.sqrt(lasCV).std() )

riCV=cross_val_score(ri,X_colli,y,scoring=MSE,cv=10)
print('Ridge')
print('Mean RMSE is:',np.sqrt(riCV).mean() )
print('Std is:', np.sqrt(riCV).std() )

rfCV=cross_val_score(rf,X_colli,y,scoring=MSE,cv=10)
print('Random Forest')
print('Mean RMSE is:',np.sqrt(rfCV).mean() )
print('Std is:', np.sqrt(rfCV).std() )


# Makes quite a difference!  (Although Random Forest still doesn't seem to be impressed..)  
# Let's see how it would look like with all the PCA stuff:  
# (I'll remove Random Forest because it doesn't seem to be affected and because it takes quite a while to run..)

# In[ ]:


df_col = calc_pca_models(X_colli,y,)  # Get the numbers.

o_las = np.sqrt(lasCV).mean()*np.ones(df_col.shape[0]) 
o_las_std = np.sqrt(lasCV).std()*np.ones(df_col.shape[0]) 
o_ri = np.sqrt(riCV).mean()*np.ones(df_col.shape[0]) 
o_ri_std = np.sqrt(riCV).std()*np.ones(df_col.shape[0])


# Plot the numbers.
fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
x=df_col['PCA']
plt.ylim(0.5,1.5)
plt.xlim(0,df_col.shape[0]+1)

ax.plot(x, df_col['lr'], 'b', label='Regression',)  
ax.plot(x, df_col['lr']+df_col['lr_std'], 'bv')

ax.plot(x, df_col['las']-df_col['las_std'], '^', color='orange')
ax.plot(x, df_col['las'],'orange', label='Lasso')

ax.plot(x, df_col['ri'], 'g', label='Ridge')
ax.plot(x, df_col['ri']+df_col['ri_std'], 'gv', label = 'Ridge std')

ax.plot(x, o_ri+o_ri_std, '--', color='g', label = 'Original Ridge std')
ax.plot(x, o_ri, '.', color='g', label = 'Original Ridge')
ax.plot(x, o_ri-o_ri_std, '--', color='g')

ax.plot(x, o_las+o_las_std, '--', color='orange', label = 'Original Lasso std')
ax.plot(x, o_las, '.', color='orange', label = 'Original Lasso')
ax.plot(x, o_las-o_las_std, '--', color='orange')

ax.set_xlabel('Number of PCA Vectors')
ax.set_ylabel('RMSE')
ax.plot()

ax.legend()


# Apart from improving regular regression PCA had no effect here.
# On both Ridge and Lasso best results are acheived only while using all vectors and are than equal to those prior to using PCA.  

# # In Conclusion  
# In this data set, PCA is effective in improving predictions using several components in Linear Regression, Ridge and Lasso.  
# But, after adjusting for multicollinearity, PCA only improved Linear Regression and had inferior results to those achieved by using all of the data using Ridge and Lasso.  
# Random Forest did not seem to be affected by PCA or multicollinearity on this data set and it seems to yield the best results regardless.

# In[ ]:




