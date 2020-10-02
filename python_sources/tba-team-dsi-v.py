#!/usr/bin/env python
# coding: utf-8

# # House Price Predictions Project

#     

# # TBA TEAM <br><br><br>
# <font size="4">
#     
# - Bilal Yussef.<br><br><br>
# - Talal Al-Mutairi.<br><br><br>
# - Abdulrahman Al-Salamah.<br><br><br>
# </font>

# ## Kaggle Kernal Link
# 

# https://www.kaggle.com/bilalyussef/kernel8011b2c7e5/edit

# ## Problem Statment

# <div class="alert alert-block alert-success">
#     
#    For most citizens around the globe, buying a house is a lifetime project that affects their lifes from that point onward. Despite the crucial importance of this task, there aren't many predictive models to assess the price of a house given its features.<br>
#     
#    The aim of this project is **to devolope a predictive model to estimate the price of a house in Ames, Iowa in the United States**, Based on data and features giving about the house.<br>
#     
#    Through this project, different models will be implemented for the sake of sale price prediction. We'll start with the **simple Linear regression** then we will use **regulerization techniques** to help reduce the expected overfitting of the simple linear regression. will use **Lasso, Ridge and Elastic Net** teqniques, with and without CV (cross validation) to penelize the overfitted features. We'll be using also other advanced Regression teqniques such as **Random Forest Regressor, Support Vector Machine (SVM) regressor and the K Nearest Neighbor Regressor.** Inaddition, the Grid Search teqnique will be used to optimize the parameters in the models.
#    
#    At the End of this project, the model with the highest score will be chosen. The success of the project will be based on scoring at least 0.9 in the test split of the data.<br>
#    
#    A model with at least 90% accuracy would be beneficial for both people working in the houses sales sector and ordinary people who wants to buy new house.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.style as style
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge,ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score, train_test_split,KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
style.use('fivethirtyeight')
warnings.filterwarnings('ignore')


# In[ ]:


pd.set_option('display.max_rows',300)
pd.set_option('display.max_columns',300)


# In[ ]:


hp = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
hp_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


hp.head()


# In[ ]:


hp_test.head()


# In[ ]:


hp.shape,hp_test.shape


# <div class="alert alert-block alert-info">
#     
#    we have **1460 rows** and **81 columns** in the **training dataset**.<br>
#    we have **1459 rows** and **80 columns** in the **testing dataset**.

# In[ ]:


hp.info()


# In[ ]:


hp_test.info()


# <div class="alert alert-block alert-info">
#     
#    ### Training Dataset
#    we have **3 columns** with **float64** data type.<br>
#    we have **35 columns** with **int64** data type.<br>
#    we have **43 columns** with **object** data type.<br>
#    Total = **81** column

# <div class="alert alert-block alert-info">
#     
#    ### Training Dataset
#    we have **11 columns** with **float64** data type.<br>
#    we have **23 columns** with **int64** data type.<br>
#    we have **46 columns** with **object** data type.<br>
#     Total = **80** column

# ## Percentage of Missing Values.

# In[ ]:


def missing_percentage(df):
    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""
    ## the two following line may seem complicated but its actually very simple. 
    if df.isnull().sum().sum() != 0:
        total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]
        percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]
        return pd.concat([total, percent], axis=1, keys=['Total','Percent'])
    else:
        print (f'Congrats, No null values in your dataframe')


# In[ ]:


missing_percentage(hp)


# In[ ]:


missing_percentage(hp_test)


# ## Visualizing the missing data

# In[ ]:


f,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
f.set_figheight(7)
f.set_figwidth(16)
sns.heatmap(hp.isnull(),ax=ax1,cbar=False, yticklabels=False,cmap='viridis')
sns.heatmap(hp_test.isnull(),ax=ax2,cbar=False, yticklabels=False,cmap='viridis')
ax1.set_title('Train Data')
ax2.set_title('Test Data')


# ## Obsevations

# <div class="alert alert-block alert-danger">
#    
#    - Many columns has NaN values.<br><br>
#    - Refering to the data description, it was found that the NaN values in many columns are not actually missing data. Rather they are NaNs either because the feature is not applicable **(i.e. BsmtQual includeds NaN values because some houses do not have basements. Consequently, it a value for basement quality can not be reported).**<br><br>
#    - The features like (BsmtQual) were searched, investegated and then a list of all such columns were saved in a list named **nnan_col** (Not NaN columns) .<br><br>
#    - also it was found that 3 catogrical variables were assigned a numercal values (ratings variables), and thus it was found as either float64 or int64. These columns were saved in a list named c_columns (catogrical columns).

# In[ ]:


nnan_col = ["BsmtQual",'BsmtCond','BsmtFinType1','BsmtFinType2','BsmtExposure','GarageQual','GarageFinish','GarageType',
            'GarageCond','FireplaceQu','Fence','Alley','MiscFeature','PoolQC']


# In[ ]:


def fill_fun(df,columns_list):
    for col in columns_list:
        df[col].fillna(value='NA',inplace = True)


# In[ ]:


fill_fun(hp,nnan_col)
fill_fun(hp_test,nnan_col)


# In[ ]:


def fill_missing(df):
    for col in df.columns:
        if df[col].dtypes == 'O':
            df[col].fillna(value=df[col].mode(dropna=True)[0],inplace=True)
        else:
            df[col].fillna(value=df[col].median(),inplace=True)


# In[ ]:


fill_missing(hp)
fill_missing(hp_test)


# In[ ]:


hp.GarageYrBlt = hp.GarageYrBlt.fillna(value=0.0)
hp_test.GarageYrBlt = hp_test.GarageYrBlt.fillna(value=0.0)


# <div class="alert alert-block alert-warning">
# 
# #### **Note: Houses that has no garages are assigned a zero value in the GarageYrBlt column

# In[ ]:


missing_percentage(hp)


# In[ ]:


missing_percentage(hp_test)


# ## Type changing

# In[ ]:


# c_columns = ['MSSubClass', 'OverallQual' , 'OverallCond' ]


# In[ ]:


# def change_columns(df):
#     column_wrong_type = c_columns
#     for col in column_wrong_type:
#         df[col]=df[col].astype(str)


# In[ ]:


# change_columns(hp)
# change_columns(hp_test)


# <div class="alert alert-block alert-success">
#     
#    ### Now that we have no missing values, we can start investigating our data.
#    - We'll start with the Target Feature **(SalePrice)**

# In[ ]:


def plotting_3_chart(df, feature):

    ## Creating a customized chart. and giving in figsize and everything. 
    fig = plt.figure(constrained_layout=True, figsize=(15,10))
    ## creating a grid of 3 cols and 3 rows. 
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    #gs = fig3.add_gridspec(3, 3)

    ## Customizing the histogram grid. 
    ax1 = fig.add_subplot(grid[0, :2])
    ## Set the title. 
    ax1.set_title('Histogram')
    ## plot the histogram. 
    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)

    # customizing the QQ_plot. 
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title. 
    ax2.set_title('QQ_plot')
    ## Plotting the QQ_Plot. 
    stats.probplot(df.loc[:,feature], plot = ax2)

    ## Customizing the Box Plot. 
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set title. 
    ax3.set_title('Box Plot')
    ## Plotting the box plot. 
    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );
    


# In[ ]:


plotting_3_chart(hp, 'SalePrice')


# In[ ]:


stats.kurtosis(hp.SalePrice)


# <div class="alert alert-block alert-info">
#     
#   #### We can coclude the following from the above graphs.
#   - We have got many outliers in the target variable.
#   - We can see also that the Sale Price is not normally distibuted (referring to the Q-Q plot above). The Sale Price data is skewed to the right, we can see also the value of the kurtosis that indicates heavy tailed data.

# In[ ]:


## Plot fig sizing. 
style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize = (30,20))
## Plotting heatmap. 

# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(hp.drop(columns=['Id']).corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


sns.heatmap(hp.drop(columns=['Id']).corr(), cmap=sns.diverging_palette(20, 220, n=200), mask = mask, annot=True, center = 0, cbar=False);
## Give title. 
plt.title("Heatmap of all the Features", fontsize = 30);


# <div class="alert alert-block alert-info">
#    
#    - The highest correlation is **0.88** between the **GarageCars** and  **GarageArea**.
#    - There is also high correlation between the **TotRmsAbvGrd** and **GrLivArea**, **0.83**.
#    - For the Target variable (SalePrice), some of the features has high correlation with the target while other has low correlation. below is a list of all the values of the correlations for the target sorted from highest to lowest

# In[ ]:


feat_corr = abs(hp.corr().SalePrice).sort_values(ascending=False)[1:]
feat_corr


# ## Let's now detect and deal with outliers
# - We'll be using the Tukey's method to detect the outliers.<br>
# Link: __[Tukey's Fence](http://http://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_summarizingdata/bs704_summarizingdata7.html)__

# In[ ]:


def outliers_nan(df):
    dff = df.drop(columns=['Id','SalePrice'])
    for col in dff.columns:
        if dff[col].dtypes != 'O':
            IQR = np.percentile(dff[col],75) - np.percentile(dff[col],25)
            upper_limit = np.percentile(dff[[col]],75)+(3*IQR)
            lower_limit = np.percentile(dff[[col]],25)-(3*IQR)
            df[col] = dff[col].apply(lambda x: np.nan if x > upper_limit or x < lower_limit else x) 


# In[ ]:


outliers_nan(hp)


# In[ ]:


hp.isnull().sum().sort_values()


# <div class="alert alert-block alert-warning">
#     
#    ### Outliers, were assigned a NaN value and it will be dealt with later on. 

# ## Combining the two datasets together to generate dummies.

# In[ ]:


hp_combined = pd.concat([hp,hp_test],join='inner')


# In[ ]:


y = hp.SalePrice


# In[ ]:


hp_combined = hp_combined[hp.columns[:-1]]


# In[ ]:


hp_combined.head()


# In[ ]:


hp_combined.info()


# In[ ]:


hp_combined_dum = pd.get_dummies(hp_combined, drop_first=True)
hp_combined_dum.shape


# In[ ]:


X_train = hp_combined_dum.iloc[0:hp.shape[0],:]
X_test =  hp_combined_dum.iloc[hp.shape[0]:,:]
y= hp.SalePrice
X_train['SalePrice']=y


# <div class="alert alert-block alert-warning">
#     
#    ### We'll be removing any column that has no significant correlation with the target variable (SalePrice). 

# In[ ]:


L = []
for col in X_train.columns:
    try:
        if (abs(X_trian.corr().SalePrice[col])>0.5):
            L.append(col)
    except:
        L.append(col)
        
        
L.remove('Id')
c=L


# In[ ]:


X_train = X_train[L].dropna()
L.remove('SalePrice')


# In[ ]:


X_test = X_test[L]
y = X_train[['SalePrice']]
X_train.drop(columns='SalePrice', inplace=True)


# In[ ]:


X_train.shape,X_test.shape,y.shape


# In[ ]:


ss = StandardScaler()
X_train_ss = ss.fit_transform(X_train)
X_test_ss = ss.fit_transform(X_test)


# In[ ]:


lr_model = LinearRegression()
lr_model.fit(X_train_ss,y)


# In[ ]:


cross_val_score(lr_model,X_train_ss,y).mean()


# # Lasso 

# In[ ]:


ls_model = Lasso(alpha=5)
cross_val_score(ls_model.fit(X_train_ss,y),X_train_ss,y).mean()


# In[ ]:


lscv_model = LassoCV()


# In[ ]:


cross_val_score(lscv_model.fit(X_train_ss,y),X_train_ss,y).mean()


# # Ridge

# In[ ]:


rg_model = Ridge(alpha=5)


# In[ ]:


cross_val_score(rg_model.fit(X_train_ss,y),X_train_ss,y).mean()


# In[ ]:


rgcv_model = RidgeCV(alphas=np.arange(0.1,10,0.1))


# In[ ]:


cross_val_score(rgcv_model.fit(X_train_ss,y),X_train_ss,y).mean()


# # ElasticNet

# In[ ]:


encv_model = ElasticNetCV()


# In[ ]:


cross_val_score(encv_model.fit(X_train_ss,y),X_train_ss,y).mean()


# # Randomforest

# In[ ]:


rf_model = RandomForestRegressor(max_depth=15, random_state=101)


# In[ ]:


cross_val_score(rf_model.fit(X_train_ss,y),X_train_ss,y).mean()


# In[ ]:





# In[ ]:


grrf = GridSearchCV(rf_model, param_grid={'n_estimators':np.arange(1,50,1),'max_depth':np.arange(1,50,1)},
                   n_jobs=-1,verbose=1)


# In[ ]:


# grrf.fit(X_train_ss,y)


# In[ ]:



# cross_val_score(grrf,X_train_ss,y,cv=5).mean()


# # KNN

# In[ ]:


knn_model= KNeighborsRegressor(n_neighbors=5)


# In[ ]:


cross_val_score(knn_model.fit(X_train_ss,y),X_train_ss,y).mean()


# # SVM

# In[ ]:


from sklearn.svm import SVR


# In[ ]:


svm_model=SVR()


# In[ ]:


cross_val_score(svm_model.fit(X_train_ss,y),X_train_ss,y).mean()


# In[ ]:


submit = pd.DataFrame(columns=['Id','SalePrice'])


# In[ ]:


submit.Id = hp_test.Id
submit.SalePrice =( rgcv_model.predict(X_test_ss))
submit.head()


# # Conclusion & Recommendations.

# <div class="alert alert-block alert-success">
# 
# - The best model is the LassoCV model, it scored 0.9101783479689268 on a cross validation. This result satisfys the original objective in the problem statments, since it is more than 0.9 accuracy score on cross validation scores mean.
# - LassoCV was able to score the best, because it was able to eleminate un-important features from the models.
# - Removing the outliers has positivly impaced the scores of the models. 
# - Doing the grid search enhanced the results of the Random forest and the SVM Regressors. However, the score for the LassoCV is still better. This is essentially because in LassoCV there is a built in optimizer for the alpha value (kind of grid search).
# - It's recommended to blend the results of more than one model together to improve the accuracy.

# In[ ]:


submit.to_csv('submission_rgcv_Final_t.csv',index=False)


# In[ ]:


pd.read_csv('submission_rgcv_Final_t.csv')


# In[ ]:




