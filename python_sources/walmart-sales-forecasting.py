#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Data
import pandas as pd
import numpy as np

#Sklearn Libraries
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, train_test_split


#LIghtGBM
import lightgbm as lgb

#Category Encoders
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.woe import WOEEncoder

#SHAP
import shap

#Plots
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')


# ## <a class="anchor">Table of Content</a>
# 
# [1. Import data](#import)
#    
# [2. Explanatory Data Analysis](#eda)
#    
# [3. Searching for Outliers](#outliers)
# 
# [4. Feature Engineering](#fe)
# 
# [5. Machine Learning Modeling](#ML)
#   - [5.1. Defining Metric](#me_metric)
#   - [5.2. Checking Correlations](#me_corr)
#   - [5.3. Choosing Best Encoder](#me_encod)
#   - [5.4. Choosing Best Model](#me_models)
#   - [5.5. Shapley Values](#me_shap)
#   
# [6. Submission](#sub)

# ### <a class="anchor" id="import">1. Import Data</a>

# In[ ]:


features_df = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/features.csv.zip", compression='zip')
stores_df = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/stores.csv")
test_df = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/test.csv.zip", compression='zip')
train_df = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/train.csv.zip", compression='zip')


# **Let's creat a dataframe that will be used for the modeling task!**

# In[ ]:


modeling_df = pd.merge(train_df, stores_df, on='Store', how='inner')
modeling_df = pd.merge(modeling_df, 
                       features_df,
                       on=['Store', 'Date','IsHoliday'],
                       how='inner')


# In[ ]:


modeling_df.head()


# ### <a class="anchor" id="eda">2. Explanatory Data Analysis</a>

# **Let's firt take a quick look at how the data is distributed!**

# In[ ]:


modeling_df.shape


# In[ ]:


modeling_df.describe()


# In[ ]:


pd.DataFrame(100*(modeling_df.isnull().sum()/modeling_df.shape[0])).rename(columns={0:'Percentual of Missing Values (%)'})


# **It is interesting to notice that all Markdown features contain more than 64% of missing values.**

# In[ ]:


modeling_df.groupby(['Store','Dept'])['Date'].count().reset_index().set_index(['Store','Dept']).describe()


# **We can see from the above dataframe that not all Departments contain 143 weeks. In fact, there is at least one department with only 1 week available.**

# In[ ]:


plt.figure(figsize=(30,10));

plt.subplot(2,1,1);
weekly_plot = modeling_df.sort_values(by='Date',ascending=True).groupby('Date')['Weekly_Sales'].sum().reset_index();
weekly_plot['Date'] = pd.to_datetime(weekly_plot['Date']);
sns.lineplot(x="Date", y="Weekly_Sales", markers=True, dashes=False, data=weekly_plot, color='darkorange');
plt.title("Total Weekly Sales", fontsize=20, fontweight="bold");
plt.xticks(rotation=20, fontweight='bold', fontsize=16);
plt.yticks(fontweight='bold', fontsize=16);
plt.xlabel("Date (Week)", fontsize=18, fontweight='bold');
plt.ylabel("Total Number of Sales", fontsize=18, fontweight='bold');

plt.subplot(2,1,2);
weekly_plot = modeling_df.sort_values(by='Date',ascending=True).groupby('Date')['Weekly_Sales'].mean().reset_index();
weekly_plot['Date'] = pd.to_datetime(weekly_plot['Date']);
sns.lineplot(x="Date", y="Weekly_Sales", markers=True, dashes=False, data=weekly_plot, color='darkorange');
plt.title("Average Weekly Sales", fontsize=20, fontweight="bold");
plt.xticks(rotation=20, fontweight='bold', fontsize=16);
plt.yticks(fontweight='bold', fontsize=16);
plt.xlabel("Date (Week)", fontsize=18, fontweight='bold');
plt.ylabel("Average Number of Sales", fontsize=18, fontweight='bold');

plt.tight_layout()


# In[ ]:


plt.figure(figsize=(30,10));
weekly_store_plot = modeling_df.sort_values(by='Date',ascending=True).groupby(['Date','Store'])['Weekly_Sales'].mean().reset_index();
weekly_store_plot['Date'] = pd.to_datetime(weekly_store_plot['Date']);

sns.lineplot(x="Date", y="Weekly_Sales",hue='Store', markers=True, dashes=False, data=weekly_store_plot);
plt.title("Average Sales By Week for each Store", fontsize=20, fontweight="bold");
plt.xticks(rotation=20, fontweight='bold', fontsize=16);
plt.yticks(fontweight='bold', fontsize=16);
plt.legend(fontsize=14);
plt.xlabel("Date (Week)", fontsize=18, fontweight='bold');
plt.ylabel("Average Number of Sales", fontsize=18, fontweight='bold');


# ### Some conclusions so far:
# 
# - We can clearly notice the seasonality peaks. In the original dataset, the holidays are marked on a single day, i.e., in 2010, for example, Christmas was on dezember 31. However, the figures above show that the effects on sales caused by the hollidays can occur days before and even last for several days. For this reason, I will be adding for each holiday 4 additional days: 2 before and 2 after the holiday's original date. (This is done in the function create_holidays).
# 
# - It is very clear that there is a huge difference in sales for each store. We have stores with an average sales of 10000 and stores with an average sale of around 30000.

# In[ ]:


def create_holidays(x):
    """
    This function defines each holiday based on its corresponding dates.
    Notice that 4 additional days are included for each holiday: 2 days before
    and 2 days after the original date.
    
    Arguments:
        x: an input date
    
    Output:
        the corresponding date
    """
    if x in ['2010-02-10','2010-02-11','2010-02-12','2010-02-13','2010-02-14',
             '2011-02-09','2011-02-10','2011-02-11','2011-02-12','2011-02-13',
             '2012-02-08','2012-02-09','2012-02-10','2012-02-11','2012-02-12',
             '2013-02-06','2013-02-07','2013-02-08','2013-02-09','2013-02-10']:
        
        return 'super_bowl'
    
    elif x in ['2010-09-08','2010-09-09','2010-09-10','2010-09-11','2010-09-12',
               '2011-09-09','2011-09-10','2011-09-11','2011-09-12','2011-09-13',
               '2012-09-05','2012-09-06','2012-09-07','2012-09-08','2012-09-09',
               '2013-09-04','2013-09-05','2013-09-06','2013-09-07','2013-09-08']:
        
        return 'labor_day'
    
    elif x in ['2010-11-24','2010-11-25','2010-11-26','2010-11-27','2010-11-28',
               '2011-11-23','2011-11-24','2011-11-25','2011-11-26','2011-11-27',
               '2012-11-21','2012-11-22','2012-11-23','2012-11-24','2012-11-25',
               '2013-11-27','2013-11-28','2013-11-29','2013-11-30','2013-12-01']:
        
        return 'thanksgiving'
    
    elif x in ['2010-12-29','2010-12-30','2010-12-31','2010-12-31','2010-12-31',
               '2011-12-30','2011-12-30','2011-12-30','2011-12-30','2011-12-30',
               '2012-12-28','2012-12-28','2012-12-28','2012-12-28','2012-12-28',
               '2013-12-27','2013-12-27','2013-12-27','2013-12-27','2013-12-27']:
        
        return 'christmas'


# In[ ]:


def update_isholiday(x):
    """
    This function is used for update the IsHoliday field
    from the original dataframe.
    
    Arguments:
        x: a holiday produced by the function create_holidays
    
    Output:
        1 if a holiday, 0 otherwise.
    """
    if x is not None:
        return 1
    else:
        return 0


# #### Updating IsHoliday and Creating a Holiday feature

# In[ ]:


modeling_df['holiday'] = modeling_df['Date'].apply(create_holidays)
modeling_df['IsHoliday'] = modeling_df['holiday'].apply(update_isholiday)


# In[ ]:


modeling_df[modeling_df['IsHoliday']==1].head(2)


# #### Let's see the impact of each holiday in each store and in overall sales

# In[ ]:


store_avg_sales_holidays = modeling_df[modeling_df['IsHoliday'] == 1][['Date','Store','Dept','Weekly_Sales']]
store_avg_sales_not_holidays = modeling_df[modeling_df['IsHoliday'] == 0][['Date','Store','Dept','Weekly_Sales']]

store_avg_sales_holidays = store_avg_sales_holidays.groupby('Store').mean().reset_index()
store_avg_sales_not_holidays = store_avg_sales_not_holidays.groupby('Store').mean().reset_index()


# In[ ]:


plt.figure(figsize=(20,10));
plt.subplot(1,2,1);
sns.barplot(x="Weekly_Sales",
            y="Store",
            data=store_avg_sales_holidays,
            color='darkmagenta',
            order=store_avg_sales_holidays.sort_values(by='Weekly_Sales',ascending=False)['Store'],
            orient='h');
plt.title("Average Sales Per Store on Holidays", fontsize=16, fontweight="bold");
plt.ylabel("Store", fontsize=18, fontweight='bold');
plt.xlabel("Average Sales", fontsize=18, fontweight='bold');
plt.legend(fontsize=16);

plt.subplot(1,2,2);
sns.barplot(x="Weekly_Sales",
            y="Store",
            data=store_avg_sales_not_holidays,
            color='darkmagenta',
            order=store_avg_sales_not_holidays.sort_values(by='Weekly_Sales',ascending=False)['Store'],
            orient='h');
plt.title("Average Sales Per Store on Regular Days", fontsize=16, fontweight="bold");
plt.ylabel("Store", fontsize=18, fontweight='bold');
plt.xlabel("Average Sales", fontsize=18, fontweight='bold');
plt.legend(fontsize=16);


# In[ ]:


average_sales_holidays = modeling_df.groupby(['Store','holiday'])['Weekly_Sales'].mean().reset_index()

plt.figure(figsize=(10,30))
sns.barplot(x="Weekly_Sales",
            y="Store",
            data=average_sales_holidays,
            color='sandybrown',
            hue='holiday',
            order=average_sales_holidays.groupby('Store').mean().reset_index().sort_values(by='Weekly_Sales', ascending=False)['Store'],
            orient='h');
plt.title("Average Sales Per Store for Each Holiday", fontsize=20, fontweight="bold");
plt.ylabel("Store", fontsize=18, fontweight='bold');
plt.xlabel("Average Sales", fontsize=18, fontweight='bold');
plt.legend(fontsize=16);


# #### We can see that store 20 is the one with the highest average of sales, while store 33 is the one with the lowest one. Besides, clearly each holiday have a different impact in sales: while thanksgiving appears to be the holiday with highest impact, on averagem christmas appears to be the one with the lowest. However, this is not true for every store: in store 7, for example, christmas appears as an important sales driver. For that reason, one can expect that the holidays feature is an important one to predict sales for each store. 

# ### <a class="anchor" id="outliers">3. Searching for Outliers</a>

# #### In this section, I will be searching for outliers in the following features: size, temperature, fuel_price, cpi and unemployment. Notice that the feature Weekly_Sales can contain outliers itself. However, since this is a time-series variable, highly influenced by sasonality, finding outliers it is not a trivial task and could result in model bias. Therefore, I will be only looking at the features variables.

# In[ ]:


outliers_df = modeling_df[modeling_df['IsHoliday']==0]


# In[ ]:


plt.figure(figsize=(10,5));

plt.subplot(1,5,1);
sns.boxplot(x=outliers_df["Size"].values, orient='v');
plt.title("Size");
plt.subplot(1,5,2);
sns.boxplot(x=outliers_df["Temperature"].values, orient='v');
plt.title("Temperature");
plt.subplot(1,5,3);
sns.boxplot(x=outliers_df["Fuel_Price"].values, orient='v');
plt.title("Fuel Price");
plt.subplot(1,5,4);
sns.boxplot(x=outliers_df["CPI"].values, orient='v');
plt.title("CPI");
plt.subplot(1,5,5);
sns.boxplot(x=outliers_df["Unemployment"].values, orient='v');
plt.title("Unemployment");

plt.tight_layout()


# ** From the boxplot above, there is no apparent reason for removing any data.**

# ### <a class="anchor" id="fe">4. Feature Engineering</a>

# In this section, I will be engineering features related to date. Instead of using the date itself and applying some kind of encoder to it, I will be making a cyclical transformation and select the best approach through feature selection.
# 
# A cyclical transformation of a certain date corresponds to its convertion into a sinusoidal (or cosinusoidal) function. The advantage of that is the ability of adding a new information to the date feature: the knowledge that monday is closer to sunday than it is to wednesday, for example;
# 
# If we want to conver the hours of a day into a cyclical variable, for example, we would need to do the following:
# 
# $cyclical\_hour = sin(\frac{2 \cdot \pi \cdot hour}{24})$
# 
# Some interesting articles about this subject can be found here:
# 
# [1] https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning
# 
# [2] https://datascience.stackexchange.com/questions/5990/what-is-a-good-way-to-transform-cyclic-ordinal-attributes

# In[ ]:


def create_cyclical_dates():
    """
    This function creates a dataframe of cyclical dates
    
    Output:
        a dataframe with dates and its cyclical transformations.
    """
    df = pd.DataFrame({"Date": pd.date_range('2000-01-01', '2050-12-31')})

    df['day_of_year'] = df['Date'].dt.dayofyear
    df['day_of_month'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['week_of_year'] = df['Date'].dt.weekofyear
    df['month_of_year'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year

    df = pd.merge(df,
                  df.groupby('year')['day_of_year'].count(). \
                      reset_index().rename(columns={'day_of_year':'num_days_year'}),
                  on='year',
                  how='inner')
    df = pd.merge(df,
                  df.groupby(['year','month_of_year'])['day_of_month'].count(). \
                      reset_index().rename(columns={'day_of_month':'num_days_month'}),
                  on=['year','month_of_year'],
                  how='inner')

    df['sine_day_of_month'] = np.sin((2*np.pi*df['day_of_month'])/df['num_days_month'])
    df['sine_day_of_year'] = np.sin((2*np.pi*df['day_of_year'])/df['num_days_year'])
    df['sine_day_of_week'] = np.sin((2*np.pi*df['day_of_week'])/7)
    df['sine_week_of_year'] = np.sin((2*np.pi*df['week_of_year'])/52)
    df['sine_month_of_year'] = np.sin((2*np.pi*df['month_of_year'])/12)
    
    df['Date'] = df['Date'].apply(lambda x: x.strftime("%Y-%m-%d"))
    
    return df[['Date','day_of_year',
               'day_of_month','day_of_week',
               'week_of_year','month_of_year',
               'year','sine_day_of_month',
               'sine_day_of_year','sine_day_of_week',
               'sine_week_of_year','sine_month_of_year']]


# In[ ]:


cyclical_df = create_cyclical_dates()


# In[ ]:


cyclical_df.head()


# In[ ]:


plt.figure(figsize=(15,5))

plt.subplot(1,5,1);
plt.plot(cyclical_df['sine_day_of_month']);
plt.xlim([0, 100]);
plt.title("Day of Month");

plt.subplot(1,5,2);
plt.plot(cyclical_df['sine_day_of_year']);
plt.xlim([0, 1000]);
plt.title("Day of Year");

plt.subplot(1,5,3);
plt.plot(cyclical_df['sine_day_of_week']);
plt.xlim([0, 100]);
plt.title("Day of Week");

plt.subplot(1,5,4);
plt.plot(cyclical_df['sine_week_of_year']);
plt.xlim([0, 1000]);
plt.title("Week of Year");

plt.subplot(1,5,5);
plt.plot(cyclical_df['sine_month_of_year']);
plt.xlim([0, 1000]);
plt.title("Month of Year");

plt.tight_layout()


# In[ ]:


modeling_df = pd.merge(modeling_df,
                       cyclical_df,
                       on='Date',
                       how='inner')


# In[ ]:


modeling_df.head()


# ### <a class="anchor" id="fe">5. Machine Learning Modeling</a>

# #### <a class="anchor" id="me_metric">5.1. Defining Metric</a>
# 
# The first thing that I will be doing is defining the validation metric as proposed by Kaggle. The metric used for evaluation in the competition is the Weighted Mean Absolute Error (WMAE), as follows:
# 
# $$WMAE = \frac{1}{\sum w_{i}} \sum_{i=1}^{n} w_{i}|y_{i} - \hat{y_{i}}|$$
# 
# where:
# 
# - $n$ is the number of rows
# - $\hat{y_{i}}$ is the predicted sales
# - $y_{i}$ is the actual sales
# - $w_{i}$ are weights. $w$ = 5 if the week is a holiday week, 1 otherwise
# 
# **Notice: I have included more days in each holiday (4 days for each), such that the metric woudn't match the Kaggle's validation. Therefore, I will be considering only the default dates in the validation function.**

# In[ ]:


def holiday_weights(x):
    """
    This function returns the holiday weights according
    to the metric definitions.
    
    Arguments:
        x: a date input
        
    Output:
        Either 5 or 1, depending on the date.
    """
    if x in ['2010-02-12','2011-02-11','2012-02-10','2013-02-08',
             '2010-09-10','2011-09-11','2012-09-07','2013-09-06',
             '2010-11-26','2011-11-25','2012-11-23','2013-11-29',
             '2010-12-31','2011-12-30','2012-12-28','2013-12-27']:

        return 5
    else:
        return 1


# In[ ]:


def weighted_mean_absolute_error(y_true, y_pred, W):
    """
    This function defines the Weighted Mean Absolute Error
    
    Arguments:
        y_true: the true output value array
        y_pred: the prediction array
        W: the correspondig weights
    """
    return np.sum(W*np.abs(np.subtract(np.squeeze(y_true.values),
                                       np.squeeze(y_pred))))/np.sum(W)


# #### Let's first verify the features most correlated with the output (Weekly Sales)

# #### <a class="anchor" id="me_corr">5.2. Checking Correlations</a>

# In[ ]:


def plot_diagonal_correlation_matrix(X,y):
    """
    This functions plots a diagonal correlation matrix.
    Ref: https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    
    Arguments:
        df: the input dataframe
    """
    
    #Compute Correlations
    corr = pd.concat([X,y],axis=1).corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20, 20))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    
    plt.show()


# In[ ]:


modeling_df.head()


# In[ ]:


X = modeling_df.drop(columns=['Weekly_Sales',
                              'MarkDown1',
                              'MarkDown2',
                              'MarkDown3',
                              'MarkDown4',
                              'MarkDown5'])
y = modeling_df[['Weekly_Sales']]


# In[ ]:


encoder = JamesSteinEncoder()
X_corr = encoder.fit_transform(X,y)


# In[ ]:


plot_diagonal_correlation_matrix(X_corr,y)


# #### From the correlation plot, we can see that the variables that are most correlated with the sales in each store are the following:
# 
# - Store and Department
# - Date
# - Store type
# - Store Size
# - Holiday

# #### <a class="anchor" id="me_encod">5.3. Choosing Best Encoder</a>

# #### Let's try different types of encoders to see which one performs best. For this task, I will be building a simple Random Forest Regressor, with default tunning. The idea here is just to see the performance of each encoder.

# In[ ]:


def compare_encoders(encoder_list, X_train, X_test, y_test):
    """
    This function compare different type of encoders available
    in the library category_encoder.
    
    Arguments:
        encoder_list: a list of encoders to be comapred
        X_train: training features
        y_train: training target
    
    Output:
        y_test: validation target
    """
    for encoder in encoder_list:
        
        lr_pipe = Pipeline([('encoder',encoder),
                            ('scaler',StandardScaler()),
                            ('clf',RandomForestRegressor(n_jobs=-1))])
        
        #Create Weights for validation metric
        W = X_test['Date'].apply(holiday_weights).values
        
        #Remove Date and year and fit model
        lr_pipe.fit(X_train, np.squeeze(y_train))
        lr_pred = lr_pipe.predict(X_test)
        
        #Print encoder and validation metric
        score = weighted_mean_absolute_error(y_test, lr_pred, W=W)
        
        print("{} wmae-score: {}".format(str(encoder).split('(')[0],score))


# In[ ]:


from sklearn.metrics import make_scorer


# In[ ]:


encoder_list = [OrdinalEncoder(),
                TargetEncoder(),
                MEstimateEncoder(),
                JamesSteinEncoder(),
                LeaveOneOutEncoder(),
                CatBoostEncoder()]


# In[ ]:


X['holiday'] = X['holiday'].fillna('not_holiday')
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33)


# In[ ]:


compare_encoders(encoder_list=encoder_list,
                 X_train=X_train,
                 X_test=X_test,
                 y_test=y_test)


# #### <a class="anchor" id="me_models">5.4. Choosing Best Model</a>

# In[ ]:


def plot_feature_importances(model, X , num = 20):
   """
   This function plots the Feature Importances.
   
   Arguments:
       model: the trained tree-based model
       X: the training features
       num: the number of features to be displayed
   """
   feature_imp = pd.DataFrame({'Value':model.feature_importances_,
                               'Feature':X.columns})
   
   plt.figure(figsize=(15, 10))
   sns.barplot(x="Value",
               y="Feature",
               data=feature_imp.sort_values(by="Value",ascending=False)[0:num],
               color='darkred')
   plt.title('Features Importance', fontsize=18, fontweight='bold')
   plt.yticks(fontsize=14, fontweight='bold')
   plt.tight_layout()
   plt.show()


# #### Random Forest Regressor

# In[ ]:


def random_forest_pipeline(encoder,
                           n_estimators,
                           max_depth,
                           min_samples_split,
                           min_samples_leaf,
                           use_selectkbest = False,
                           k = None):
    """
    Creates a Random Forest regressor pipeline with the following steps:
    
    1) Encoder
    2) Scaler
    3) Random Forest Regressor
    """
    
    if use_selectkbest:
        rf_pipe = Pipeline([('encoder',encoder),
                            ('scaler',StandardScaler()),
                            ('kbest',SelectKBest(f_regression)),
                            ('clf',RandomForestRegressor(n_jobs=-1))])
        param_grid = {
        'clf__n_estimators':n_estimators,
        'clf__max_depth':max_depth,
        'clf__min_samples_split':min_samples_split,
        'clf__min_samples_leaf':min_samples_leaf,
        'kbest__k':k
        }
        
    else:
        rf_pipe = Pipeline([('encoder',encoder),
                            ('scaler',StandardScaler()),
                            ('clf',RandomForestRegressor(n_jobs=-1))])      
        param_grid = {
        'clf__n_estimators':n_estimators,
        'clf__max_depth':max_depth,
        'clf__min_samples_split':min_samples_split,
        'clf__min_samples_leaf':min_samples_leaf
        }
        
    return rf_pipe, param_grid


# In[ ]:


rf_encoder = MEstimateEncoder()
rf_n_estimators = [10]
rf_max_depth = [None, 2, 4, 8, 10]
rf_min_samples_split = [2, 4, 8, 10, 20]
rf_min_samples_leaf = [1, 2, 4, 8, 10]

rf_pipe, rf_grid = random_forest_pipeline(encoder = rf_encoder,
                                          n_estimators = rf_n_estimators,
                                          max_depth = rf_max_depth,
                                          min_samples_split = rf_min_samples_split,
                                          min_samples_leaf = rf_min_samples_leaf)

rf_gd = RandomizedSearchCV(estimator=rf_pipe,
                           param_distributions=rf_grid,
                           scoring='neg_mean_absolute_error',
                           cv=3,
                           n_iter=10,
                           n_jobs=-1,
                           verbose=10)


# In[ ]:


rf_gd.fit(X_train, y_train);


# In[ ]:


rf_gd.best_estimator_


# In[ ]:


rf_pred = rf_gd.predict(X_test)
score = weighted_mean_absolute_error(y_test,
                                     rf_pred,
                                     W = X_test['Date'].apply(holiday_weights).values)
print("wmae-score: {}".format(score))


# In[ ]:


plot_feature_importances(model=rf_gd.best_estimator_['clf'],
                         X=X_train,
                         num=20);


# **I will be selecting just the top features and reapplying a grid search for parameters**

# In[ ]:


rf_encoder = MEstimateEncoder()
rf_n_estimators = [10]
rf_max_depth = [None, 2, 4, 8, 10]
rf_min_samples_split = [2, 4, 8, 10, 20]
rf_min_samples_leaf = [1, 2, 4, 8, 10]
rf_k = list(range(1,X_train.shape[1]))

rf_pipe_2, rf_grid_2 = random_forest_pipeline(encoder = rf_encoder,
                                          n_estimators = rf_n_estimators,
                                          max_depth = rf_max_depth,
                                          min_samples_split = rf_min_samples_split,
                                          min_samples_leaf = rf_min_samples_leaf,
                                          use_selectkbest=True,
                                          k=rf_k)

rf_gd_2 = RandomizedSearchCV(estimator=rf_pipe_2,
                           param_distributions=rf_grid_2,
                           scoring='neg_mean_absolute_error',
                           cv=3,
                           n_iter=10,
                           n_jobs=-1,
                           verbose=10)


# In[ ]:


rf_gd_2.fit(X_train, np.squeeze(y_train.values));
rf_pred_2 = rf_gd_2.predict(X_test)


# In[ ]:


rf_gd_2.best_estimator_


# In[ ]:


score = weighted_mean_absolute_error(y_test,
                                     rf_pred_2,
                                     W = X_test['Date'].apply(holiday_weights).values)

print("wmae-score: {}".format(score))


# #### Polynomial Regression - Lasso

# In[ ]:


def non_linear_multiple_regressor_pipeline(encoder,
                                           max_poly_degree,
                                           alpha,
                                           use_selectkbest = False,
                                           k = None):                                           
    """
    Creates a non linear multiple regressor pipeline with the following steps:
    
    1) Encoder
    2) Polynomial Transformer
    3) Scaler
    3) Lasso Regressor
    
    Arguments:
        encoder: the encoder object
        max_poly_degree: the maximum degree of the polynomial
        alpha: penalty value for l1 regularization
    
    Output:
        pipe: the training pipeline
        param_grid: the param grid for training the pipeline
    """
    
    if use_selectkbest:
        pipe = Pipeline([("encoder",encoder),
                         ("polynomial",PolynomialFeatures()),
                         ('scaler',StandardScaler()),
                         ('kbest',SelectKBest(f_regression)),
                         ('clf',Lasso())])
        param_grid = {
        'polynomial__degree':list(range(1,max_poly_degree)),
        'clf__alpha':list(alpha),
        'clf__fit_intercept':[True],
        'clf__normalize':[False],
        'kbest__k':list(k)
        }
        
    else:
        pipe = Pipeline([("encoder",encoder),
                         ("polynomial",PolynomialFeatures()),
                         ('scaler',StandardScaler()),
                         ('clf',Lasso())])
        param_grid = {
        'polynomial__degree':list(range(1,max_poly_degree)),
        'clf__alpha':list(alpha),
        'clf__fit_intercept':[True],
        'clf__normalize':[False]
        }

    return pipe, param_grid


# In[ ]:


lr_encoder = MEstimateEncoder()
lr_max_poly_degree = 3
lr_alpha = [0.00000001, 0.001, 0.1, 1, 2, 4, 10]
lr_k = list(range(1,X_train.shape[1]))

lr_pipe, lr_grid = non_linear_multiple_regressor_pipeline(
                                          encoder = lr_encoder,
                                          max_poly_degree = lr_max_poly_degree,
                                          alpha = lr_alpha,
                                          use_selectkbest=True,
                                          k=lr_k)

lr_gd = RandomizedSearchCV(estimator=lr_pipe,
                           param_distributions=lr_grid,
                           scoring='neg_mean_absolute_error',
                           cv=3,
                           n_iter=10,
                           n_jobs=-1,
                           verbose=10)


# In[ ]:


lr_gd.fit(X_train, np.squeeze(y_train.values));
lr_pred = lr_gd.predict(X_test)


# In[ ]:


score = weighted_mean_absolute_error(y_test, lr_pred, W = X_test['Date'].apply(holiday_weights).values)
print("wmae-score: {}".format(score))


# #### Light GBM - An approach considering the Markdown features

# In[ ]:


def lgbm_pipeline(encoder,
                  n_estimators,
                  max_depth,
                  min_child_samples,
                  boosting_type):   

    """
    Creates a Light GBM regressor pipeline with the following steps:
    
    1) Econder
    2) Scaler
    3) LightGBM Regressor
    
    Arguments:
        n_estimators: the number of estimators.
        max_depth: the maximum tree depth.
        min_child_samples: the minimum number of samples accepted in a leaf.
        boosting_type: defines the type of algorithm you want to run, default=gdbt
                       - gbdt: traditional Gradient Boosting Decision Tree
                       - rf: random forest
                       - dart: Dropouts meet Multiple Additive Regression Trees (Recommended)
                       - goss: Gradient-based One-Side Sampling
                       
    Output:
        pipe: the training pipeline
        param_grid: the param grid for training the pipeline
    """
    
    if use_selectkbest:
        pipe = Pipeline([("encoder",encoder),
                         ("scaler",StandardScaler()),
                         ('kbest',SelectKBest(f_regression, k=k)),
                         ("clf", lgb.LGBMRegressor(n_jobs=-1))])
    else:
        pipe = Pipeline([("encoder",JamesSteinEncoder()),
                         ("scaler",StandardScaler()),
                         ("clf", lgb.LGBMRegressor(n_jobs=-1))])
    
    return pipe, param_grid


# #### LightGBM Without Markdown Features

# In[ ]:


lgbm_encoder = MEstimateEncoder()
lgbm_n_estimators = [5000]
lgbm_max_depth = [None, 2, 4, 8, 10]
lgbm_min_child_samples = [2, 4, 8, 10, 20]
lgbm_boosting_type = ['dart','gbdt','rf']

lgbm_pipe, lgbm_grid = lgbm_pipeline(
                            encoder=lgbm_encoder,
                            n_estimators=lgbm_n_estimators,
                            max_depth=lgbm_max_depth,
                            min_child_samples=lgbm_min_child_samples,
                            boosting_type=lgbm_boosting_type
                        )

lgbm_gd = RandomizedSearchCV(estimator=lgbm_pipe,
                             param_distributions=lgbm_grid,
                             scoring='neg_mean_absolute_error',
                             cv=3,
                             n_iter=2,
                             n_jobs=-1,
                             verbose=10)


# In[ ]:


lgbm_gd.fit(X_train, np.squeeze(y_train.values));
lgbm_pred = lgbm_gd.predict(X_test)


# In[ ]:


lgbm_gd.best_estimator_


# In[ ]:


score = weighted_mean_absolute_error(y_test, lgbm_pred, W = X_test['Date'].apply(holiday_weights).values)
print("wmae-score: {}".format(score))


# In[ ]:


plot_feature_importances(model=lgbm_gd.best_estimator_['clf'],
                         X=X_train,
                         num=30);


# #### Including Markdown Features

# In[ ]:


X = modeling_df.drop(columns=['Weekly_Sales'])
y = modeling_df[['Weekly_Sales']]

X['holiday'] = X['holiday'].fillna('not_holiday')
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33)


# In[ ]:


lgbm_encoder = MEstimateEncoder()
lgbm_n_estimators = [5000]
lgbm_max_depth = [None, 2, 4, 8, 10]
lgbm_min_child_samples = [2, 4, 8, 10, 20]
lgbm_boosting_type = ['dart','gbdt','rf']

lgbm_pipe_2, lgbm_grid_2 = lgbm_pipeline(
                            encoder=lgbm_encoder,
                            n_estimators=lgbm_n_estimators,
                            max_depth=lgbm_max_depth,
                            min_child_samples=lgbm_min_child_samples,
                            boosting_type=lgbm_boosting_type
                        )

lgbm_gd_2 = RandomizedSearchCV(estimator=lgbm_pipe,
                             param_distributions=lgbm_grid,
                             scoring='neg_mean_absolute_error',
                             cv=3,
                             n_iter=2,
                             n_jobs=-1,
                             verbose=10)


# In[ ]:


lgbm_gd_2.fit(X_train, np.squeeze(y_train.values));
lgbm_pred_2 = lgbm_gd_2.predict(X_test)


# In[ ]:


lgbm_gd_2.best_estimator_


# In[ ]:


score = weighted_mean_absolute_error(y_test, lgbm_pred_2, W = X_test['Date'].apply(holiday_weights).values)
print("wmae-score: {}".format(score))


# In[ ]:


plot_feature_importances(model=lgbm_gd_2.best_estimator_['clf'],
                         X=X_train,
                         num=30);


# #### Clearly the best model was the LightGBM Regressor not considering the Markdown features!

# #### <a class="anchor" id="me_shap">5.4. Shapley Values</a>

# In[ ]:


encoder = MEstimateEncoder()
plot_df = encoder.fit_transform(X_test,
                                y_test)


# In[ ]:


explainer = shap.TreeExplainer(lgbm_gd.best_estimator_['clf'])
shap_values = explainer.shap_values(plot_df)


# In[ ]:


plot_df.columns = X_test.columns


# In[ ]:


shap.summary_plot(shap_values, plot_df,max_display=20);


# In[ ]:


shap.summary_plot(shap_values, plot_df,max_display=20, plot_type="bar");


# ### <a class="anchor" id="sub">6. Submission</a>

# In[ ]:


validation_data = test_df.copy()


# In[ ]:


def make_data_transformations(df):
    """
    This function makes all the necessary transformations
    for predicting the testing data.
    """
    df = pd.merge(df, stores_df, on='Store', how='inner')
    
    df = pd.merge(df, 
                  features_df,
                  on=['Store', 'Date','IsHoliday'],
                  how='inner')

    df = pd.merge(df,
                  cyclical_df,
                  on='Date',
                  how='inner')

    df['holiday'] = df['Date'].apply(create_holidays)
    df['IsHoliday'] = df['holiday'].apply(update_isholiday)

    df['holiday'] = df['holiday'].fillna('not_holiday')

    df.drop(columns=['MarkDown1',
                     'MarkDown2',
                     'MarkDown3',
                     'MarkDown4',
                     'MarkDown5'],inplace=True)
    
    return df


# In[ ]:


validation_data = make_data_transformations(validation_data)


# In[ ]:


validation_pred = lgbm_gd.best_estimator_.predict(validation_data)


# In[ ]:


def make_submission_df(validation_data, y_pred):
    """
    This functiont takes the testing data and its 
    correspondig predictions and create a submission
    dataframe.
    
    Arguments:
        validation_data: the validation_data
        y_pred: the validation_data predictions
        
    Output:
        submission_df: the submission dataframe
    """
    submission_df = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip", compression='zip')
    
    validation_data['Id'] = validation_data['Store'].astype(str) + "_" +                             validation_data['Dept'].astype(str) + "_" +                             validation_data["Date"]
    
    validation_data['Weekly_Sales'] = y_pred
    
    submission_df = pd.merge(submission_df,
                             validation_data[['Id','Weekly_Sales']],
                             on='Id',
                             how='inner')
    
    submission_df.drop(columns='Weekly_Sales_x', inplace = True)
    submission_df.rename(columns={'Weekly_Sales_y':'Weekly_Sales'}, inplace=True)
    
    return submission_df


# In[ ]:


submission_df = make_submission_df(validation_data=validation_data,
                                   y_pred=validation_pred)


# In[ ]:


submission_df.head()


# In[ ]:


submission_df.to_csv('submission.csv')

