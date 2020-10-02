#!/usr/bin/env python
# coding: utf-8

# # Motivation
# Tree-based models like Random Forest and XGBoost has become very poplular to address tabular(structured) data problems and gained a lot of tractions in Kaggle competitions. It has its very deserving reasons. A lot of the notebooks for this competition is inspired by fast.ai ML course. This notebook will also try to use fast.ai, but another approach: **Deep Learning**. 
# This is a bit against industry consensous that Deep Learning is more for unstructured data like image, audio or NLP, and usually won't be very good at handling tabular data. Yet, the introduction of embedding for the categorical data changed this perspective and we'll try to use fast.ai's tabular model to tackle this competition and see how well a Deep Learning approach can do. 

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


from fastai import *
from fastai.tabular import *


# # Load Data
# After imported the necessary fast.ai modules, mainly 'fastai.tabular'. Let's load the data in. 

# In[ ]:


path = Path('data')
dest = path
dest.mkdir(parents=True, exist_ok=True)


# In[ ]:


# copy the data over to working directory for easier manipulations
get_ipython().system('cp -r ../input/* {path}/')


# In[ ]:


path.ls()


# In[ ]:


ls data/bluebook-for-bulldozers


# In[ ]:


# read in the dataset. Since the Test.csv and Valid.csv doesn't have label, it will be used to create our own validation set. 
train_df = pd.read_csv('/kaggle/working/data/bluebook-for-bulldozers/train/Train.csv', low_memory=False, parse_dates=["saledate"])
valid_df = pd.read_csv('/kaggle/working/data/bluebook-for-bulldozers/Valid.csv', low_memory=False, parse_dates=["saledate"])
test_df = pd.read_csv('/kaggle/working/data/bluebook-for-bulldozers/Test.csv', low_memory=False, parse_dates=["saledate"])


# In[ ]:


len(train_df), len(test_df)


# In[ ]:


train_df.head()


# In[ ]:


len(train_df),len(valid_df), len(test_df)


# # Sort the Training Set
# This is to create a good validation set. It cannot be emphasised enough how important a good validation set is to making a successful model. Since we are predicting sales data in the future, we need to make a validation set that all data is collected in the 'future' of the training set. So we need to sort the training set first, then split the 'future' part as the validation set. 

# In[ ]:


# Sort the dataframe on 'saledate' so we can easily create a validation set that data is in the 'future' of what's in the training set
train_df = train_df.sort_values(by='saledate', ascending=False)
train_df = train_df.reset_index(drop=True)


# # Data Pre-processing
# The competition's evaluation methods uses RMSLE (root mean squared log error). So if we take the log of our prediction, we can just use the good old RMSE as our loss function. It's just easier this way.

# In[ ]:


# The evaluation method for this Kaggle competition is REMLE, so if we take the log on dependant variable, we can just use RSME as evaluation metrics. 
# Simpler handling this way. 
train_df.SalePrice = np.log(train_df.SalePrice)


# For **Feature Engineering**, we'll just do it on the 'saledate'. We'll use the fast.ai's *add_datepart* function to achieve that. 

# In[ ]:


# The only feature engineering we do is add some meta-data from the sale date column, using 'add_datepart' function in fast.ai
add_datepart(train_df, "saledate", drop=False)
add_datepart(test_df, "saledate", drop=False)


# In[ ]:


# check and see whether all date related meta data is added.
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
        
display_all(train_df.tail(10).T)


# It's easy to do data pre-processing in fast.ai, we just specify the pre-processing methods we want to use in a list:

# In[ ]:


# Defining pre-processing we want for our fast.ai DataBunch
procs=[FillMissing, Categorify, Normalize]


# Namely, we'll fix the missing values, categorify all categorical columns, then normalize. Plain and simple. 

# # Building the Model
# 

# In[ ]:


train_df.dtypes
g = train_df.columns.to_series().groupby(train_df.dtypes).groups
g


# Have a look at all the column types and see which are categorical and continuous. We'll use it to build the fast'ai DataBunch for training our learner. 

# In[ ]:


# prepare categorical and continous data columns for building Tabular DataBunch.
cat_vars = ['SalesID', 'YearMade', 'MachineID', 'ModelID', 'datasource', 'auctioneerID', 'UsageBand', 'fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor', 'ProductSize', 
            'fiProductClassDesc', 'state', 'ProductGroup', 'ProductGroupDesc', 'Drive_System', 'Enclosure', 'Forks', 'Pad_Type', 'Ride_Control', 'Stick', 'Transmission', 'Turbocharged', 'Blade_Extension', 
            'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower', 'Hydraulics', 'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 'Tire_Size', 'Coupler', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow', 
            'Track_Type', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb', 'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type', 'Travel_Controls', 'Differential_Type', 'Steering_Controls', 
            'saleYear', 'saleMonth', 'saleWeek', 'saleDay', 'saleDayofweek', 'saleDayofyear', 'saleIs_month_end', 'saleIs_month_start', 'saleIs_quarter_end', 'saleIs_quarter_start', 'saleIs_year_end', 
            'saleIs_year_start'
           ]

cont_vars = ['MachineHoursCurrentMeter', 'saleElapsed']


# In[ ]:


# rearrange training set before feed into the databunch
dep_var = 'SalePrice'
df = train_df[cat_vars + cont_vars + [dep_var,'saledate']].copy()


# In[ ]:


# Look at the time period of test set, make sure it's more recent
test_df['saledate'].min(), test_df['saledate'].max()


# Time to create our validation set. The most important step. Since this dataset is somewhat time series, we need to make sure validation set entries happens AFTER all entries in the training set, otherwise the model will be cheating and won't generalize well. 

# In[ ]:


# Calculate where we should cut the validation set. We pick the most recent 'n' records in training set where n is the number of entries in test set. 
cut = train_df['saledate'][(train_df['saledate'] == train_df['saledate'][len(test_df)])].index.max()
cut


# In[ ]:


valid_idx = range(cut)


# In[ ]:


df[dep_var].head()


# Utilize fast.ai's Datablock API, it's very easy to put all the training/validation dataset together for training. We pass the dataframe, categorical columns list, continous columns list, pre-processing methods list, then splid the training set into training and validation set. Specified the dependent varialble as 'SalePrice', put everything into a 'DataBunch' and get ready for training time.

# In[ ]:


# Use fast.ai datablock api to put our training data into the DataBunch, getting ready for training
data = (TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=procs)
                   .split_by_idx(valid_idx)
                   .label_from_df(cols=dep_var, label_cls=FloatList)
                   .databunch())


# # Model

# Finally, it's time for some training. We will fire up a fast.ai 'tabular.learner' from the DataBunch we just created.

# In[ ]:


# We want to limit the price range for our prediction to be within the history sale price range, so we need to calculate the y_range
# Note that we multiplied the maximum of 'SalePrice' by 1.2 so when we apply sigmoid, the upper limit will also be covered. 
max_y = np.max(train_df['SalePrice'])*1.2
y_range = torch.tensor([0, max_y], device=defaults.device)
y_range


# In[ ]:


# Create our tabular learner. The dense layer is 1000 and 500 two layer NN. We used dropout, hai 
learn = tabular_learner(data, layers=[1000,500], ps=[0.001,0.01], emb_drop=0.04, 
                        y_range=y_range, metrics=rmse)


# The single most important thing about fast.ai tabular_learner is the use of embedding layers for categorical data. This is the 'secret sause' that enable Deep Learning to be competitive on handling tabular data. With one embedding layer for each categorical variable, we introduced good interaction for the categorical variables and leverage Deep Learning's biggest strengh: Automatic Feature Finding. We also used Drop Out for both the dense layers and embedding layers for better regularization. The metrics of the learner is RMSE since we've already take the log of SalePrice. Let's look at the model. 

# In[ ]:


learn.model


# As can be seen from the above, we have embedding layers for categorical columns, then followed by a drop out layer. We have batch norm layer for the continuous columns, then we put all of them into two fully connected layers with 1000 and 500 nodes, with Relu, BatchNorm, and Dropout in between. Quite standard.

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# User fast.ai's *lr_find* function to find the proper learning rate, then do a 'fit one cycle' training. 

# In[ ]:


learn.fit_one_cycle(2, 1e-2, wd=0.2)


# In[ ]:


learn.fit_one_cycle(5, 3e-4, wd=0.2)


# In[ ]:


# learn.fit_one_cycle(5, 3e-4, wd=0.2)


# Best result reaches 0.227 RMSLE, I think it beats the #1 in Kaggle leaderboard. 

# # Conlusion
# I think overall people still prefer XGBoost or Random Forest for tabular Kaggle competitions since it usually will yield the best scores. However, Deep Learning is also a viable approach, though lacking a bit on the explainability side. At least it could be used for ensamble purpose so it's worth exploring. 

# In[ ]:




