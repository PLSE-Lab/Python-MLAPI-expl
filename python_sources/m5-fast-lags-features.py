#!/usr/bin/env python
# coding: utf-8

# # The point of this notebook: 
# Create a starting module of functions that we can easily customize to make many features. 
# Start with this basic lags feature engineering framework, customize it with new functions (we will have to do it in batches though due to memory issues). You could also alter the functions to create features for different aggregation levels (must aggregate sales before processing). 
# 
# # What you can get out of this notebook
# 
# 1. Know how to make lag features from the horizontal "rectangle" data representation, which is how the data starts.
# 2. A flexible, copy-pastable, customizable, pipeline-insertionable, mini module of functions at the end of the notebook. 
# 3. Knoweldge of how to utilize numpy to do quick rolling window aggregations.
# 
# 
# # RAM issues 
# My notebook must have crashed 100 times while I was trying to finish this and make it nice. 
# #### Things to mind when doing these kinds of computations: 
# * Datatypes matter: We use this info by setting features to float16. We should be careful if there will be many caculations that demand finer details that float64 provides. EXAMPLE: feature.astype(np.float16). Objects and float64 seems to eat up memory and cause "allocating too much memory" crashes.
# * Numpy functions on rolling windows: I do a technique to make the rolling window mean and std calculations fast. I think it calculates all the windows at once in parallel or something. But this std will use too much ram very easily. Therefore I had to do batches of size 10. Even with this, the RAM almost maxes out when calculating the standard deviation of the 180 sized window. Keep this in mind when using other numpy functions, custom functions, or window sizes for calculations. 

# In[ ]:


import numpy as np 
import pandas as pd
from time import time 
import gc


# In[ ]:


################## Load data ####################
train_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')


# In[ ]:


################## Create Grid ##################
#
# We want our data in a 
# "grid" form, where we have a row for every 
# product id on every day. This is the proper 
# data representation for an lgbm (at least that 
# I know). 
s = time()
start_time = time()
DROP_COLS = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
grid_df = train_df.drop(DROP_COLS, axis=1).melt(id_vars='id', var_name='d', value_name='sales')
print(f"Total time for melt: {(time() - start_time)/60} min")

# Saving space
start_time = time()
grid_df['d'] = grid_df.d.str[2:].astype(np.int16)
print(f"Total time for day col change: {(time() - start_time)/60} min")


start_time = time()
grid_df['id'] = grid_df.id.astype('category')
print(f"Total time for category: {(time() - start_time)/60} min")

print(f'Total time: {(time() - s)/60}')
grid_df

del s


# In[ ]:


get_ipython().run_cell_magic('time', '', "################# Faster grid ceation #####################\n# BE CAREFUL ABOUT DTYPES. I don't set sales and d columns \n# dtypes but I have to in order to conserve memory and \n# prevent my notebooks from crashing. My final functions \n# do have the adjustments. \ndays = 1913\nd_cols = [f'd_{i}' for i in range(1, days + 1)]\nindex = train_df.id.astype('category')\nsales = train_df[d_cols].values.T.reshape(-1,)\n\n\ng = pd.DataFrame({'id': np.tile(index, days), \n                  'd': np.concatenate([[i] * 30490 for i in range(1, days + 1)]), \n                  'sales': np.float64(sales)})\n\ndisplay((g == grid_df).all().all())\ndel index, sales, days")


# In[ ]:


################### Rectangle ###################
#
# I will take the sales values as they are to 
# form my base "rectangle" of sales. 
# I think I can take this recatangle and 
# quickly reshape it so that it lines up 
# with grid_df. If I am correct we can use this 
# to create any lags we want super fast. 

d_cols = [f'd_{i}' for i in range(1,1914)]
rec = train_df[d_cols].values


################## Test ########################

# I will test my idea by reshaping the basic 
# rectangle so that it matches sales.
test_sales = rec.T.reshape(-1)
print('test_sales matches sales?? ', (test_sales == grid_df['sales']).all())


# In[ ]:


############# Make lag_1 feature ###############
lag_day = 1

# We need to take off the last (lag_day) columns
# from our rectangle. Then we can reshape the 
# sales to long format.
lag = rec[:, :-lag_day].T.reshape(-1,)

# The new column must be prepended with np.nans
# to make up for the data we have cut off 
# our rectangle. Therefore, all the d_1 products 
# in grid_df will have np.nan for lag_1. In 
# fact, as we carry out this process for all 
# lag days, rows with sales on d_x will have 
# np.nan values for all lags lag_y where y >= x.
grid_df[f'lag_{lag_day}'] = np.append(np.zeros(30490 * lag_day) + np.nan, lag).astype(np.float16)


###### Checking work
# Lets check our work. Looking at day 1912
# of train_df.tail() should be the same as 
# grid_df''lag_1'].tail() 
print('Checking our work')
display(train_df[['d_1912']].tail(10))
display(grid_df[['lag_1']].tail(10))
print('They are the same. Fantastic!')

del lag_day, lag


# In[ ]:


################ Make lag function #################
def make_lag_col(rec, lag_day=1):
    """rec is just train_df[d_cols].values"""
    
    # We need to take off the last lag_day columns
    lag = rec[:, :-lag_day].T.reshape(-1,)

    # The new column must be prepended with np.nans
    return np.append(np.zeros(30490 * lag_day) + np.nan, lag).astype(np.float16)


# In[ ]:


get_ipython().run_cell_magic('time', '', "############### Make lags for 14 days ###############\nfor i in range(1, 16): \n    grid_df[f'lag_{i}'] = make_lag_col(rec=rec, lag_day=i)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "############ Pandas shift ##############\n# I realized later that we could also \n# just use pandas shift.  easier to implement. \n# Here we will do it for g, which was the\n# same as grid_df before adding lags. \n# So I think this is the better way of adding \n# lags. Our time was not wasted though, \n# because we learned skills that we will \n# need for making rolling windows. \nfor i in range(1,16):\n    g[f'lag_{i}'] = g['sales'].shift(30490 * i).astype(np.float16)")


# In[ ]:


del g
gc.collect()


# In[ ]:


################# Rolling features #################
####################################################

######## rolling window ##############
#
# Lets again utilize our sales rectangel rec, and 
# do some fast rolling calculations. 
# 
########### rolling window function ############
# Please check
# out this article: 
## https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html
# it shows how to create rolling windows that you can
# use to do really fast numpy calculations with. 
def rolling_window(a, window):
    """Reference: https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html
    A super fast way of getting rolling windows on a numpy array. """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

###### Example to see it works  #######
x = np.array([[1,2,3,4,5] for i in range(3)])
print("Here is our array x")
display(x)
rw = rolling_window(x, 3)

print(f"Here is our rw array, with shape {rw.shape} made from x")
display(rw)

print("Here is our rolling mean with window 3")
display(np.mean(rw, axis=-1))

print("Here is our rolling std with window 3")
np.std(rw, axis=-1)


# In[ ]:


del x, rw


# In[ ]:


############# Rolling features funciton ################

####### Walk through ##########
# Lets make rolling_mean_3
## Variables for function 
window = 3
rw = rolling_window(rec, 3)
print(f'shape of rw is {rw.shape}')
function = np.mean

# We need to take off the last columns so 
# get the rolling feature shifted one day. 
col = function(rw, -1)[:, :-1].T.reshape(-1,)

# The new column must be prepended with np.nans
col = np.append(np.zeros(30490 * window) + np.nan, col).astype(np.float16)

# Make sure the shape matches grid_df
display(grid_df.shape[0])
display(col.shape[0])
display(col)


# In[ ]:


del rw, window, function, col


# In[ ]:


################ make rolling col helper ###############

#### version 1 ######

# def make_rolling_col_v1(rw, window, function): 
#     # We need to take off the last columns to
#     # get the rolling feature shifted one day. 

#     col = function(rw, -1)[:, :-1].T.reshape(-1,)

#     # The new column must be prepended with np.nans 
#     # to account for missing gaps

#     return np.append(np.zeros(30490 * window) + np.nan, col).astype(np.float16)

# This version is commented out because it breaks my 
# notebook session. I get a message saying I have tried 
# to allocate too much memory. I discovered that the 
# problem was with np.std when the window was 30 or 
# above. I believe the problem was np was trying to 
# calculate std for all windows, and that was just 
# too much. But I experimented with np.split(rw), and 
# found that there was no problem calculating std in 
# 10 batches, even for window 180. I have set splits 
# to 10. If you have a function or window that still 
# causes a crash, you can increase splits to 3049, the 
# next factor of 30490. 
# I have noticed a slight slow down 
# when doing this, so I will leave it at 10 for now. 

##### experiment code to show problem #####
## This will break 
# rw = rolling_window(rec, 180)
# np.std(rw, -1) 

## This will not break
# rw = rolling_window(rec, 180)
# x= np.split(rw, 10, axis=0)
# x = [np.std(rw, -1) for rw in x]


#### Final version #####
def make_rolling_col(rw, window, function): 
    # We need to take off the last columns to
    # get the rolling feature shifted one day.
    
    split_rw = np.split(rw, 10, axis=0)
    split_col = [function(rw, -1) for rw in split_rw]
    col = np.concatenate(split_col)
    col = col[:, :-1].T.reshape(-1,)

    # The new column must be prepended with np.nans 
    # to account for missing gaps
    return np.append(np.zeros(30490 * window) + np.nan, col).astype(np.float16)


# In[ ]:


def add_rolling_cols(df: pd.DataFrame, rec: np.array, windows: list, functions: list, function_names: list): 
    """Adds rolling features to df."""
    
    print( 72 * '#', '\nAdding rolling columns\n',  )
    start_time = time()
    f = list(zip(functions, function_names))
    
    for window in windows: 
        rw = rolling_window(rec, window)
        for function in f: 
            s_time = time()
            df[f'shift_1_rolling_{function[1]}_{str(window)}'] = make_rolling_col(rw, window, function[0])
            print(f'{function[1]} with window {window} time: {(time() - s_time):.2f} seconds')
            
    print(f'Total time for rolling cols: {(time() - start_time)/60:.2f}')


# In[ ]:


################ Adding rolling features ###############
add_rolling_cols(grid_df, 
                 rec, 
                 windows=[7, 14, 30, 60, 180], 
                 functions=[np.mean, np.std], 
                 function_names=['mean', 'std'])


# In[ ]:


grid_df.info()


# In[ ]:


get_ipython().run_cell_magic('time', '', "################ Shifted lag rolling features ###################\n#\n# Perhaps I want to also want to know 7 day rolling \n# mean, but from 7 seven days ago. This could go \n# directly into a model, or we could create a weekly\n# momentum feature = shift_1_rolling_mean_7/shift_8_rolling_mean_7. \n# I propose we have already calculated these features, \n# we just need to shift the columns by 30490 * (shift_days - 1).\n# We subtract 1 from shift_days because the column shift_1_rolling_mean_7\n# is already shifted 1 day. \n######## prototype #########\n### Objective ###\n# create col shift_8_rolling_mean_7: shift 7, rolling mean with window 7.\n\n### Features check ###\n# shift_8_rolling_mean_7[-30490:] == grid_df[grid_df.d == 1913 - 7]['rolling_mean_7']\n\n### x ###\nshift_8_rolling_mean_7 = grid_df['shift_1_rolling_mean_7'].shift((8-1) * 30490)\n\n### test ###\n(shift_8_rolling_mean_7[-30490:] == grid_df[grid_df.d == 1913 - 7]['shift_1_rolling_mean_7'].values).all()")


# In[ ]:


############ Shifting function ###############
def add_shift_cols(grid_df, shifts, cols, num_series=30490): 
    for shift in shifts: 
        for col in cols: 
            grid_df[f"{col.replace('shift_1', f'shift_{shift}')}"] = grid_df[col].shift((shift - 1) * num_series)
            


# In[ ]:


############## Adding shifted rolling mean ###############
shifts = [7, 14, 21, 28]
cols = [f'shift_1_rolling_mean_{i}' for i in [7, 14]]
add_shift_cols(grid_df, shifts, cols, num_series=30490)


# In[ ]:


list(grid_df)


# In[ ]:


del grid_df, shift_8_rolling_mean_7, shifts, cols
gc.collect()


# # Module of functions

# In[ ]:


################## Helper functions ########################
############################################################

################## Load data ####################
# train_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

############################################################
######################### Imports ##########################
import numpy as np 
import pandas as pd
from time import time
import gc

############################################################
#################### Making grid_df ########################

def nan_leading_zeros(rec):
    rec = rec.astype(np.float64)
    zero_mask = rec.cumsum(axis=1) == 0
    rec[zero_mask] = np.nan
    return rec

def make_grid_df(train_df, pred_horizon=True): 
    """Returns a grid """
    
    start_time = time()
    print("#" * 72, "\nMaking grid_df")
    # Add 28 days for the predicton horizon 
    
    last_day = int(train_df.columns[-1][2:])
    if pred_horizon: 
        for i in range(last_day + 1, last_day + 29): 
            train_df[f'd_{i}'] = np.nan
            
            
    d_cols = [col for col in train_df.columns if 'd_' in col]
    index = train_df.id
    index = pd.Series(np.tile(index, last_day + 28)).astype('category')
    
    # Turn leading zeros into np.nan
    rec = nan_leading_zeros(train_df[d_cols].values)
    sales = rec.T.reshape(-1,)

    
    grid_df = pd.DataFrame({'id': index, 
                      'd': np.concatenate([[i] * 30490 for i in range(1, last_day + 28 + 1)]).astype(np.int16), 
                      'sales': sales})
    print(f'Time: {(time() - start_time):.2f} seconds')
    return grid_df, rec

############################################################
#####################@ Basic lags ##########################

def add_lags(grid_df, lags = range(1,16)):
    
    start_time = time()
    print( 72 * '#', '\nAdding lag columns')
    for i in lags:
        grid_df[f'lag_{i}'] = grid_df['sales'].shift(30490 * i).astype(np.float16)
    
    print(f'Time: {(time() - start_time):.2f} seconds')
        
        
############################################################       
################# Rolling window columns ###################

def rolling_window(a, window):
    """Reference: https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html
    A super fast way of getting rolling windows on a numpy array. """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def make_rolling_col(rw, window, function): 
    # We need to take off the last columns to
    # get the rolling feature shifted one day.
    
    split_rw = np.split(rw, 10, axis=0)
    split_col = [function(rw, -1) for rw in split_rw]
    col = np.concatenate(split_col)
    col = col[:, :-1].T.reshape(-1,)

    # The new column must be prepended with np.nans 
    # to account for missing gaps
    return np.append(np.zeros(30490 * window) + np.nan, col).astype(np.float16)


def add_rolling_cols(df: pd.DataFrame, rec: np.array, windows: list, functions: list, function_names: list): 
    """Adds rolling features to df."""
    
    print( 72 * '#', '\nAdding rolling columns\n',  )
    start_time = time()
    f = list(zip(functions, function_names))
    
    for window in windows: 
        rw = rolling_window(rec, window)
        for function in f: 
            s_time = time()
            df[f'shift_1_rolling_{function[1]}_{str(window)}'] = make_rolling_col(rw, window, function[0])
            print(f'{function[1]} with window {window} time: {(time() - s_time):.2f} seconds')
            
    print(f'Total time for rolling cols: {(time() - start_time)/60:.2f}')
    
    
    
############################################################       
################# Shifting function ########################
def add_shift_cols(grid_df, shifts, cols, num_series=30490): 
    
    print( 72 * '#', '\nAdding shift columns',  )
    start_time = time()
    for shift in shifts: 
        for col in cols: 
            grid_df[f"{col.replace('shift_1', f'shift_{shift}')}"] = grid_df[col].shift((shift - 1) * num_series)
    print(f'Time: {(time() - start_time):.2f} seconds')


            
            
            
############################################################       
################# Create lags df ###########################
def make_lags_df(train_df): 
    
    start_time = time()
    grid_df, rec = make_grid_df(train_df)
    add_lags(grid_df)
    add_rolling_cols(grid_df, 
                     rec, 
                     windows=[7, 14, 30, 60, 180], 
                     functions=[np.mean, np.std], 
                     function_names=['mean', 'std'])
    
    
    shifts = [8, 15]
    cols = [f'shift_1_rolling_mean_{i}' for i in [7, 14, 30, 60]]
    add_shift_cols(grid_df, shifts, cols, num_series=30490)
    
    print(72 * '#', f'Total time: {(time() - start_time)//60:} : {(time() - start_time)%60:.2f}')
    return grid_df


# In[ ]:


grid_df = make_lags_df(train_df)


# In[ ]:


grid_df.to_pickle('lags.pkl')


# In[ ]:


grid_df.info()

