#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from typing import Union
from tqdm.notebook import tqdm


# In[ ]:


# reading data
path_folder = "../input/m5-forecasting-accuracy/"

df_train_full = pd.read_csv(path_folder + "sales_train_evaluation.csv")
df_calendar = pd.read_csv(path_folder + "calendar.csv")
df_prices = pd.read_csv(path_folder + "sell_prices.csv")
df_sample_submission_original = pd.read_csv(path_folder + "sample_submission.csv")


# # Code to verify score and public ranking (used to validate model during testing)
# Source: https://www.kaggle.com/rohanrao/m5-how-to-get-your-public-lb-score-rank

# In[ ]:


## evaluation metric
## from https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834 and edited to get scores at all levels
class WRMSSEEvaluator(object):

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 0  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices

        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        weight_df = self.get_weight_df()

        self.group_ids = (
            'all_id',
            'cat_id',
            'state_id',
            'dept_id',
            'store_id',
            'item_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )

        for i, group_id in enumerate(tqdm(self.group_ids)):
            train_y = train_df.groupby(group_id)[train_target_columns].sum()
            scale = []
            for _, row in train_y.iterrows():
                series = row.values[np.argmax(row.values != 0):]
                scale.append(((series[1:] - series[:-1]) ** 2).mean())
            setattr(self, f'lv{i + 1}_scale', np.array(scale))
            setattr(self, f'lv{i + 1}_train_df', train_y)
            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum())

            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())

    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f'lv{lv}_scale')
        return (score / scale).map(np.sqrt)

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        group_ids = []
        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            group_ids.append(group_id)
            all_scores.append(lv_scores.sum())

        return group_ids, all_scores


# In[ ]:


## public LB rank
def get_lb_rank(score):
    """
    Get rank on public LB as of 2020-05-31 23:59:59
    """
    df_lb = pd.read_csv("../input/m5-accuracy-final-public-lb/m5-forecasting-accuracy-publicleaderboard-rank.csv")

    return (df_lb.Score <= score).sum() + 1

def get_ranking_from_submission(submission_original, evaluator, df_sample_submission_original=df_sample_submission_original):
    submission = submission_original.copy()
    submission = submission[submission.id.str.contains("validation")]
    
    df_sample_submission = df_sample_submission_original.copy()
    df_sample_submission["order"] = range(df_sample_submission.shape[0])

    submission = submission.merge(df_sample_submission[["id", "order"]], on = "id").sort_values("order").drop(["id", "order"], axis = 1).reset_index(drop = True)
    submission.rename(columns = {
        "F1": "d_1914", "F2": "d_1915", "F3": "d_1916", "F4": "d_1917", "F5": "d_1918", "F6": "d_1919", "F7": "d_1920",
        "F8": "d_1921", "F9": "d_1922", "F10": "d_1923", "F11": "d_1924", "F12": "d_1925", "F13": "d_1926", "F14": "d_1927",
        "F15": "d_1928", "F16": "d_1929", "F17": "d_1930", "F18": "d_1931", "F19": "d_1932", "F20": "d_1933", "F21": "d_1934",
        "F22": "d_1935", "F23": "d_1936", "F24": "d_1937", "F25": "d_1938", "F26": "d_1939", "F27": "d_1940", "F28": "d_1941"
    }, inplace = True)

    groups, scores = evaluator.score(submission)

    score_public_lb = np.mean(scores)
    score_public_rank = get_lb_rank(score_public_lb)

    for i in range(len(groups)):
        print("Score for group {}: {}".format(groups[i],round(scores[i], 5)))
        
    print("\nPublic LB Score: {}".format(round(score_public_lb, 5)))
    print("Public LB Rank: {}".format(score_public_rank))
    
    return None


# ## Approach: 'GROUPED' modelling per store and department data day-by-day approach
# - 70 groups of store + department -> 70 models to build (between 149 and 532 items for each)
# - Inputs (for each store-department combination): previous sales, day + price data (only for the day to predict)
# 
# Reasons for such division:
# - kaggle notebook doesn't have enough RAM to handle the whole dataset and a correspondingly complex model at once
# - our hypothesis here is that the sale of an item in a store in one of the departments has an impact on a different item in that same department and store (e.g. a customer picks up an item and it reminds him of buying another one close to that item)

# ## Feature engineering
# we will transpose the day data (calendar and prices) and then stack it to our current input X (or X_reshaped)
# 
# Other possible ideas:
# - scale the input data before modelling (only X should be scaled, and Y needs to stay as is) 
# - start training when the item is first sold (different timeframes for each items / store)
# - average price accross all items for each day to add to the calendar data (to account for nationwide promotions)
# - temperature and weather on each day (e.g. if raining, probably not much visits to the store -> fewer sales)

# In[ ]:


#global variables
T_predict = 28
T_train = 30 #each sample will have 30 days of history (sales and daily data)
timeframe = 1*365 + 1 #we consider data from more than 1 year old is too outdated and useless in our modelling
nodes_initial = 1000
initial_learning_param = 0.0004
epoch_max = 50
drop_out = 0.01
# batch_size = 1 #adding a batch doesn't seem helpful improving the model performance

test = False


# In[ ]:


def build_features_per_day(df_calendar, timeframe, T_predict, df_prices, dept, store, test=test):
    '''
    Create features based on the day, events and each item's price (for the ones in the dept and store)
    '''
    df_calendar_edited = calendar_data(df_calendar, timeframe, T_predict, test)
    df_prices_limited = prices_data(df_prices, df_calendar_edited, dept, store)
    df_daily_data = pd.merge(df_calendar_edited, df_prices_limited, left_on='wm_yr_wk', right_index=True)
    df_daily_data = df_daily_data.drop('wm_yr_wk', axis=1)
    
    return df_daily_data


def calendar_data(df_calendar, timeframe, T_predict, test):
    '''
    Builds the calendar data
    '''
    if test:
        df_calendar_edited = df_calendar.iloc[-timeframe-2*T_predict:-T_predict].copy() ###FOR TESTING COMPARED TO VALIDATION SET
    else:
        df_calendar_edited = df_calendar.iloc[-timeframe-T_predict:].copy() ###FOR FINAL EVALUATION

    #one hot encoding on the weekday and event_type_1 (because few values but these should be interesting for high spending days)
    list_col_one_hot = ['weekday', 'event_type_1']
    list_drop_or_not = [True, False] #removes one of the weekday to avoid multicollinearity later on
    for col_one_hot,drop_or_not in zip(list_col_one_hot,list_drop_or_not):
        df_calendar_edited = pd.concat([df_calendar_edited, pd.get_dummies(df_calendar_edited[col_one_hot], prefix=col_one_hot, drop_first=drop_or_not)], axis=1)

    df_calendar_edited = df_calendar_edited.set_index('d')

    list_col_calendar_keep = ['wm_yr_wk']
    for col_to_add in list_col_one_hot:
        list_col_keep_encoded = [col for col in df_calendar_edited.columns if col_to_add in col and col!=col_to_add]
        list_col_calendar_keep += list_col_keep_encoded

    df_calendar_edited = df_calendar_edited[list_col_calendar_keep]
    
    return df_calendar_edited


def prices_data(df_prices, df_calendar_edited, dept, store):
    '''
    Builds the pricing data
    '''
    #define the prices for each of the weeks
    df_prices_limited = df_prices[df_prices['wm_yr_wk'].isin(df_calendar_edited['wm_yr_wk'].unique())].copy()
    df_prices_limited['dept_id'] = df_prices_limited['item_id'].astype(str).str[:-4]
    df_prices_limited = df_prices_limited[(df_prices_limited['dept_id'] == dept) & (df_prices_limited['store_id'] == store)]

    #get the price for each item on each week in the separated one-hot encoded columns
    df_prices_limited = pd.concat([df_prices_limited, pd.get_dummies(df_prices_limited["item_id"], prefix="")], axis=1)

    list_col_items_encoded = [col for col in df_prices_limited.columns if col.startswith('_')]
    for col_item_encoded in list_col_items_encoded:
        df_prices_limited[col_item_encoded] *= df_prices_limited["sell_price"]

    #remove duplicate weeks
    df_prices_limited = df_prices_limited[list_col_items_encoded + ["wm_yr_wk"]].groupby("wm_yr_wk").sum()
    
    return df_prices_limited


# ## Modeling
# 
# Simple FFN (Feed-Forward Network) 

# In[ ]:


def approach3(df_train_full, df_sample_submission_original=df_sample_submission_original, df_calendar=df_calendar,               df_prices=df_prices, T_predict=T_predict, T_train=T_train, timeframe=timeframe,               nodes_initial=nodes_initial, initial_learning_param=initial_learning_param, epoch_max=epoch_max, drop_out=drop_out,              test=test):
    '''
    For each department+store combination (70 groups), get the data input, train a model and edit the submission file for this group
    Note: we predict T_predict (default=28) days based on samples of T_train (default=365) days from all items
    '''
    
    list_unique_dept_id = list(df_train_full['dept_id'].unique())
    list_unique_store_id = list(df_train_full['store_id'].unique())

    submission_grouped = df_sample_submission_original.set_index('id')
    
    counter = 1
    for dept in list_unique_dept_id:
        for store in list_unique_store_id:
#     for dept, store in zip([list_unique_dept_id[3]], [list_unique_store_id[2]]):
            print("counter:", counter, " - DEPARTMENT, STORE:", dept, store)
            df_daily_data = build_features_per_day(df_calendar, timeframe, T_predict, df_prices, dept, store)
            df_train_limited = df_train_full[(df_train_full['dept_id'] == dept) & (df_train_full['store_id'] == store)]
            
            submission_grouped = approach_per_store(df_train_limited, submission_grouped, df_daily_data,                                                     T_predict, T_train, timeframe, nodes_initial, initial_learning_param,                                                     epoch_max, drop_out, test)
            counter += 1
    
    if test:
        ###FOR TESTING COMPARED TO VALIDATION SET        
        get_evaluation(submission_grouped, df_train_full, df_calendar, df_prices)
    else:
        #add the numbers for the validation part (in case the evaluation is also based on these values)
        submission_grouped.iloc[:30490,:] = df_train_full.iloc[:,-T_predict:].values
        
    return submission_grouped


def approach_per_store(df_train_limited, submission_grouped, df_daily_data, T_predict, T_train, timeframe,                        nodes_initial, initial_learning_param, epoch_max, drop_out, test):
    
    '''
    Get the department+store input data, train the model, then predict for the next 28 days and edit the submission
    '''
    
    if test:
        df_train_grouped = df_train_limited.set_index("id").iloc[:, 5:-28] ###FOR TESTING COMPARED TO VALIDATION SET
    else:
        df_train_grouped = df_train_limited.set_index("id").iloc[:, 5:] ###FOR FINAL EVALUATION
    
    samples = timeframe - T_train
    
    X_reshaped, Y, in_dim, out_dim = make_input_data(df_train_grouped, df_daily_data, samples, timeframe)
    history, model = build_train_model(X_reshaped, Y, in_dim, out_dim, nodes_initial, initial_learning_param,epoch_max,drop_out)
    
    #fill in the evaluation part in the submission file
    submission_grouped = build_validation_predictions_per_dept_store(model, df_train_grouped, df_daily_data,                                                                      timeframe, T_train, T_predict, submission_grouped)
    
    return submission_grouped


def make_input_data(df_train_grouped, df_daily_data, samples, timeframe):
    '''
    Get model input data for the store+department group
    Sales and days data are combined together the model input
    '''
    
    X_reshaped = []
    X_part1_sales = []
    X_part2_days = []
    Y = []

    for col_index in range(samples):
        
        #first type of data: item sales
        sales_array = df_train_grouped.iloc[:, -(timeframe - col_index):-(samples - col_index)].to_numpy()
        output_sales_array = df_train_grouped.iloc[:, -(samples - col_index)].to_numpy()

        #second type of data: daily data
        day_data_array = df_daily_data.iloc[col_index + 1 : col_index + 1 + T_train + 1].transpose().to_numpy()
        
        #concatenate both inputs and reshape so that the ML model can handle it
        concatenation_input_array = np.concatenate([sales_array.reshape(-1),day_data_array.reshape(-1)])

        X_reshaped.append(concatenation_input_array)
        Y.append(output_sales_array)

    X_reshaped = np.stack(X_reshaped, axis=0)
    Y = np.stack(Y, axis=0)
    in_dim = X_reshaped.shape[1]
    out_dim = Y.shape[1]
    
    return X_reshaped, Y, in_dim, out_dim


def build_train_model(X_reshaped, Y, in_dim, out_dim, nodes_initial, initial_learning_param, epoch_max, drop_out):
    '''
    Build a model based on the department+store group of data
    Simple FFN, with a minor dropout to prevent overfitting (the model doesn't overfit much as its capacity is relatively low)
    '''
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(nodes_initial, activation='relu', input_dim=in_dim),
        tf.keras.layers.Dropout(drop_out),
        tf.keras.layers.Dense(nodes_initial, activation='relu'),
        tf.keras.layers.Dense(out_dim)
    ])

    optimizer_model = tf.keras.optimizers.Adam(initial_learning_param)
    model.compile(loss = 'mse', optimizer = optimizer_model)

    #train model
    history = model.fit(X_reshaped, Y,epochs=epoch_max)
    
    return history, model


def get_evaluation(submission_grouped, df_train_full, df_calendar, df_prices):
    '''
    When testing the model, edit the submission file so that the validation rows are the same as the evaluation rows
    We then use the existing public leaderboard (as it was before the validation data was released) to get the predicted ranking
    '''
    
    #NEEDS TO EDIT submission_grouped (now we edit the "evaluation" index and the "validation" one needs to be edited to get the evaluation)
    submission_grouped_full = submission_grouped.copy()
    submission_grouped_full.iloc[:30490,:] = submission_grouped_full.iloc[30490:,:].values
    submission_grouped_full.reset_index(inplace=True)
    
    #get current ranking (to change when building the final submission)
    evaluator = WRMSSEEvaluator(df_train_full.iloc[:, :-28], df_train_full.iloc[:, -28:], df_calendar, df_prices)
    get_ranking_from_submission(submission_grouped_full, evaluator)
    
    return None


def build_validation_predictions_per_dept_store(model, df_train_grouped, df_daily_data, timeframe, T_train, T_predict, submission_grouped):
    '''
    Edit the submission file based on the predictions for the department and store group
    '''
    
    validation_predictions = predictions(model, df_train_grouped, df_daily_data, timeframe, T_train, T_predict)
    
    #update submission (i.e. output file)
    submission_grouped.loc[df_train_grouped.index,:] = validation_predictions.transpose()
    
    return submission_grouped


def predictions(model, df_train_grouped, df_daily_data, timeframe, T_train, T_predict):
    '''
    Build predictions arrays based on the model (day by day)
    Note: we need to roll the T_predict (default=30 days) predicted arrays for both the sales data and the daily data
    '''
    
    validation_predictions = []

    #initialize sales data
    rolling_sales_data = df_train_grouped.iloc[:, -T_train:].to_numpy().astype(float)

    for prediction_day in range(1,T_predict+1):
        # prediction_day = 1
        #prediction_day = 1 is for d_1914, all the way to prediction_day = T_predict (default = 28) for d_1941


        #daily data: simple index change
        rolling_day_data = df_daily_data.iloc[timeframe - T_train + 1 + prediction_day - 2:                                               timeframe + 1 + prediction_day -1].transpose().to_numpy()

        X_input_for_prediction = np.concatenate([rolling_sales_data.reshape(-1),rolling_day_data.reshape(-1)]).reshape(1,-1)

        p = model.predict(X_input_for_prediction)[0]
        p[p<0] = 0

        # update the predictions list
        validation_predictions.append(p)

        #sales data: need to roll using the predicted next day's sales data
        rolling_sales_data = np.roll(rolling_sales_data.transpose(), -1, axis=0).transpose()
        rolling_sales_data.transpose()[-1] = p

    validation_predictions = np.stack(validation_predictions, axis=0)

    return validation_predictions


# In[ ]:


submission_final = approach3(df_train_full)


# In[ ]:


# submission_final.to_csv("submission_M5_20200620_v2.csv")


# # Other approaches tried:
# 
# - item by item: too time-consuming without even using daily data (about 9h to compute with a simple FFN)
# - all items together (not grouping by store and department) with a simple FFN: too memory-heavy and lower performance
# - custom LSTM model (please see below): too time-consuming and lower performance
#     - Past sales -> go through an initial cell: simple RNN or LSTM
#     - The output of 1. and the array of next day data together with the array of price data -> go through a FFN
#     - The output of 2. -> goes through a final Dense layer to predict the next day sales for all items in the group
# 

# In[ ]:


def make_input_data(df_train_grouped, df_daily_data, samples, timeframe):
    '''
    We have 2 inputs to our model, one X_main going through a LSTM cell, the other X_day going through a standard Dense layer
    The model output to predict is Y
    '''
    
    #get model input data
    X_main = []
    X_day = []
    Y = []

    for col_index in range(samples):
        # col_index = 335 ###THIS IS JUST FOR TESTING (within the make_input_data function)

        #first type of data: item sales
        sales_array = df_train_grouped.iloc[:, -(timeframe - col_index):-(samples - col_index)].to_numpy()
        output_sales_array = df_train_grouped.iloc[:, -(samples - col_index)].to_numpy()

        #second type of data: daily data
        day_data_array = df_daily_data.iloc[col_index,:].to_numpy()
        
        #concatenate both inputs and reshape so that the ML model can handle it
#         concatenation_input_array = np.concatenate([sales_array.reshape(-1),day_data_array.reshape(-1)])
        
        #reshape input sales array
#         reshape_input_array = sales_array.reshape(-1)

        X_main.append(sales_array)
        X_day.append(day_data_array)
        Y.append(output_sales_array)

    X_main = np.stack(X_main, axis=0)
    X_day = np.stack(X_day, axis=0)
    Y = np.stack(Y, axis=0)
    in_dim_main = (X_main.shape[1], X_main.shape[2])
    in_dim_day = X_day.shape[1]
    out_dim = Y.shape[1]
    
    return X_main, X_day, Y, in_dim_main, in_dim_day, out_dim


def custom_recurrent_model(in_dim_main, in_dim_day, out_dim, nodes_initial):
    '''
    Custom LSTM model
    X_main goes through an LSTM cell
    X_day goes through a Dense layer
    Then join together through another Dense layer
    Then output array
    '''
    
    X_main = Input(in_dim_main)
    X_day = Input((in_dim_day,))
    
    #use recursive model on X_main
    X_main_edited = LSTM(nodes_initial, dropout=drop_out, recurrent_dropout=drop_out)(X_main)
    
    #use a simple Dense layer on the daily data
    X_day_edited = Dense(nodes_initial, activation = 'relu')(X_day)
    
    #combine the two X
    X = Add()([X_main_edited, X_day_edited])
    X = Activation('relu')(X)
    
    #add another FFN
    X = Dense(nodes_initial, activation = 'relu')(X)
    
    #add one final layer for the output (regression)
    X = Dense(out_dim)(X)
    
    model = Model(inputs=[X_main, X_day], outputs=X, name="Custom Reccurent Model")
    
    return model


# # Final Ranking:
# - Private LB Score: 0.72949
# - Private LB Rank: 729 / 5558 (top 14%)
# 
# 
# # Evolution of public rankings (for reference)
# 
# 
# ## 1. All items together
# 
# drop_out = 0.2
# - Public LB Score: 1.72354
# - Public LB Rank: 4209
# 
# drop_out = 0.1
# - Public LB Score: 1.35338
# - Public LB Rank: 4103
# 
# ## 2. Grouped items by store and department
# 
# drop_out = 0.01, learning_rate = 0.0004, per group, 200 nodes
# - Public LB Score: 0.9927
# - Public LB Rank: 3673
# 
# adding daily data and increasing capacity (1000 nodes, 50 epochs, no batch size)
# - Public LB Score: 0.85322
# - Public LB Rank: 3546
