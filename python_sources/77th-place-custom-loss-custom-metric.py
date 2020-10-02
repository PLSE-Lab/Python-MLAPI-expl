#!/usr/bin/env python
# coding: utf-8

# ### Updated: Link to [notebook](https://www.kaggle.com/chrisrichardmiles/m5-wrmsse-custom-objective-and-custom-metric?scriptVersionId=38218310) that shows how to to create custom loss and metric functions. 
# 
# Above, I have linked to a notebook that shows how to get everything for the custom loss and custom metric, but I wanted to post my final notebook in its raw form. I am super proud of how far I have come in the past months, and I definitely needed the help of everyones notebooks and ideas. If you posted any notebooks or comments in this competition, I read them and used them. Thank you.
# 
# ### Highlights and defining characteristics
# * **Custom loss function** - discovered at 11:37 pm, with about 17 hours to go. 
# * **Custom metric** - essentially an RMSSE function, but with the scales calculated with out of stock taken into account.
# * **Training and validation setup:** 28 models, trained for each prediction day.  For a training set, I only used days that were the same day of the week as prediciton day. This allowed me to use all the trianing data, and a good amount of features. For early stopping validation, I used a single day 28 days before the prediction day. So for predicting day 1942, validation day was 1914, and last training days were ...1893, 1900, 1907. The inspiration for this setup was from the winners of the grocery favorita competition and [their source code](https://www.kaggle.com/shixw125/1st-place-lgb-model-public-0-506-private-0-511). I studied this code heavily, and also used a lot of the same features. I was a bit worried about not using the last month for trianing, and only using a single validation day, but this grocery solution gave me the confidence that this approach was possible.
# 
# ### Most obvious area for improvement: LGBM parameter tuning
# I literally just used the parameters that were in [Custom validation](https://www.kaggle.com/kyakovlev/m5-custom-validation) by [Konstantin](https://www.kaggle.com/kyakovlev). With 10 hours left in the competition, I was about to start testing new parameters but thought "dude, you better start running this notebook or it might not run in time", so I just left them.  I have no idea really about tuning the parameters, so maybe theres a big boost that could be made here. 
# 
# Actually, I changed learning rate to .03 from .05, but also ran the .05 version in case the .03 version didn't finish in time. I did this because I think it is true that lower learning rate cannot make results less accurate, but just slower training. Also, I ran two copies of this notebook, one for days 1-14 and the other for 15-28. 
# 

# In[ ]:


START_TEST = 1942
MODEL_NAME = f'LGBM_002_{START_TEST}'

from m5_helpers import *
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split

########################### Global variables ###############################
TARGET = 'sales'

START_TRAIN = 140
P_HORIZON = 28
SEED      = 42                     # Seed for deterministic processes
NUM_ITERATIONS = 6000              # Rounds for lgbm
ITEM_ID_SAMPLE_SIZE = 1            # In case we want to do fast training or fit more features for testing 

DATA_RAW_PATH = '../input/m5-forecasting-accuracy/'


# In[ ]:


############################# Original features ###############################
###############################################################################
######################### DO NOT ALTER THIS LIST ##############################
original_features = ['id',
 'd',
 'sales',
                     ###### BASIC FEATURES ######
 'item_id',
 'dept_id',
 'cat_id',
 'store_id',
 'state_id',
 'sell_price',
 'price_min',
 'price_max',
 'price_median',
 'price_mode',
 'price_mean',
 'price_std',
 'price_norm_max',
 'price_norm_mode',
 'price_norm_mean',
 'price_momentum',
 'price_roll_momentum_4',
 'price_roll_momentum_24',
 'price_end_digits',
 'event_name_1',
 'event_type_1',
 'event_name_2',
 'event_type_2',
 'snap_CA',
 'snap_TX',
 'snap_WI',
 'tm_d',
 'tm_w',
 'tm_m',
 'tm_y',
 'tm_wm',
 'tm_dw',
 'tm_w_end',
 'snap_transform_1',
 'snap_transform_2',
 'next_event_type_1',
 'last_event_type_1',
 'days_since_event',
 'days_until_event',
                          ###### ENCODING FEATURES ######
 'enc_state_id_mean',
 'enc_state_id_std',
 'enc_store_id_mean',
 'enc_store_id_std',
 'enc_cat_id_mean',
 'enc_cat_id_std',
 'enc_dept_id_mean',
 'enc_dept_id_std',
 'enc_state_id_cat_id_mean',
 'enc_state_id_cat_id_std',
 'enc_state_id_dept_id_mean',
 'enc_state_id_dept_id_std',
 'enc_store_id_cat_id_mean',
 'enc_store_id_cat_id_std',
 'enc_store_id_dept_id_mean',
 'enc_store_id_dept_id_std',
 'enc_item_id_mean',
 'enc_item_id_std',
 'enc_item_id_state_id_mean',
 'enc_item_id_state_id_std',
 'enc_item_id_store_id_mean',
 'enc_item_id_store_id_std',
 'enc_store_id_dept_id_snap_transform_1_mean',
 'enc_store_id_dept_id_snap_transform_1_std',
 'enc_item_id_store_id_snap_transform_1_mean',
 'enc_item_id_store_id_snap_transform_1_std',
 'enc_store_id_dept_id_tm_m_mean',
 'enc_store_id_dept_id_tm_m_std',
 'enc_item_id_store_id_tm_m_mean',
 'enc_item_id_store_id_tm_m_std',
 'enc_store_id_dept_id_snap_transform_1_tm_m_mean',
 'enc_store_id_dept_id_snap_transform_1_tm_m_std',
 'enc_item_id_store_id_snap_transform_1_tm_m_mean',
 'enc_item_id_store_id_snap_transform_1_tm_m_std',
 '0.5%',
 '2.5%',
 '16.5%',
 '25%',
 '50%',
 '75%',
 '83.5%',
 '97.5%',
 '99.5%',
 'max',
 'fraction_0',
 'max_streak_allowed',
 'probability_0',
                     ####### LAG FEATURES #######
 'mean_4_dow_0',
 'mean_4_dow_1',
 'mean_4_dow_2',
 'mean_4_dow_3',
 'mean_4_dow_4',
 'mean_4_dow_5',
 'mean_4_dow_6',
 'mean_20_dow_0',
 'mean_20_dow_1',
 'mean_20_dow_2',
 'mean_20_dow_3',
 'mean_20_dow_4',
 'mean_20_dow_5',
 'mean_20_dow_6',
 'last_sale_day',
 'lag_1',
 'lag_2',
 'lag_3',
 'lag_4',
 'lag_5',
 'lag_6',
 'lag_7',
 'lag_8',
 'lag_9',
 'lag_10',
 'lag_11',
 'lag_12',
 'lag_13',
 'lag_14',
 'lag_15',
 'lag_16',
 'lag_17',
 'lag_18',
 'lag_19',
 'lag_20',
 'lag_21',
 'lag_22',
 'lag_23',
 'lag_24',
 'lag_25',
 'lag_26',
 'lag_27',
 'lag_28',
 'lag_29',
 'lag_30',
 'lag_31',
 'lag_32',
 'lag_33',
 'lag_34',
 'lag_35',
 'lag_36',
 'lag_37',
 'lag_38',
 'lag_39',
 'lag_40',
 'lag_41',
 'lag_42',
 'lag_43',
 'lag_44',
 'lag_45',
 'lag_46',
 'lag_47',
 'lag_48',
 'lag_49',
 'lag_50',
 'lag_51',
 'lag_52',
 'lag_53',
 'lag_54',
 'lag_55',
 'lag_56',
 'lag_57',
 'lag_58',
 'lag_59',
 'lag_60',
 'lag_61',
 'lag_62',
 'lag_63',
 'lag_64',
 'lag_65',
 'lag_66',
 'lag_67',
 'lag_68',
 'lag_69',
 'lag_70',
 'lag_71',
 'lag_72',
 'lag_73',
 'lag_74',
 'lag_75',
 'lag_76',
 'lag_77',
 'lag_78',
 'lag_79',
 'lag_80',
 'lag_81',
 'lag_82',
 'lag_83',
 'lag_84',
 'shift_1_rolling_nan_mean_3',
 'shift_1_rolling_nan_median_3',
 'shift_1_rolling_mean_decay_3',
 'shift_1_rolling_diff_nan_mean_3',
 'shift_1_rolling_nan_min_3',
 'shift_1_rolling_nan_max_3',
 'shift_1_rolling_nan_std_3',
 'shift_1_rolling_nan_mean_7',
 'shift_1_rolling_nan_median_7',
 'shift_1_rolling_mean_decay_7',
 'shift_1_rolling_diff_nan_mean_7',
 'shift_1_rolling_nan_min_7',
 'shift_1_rolling_nan_max_7',
 'shift_1_rolling_nan_std_7',
 'shift_1_rolling_nan_mean_14',
 'shift_1_rolling_nan_median_14',
 'shift_1_rolling_mean_decay_14',
 'shift_1_rolling_diff_nan_mean_14',
 'shift_1_rolling_nan_min_14',
 'shift_1_rolling_nan_max_14',
 'shift_1_rolling_nan_std_14',
 'shift_1_rolling_nan_mean_30',
 'shift_1_rolling_nan_median_30',
 'shift_1_rolling_mean_decay_30',
 'shift_1_rolling_diff_nan_mean_30',
 'shift_1_rolling_nan_min_30',
 'shift_1_rolling_nan_max_30',
 'shift_1_rolling_nan_std_30',
 'shift_1_rolling_nan_mean_60',
 'shift_1_rolling_nan_median_60',
 'shift_1_rolling_mean_decay_60',
 'shift_1_rolling_diff_nan_mean_60',
 'shift_1_rolling_nan_min_60',
 'shift_1_rolling_nan_max_60',
 'shift_1_rolling_nan_std_60',
 'shift_1_rolling_nan_mean_140',
 'shift_1_rolling_nan_median_140',
 'shift_1_rolling_mean_decay_140',
 'shift_1_rolling_diff_nan_mean_140',
 'shift_1_rolling_nan_min_140',
 'shift_1_rolling_nan_max_140',
 'shift_1_rolling_nan_std_140',
 'shift_8_rolling_nan_mean_7',
 'shift_8_rolling_mean_decay_7',
 'shift_8_rolling_diff_nan_mean_7',
 'shift_29_rolling_nan_mean_7',
 'shift_29_rolling_mean_decay_7',
 'shift_29_rolling_diff_nan_mean_7',
 'momentum_7_rolling_nan_mean_7',
 'momentum_28_rolling_nan_mean_7',
 'momentum_7_rolling_mean_decay_7',
 'momentum_28_rolling_mean_decay_7',
 'momentum_7_rolling_diff_nan_mean_7',
 'momentum_28_rolling_diff_nan_mean_7',
 'shift_8_rolling_nan_mean_30',
 'shift_8_rolling_mean_decay_30',
 'shift_8_rolling_diff_nan_mean_30',
 'shift_29_rolling_nan_mean_30',
 'shift_29_rolling_mean_decay_30',
 'shift_29_rolling_diff_nan_mean_30',
 'momentum_7_rolling_nan_mean_30',
 'momentum_28_rolling_nan_mean_30',
 'momentum_7_rolling_mean_decay_30',
 'momentum_28_rolling_mean_decay_30',
 'momentum_7_rolling_diff_nan_mean_30',
 'momentum_28_rolling_diff_nan_mean_30',
 'shift_29_rolling_nan_mean_60',
 'shift_29_rolling_mean_decay_60',
 'shift_29_rolling_diff_nan_mean_60',
 'shift_91_rolling_nan_mean_60',
 'shift_91_rolling_mean_decay_60',
 'shift_91_rolling_diff_nan_mean_60',
 'momentum_28_rolling_nan_mean_60',
 'momentum_90_rolling_nan_mean_60',
 'momentum_28_rolling_mean_decay_60',
 'momentum_90_rolling_mean_decay_60',
 'momentum_28_rolling_diff_nan_mean_60',
 'momentum_90_rolling_diff_nan_mean_60',
                    
                    ############# PCA ########################### 
 '29_84_lags_ipca_comp_1',
 '29_84_lags_ipca_comp_2',
 '29_84_lags_ipca_comp_3',
 '29_84_lags_ipca_comp_4',
 '29_84_lags_ipca_comp_5',
 '29_84_lags_ipca_comp_6',
 '29_84_lags_ipca_comp_7',
 '29_84_lags_ipca_comp_8',
 '29_84_lags_ipca_comp_9',
 '29_84_lags_ipca_comp_10',
 '29_84_lags_ipca_comp_11',
 '29_84_lags_ipca_comp_12',
 '29_84_lags_ipca_comp_13',
 '29_84_lags_ipca_comp_14', 
                     
 '15_84_lags_ipca_comp_1',
 '15_84_lags_ipca_comp_2',
 '15_84_lags_ipca_comp_3',
 '15_84_lags_ipca_comp_4',
 '15_84_lags_ipca_comp_5',
 '15_84_lags_ipca_comp_6',
 '15_84_lags_ipca_comp_7',
 '15_84_lags_ipca_comp_8',
 '15_84_lags_ipca_comp_9',
 '15_84_lags_ipca_comp_10',
 '15_84_lags_ipca_comp_11',
 '15_84_lags_ipca_comp_12',
 '15_84_lags_ipca_comp_13',
 '15_84_lags_ipca_comp_14']


# In[ ]:


keep_features = [
    
 'id', 
 'd', 
 'sales',
#                      ###### BASIC FEATURES ######
#  'item_id', #####################
 'dept_id',
#  'cat_id',#####################################################################
 'store_id',
#  'state_id',#####################################################################
 'sell_price',
 'price_min',
 'price_max',
 'price_median',
 'price_mode',
 'price_mean',
 'price_std',
 'price_norm_max',
 'price_norm_mode',
 'price_norm_mean',
 'price_momentum',
 'price_roll_momentum_4',
 'price_roll_momentum_24',
 'price_end_digits',
#  'event_name_1',#######################################################################################
#  'event_type_1',#######################################################################################
#  'event_name_2',#######################################################################################
#  'event_type_2',#######################################################################################
#  'snap_CA',#######################################################################################
#  'snap_TX',#######################################################################################
#  'snap_WI',#######################################################################################
 'tm_d',
 'tm_w',
 'tm_m',
#  'tm_y',
#  'tm_wm',#########################################################
 'tm_dw',
 'tm_w_end',
 'snap_transform_1',
 'snap_transform_2',
 'next_event_type_1',
 'last_event_type_1',
 'days_since_event',
 'days_until_event',
#                           ###### ENCODING FEATURES ######
#  'enc_state_id_mean', ####################################################################
#  'enc_state_id_std',####################################################################
#  'enc_store_id_mean',####################################################################
#  'enc_store_id_std',####################################################################
#  'enc_cat_id_mean',####################################################################
#  'enc_cat_id_std',####################################################################
#  'enc_dept_id_mean',####################################################################
#  'enc_dept_id_std',####################################################################
#  'enc_state_id_cat_id_mean',####################################################################
#  'enc_state_id_cat_id_std',####################################################################
#  'enc_state_id_dept_id_mean',####################################################################
#  'enc_state_id_dept_id_std',####################################################################
#  'enc_store_id_cat_id_mean',####################################################################
#  'enc_store_id_cat_id_std',####################################################################
#  'enc_store_id_dept_id_mean',####################################################################
#  'enc_store_id_dept_id_std',####################################################################
#  'enc_item_id_mean',####################################################################
#  'enc_item_id_std',####################################################################
#  'enc_item_id_state_id_mean',####################################################################
#  'enc_item_id_state_id_std',####################################################################
#  'enc_item_id_store_id_mean',####################################################################
#  'enc_item_id_store_id_std',####################################################################
#  'enc_store_id_dept_id_snap_transform_1_mean',####################################################################
#  'enc_store_id_dept_id_snap_transform_1_std',####################################################################
#  'enc_item_id_store_id_snap_transform_1_mean',####################################################################
#  'enc_item_id_store_id_snap_transform_1_std',####################################################################
#  'enc_store_id_dept_id_tm_m_mean',####################################################################
#  'enc_store_id_dept_id_tm_m_std',####################################################################
#  'enc_item_id_store_id_tm_m_mean',####################################################################
#  'enc_item_id_store_id_tm_m_std',####################################################################
#  'enc_store_id_dept_id_snap_transform_1_tm_m_mean',####################################################################
#  'enc_store_id_dept_id_snap_transform_1_tm_m_std',####################################################################
#  'enc_item_id_store_id_snap_transform_1_tm_m_mean',####################################################################
#  'enc_item_id_store_id_snap_transform_1_tm_m_std',####################################################################
#  '0.5%',#########################################################################################
#  '2.5%',#########################################################################################
#  '16.5%',#########################################################################################
#  '25%',#########################################################################################
#  '50%',#########################################################################################
#  '75%',#########################################################################################
#  '83.5%',#########################################################################################
#  '97.5%',#########################################################################################
#  '99.5%',#########################################################################################
#  'max',
 'fraction_0',
#  'max_streak_allowed', 
#  'probability_0',
#                      ####### LAG FEATURES #######
 'mean_4_dow_0',
 'mean_4_dow_1',
 'mean_4_dow_2',
 'mean_4_dow_3',
 'mean_4_dow_4',
 'mean_4_dow_5',
 'mean_4_dow_6',
 'mean_20_dow_0',
 'mean_20_dow_1',
 'mean_20_dow_2',
 'mean_20_dow_3',
 'mean_20_dow_4',
 'mean_20_dow_5',
 'mean_20_dow_6',
 'last_sale_day',
 'lag_1',
 'lag_2',
 'lag_3',
 'lag_4',
 'lag_5',
 'lag_6',
 'lag_7',
 'lag_8',
 'lag_9',
 'lag_10',
 'lag_11',
 'lag_12',
 'lag_13',
 'lag_14',
#  'lag_15',
#  'lag_16',
#  'lag_17',
#  'lag_18',
#  'lag_19',
#  'lag_20',
#  'lag_21',
#  'lag_22',
#  'lag_23',
#  'lag_24',
#  'lag_25',
#  'lag_26',
#  'lag_27',
#  'lag_28',
#  'lag_29',
#  'lag_30',
#  'lag_31',
#  'lag_32',
#  'lag_33',
#  'lag_34',
#  'lag_35',
#  'lag_36',
#  'lag_37',
#  'lag_38',
#  'lag_39',
#  'lag_40',
#  'lag_41',
#  'lag_42',
#  'lag_43',
#  'lag_44',
#  'lag_45',
#  'lag_46',
#  'lag_47',
#  'lag_48',
#  'lag_49',
#  'lag_50',
#  'lag_51',
#  'lag_52',
#  'lag_53',
#  'lag_54',
#  'lag_55',
#  'lag_56',
#  'lag_57',
#  'lag_58',
#  'lag_59',
#  'lag_60',
#  'lag_61',
#  'lag_62',
#  'lag_63',
#  'lag_64',
#  'lag_65',
#  'lag_66',
#  'lag_67',
#  'lag_68',
#  'lag_69',
#  'lag_70',
#  'lag_71',
#  'lag_72',
#  'lag_73',
#  'lag_74',
#  'lag_75',
#  'lag_76',
#  'lag_77',
#  'lag_78',
#  'lag_79',
#  'lag_80',
#  'lag_81',
#  'lag_82',
#  'lag_83',
#  'lag_84',
 'shift_1_rolling_nan_mean_3',
#  'shift_1_rolling_nan_median_3',
 'shift_1_rolling_mean_decay_3',
#  'shift_1_rolling_diff_nan_mean_3',
# #  'shift_1_rolling_nan_min_3',
# #  'shift_1_rolling_nan_max_3',
#  'shift_1_rolling_nan_std_3',
 'shift_1_rolling_nan_mean_7',
#  'shift_1_rolling_nan_median_7',
 'shift_1_rolling_mean_decay_7',
#  'shift_1_rolling_diff_nan_mean_7',
#  'shift_1_rolling_nan_min_7',
#  'shift_1_rolling_nan_max_7',
 'shift_1_rolling_nan_std_7',
 'shift_1_rolling_nan_mean_14',
#  'shift_1_rolling_nan_median_14',
 'shift_1_rolling_mean_decay_14',
 'shift_1_rolling_diff_nan_mean_14',
#  'shift_1_rolling_nan_min_14',
#  'shift_1_rolling_nan_max_14',
 'shift_1_rolling_nan_std_14',
 'shift_1_rolling_nan_mean_30',
#  'shift_1_rolling_nan_median_30',
 'shift_1_rolling_mean_decay_30',
#  'shift_1_rolling_diff_nan_mean_30',
#  'shift_1_rolling_nan_min_30',
#  'shift_1_rolling_nan_max_30',
#  'shift_1_rolling_nan_std_30',
 'shift_1_rolling_nan_mean_60',
 'shift_1_rolling_nan_median_60',
 'shift_1_rolling_mean_decay_60',
#  'shift_1_rolling_diff_nan_mean_60',
#  'shift_1_rolling_nan_min_60',
#  'shift_1_rolling_nan_max_60',
 'shift_1_rolling_nan_std_60',
 'shift_1_rolling_nan_mean_140',
#  'shift_1_rolling_nan_median_140',
 'shift_1_rolling_mean_decay_140',
#  'shift_1_rolling_diff_nan_mean_140',
#  'shift_1_rolling_nan_min_140',
#  'shift_1_rolling_nan_max_140',
 'shift_1_rolling_nan_std_140',
 'shift_8_rolling_nan_mean_7',
 'shift_8_rolling_mean_decay_7',
#  'shift_8_rolling_diff_nan_mean_7',
 'shift_29_rolling_nan_mean_7',
#  'shift_29_rolling_mean_decay_7',
#  'shift_29_rolling_diff_nan_mean_7',
 'momentum_7_rolling_nan_mean_7',
 'momentum_28_rolling_nan_mean_7',
 'momentum_7_rolling_mean_decay_7',
#  'momentum_28_rolling_mean_decay_7',
 'momentum_7_rolling_diff_nan_mean_7',
 'momentum_28_rolling_diff_nan_mean_7',
 'shift_8_rolling_nan_mean_30',
 'shift_8_rolling_mean_decay_30',
#  'shift_8_rolling_diff_nan_mean_30',
 'shift_29_rolling_nan_mean_30',
 'shift_29_rolling_mean_decay_30',
#  'shift_29_rolling_diff_nan_mean_30',
 'momentum_7_rolling_nan_mean_30',
 'momentum_28_rolling_nan_mean_30',
#  'momentum_7_rolling_mean_decay_30',
#  'momentum_28_rolling_mean_decay_30',
#  'momentum_7_rolling_diff_nan_mean_30',
#  'momentum_28_rolling_diff_nan_mean_30',
 'shift_29_rolling_nan_mean_60',
#  'shift_29_rolling_mean_decay_60',
#  'shift_29_rolling_diff_nan_mean_60',
 'shift_91_rolling_nan_mean_60',
 'shift_91_rolling_mean_decay_60',
#  'shift_91_rolling_diff_nan_mean_60',
#  'momentum_28_rolling_nan_mean_60',
#  'momentum_90_rolling_nan_mean_60',
#  'momentum_28_rolling_mean_decay_60',
#  'momentum_90_rolling_mean_decay_60',
#  'momentum_28_rolling_diff_nan_mean_60',
#  'momentum_90_rolling_diff_nan_mean_60',
    
#  '1_84_lags_ipca_comp_1',
#  '1_84_lags_ipca_comp_2',
#  '1_84_lags_ipca_comp_3',
#  '1_84_lags_ipca_comp_4',
#  '1_84_lags_ipca_comp_5',
#  '1_84_lags_ipca_comp_6',
#  '1_84_lags_ipca_comp_7',
#  '1_84_lags_ipca_comp_8',
#  '1_84_lags_ipca_comp_9',
#  '1_84_lags_ipca_comp_10',
#  '1_84_lags_ipca_comp_11',
#  '1_84_lags_ipca_comp_12',
#  '1_84_lags_ipca_comp_13',
#  '1_84_lags_ipca_comp_14',
 '15_84_lags_ipca_comp_1',
 '15_84_lags_ipca_comp_2',
 '15_84_lags_ipca_comp_3',
 '15_84_lags_ipca_comp_4',
 '15_84_lags_ipca_comp_5',
 '15_84_lags_ipca_comp_6',
 '15_84_lags_ipca_comp_7',
#  '15_84_lags_ipca_comp_8',
#  '15_84_lags_ipca_comp_9',
#  '15_84_lags_ipca_comp_10',
#  '15_84_lags_ipca_comp_11',
#  '15_84_lags_ipca_comp_12',
#  '15_84_lags_ipca_comp_13',
#  '15_84_lags_ipca_comp_14'
]


# In[ ]:


drop_features_from_training = [fe for fe in original_features if fe not in keep_features]
print(len(drop_features_from_training))
print(f'we have {len(keep_features)} features')


# In[ ]:


########################## w_12 ################################
DATA_INTERIM_PATH = '../input/m5-w-df-with-all-scaled-weights/'
w_12 = pd.read_pickle(F'{DATA_INTERIM_PATH}w_12_{START_TEST}.pkl')
if START_TEST < 1942: 
    df = pd.read_pickle(F'{DATA_INTERIM_PATH}w_12_1942.pkl')
    w_12 = w_12.join(df[['oos_level_12_scale']])

####################### For validation #########################
def get_actuals(train_df, days):
    return train_df[[f'd_{d}' for d in days]].values

################################################################
###################### Evaluators  #############################
class WRMSSELGBM(WRMSSE): 
    def feval(self, preds, train_data):
        preds = preds.reshape(self.actuals.shape[1], -1).T
        score = self.score(preds)
        return 'WRMSSE', score, False
    
    def full_score(self, preds, train_data):
        preds = preds.reshape(self.actuals.shape[1], -1).T
        score = self.score_all(preds)
        return 'WRMSSE', score, False
    
def get_evaluators(keep_id, start_test_days=[1914], keep_all=False): 
    train_df, cal_df, prices_df, _ = read_data(F'{DATA_RAW_PATH}')
    if not keep_all:
        train_df = train_df[train_df.item_id.isin(keep_id)]

    evaluators = []
    for start_test in start_test_days: 
        e = WRMSSELGBM(train_df, cal_df, prices_df, start_test)
        evaluators.append(e)
    return evaluators

################################################################
###################### Objective functions #####################

def get_wrmsse(w_12_train):
    """w_12_train must be aligned with grid_df like
    w_12_train = w_12.reindex(grid_df[train_mask].id)
    """
    weight = w_12_train['total_sw'] / w_12_train['total_sw'].mean()
    
    def wrmsse(preds, train_data): 
        actuals = train_data.get_label()
        diff = actuals - preds
        grad = -diff * weight
        hess = np.ones_like(diff)
        return grad, hess
    return wrmsse

def get_oos_rmsse(w_12_train): 
    
    oos_scale = 1/w_12_train.oos_level_12_scale
    
    def oos_rmsse(preds, train_data): 
        actuals = train_data.get_label()
        diff = actuals - preds
        grad = -diff * oos_scale
        hess = np.ones_like(diff)
        return grad, hess
    return oos_rmsse


################################################################
######################## Custom metrics ########################

def get_wrmsse_metric(w_12_valid):
    weight = w_12_valid['total_sw'] / w_12_valid['total_sw'].mean()

    def wrmsse_metric(preds, train_data): 
        actuals = train_data.get_label()
        diff = actuals - preds
        res = np.sum(diff**2 * weight)
        return 'custom_wrmsse_metric', res, False
    return wrmsse_metric

def get_oos_rmsse_metric(w_12_valid): 
    oos_scale = 1/w_12_valid.oos_level_12_scale
    
    def oos_rmsse_metric(preds, train_data): 
        actuals = train_data.get_label()
        diff = actuals - preds
        res = np.sum(diff**2 * oos_scale**2)
        return 'oos_rmsse', res, False
    return oos_rmsse_metric

def get_rmsse_metric(w_12_valid):
    scale = 1/np.sqrt(w_12_valid.scale)
    
    def rmsse_metric(preds, train_data): 
        actuals = train_data.get_label()
        diff = actuals - preds
        res = np.sum(diff**2 * scale**2)
        return 'rmsse', res, False
    return rmsse_metric

def get_l12_wrmsse_metric(w_12_valid):
    scale = w_12_valid.scaled_weight
    
    def l12_wrmsse_metric(preds, train_data): 
        actuals = train_data.get_label()
        diff = actuals - preds
        res = np.sum(np.abs(diff) * scale)
        return 'rmsse', res, False
    return l12_wrmsse_metric

################################################################
################# Simple model training function ###############
def train_lgbm(grid_df,             # All data, train and valid
               lgb_params,     
               remove_features,
               start_train=140, 
               test_day=1914,
               TARGET='sales',
               objective=None, 
               metric=None,
               fobj=None,           # Custom objective function
               feval=None,          # Custom metric function
               verbose_eval=25, 
               categories=None,     # Must use if categories are int type!!!!!
               w_12=None, 
               estimator=None,
               ):    
    """returns an lgbm estimator: this version is made for day
    to day models. """
    
    feature_cols = [col for col in list(grid_df) if col not in remove_features]
    lgb_params = lgb_params.copy()
    
    ####################### test set #########################
    ##########################################################
    test_mask = (grid_df.d == test_day) 
    test_x = grid_df[test_mask][feature_cols]
    
    ############ train, valid and test masks #################
    start_valid = test_day - 28
    valid_mask = (grid_df.d == start_valid) 
    train_mask = (grid_df.d >= start_train) & (grid_df.d < start_valid) & (grid_df[TARGET].notnull())
    
    ########################### train, valid and test #########################
    train_x, train_y = grid_df[train_mask][feature_cols], grid_df[train_mask][TARGET]
    valid_x, valid_y = grid_df[valid_mask][feature_cols], grid_df[valid_mask][TARGET]
    
    ######## Switching to numpy array to avoid RAM usage spike ################
#     features = list(train_x)
#     train_x = train_x.values.astype(np.float32)
#     valid_x = valid_x.values.astype(np.float32)

#     train_data = lgb.Dataset(train_x, feature_name=features, categorical_feature=categories, label=train_y)
#     valid_data = lgb.Dataset(valid_x, feature_name=features, categorical_feature=categories, label=valid_y)
    train_data = lgb.Dataset(train_x, label=train_y)
    valid_data = lgb.Dataset(valid_x, label=valid_y)

    print('\n', '#' * 72,'\n', f'Training set: {start_train} to {start_valid - 1}\n', 
         f'Valid set: {test_day - 28}')
    
    ###################### Update objective ########################
    if objective == 'tweedie': 
        lgb_params['objective'] = 'tweedie'
        lgb_params['tweedie_variance_power'] = 1.1
        
    
        

    
    estimator = lgb.train(
                                lgb_params,
                                train_set=train_data,
                                valid_sets=[valid_data],
                                fobj = fobj,
                                feval = feval,
                                verbose_eval=verbose_eval, 
                                
    )
    gc.collect()    
    
    estimator.save_model(f'{MODEL_NAME}_{test_day - START_TEST}.bin', -1)
    preds = estimator.predict(test_x)
    
    ax = lgb.plot_importance(estimator, max_num_features=266, figsize=(15,40), title=f'{MODEL_NAME}_{test_day}')
    plt.show()
    
    return estimator, preds, test_x, test_mask, feature_cols


# In[ ]:


###########################################################################
######################## Process before training ##########################

########################### Load competition data #########################
cal_df = pd.read_csv(F'{DATA_RAW_PATH}calendar.csv')
prices_df = pd.read_csv(F'{DATA_RAW_PATH}sell_prices.csv')
train_df = pd.read_csv(F'{DATA_RAW_PATH}sales_train_evaluation.csv')

######################### Find items to drop ##############################
# We don't want items that haven't had sales at all stores for at least 
# 72 days, because this is the situation we had for the public validation
# set, and it could make our evaluator not behave correclty

###########################################################################
DATA_FEATURES_PATH = '/kaggle/input/m5-fe-basic-features/'
grid_df = pd.read_pickle(F'{DATA_FEATURES_PATH}grid_df_base_fe.pkl')
first_sale = grid_df[grid_df.sales.notnull()].drop_duplicates('id')
first_sale = first_sale.groupby('item_id')['d'].max()
keep_id = first_sale[(START_TEST - first_sale) >= 68].index

###########################################################################
##################### START_TEST specific processing ######################

######################## Truncate train_df ################################
# Create a truncated train_df so we can easily get accurate data
train_df_truncated = train_df[train_df.item_id.isin(keep_id)]

####################### Raw data for validation ###########################
valid_days = [d for d in range(START_TEST - 28, START_TEST)]
valid_actuals = get_actuals(train_df_truncated, valid_days)


# In[ ]:


############################## Evaluator ##################################
e = WRMSSELGBM(train_df_truncated, cal_df, prices_df, START_TEST)
if START_TEST!=1942:
    e_actuals = e.actuals.copy()

############################## prediction df ##############################
prediction_df = train_df_truncated[['id']]


# In[ ]:


for i in range(14):
    ###########################################################################
    ##################### test_day specific processing ########################
    # (START_TEST + test_day) is the day we are building a model to predict. 
    # for i in range(28):
    #     print(f'day {i + 1}')
    test_day = i

    # Validation set only a single day!... a month before predicting...weird. 
    # The thought is that this the best single day I could pick. This is also
    # what the grocery favorita winners did... I think. 
    e.actuals = valid_actuals[:, test_day].reshape((-1,1))



    ######################### Features #############################
    ################################################################

    ##################### Feature selection ########################

    #################### Start with the basics #####################
    DATA_FEATURES_PATH = '/kaggle/input/m5-fe-basic-features/'
    grid_df = pd.read_pickle(F'{DATA_FEATURES_PATH}grid_df_base_fe.pkl')
    grid_df['sales'] = grid_df['sales'].astype(np.float16)

    ####################### Truncate data  #########################
    # Item_id
    grid_df = grid_df[grid_df.item_id.isin(keep_id)]

    # Days
    days = [d for d in range(START_TRAIN, START_TEST + test_day + 1) if d%7 == (START_TEST + test_day)%7]
    grid_df = grid_df[grid_df.d.isin(days)]

    # Features
    keep_cols = [col for col in list(grid_df) if col not in drop_features_from_training]
    grid_df = grid_df[keep_cols]



    ############ Features that don't need shifting #################
    # These will be features that "stay" with the id and sale date, 
    # while other features such as lags and rolling windows will 
    # need to be shifted so that we prevent leakage. All lags will 
    # look the same 

    ###### add price features ######
    df = pd.read_pickle(F'{DATA_FEATURES_PATH}grid_df_price_fe.pkl').iloc[:, 3:]
    keep_cols = [col for col in list(df) if col not in drop_features_from_training]
    df = df[keep_cols]
    df = df[df.index.isin(grid_df.index)]
    grid_df = pd.concat([grid_df, df], axis=1)
    del df
    gc.collect() 

    ###### add calendar features ######
    DATA_FEATURES_PATH = '../input/m5-nb-fe-snap-and-event-features/'
    df = pd.read_pickle(f'{DATA_FEATURES_PATH}grid_df_cal_fe_2.pkl').iloc[:, 3:]
    keep_cols = [col for col in list(df) if col not in drop_features_from_training]
    df = df[keep_cols]
    df = df[df.index.isin(grid_df.index)]
    grid_df = pd.concat([grid_df, df], axis=1)
    del df
    gc.collect()



    ################## Stats encodings #############################
    DATA_FEATURES_PATH = '../input/m5-fe-dow-encodings/'
    df = pd.read_pickle(f'{DATA_FEATURES_PATH}fe_stats_encodings.pkl')
    keep_cols = [col for col in list(df) if col not in drop_features_from_training]
    df = df[keep_cols]
    df = df[df.index.isin(grid_df.index)]
    grid_df = pd.concat([grid_df, df], axis=1)
    del df
    gc.collect()

    ################################################################
    ############ Features that need index to be shifted ############
    # This way, we can use the same lagged features and be sure that
    # we don't have leakage and everything is in line. 
    shift = 30490 * test_day

    ##### Day of week means and last sale day ######
    DATA_FEATURES_PATH = '../input/m5-fe-dow-encodings/'
    df = pd.read_pickle(f'{DATA_FEATURES_PATH}fe_dow_means_4_20_and_last_sale_day.pkl')
    keep_cols = [col for col in list(df) if col not in drop_features_from_training]
    df = df[keep_cols]
    df.index = df.index + shift
    df = df[df.index.isin(grid_df.index)]
    grid_df = pd.concat([grid_df, df], axis=1)
    del df
    gc.collect()

    ############################ Normal lags part 1 ############################
    DATA_FEATURES_PATH = '../input/m5-fe-basic-lags-part-1-oos-fixed/'

    df = pd.read_pickle(F'{DATA_FEATURES_PATH}oos_fe_lags_1_14.pkl')
    keep_cols = [col for col in list(df) if col not in drop_features_from_training]
    df = df[keep_cols]
    df.index = df.index + shift
    df = df[df.index.isin(grid_df.index)]
    grid_df = pd.concat([grid_df, df], axis=1)
    del df

    # df = pd.read_pickle(F'{DATA_FEATURES_PATH}oos_fe_lags_15_28.pkl')
    # keep_cols = [col for col in list(df) if col not in drop_features_from_training]
    # df = df[keep_cols]
    # df.index = df.index + shift
    # df = df[df.index.isin(grid_df.index)]
    # grid_df = pd.concat([grid_df, df], axis=1)
    # del df

    ##################################################################################
    DATA_FEATURES_PATH = '../input/m5-fe-oos-ipca-with-nan-part-4/'

    df = pd.read_pickle(F'{DATA_FEATURES_PATH}fe_pca5_oos_lags_15_84.pkl')
    keep_cols = [col for col in list(df) if col not in drop_features_from_training]
    df = df[keep_cols]
    df.index = df.index + shift
    df = df[df.index.isin(grid_df.index)]
    grid_df = pd.concat([grid_df, df], axis=1)
    del df
    gc.collect()

    ########################## Rolling window features ###########################
    DATA_FEATURES_PATH = '../input/m5-oos-fe-00-rolling-windows-basic-stats/'

    df = pd.read_pickle(F'{DATA_FEATURES_PATH}oos_fe_rw_basic_3_7.pkl')
    keep_cols = [col for col in list(df) if col not in drop_features_from_training]
    df = df[keep_cols]
    df.index = df.index + shift
    df = df[df.index.isin(grid_df.index)]
    grid_df = pd.concat([grid_df, df], axis=1)
    del df
    gc.collect()

    df = pd.read_pickle(F'{DATA_FEATURES_PATH}oos_fe_rw_basic_14_30.pkl')
    keep_cols = [col for col in list(df) if col not in drop_features_from_training]
    df = df[keep_cols]
    df.index = df.index + shift
    df = df[df.index.isin(grid_df.index)]
    grid_df = pd.concat([grid_df, df], axis=1)
    del df
    gc.collect()

    df = pd.read_pickle(F'{DATA_FEATURES_PATH}oos_fe_rw_basic_60_140.pkl')
    keep_cols = [col for col in list(df) if col not in drop_features_from_training]
    df = df[keep_cols]
    df.index = df.index + shift
    df = df[df.index.isin(grid_df.index)]
    grid_df = pd.concat([grid_df, df], axis=1)
    del df
    gc.collect()

    ##################### Shifted rolling window features ######################
    DATA_FEATURES_PATH = '../input/m5-fe-shift-rw-and-momentum/'

    df = pd.read_pickle(F'{DATA_FEATURES_PATH}fe_rw_shifts_and_momentum_7.pkl')
    keep_cols = [col for col in list(df) if col not in drop_features_from_training]
    df = df[keep_cols]
    df.index = df.index + shift
    df = df[df.index.isin(grid_df.index)]
    grid_df = pd.concat([grid_df, df], axis=1)
    del df
    gc.collect()

    df = pd.read_pickle(F'{DATA_FEATURES_PATH}fe_rw_shifts_and_momentum_30.pkl')
    keep_cols = [col for col in list(df) if col not in drop_features_from_training]
    df = df[keep_cols]
    df.index = df.index + shift
    df = df[df.index.isin(grid_df.index)]
    grid_df = pd.concat([grid_df, df], axis=1)
    del df
    gc.collect()

    df = pd.read_pickle(F'{DATA_FEATURES_PATH}fe_rw_shifts_and_momentum_60.pkl')
    keep_cols = [col for col in list(df) if col not in drop_features_from_training]
    df = df[keep_cols]
    df.index = df.index + shift
    df = df[df.index.isin(grid_df.index)]
    grid_df = pd.concat([grid_df, df], axis=1)
    del df
    gc.collect()

    grid_df.info()


    ########################## Save predictions #####################################
    # prediction_df.to_csv(f'preds_{model_name}.csv')
    display(list(grid_df))
    display(grid_df.shape)

    ####################### test set #########################
    ##########################################################
    remove_features = ['id', 'd', TARGET, 'item_id']
    feature_cols = [col for col in list(grid_df) if col not in remove_features]
    test_mask = (grid_df.d == (START_TEST + test_day)) 
    test_x = grid_df[test_mask][feature_cols]

    ############ train, valid and test masks #################
    start_valid = (START_TEST + test_day) - 28
    valid_mask = (grid_df.d == start_valid) 
    train_mask = (grid_df.d >= START_TRAIN) & (grid_df.d < start_valid) & (grid_df[TARGET].notnull())

    ################## Fit custom functions ##################
    w_12_train = w_12.reindex(grid_df[train_mask].id)
    w_12_valid = w_12.reindex(grid_df[valid_mask].id)

    ################## Objective #############################
    wrmsse = get_wrmsse(w_12_train)

    ######################### Metrics ########################
    oos_rmsse_metric = get_oos_rmsse_metric(w_12_valid)

    ################################################################################
    remove_features = ['id', 'd', TARGET, 'item_id']

    lgb_params = {
                        'boosting_type': 'gbdt',                      
                        'subsample': 0.5,
                        'metric': 'None',
                        'subsample_freq': 1,
                        'learning_rate': 0.03,           
                        'num_leaves': 2**8-1,            
                        'min_data_in_leaf': 2**8-1,     
                        'feature_fraction': 0.8,
                        'n_estimators': 5000,            
                        'early_stopping_rounds': 50,     
                        'seed': SEED,
                        'verbose': -1,
                    } 

    ######################## Same as above with item_id ##########################
    # I removed item_id from remove features
    estimator, preds, test_x, test_mask, feature_cols = train_lgbm(grid_df,             # All data, train and valid
               lgb_params,     
               remove_features,
               test_day = START_TEST + test_day,
               TARGET='sales',
               objective=None,
               fobj=wrmsse,           # Custom objective function
               feval=oos_rmsse_metric,          # Custom metric function
               verbose_eval=25, 
               categories=None,     # Must use if categories are int type!!!!!
               w_12 = w_12
               )

    prediction_df[f'F{test_day + 1}'] = preds
prediction_df.to_csv(f'sub{test_day}.csv')


# In[ ]:




