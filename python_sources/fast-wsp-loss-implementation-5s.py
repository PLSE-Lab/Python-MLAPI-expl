#!/usr/bin/env python
# coding: utf-8

# The M5 competition loss is a little ugly, epecially when it comes to the Uncertainty's one with it's WSPL! So far, there is no, if not little, public implementation of that *beast* (I still talking about the WSPL and not *the american  crime drama series*).
# 
# As I was saying it on [M5  Accuracy](https://www.kaggle.com/kneroma/faster-loss-function-with-lb-position-500ms) my implementation is based on a deep inspection of the loss combined with an intelligent choice of pandas' components. 
# 
# **Too much talk, let's dive in it !**

# <h3 style="color: red;">If you find this work helpful, please don't forget upvoting in order to get me motivated in sharing my hard work<h3/>

# In[ ]:


from pathlib import Path
import pandas as pd, numpy as np
np.set_printoptions(suppress=True)


# In[ ]:


get_ipython().system(' ls ../input')


# In[ ]:


DATA_ROOT = Path("../input/m5-forecasting-uncertainty")


# In[ ]:





# # Let's configure everything here

# In[ ]:


class M5Config:
    def __init__(self,cat_cols=None,sales_path=None, add_fake_categories=True, start=1,
                     end = 1913, days=None, evaluation=False,
                read_calendar=True, read_sales=True, read_prices=True, read_sample_submission=False):
        self.cat_cols = ["id", "cat_id", "state_id", "dept_id", "store_id", "item_id"] if cat_cols is None else cat_cols
        self.col_groups = [
                ('Total', 'X'),
                ('cat_id', 'X'),
                ('state_id', 'X'),
                ('dept_id', 'X'),
                ('store_id', 'X'),
                ('item_id', 'X'),
                ('state_id', 'cat_id'),
                ('state_id', 'dept_id'),
                ('store_id', 'cat_id'),
                ('store_id', 'dept_id'),
                ('state_id','item_id'),
                ('item_id', 'store_id')]

        self.evaluation = False
        self.suffix = "evaluation" if self.evaluation else "validation"

        self.sales_path = DATA_ROOT/f'sales_train_{self.suffix}.csv' if sales_path is None else sales_path
        self.calendar_path = DATA_ROOT/"calendar.csv"
        self.prices_path = DATA_ROOT/"sell_prices.csv"
        self.sample_submission_path = DATA_ROOT/"sample_submission.csv"

        self.add_fake_categories = add_fake_categories

        self.start = start
        self.end = end

        
        if days is None:
            self.set_days()
        else:
            self.days = days

        assert end > 28
        self.set_weight_days()
        
        self.read_calendar = read_calendar
        self.read_sales = read_sales
        self.read_prices = read_prices
        self.read_sample_submission = False
        
    def set_days(self):
        self.days = [f"d_{i}" for i in range(self.start,self.end+1)]
    
    def set_weight_days(self):
        self.weight_days = [f"d_{i}" for i in range(self.end-27, self.end+1)]


# In[ ]:





# # And what if we make a generic class to laod the data when needed ?

# In[ ]:


class M5Data:
    def __init__(self, config=None):
        self.config = config if config is not None else M5Config()
        
        self.cal = self.read(self.config.calendar_path) if self.config.read_calendar else None
        self.sales = self.read(self.config.sales_path, usecols = self.config.cat_cols
                               + self.config.days) if self.config.read_sales else None
        self.prices = self.read(self.config.prices_path) if self.config.read_prices else None
        self.sample_submission = self.read(self.config.sample_submission_path)                                            if self.config.read_sample_submission else None
        
        if self.config.add_fake_categories:
            self.add_fake_categories()
            
        
    def read(self, path, usecols=None):
        return pd.read_csv(path.as_posix(), usecols=usecols)
    
    
    def add_fake_categories(self):
        self.sales["Total"] = "Total"
        self.sales["X"] = "X"
        
        
        
    
    def _to_42840(self, group):
        assert group in self.config.col_groups
#         print(group, self.sales.columns)
        group = list(group)
        df = self.sales[group+self.config.days].groupby(group)[self.config.days].sum()
        df.reset_index(inplace=True)
        df.rename(columns={group[0]:"level1_val", group[1]:"level2_val"}, inplace=True)
        df["level1_name"] = group[0]
        df["level2_name"] = group[1]
        df["group_id"] = self.config.col_groups.index(tuple(group))
        df = df[["group_id", "level1_name", "level2_name", "level1_val", "level2_val"]+self.config.days]
        
        return df
    
    def to_42840(self):
        
        df = pd.concat([self._to_42840(group) for group in self.config.col_groups], axis=0,sort=False)
        df.sort_values([ "level1_val", "level2_val"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df


# ### You can give a try to the M5Data API here :

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndata = M5Data()')


# In[ ]:


data.sales.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndata._to_42840(("item_id", "store_id")).head()')


# In[ ]:





# # Powerful API for Weights & Scales computation

# In[ ]:


class WeightAndScaleComputer:
    def __init__(self, config=None, data=None):
        self.config = config if config is not None else M5Config()
        self.data = data if data is not None else M5Data(config=self.config)
        
        self.df_42840 = self.data.to_42840()
        
    def get_prices(self):
        prices = self.data.prices.copy()
        cal = self.data.cal

        prices["id"] = ["{}_{}_{}".format(item_id, store_id, self.config.suffix) 
                            for item_id, store_id in zip(prices.item_id, prices.store_id)] 
        
        day_count = cal["d"].str.replace("d_", "").astype(int)

        prices = prices[["wm_yr_wk", "id", "sell_price"]].merge(
                cal.loc[(day_count>= self.config.end-27) & (day_count<= self.config.end), ["wm_yr_wk", "d"]],
                                                on = ["wm_yr_wk"])
        prices = prices.set_index(["id", "d"]).sell_price.fillna(0.).unstack().fillna(0.)
        return prices
        
    
    def get_weights(self):
        
        # Backup old data
        sales_backup = self.data.sales
        df_42840_backup  = self.df_42840

        data = self.data
        sales = data.sales
        
        sales.sort_values("id",inplace=True)
        sales.reset_index(inplace=True, drop=True)
        
        prices = self.get_prices()
        prices.sort_index(inplace=True)
        prices.reset_index(inplace=True, drop=True)
        prices = prices[self.config.weight_days]
        
        for i,col in enumerate(self.config.weight_days):
            sales[col] = sales[col]*prices[col].values
            
        data.sales = sales
        df_42840 = data.to_42840()
        
        df_42840["turnover"] = df_42840[self.config.weight_days].sum(axis=1)
        df_42840["level_turnover"] = df_42840.groupby(["level1_name","level2_name"]).turnover.transform("sum")
        df = df_42840[["group_id", "level1_name", "level2_name", "level1_val", "level2_val"]].copy()
        df["weights"] = df_42840["turnover"]/df_42840["level_turnover"].values
        
        df.sort_values(["level1_val", "level2_val"], inplace=True)
        df.reset_index(drop=True,inplace=True)
        
        # Restore old data
        self.data.sales = sales_backup
        self.df_42840 = df_42840_backup
        
        return df
    
    
    def get_scales(self, kind="mae"): # kind in ['mae', 'mse']
        assert kind in ['mae', 'mse']
        
        df = self.df_42840[self.config.days].values
        
        diff = (np.abs if kind == "mae" else np.square )(df[:, 1:] - df[:, :-1]) 
        
        is_start = df[:, :-1].cumsum(1) >= 1
        
        diff *= is_start
        
        starts = is_start.argmax(1)
        size = df.shape[1] - starts - 1
        
        scales = diff.sum(1)/size
        
        df = self.df_42840[["level1_val", "level2_val"]].copy()
        df["scales"] = scales
        
        df.sort_values(["level1_val", "level2_val"], inplace=True)
        df.reset_index(drop=True,inplace=True)
        
        return df


# In[ ]:





# # Weighted Scaled Pinball Loss

# In[ ]:


class WeightedScaledPinballLoss:
        """A fast routine for the  Weighted Scaled Pinball Loss (WSPL).

This might be slow (one to two minutes) at initialisation in order to initiate all the routines required 
to accelerate on-the-fly WSPL computation.
"""
        
        def __init__(self, scales, weights, data=None, qs=None):
            self.set_weights_and_scales(weights = weights, scales =  scales)
            self._scales = self.weights_and_scales["scales"].values
            self._weights = self.weights_and_scales["weights"].values
            
            self.qs = np.array([0.005, 0.025, 0.165, 0.25 , 0.5  , 0.75 , 0.835, 0.975, 0.995])                                                        if qs is None else np.array(qs)
            
            if data is None:
                config = M5Config(sales_path=DATA_ROOT/"sales_train_evaluation.csv", start = 1914, end=1914+27,
                                        read_calendar=False,read_prices=False,read_sample_submission=False)
                data = M5Data(config)
                data.sales["id"] = data.sales["id"].str.replace("evaluation", "validation")
                
            self.data = data
            
            df_42840 = data.to_42840()

            df_42840["id_no_q"] = df_42840["level1_val"].str.cat(df_42840["level2_val"], 
                                                                 sep="_").str.cat(["_validation"]*len(df_42840))
            df_42840 = df_42840.sort_values("id_no_q").reset_index(drop=True)

            self.df_42840 = df_42840[self.data.config.days].values
            
            self.submission_config = M5Config(cat_cols = ["id"], days = [f"F{i}" for  i in range(1,29)],
                                read_calendar=False,read_prices=False,read_sample_submission=False)
            
            
        def set_weights_and_scales(self, weights, scales):
            weights = weights.copy()
            weights = weights.merge(scales,on=["level1_val", "level2_val"])
            weights["id_no_q"] = weights["level1_val"].str.cat(weights["level2_val"], 
                                                               sep="_").str.cat(["_validation"]*len(weights))
            weights = weights.sort_values("id_no_q").reset_index(drop=True)
            
            self.weights_and_scales = weights
                
            
        def score(self, y_pred ):
            """Compute the WSPL.
            
            Parameters:
            -----------
            y_true: pd.DataFrame, Path,str-path 
                pd.DataFram or path to a pd.DataFrame that consists of daily 42840x28 evaluation data.
                This dataframe must includes the 'id' column. 
                
            y_pred: pd.DataFrame, Path,str-path 
                pd.DataFram or path to a pd.DataFrame that consists of daily 4(2840x9)x28 prediction data.
                This dataframe must includes the 'id' column.
            """
            y_true = self.df_42840
            y_pred = self.get_sub_data(y_pred)
            assert (9, *y_true.shape) == y_pred.shape
            
            diff =   y_true[None] - y_pred
            diff = np.maximum(diff*self.qs[:, None, None], diff*(self.qs[:, None, None]-1)).mean(2)
            wspl = np.sum(diff*self._weights[None]/self._scales[None])/(9*12)
            
            return wspl
        
        def get_sub_data(self, data):
            
            sales = data
        
            if isinstance(sales, (str, Path)):
                sales = pd.read_csv(Path(sales).as_posix())
            else:
                assert isinstance(sales, pd.DataFrame)
                
            sales = sales[sales["id"].str.endswith("validation")].copy()
            
            if not "_q_" in sales.columns:
                sales["_q_"] = sales["id"].str.extract(r"_(\d+\.\d+)_").astype(float)
            if not "id_no_q" in sales.columns:
                sales["id_no_q"] = sales["id"].str.replace(r"_(\d+\.\d+)_", "_")

            sales.sort_values(["_q_", "id_no_q"], inplace=True)
            
            sales = sales[[f"F{i}" for i in range(1, 29)]].values.reshape((9, 42840, 28 ))

            return sales


# # Some tests for our loss API

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nwsc = WeightAndScaleComputer()\nwspl_scales = wsc.get_scales(kind="mae")\nweights = wsc.get_weights()\nwspl_scales.shape, weights.shape')


# In[ ]:


weights[(weights.level1_name=="state_id")&(weights.level2_name=="cat_id")]


# ### Let's test it !
# Now we can give a real-world test to our API by using [this best uncertainty public kernels dataset](https://www.kaggle.com/kneroma/m5-uncertainty-best-public-lbs). Please upvote the dataset to make it more visible for all.

# In[ ]:





# In[ ]:


WSPL =  WeightedScaledPinballLoss(scales = wspl_scales, weights=weights )


# In[ ]:


get_ipython().system(' ls ../input/m5-uncertainty-best-public-lbs')


# ### My older submission (public kernel)

# In[ ]:


kkiller_017921 = pd.read_csv("../input/m5-uncertainty-best-public-lbs/Kkiller_FromPointToUncertainty_017921.csv")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nWSPL.score(kkiller_017921)')


# In[ ]:





# ###  Ulrich's one

# In[ ]:


ulrich_012565 = pd.read_csv("../input/m5-uncertainty-best-public-lbs/Ulrich_QuantileRegressionWithKeras_012565.csv")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nWSPL.score(ulrich_012565)')


# ### Krisztiansz's one

# In[ ]:


krisztiansz_015905 = pd.read_csv("../input/m5-uncertainty-best-public-lbs/"+
                                 "KrisztianSz_PointToUncertaintyDifferentRangesPerLevel_015905.csv")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nWSPL.score(krisztiansz_015905)')


# In[ ]:





# # Notes
# * Even if I put a lot of efforts making the evaluation function faster, there could still some room for improvements so your contribs' would be highly appreciated
# * There still some tiny differences between our API results and the Kaggle's one. I'm on it !
# * I'm hard-working currently to port this implementation on Pytorch/Tensorflow in order to be able to use it directly as loss function. Stay tuned !

# <p style="text-align:center;size:1.2rem;weight:200">Thank you for reading my Kernel through the end!</p>

# <h1 style="color:pink;text-align:center">Good Luck Folks</h1>

# In[ ]:




