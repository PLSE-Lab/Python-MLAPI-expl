#!/usr/bin/env python
# coding: utf-8

# Check the original I forked

# In[ ]:


get_ipython().system('pip install fastai2 --quiet')


# In[ ]:


from fastai2.basics import *
from fastai2.tabular.all import *

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Load input data
# 
# No additional metadata here - let the model learn everything from scratch from competition data

# In[ ]:


PATH = '/kaggle/input/covid19-global-forecasting-week-5/'
train_df = pd.read_csv(PATH + 'train.csv', parse_dates=['Date'])
test_df = pd.read_csv(PATH + 'test.csv', parse_dates=['Date'])
example_submit = pd.read_csv(PATH + 'submission.csv')


# In[ ]:


assert test_df.shape[0] * 3 == example_submit.shape[0]


# In[ ]:


loc_key = ['Country_Region', 'Province_State', 'County']
day_key = loc_key + ['Date']

def fill_unknown_state(df):
    df.fillna({x : 'Unknown' for x in day_key}, inplace=True)
    
for d in [train_df, test_df]:
    fill_unknown_state(d)


# # Train data filtering
# 
# Just a random guess - based on the disease history and having much more accurate data from March and April, targets before February the 20th are probably a total garbage.

# In[ ]:


def with_time_range(df, min_date):
    return df[df['Date'] >= min_date]

MIN_DATE = '2020-02-20'

train_df = with_time_range(train_df, MIN_DATE)
test_df = with_time_range(test_df, MIN_DATE)


# # Input DataFrame compression
# 
# For each entity: (Place, Day) where Place is determined by `['Country_Region', 'Province_State', 'County']` we want to have one row instead of two in the original DataFrame, preserving target indices in additional columns - that's what we do in the merging function below

# In[ ]:


def get_target(target_name):
    def get_cases(df):
        res = df[df.Target == target_name]
        return res.drop(columns=['Target']).rename({'TargetValue': target_name}, axis=1)
    return get_cases

get_cases = get_target('ConfirmedCases')
get_fatalities = get_target('Fatalities')

def cases_fatalities_merged(df):
    cases = get_cases(df)
    fats = get_fatalities(df).drop(columns=['Population'])
    
    cases = pd.merge(cases, fats, how='left', on=day_key, suffixes=('_c', '_f'))
    return cases
    

train_df = cases_fatalities_merged(train_df)
test_df = cases_fatalities_merged(test_df)


# # Add temporal features
# 
# Some basic features like number of days since the first case in each country/province with analogous feature for 20 cases may be particularly worth adding.

# In[ ]:


def day_reached_cases(df, name, no_cases=1):
    """For each country/province get first day of year with at least given number of cases."""
    gb = df[df['ConfirmedCases'] >= no_cases].groupby(loc_key)
    return gb.Dayofyear.first().reset_index().rename(columns={'Dayofyear': name})


# In[ ]:


def additional_features(df):
    add_datepart(df, 'Date', drop=False)
    first_nonzero = day_reached_cases(train_df, 'FirstCaseDay', 1)
    first_twenty = day_reached_cases(train_df, 'First20CasesDay', 20)
    
    df = pd.merge(df, first_nonzero, how='left')
    df = pd.merge(df, first_twenty, how='left')
    
    df['DaysSinceFirst'] = df['Dayofyear'] - df['FirstCaseDay']
    df['DaysSince20'] = df['Dayofyear'] - df['First20CasesDay']
    return df


# In[ ]:


train_df_final = additional_features(train_df)
test_df_final = additional_features(test_df)


# # Feature selection (both for ConfirmedCases and Fatalities)
# 
# In fast.ai we can easily select categorical and continuous variables for training.
# 
# I decided not to choose any external data in baseline model. Adding numerical values from country data provided in this notebook doesn't seem to improve the validation score much.
# 
# **Avoiding leakage is also very important! MAX_TRAIN_DATE is a global constant indicating minimum date of our test set, and should be used to limit rows in our train set to prevent leaky modeling.**

# In[ ]:


cat_vars = [
    'Country_Region', 'Province_State', 'County', 
]

cont_vars = [
    'DaysSinceFirst', 'DaysSince20', 'Dayofyear', 'Dayofweek',
    'Population'
]

MAX_TRAIN_DATE = test_df.Date.min()
print(MAX_TRAIN_DATE)


# # Custom loss functions
# 
# One of the most important parts of this notebook - neither fast.ai nor PyTorch have builtin losses for quantile regression, so we have to define ones on our own.
# 
# Should you notice any mistake in the code in the cell below, please write a comment since I don't feel pretty familiar with quantile regression yet.
# 
# `QuantileLossL1` function is a standard loss which, for a given quantile `q` depends on `q * abs(target - pred)`. However, it is well known that neural nets don't
# optimize well on linear functions, that's why I tried to create another loss function based on this one - `QuantileLossL2`. 
# 
# The function works as follows: if the predicted `pred` is smaller than the actual `target`, we return `(target - pred)**2 / (1 - q)`. Else we return simply
# `(target - pred)**2 / q`. I'm not sure whether this function is actually any good, it just seemed intuitively to be something reasonable and analogous to the original
# `QuantileLossL1`. 

# In[ ]:


import pdb

quants = [.05, .5, .95]

class QuantileLossL1(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = target[:, 0] - preds[:, i]

            losses.append(
                torch.max(
                   (q-1) * errors, 
                   q * errors
            ).unsqueeze(1))
            
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

    
class QuantileLossL2(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = target[:, 0] - preds[:, i]
            err_sq = errors**2
            less_than_target = errors > 0
            err_sq[less_than_target] /= (1 - q)
            err_sq[~less_than_target] /= q

            losses.append(err_sq.unsqueeze(1))

        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


# # Custom metrics - pinball
# 
# Just an attempt to implement weighted pinball loss - something is probably wrong here, but nevertheless it gives us some information in spite of being pretty different than this one from the leaderboard.

# In[ ]:


def pinball(preds, target):
    assert preds.size(0) == target.size(0)
    target_vals = target[:, 0]
    target_weights = target[:, 1]
    
    losses = []

    for i, q in enumerate(quants):
        errors = (target_vals - preds[:, i]) * target_weights
        losses.append(
            torch.max(
               (q-1) * errors, 
               q * errors
            ).unsqueeze(1)
        )

    return torch.mean(
        torch.mean(torch.cat(losses, dim=1), dim=1)
    )

def interval(preds, target):
    assert preds.size(0) == target.size(0)
    target_vals = target[:, 0]
    pred_mins = preds[:, 0]
    pred_maxs = preds[:, 2]
    
    goods1 = pred_mins <= target_vals
    goods2 = target_vals <= pred_maxs
    
    return torch.sum(goods1 & goods2).item() / target.size(0)


# # Predictor class
# 
# A convenience class to make predicting 2 values (cases and fatalities) easier - it contains all steps needed to initialize a fast.ai model, from input dataframes to predictions.

# In[ ]:


class Predictor():
    def __init__(self, train_df, test_df, target_colname, weight_colname,
                 categoricals=cat_vars, continuous=cont_vars,
                 max_train_date=MAX_TRAIN_DATE,
                 batch_size=1024):
        self._target_colname = target_colname
        self._weight_colname = weight_colname
        
        self._cat_vars = categoricals
        self._cont_vars = continuous
        self._dep_var = [target_colname, weight_colname]
        
        self._train_df = self._train_df_processed(train_df)
        
        self._MAX_TRAIN_IDX = self._train_df[self._train_df['Date'] < max_train_date].shape[0]
        self._df_wrapper = self._prepare_df_wrapper(self._train_df)
        
        self._path = '/kaggle/working/'
        self._dls = self._df_wrapper.dataloaders(bs=batch_size, path=self._path)
        
        self._dls.c = len(quants) # Number of outputs of our network is number of quantiles to be predicted.
        
        self._learn = tabular_learner(self._dls, layers=[1000, 500, 250],
                        opt_func=ranger, loss_func=QuantileLossL1(quants), metrics=[interval])
        
        self._test_dls = self._prepare_test_dl(test_df)
        
    def _train_df_processed(self, train_df):
        df = train_df[self._cont_vars + self._cat_vars + self._dep_var + ['Date']].copy().sort_values('Date')
        df = df[df[self._target_colname] >= 0] # Filter negatives - bugs in dataset
        df[self._target_colname] = np.log1p(df[self._target_colname])
        return df
    
    def _prepare_df_wrapper(self, train_df_processed):
        procs=[FillMissing, Categorify, Normalize]

        splits = list(range(self._MAX_TRAIN_IDX)), (list(range(self._MAX_TRAIN_IDX, len(train_df_processed))))

        to = TabularPandas(train_df_processed, procs,self._cat_vars.copy(), self._cont_vars.copy(), self._dep_var,y_block=TransformBlock(), splits=splits)
        return to
    
    def _prepare_test_dl(self, test_df_raw):
        to_tst = self._df_wrapper.new(test_df_raw)
        to_tst.process()
        return self._dls.valid.new(to_tst)
        
        
    @property
    def learn(self):
        return self._learn
    
    def predict(self) -> np.ndarray:
        tst_preds,_ = self._learn.get_preds(dl=self._test_dls)
        tst_preds = tst_preds.data.numpy()
        return np.expm1(tst_preds)
    
    def lc(self):
        emb_szs = get_emb_sz(self._df_wrapper); print(emb_szs)
        self._dls.show_batch()
        self._test_dls.show_batch()


# # Training
# 
# As we can see, our predictor class is pretty flexible (although probably not equally open to extensions, but that doesn't matter here). To train we simply use `learn` property of the `Predictor` objects, to get a fast.ai Learner.

# In[ ]:


train_df_final


# In[ ]:


blackbox_cases = Predictor(train_df_final, test_df_final, 'ConfirmedCases', 'Weight_c')
blackbox_fats = Predictor(train_df_final, test_df_final, 'Fatalities', 'Weight_f')


# # Loss function
# 
# The L1-like loss function is actually not that bad, we can stick with it.

# In[ ]:


blackbox_cases.learn.lr_find()


# In[ ]:


blackbox_cases.learn.fit_one_cycle(20, lr_max=0.004)


# In[ ]:


blackbox_fats.learn.lr_find()


# In[ ]:


blackbox_fats.learn.fit_one_cycle(20, lr_max=0.002)


# # Getting predictions
# 
# With our `Predictor` class, getting test set prediction values as numpy arrays is extremely simple.

# In[ ]:


pred_cases = blackbox_cases.predict()
pred_fats = blackbox_fats.predict()


# # Quick look at predicted values
# 
# We take a look at predicted cases and fatalities, just to make sure our predictions make any sense.

# In[ ]:


print(pred_cases.mean(axis=0), pred_fats.mean(axis=0), pred_cases.std(axis=0), pred_fats.std(axis=0))


# # Preparing submission
# 
# The submission format is pretty weird, so it requires some numpy/pandas code to prepare it from our predicted targets.

# In[ ]:


def prepare_submission_target(test_df, model_preds, forecast_id_col, submit_df):
    res = submit_df.copy()
    assert(len(test_df) == model_preds.shape[0])
    tmp_target = res.TargetValue.copy()
  
    preds_flattened = model_preds.flatten()
    
    indices = 3 * np.repeat(test_df[forecast_id_col].to_numpy() - 1, 3)
    indices += np.tile(np.array([0,1,2]), len(test_df))
    
    tmp_target.loc[indices] = preds_flattened
    res.TargetValue = tmp_target
    return res


# In[ ]:


def prepare_submission(cases_preds, fatality_preds):
    submit = prepare_submission_target(test_df_final, cases_preds, 'ForecastId_c', example_submit)
    submit = prepare_submission_target(test_df_final, fatality_preds, 'ForecastId_f', submit)
    return submit


# In[ ]:


submit = prepare_submission(pred_cases, pred_fats)


# In[ ]:


submit.to_csv('submission.csv', index=False)

