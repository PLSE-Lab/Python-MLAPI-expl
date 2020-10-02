#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# The goal of this notebook was to benchmark some epidemeological models across a range of evaluation metrics. This was to get a more robust sense for how the Kaggle models perform against well known epidemiological models. 
# 
# For now I have only benchmarked the leading Kaggle model. The week 3 and week 4 leading model clearly wins against IHME and LANL on RMSLE (the loss function it was evaluated against). But the advantage goes away when evaluated on other loss functions like MAE and RMSE.
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

cohort = {}


# ## Week 4

# In[ ]:



cohort['2020-04-16'] = dict()
cohort['2020-04-16']['solution_file'] = '/kaggle/input/covid19-epidemiological-benchmarking-dataset/Benchmarking Data/2020_04_16/solution_20200426_derived.csv'
cohort['2020-04-16']['test_file'] = '/kaggle/input/covid19-epidemiological-benchmarking-dataset/Benchmarking Data/2020_04_16/test.csv'


cohort['2020-04-16']['kaggle_leader'] = '/kaggle/input/covid19-epidemiological-benchmarking-dataset/Benchmarking Data/2020_04_16/Kaggle/15210154 - Pub_0_50424 Priv_0_22123.csv'
cohort['2020-04-16']['cpmp'] = '/kaggle/input/cpmpextraday/submission.csv'

cohort['2020-04-16']['ihme'] = '/kaggle/input/covid19-epidemiological-benchmarking-dataset/Benchmarking Data/IHME/2020_04_16.05/Hospitalization_all_locs.csv'
cohort['2020-04-16']['lanl'] = '/kaggle/input/covid19-epidemiological-benchmarking-dataset/Benchmarking Data/LANL/2020-04-15_deaths_quantiles_us.csv'


# # Load in solution

# In[ ]:


def load_solution(solution_file,test_file):
    df_benchmark_panel = pd.read_csv(solution_file,index_col=0)
    df_test = pd.read_csv(test_file,index_col=0)
    df_benchmark_panel = df_benchmark_panel.merge(df_test,left_index=True,right_index=True)
    return df_benchmark_panel


# # Merge Lead Kaggle model
# - Optimized for RMSLE and for predicing cases and fatalities (Disadvantage)
# - Optimized for global predictions (Disadvantage)
# - We're picking winning model ex-post. No way we would have known this model would have won ex-ante (Advantage)
# 
# 

# In[ ]:


def load_kaggle(submission_file,df_benchmark_panel):
    df_kaggle = pd.read_csv(submission_file,index_col=0)
    df_kaggle = df_kaggle.rename(columns={"Fatalities": "kaggle_leader"})
    df_benchmark_panel = df_benchmark_panel.merge(df_kaggle[['kaggle_leader']],left_index=True,right_index=True)
    return df_benchmark_panel

def load_cpmp(submission_file,df_benchmark_panel):
    df_kaggle = pd.read_csv(submission_file,index_col=0)
    df_kaggle = df_kaggle.rename(columns={"Fatalities": "cpmp"})
    df_benchmark_panel = df_benchmark_panel.merge(df_kaggle[['cpmp']],left_index=True,right_index=True)
    return df_benchmark_panel


# # Merge IHME model
# 
# - For April 10 model (Favors)
# - Not sure what loss function optimized for (Might Disadvantage)
# 

# In[ ]:


def load_ihme(ihme_file,df_benchmark_panel):
    df_ihme = pd.read_csv(ihme_file,index_col=[0])
    for location in df_benchmark_panel['Province_State'].unique():
        df_ihme.loc[(df_ihme['location_name'] == location),'ihme'] = df_ihme[df_ihme['location_name'] == location]['deaths_mean'].cumsum()
    
    df_benchmark_panel = df_benchmark_panel.merge(df_ihme[['location_name','date','ihme']],left_on=['Province_State','Date'],right_on=['location_name','date'],how='inner')

    del(df_benchmark_panel['location_name'])
    del(df_benchmark_panel['date'])
    return df_benchmark_panel


# # Merge LANL model
# 
# - April 8 forecast (disadvantage)
# 

# In[ ]:


def load_lanl(lanl_file,df_benchmark_panel):
    df_lanl = pd.read_csv(lanl_file)
    df_lanl = df_lanl.rename(columns={"q.50": "lanl"})
    df_benchmark_panel = df_benchmark_panel.merge(df_lanl[['state','dates','lanl']],left_on=['Province_State','Date'],right_on=['state','dates'],how='inner')
    del(df_benchmark_panel['state'])
    del(df_benchmark_panel['dates'])
    return df_benchmark_panel


# # Evaluate

# In[ ]:


def evaluate_models(evaluate, df_benchmark_panel):

    df_eval = df_benchmark_panel[(df_benchmark_panel['Fatalities'] >= 0) & (df_benchmark_panel['Usage'] == 'Private') 
                                 & (df_benchmark_panel['Province_State'] != 'New York') & (df_benchmark_panel['Province_State'] != 'New Jersey')].reset_index(drop=True)
    for model in evaluate:
        df_eval['%s_mae' % model] = np.abs(df_eval[model] - df_eval['Fatalities'])
        df_eval['%s_rmsle'  % model] = np.abs(np.log(1+df_eval[model])-np.log(1+df_eval["Fatalities"]))
        
    results = dict()
    for model in evaluate:
        results[model] = dict()
        results[model]['MAE'] = ((df_eval[model] - df_eval['Fatalities']).abs().mean())
        results[model]['RMSE'] = np.sqrt(((df_eval[model]-df_eval["Fatalities"])**2).mean())
        results[model]['RMSLE'] = np.sqrt(((np.log(1+df_eval[model])-np.log(1+df_eval["Fatalities"]))**2).mean())
    
    
    df_eval.to_csv('/kaggle/working/benchmark.csv')
    return (results, df_eval)
    
    
    


# In[ ]:


df_benchmarks = pd.DataFrame(columns=['Model','Forecast_Date','MAE','RMSE','RMSLE'])

for forecast_date in ['2020-04-16']:
    
    
    df_benchmark_panel = load_solution(cohort[forecast_date]['solution_file'],cohort[forecast_date]['test_file'])
    df_benchmark_panel = load_kaggle(cohort[forecast_date]['kaggle_leader'],df_benchmark_panel)
    df_benchmark_panel = load_cpmp(cohort[forecast_date]['cpmp'],df_benchmark_panel)
    df_benchmark_panel = load_ihme(cohort[forecast_date]['ihme'],df_benchmark_panel)
    df_benchmark_panel = load_lanl(cohort[forecast_date]['lanl'],df_benchmark_panel)
    results, df_eval = evaluate_models(['kaggle_leader', 'cpmp', 'ihme','lanl'], df_benchmark_panel)
    
    for model in results: 
        
        row_dict = {'Model': model,'Forecast_Date':forecast_date,'MAE':np.round(results[model]['MAE'],0),'RMSE':np.round(results[model]['RMSE'],1), 'RMSLE': np.round(results[model]['RMSLE'],3) } 
        
        df_benchmarks = df_benchmarks.append(row_dict,ignore_index=True)
        


# In[ ]:


df_benchmarks


# In[ ]:


df = df_eval[['Province_State', 'Fatalities', 'kaggle_leader', 'cpmp', 'ihme', 'kaggle_leader_mae', 'cpmp_mae', 'ihme_mae', 'lanl_mae']
             ].groupby('Province_State').last().reset_index()
df.sort_values(by='Fatalities', ascending=False)


# ## Possible Next Steps
# 
# - test Kaggle ensembles
# - benchmark all global
# - benchmark confirmed cases
# - benchmark more models
# - add additional metrics
# 
# 
