#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Exponential curve fitting exercise for basic prediction of COVID19 - week4 kaggle
#Try to learn from countries that passed the infection peak for duration
#For fatalities rate use country demographics to predict and check for feature importance - used xgboost/elastic net
#Analysis is based on daily new cases curve, normalized by population and smoothed before extracting parameters

import os
import pandas as pd
import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import curve_fit
from scipy.stats import linregress
from datetime import timedelta
from sklearn.metrics import mean_squared_log_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce
import xgboost
from catboost import Pool, CatBoostRegressor

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


EVAL_MODE = False #set to false for full analysis of train data without splitting for eval data


# In[ ]:


df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
df_test  = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
df_sub   = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')


# In[ ]:


df_train['Date'] = pd.to_datetime(df_train['Date'])
df_test['Date']  = pd.to_datetime(df_test['Date'])


# In[ ]:


df_train.tail(3)


# In[ ]:


#prevent NA rows dissapear in grouby operations
df_train['Province_State'].fillna(' ',inplace=True)
df_test['Province_State'].fillna(' ',inplace=True)


# In[ ]:


df_test.head(3)


# In[ ]:


df_sub.head(3)


# In[ ]:


#calculate daily additions to confirmed cases and fatalities in absolute values and percentages
by_ctry_prov = df_train.groupby(['Country_Region','Province_State'])[['ConfirmedCases','Fatalities']]
df_train[['NewCases','NewFatalities']]= by_ctry_prov.transform(lambda x: x.diff().fillna(0))
df_train[['NewCasesPct','NewFatalitiesPct']]= by_ctry_prov.transform(lambda x: x.pct_change().fillna(0))


# In[ ]:


df_train.sort_values('NewCases',ascending = False).head()


# In[ ]:


#check for inconsistencies in daily new cases, cumulative count should only increase
df_train[df_train.NewCases < 0].sort_values('NewCases')


# In[ ]:


#check for inconsistencies in daily new fatalities, cumulative count should only increase
df_train[df_train.NewFatalities < 0].sort_values('NewFatalities')


# In[ ]:


#more deaths than confirmed cases
df_train[df_train.Fatalities > df_train.ConfirmedCases]


# In[ ]:


#more than 40% increase in ConfirmedCases with at least 1000 new cases - Hubei 13 Feb example
df_train[(df_train.NewCasesPct > 0.4) & (df_train.NewCases > 1000)]


# In[ ]:


#more than 80% increase in Fatalities with at least 50 new cases
df_train[(df_train.NewFatalitiesPct > 0.8) & (df_train.NewFatalities > 50)]


# In[ ]:


def lin_interpolate(indx1,indx2,col,df):
    start_val = df.loc[indx1,col]
    end_val = df.loc[indx2,col]
    i = indx1+1
    while i < indx2:
        df.loc[i,col] = start_val + int((i-indx1)/(indx2-indx1)*(end_val-start_val))
        i+=1


# In[ ]:


df_train[(df_train['Country_Region']=='China') & 
         (df_train['Province_State']=='Shandong') &
         (df_train.Date > '2020-02-18')].head(5)


# In[ ]:


#Hubei 13 Feb
df_train[(df_train['Country_Region']=='China') & 
         (df_train['Province_State']=='Hubei') &
         (df_train.Date > '2020-02-8')].head(8)


# In[ ]:


#fix China Hubei 12-14 Feb reporting and Shandong 21 Feb
#need to check indexes don't change every new training set
#decided to fix on new cases only
lin_interpolate(5647,5652,'NewCases',df_train) #Hubei 12-14 Feb
lin_interpolate(6581,6583,'NewCases',df_train) #Shandong 21 Feb


# In[ ]:


df_train['Fatalities'] = df_train.groupby(['Province_State','Country_Region']).Fatalities.cummax()
df_train['NewCases'] = df_train.NewCases.clip(0)


# In[ ]:


all_ctry_lst = list(df_train.set_index(['Country_Region','Province_State']).index.unique())
len(all_ctry_lst)


# In[ ]:


df_train['Country_Region'].nunique()


# In[ ]:


c = df_train[df_train['Province_State'] != 'UniqueProvince'].groupby('Country_Region')['Province_State'].unique()
[(x,list(c.loc[x])) for x in c.index]


# In[ ]:


EVAL_DATE       = pd.to_datetime('4/1/2020')
if EVAL_MODE:
    TRAIN_LAST_DATE = EVAL_DATE
else:
    TRAIN_LAST_DATE = df_train.Date.max()
PRED_LAST_DATE  = df_test.Date.max()


# In[ ]:


df_train.drop(['Id','NewCasesPct','NewFatalitiesPct'],axis=1,inplace=True)


# In[ ]:


#remove data for evaluation
df_eval  = df_train[df_train['Date'] >  EVAL_DATE]
df_train = df_train[df_train['Date'] <= TRAIN_LAST_DATE]


# In[ ]:


#add ForecastId column
df_eval = df_eval.merge(df_test,on=['Date', 'Country_Region', 'Province_State'],how='left',validate='1:1')
df_eval


# In[ ]:


df_corona = df_train[['Province_State','Country_Region','Date','ConfirmedCases','Fatalities','NewCases']].copy()


# In[ ]:


#population per province dataset from kaggle
df_pop = pd.read_csv('/kaggle/input/covid19-global-forecasting-locations-population/locations_population.csv')


# In[ ]:


df_pop['Province.State'].fillna(' ',inplace=True)
df_pop.drop(labels=['Provenance'],axis=1,inplace=True)
df_pop


# In[ ]:


#df_pop['Country.Region'].unique()


# In[ ]:


#df_pop[df_pop['Country.Region']=='South Sudan']


# In[ ]:


#df_corona[df_corona['Country_Region']=='Botwsana'].Province_State.unique()


# In[ ]:


a = set(df_corona['Country_Region'])


# In[ ]:


b = set(df_pop['Country.Region'])


# In[ ]:


a.difference(b) #issues with country names between the files


# In[ ]:


df_corona = df_corona.merge(df_pop,"left",left_on=['Country_Region','Province_State'],
                            right_on=['Country.Region','Province.State'],indicator=True,validate='many_to_one')


# In[ ]:


df_corona._merge.value_counts()


# In[ ]:


#df_corona[df_corona._merge == 'left_only']


# In[ ]:


df_corona['new_cases_per_pop']=df_corona['NewCases']/df_corona['Population']*100000
df_corona.drop(['_merge'],axis=1,inplace=True)
df_corona


# In[ ]:


#check distribution of known total cases per country/province
df_corona.groupby(['Country_Region','Province_State']).ConfirmedCases.max().describe()


# In[ ]:


len(df_corona.groupby(['Country_Region','Province_State'])) #number of countries/provinces in dataset


# In[ ]:


#df_corona.Country_Region.unique()


# In[ ]:


def filter_out(x):
    f1 = x['ConfirmedCases'].max() > 30      #at least 30 sick cases
    f2 = x['Population'].min() > 2000000     #pop more than 2M, small countries with distorted percent of confirmed cases
    f3 = x['new_cases_per_pop'].max() > 0.1  #at least 0.1 cases per 100K pop
    return f1 & f2 & f3


# In[ ]:


#create reduced filtered dataset for analysis
df_analysis = df_corona.groupby(['Province_State','Country_Region']).filter(filter_out).copy()
len(df_analysis.groupby(['Country_Region','Province_State'])) #number of countries in dataset


# In[ ]:


#df_analysis['Country_Region'].unique()


# In[ ]:


#for each country/province, filter out all dates before there are at least 5 new daily cases
df_filt = pd.DataFrame(columns=df_analysis.columns)
for grp in df_analysis.groupby(['Province_State','Country_Region']):
    start_indx = grp[1].loc[grp[1].NewCases >= 5,:].index
    if len(start_indx) > 0:
        df_filt = pd.concat([df_filt,grp[1].loc[start_indx[0]:,:]])


# In[ ]:


#remove countries with less than 5 days of meaningful new daily cases
df_filt = df_filt.groupby(['Province_State','Country_Region']).filter(lambda x: len(x.Date) > 4)


# In[ ]:


len(df_filt.groupby(['Country_Region','Province_State'])) #final number of countries/provinces after filtering


# In[ ]:


#df_filt['Country_Region'].unique()


# In[ ]:


ctry_prov_lst = list(df_filt.set_index(['Country_Region','Province_State']).index.unique())


# In[ ]:


def country_prov_select(cntry,prov,df):
    df_sel = df[(df['Country_Region'] == cntry) & (df['Province_State'] == prov)].reset_index(drop=True)
    #consistency check, fails for China after fixing 13/2 reports
    #assert(all(df_sel.ConfirmedCases == df_sel.NewCases.cumsum()))
    return df_sel.fillna(0)


# In[ ]:


# fig, ax = plt.subplots()
# df_filt.groupby(['Province_State','Country_Region']).plot(y='ConfirmedCases',use_index=False,ax=ax,figsize = (10,6), marker='.',
#                                  legend=False,title='Confirmed cases for each country and region')
# ax.set_xlabel('Days from measurement start')
# ax.set_ylabel('Confirmed cases')
# plt.show()


# In[ ]:


#show maximum daily new cases as a percentage of population, "wall of shame" for country measures :-)
df_filt.groupby(['Province_State','Country_Region']).new_cases_per_pop.max().sort_values(ascending=False)[:30]


# In[ ]:


df_filt = df_filt.set_index(['Province_State','Country_Region'])
df_filt['max_ratio'] = df_filt.groupby(['Province_State','Country_Region']).new_cases_per_pop.max()
df_filt.reset_index(inplace=True)
df_filt.sort_values(by=['max_ratio'],ascending=False,inplace=True,kind='mergesort') #mergesort is stable


# In[ ]:


# #show graphs of daily new cases in batches of 10 to be able to see difference, sorted by severity
# i = 0
# for ctry,grp in df_filt.groupby(['Province_State','Country_Region'],sort=False):
#     if i%10==0: #new plot for group
#         fig, ax = plt.subplots()
#         ax.set_xlabel('Days from measurement start')
#         ax.set_ylabel('New daily cases per 100K')
#     grp.plot(y='new_cases_per_pop',use_index=False, ax=ax, figsize=(10,6), label=ctry[1]+" "+ctry[0], marker='.',
#                                 legend=True,title='New daily cases per 100K for each country/region')
#     i +=1

# plt.show()


# In[ ]:


#candidate functions definition for curve fitting
def exp_func(x,a,b):
    return a*np.exp(b*x)

def poly_func(x,a,b):
    return a*(x**b)

def sigmoid_func(x,a,b,c):
    return a/(1+np.exp(-b*(x-c)))

#sqrt2 = sqrt(2)
#def lognormal_c(x, s, mu, h): # x, sigma, mean, height
#    return h * 0.5 * erfc(- (log(x) - mu) / (s * sqrt2))


# In[ ]:


#loop over all countries/provinces
#check curve fit for exponential and polynomial function for countries which passed the peak of the infection
#uses new cases per day per 100K of the country population as time series values for radial basis function interpolation
#saves peak duration list for countries that seem to be after infection peak and the curve fitting params
peak_duration = {}
param_list = []
fit_list = []
for (ctry,prov) in ctry_prov_lst:
    fig, ax = plt.subplots()
    df_country = country_prov_select(ctry,prov,df_filt)
    plt.plot(df_country.new_cases_per_pop,marker='.')
    rbf=Rbf(df_country.index,df_country.new_cases_per_pop,smooth=5)(df_country.index)
    plt.plot(rbf,marker='+')
    ax.set_title(ctry+" "+prov)
    plt.show()
    peak_indx = np.argmax(rbf)
    peak_date = df_country.Date.iloc[peak_indx]
    #are we at least 5 days after peak of new cases and peak detection 5 days after start?
    peak_reached = (len(df_country) > peak_indx+4) & (peak_indx > 5)
    if peak_reached:
        print("Peak reached for {0} {1} in {2} days on {3}".format(ctry,prov,peak_indx,peak_date))
        peak_duration[(ctry,prov)]=peak_indx
        #exponential fit check
        params1, cov_params1 = curve_fit(exp_func,np.arange(peak_indx),rbf[:peak_indx],maxfev=10000,p0=[0.1,0.4])
        print("\nExponential fit parameters: a:{0:.4f} b:{1:.2f}".format(params1[0],params1[1]))
        print("Covariance matrix of parameters:")
        print(cov_params1)
        param_list.append(params1)
        fit_list.append(np.diag(cov_params1).sum())
        #polynomial fit check
        #params2, cov_params2 = curve_fit(poly_func,np.arange(peak_indx),rbf[:peak_indx],maxfev=10000,p0=[0.01,0.5])
        #print("\nPolynomial fit parameters: a:{0:.4f} b:{1:.2f}".format(params2[0],params2[1]))
        #print("Covariance matrix of parameters:")
        #print(cov_params2)
        #sigmoid fit check
        #params3, cov_params3 = curve_fit(sigmoid_func,np.arange(peak_indx)+0.1,rbf[:peak_indx],maxfev=10000)
        #print("\nSigmoid fit parameters: a:{0:.2f} b:{1:.2f} c:{2:.2f}".format(params3[0],params3[1],params3[2]))
        #print("Covariance matrix of parameters:")
        #print(cov_params3) 
        #rough check for goodness of fit
        #if np.diag(cov_params1).sum() > np.diag(cov_params2).sum():
        #    print("\nPolynomial fit seems better.")
        #else:
        #    print("\nExponential fit seems better.")
    else:
        print("Peak not yet reached for {0} {1}".format(ctry,prov))


# In[ ]:


peak_duration


# In[ ]:


pd.DataFrame(peak_duration.values(),columns=(['duration'])).describe()


# In[ ]:


peaked_countries = list(peak_duration.keys())


# In[ ]:


param_list #curve fitting result parameters


# In[ ]:


pd.DataFrame({'Duration':list(peak_duration.values()),'Params':param_list,'Fit':fit_list})


# In[ ]:


avg_param = np.average(param_list,axis=0)#,weights=fit_list)
avg_param


# In[ ]:


#taking the 50% quantile as average
avg_duration = round(np.quantile(list(peak_duration.values()),0.5))
avg_duration


# In[ ]:


max_duration = round(np.quantile(list(peak_duration.values()),0.8)) #max(list(peak_duration.values()))
max_duration


# In[ ]:


#rough predict based on average/max duration to peak using exponential fitting parameters
FLAT = 5    #guess for number of days remaining at peak TBD optimize based on fitting to peaked countries
MAX_INC = 5 #arbitrary additional days prediction till peak for countries that lasted longer than history

def predict_new_cases(ctry,prov,df):
    df_country = country_prov_select(ctry,prov,df)
    pop_ctry = df_pop[(df_pop['Country.Region']==ctry) & (df_pop['Province.State']==prov)].Population
    pop_factor = int(pop_ctry/100000)
    ctry_dates = df_country[['Date']].copy()
    ctry_confirmed = df_country[['ConfirmedCases']].copy()
    pred_stage = [0]*len(df_country)
    if len(df_country) > 3:
        rbf=Rbf(df_country.index,df_country.new_cases_per_pop,smooth=5)(df_country.index).clip(0)
    else:
        rbf=np.array(df_country.new_cases_per_pop).clip(0)
    peak_indx = np.argmax(rbf)
    peak_date = df_country.Date.iloc[peak_indx]
    #are we at least 5 days after peak of new cases and peak detection 5 days after start?
    peak_reached = (len(df_country) > peak_indx+4) & (peak_indx > 5)
    
    #exponential fitting TBD take parameters fitted from other peaked countries with similar characteristics
    if (peak_indx > 5) and (len(df_country) > 9):
        params, _ = curve_fit(exp_func,np.arange(peak_indx),rbf[:peak_indx],maxfev=10000,p0=[0.1,0.3])
    else:
        params = avg_param
    
    if peak_reached:
        print("Peak reached for {0} {1} in {2} days on {3}".format(ctry,prov,peak_indx,peak_date))
        dur = peak_indx
        if len(df_country) < dur*2+1+FLAT:
            #add predictions to decrease
            for j in range(len(df_country),dur*2+1+FLAT):
                pred = exp_func(2*dur-j+FLAT,params[0],params[1])
                rbf = np.append(rbf,pred)
                ctry_dates = ctry_dates.append(ctry_dates.iloc[-1]+timedelta(1))
                ctry_confirmed = ctry_confirmed.append(ctry_confirmed.iloc[-1]+pred*pop_factor)
                pred_stage.append(3)
    else:
        print("Peak not yet reached for {0} {1}".format(ctry,prov))
        #naive predict duration till new cases per day peak
        if len(df_country) < int(np.ceil(avg_duration)):
            dur = int(np.ceil(avg_duration))
        elif len(df_country) < max_duration:
            dur = int(np.ceil(max_duration))
        else:
            dur = len(df_country)+MAX_INC
            print("Past due peaking for {0} {1}, arbitrary {2} days prediction".format(ctry,prov,MAX_INC))
        slope = linregress(np.arange(len(rbf)),rbf).slope #check smoothed curve is at least increasing
        if (np.mean(df_country.NewCases[-5:])>5) and (df_country.ConfirmedCases.iloc[-1]>30) and (slope > 0.005):
            #add predictions till peak based on estimated function
            for j in range(len(df_country),dur+1):
                pred = exp_func(j,params[0],params[1])
                rbf = np.append(rbf, pred)
                ctry_dates = ctry_dates.append(ctry_dates.iloc[-1]+timedelta(1))
                ctry_confirmed = ctry_confirmed.append(ctry_confirmed.iloc[-1]+pred*pop_factor)
                pred_stage.append(1)
            #add predictions for remaining flat TBD curve fit to peaked countries
            rbf = np.append(rbf,np.ones(FLAT)*rbf[-1])
            for i in np.arange(FLAT):
                ctry_dates = ctry_dates.append(ctry_dates.iloc[-1]+timedelta(1))
                ctry_confirmed = ctry_confirmed.append(ctry_confirmed.iloc[-1]+pred*pop_factor)
                pred_stage.append(2)
            #add predictions to decrease
            for j in range(dur+FLAT,dur*2+1+FLAT):
                pred = exp_func(2*dur-j+FLAT,params[0],params[1])
                rbf = np.append(rbf,pred)
                ctry_dates = ctry_dates.append(ctry_dates.iloc[-1]+timedelta(1))
                ctry_confirmed = ctry_confirmed.append(ctry_confirmed.iloc[-1]+pred*pop_factor)
                pred_stage.append(3)
    
    #just assume constant last new cases per day rate from here lacking any other info
    while ctry_dates['Date'].iloc[-1] < PRED_LAST_DATE:
        rbf = np.append(rbf,np.mean(rbf[-3:]))
        ctry_dates = ctry_dates.append(ctry_dates.iloc[-1]+timedelta(1))
        ctry_confirmed = ctry_confirmed.append(ctry_confirmed.iloc[-1]+rbf[-1]*pop_factor)
        pred_stage.append(4)
    
    cases = rbf * pop_factor
    fig, ax = plt.subplots()
    plt.plot(np.arange(len(df_country)),cases[:len(df_country)],marker='+')
    plt.plot(np.arange(len(df_country),len(cases)),cases[len(df_country):],marker='.')
    ax.set_title(ctry+" "+prov+" predicted new daily cases")
    plt.show()
    print("Total predicted infected for {0} {1}: {2:d}\n".format(ctry,prov,int(ctry_confirmed.iloc[-1])))
    cases_df = pd.DataFrame(cases.astype(int),columns=['NewCases'])
    ctry_df = pd.DataFrame([ctry for i in range(len(cases))],columns=['Country_Region'])
    ctry_confirmed = ctry_confirmed.reset_index(drop=True).astype(int)
    pop_tmp = pd.DataFrame([pop_ctry.values[0] for i in range(len(cases))], columns=['Population'])
    pred_stage = pd.DataFrame(pred_stage,columns=['PredStage'])
    return(pd.concat([ctry_df,ctry_dates.reset_index(drop=True),ctry_confirmed,
                      cases_df,pop_tmp,pred_stage],axis=1))


# In[ ]:


country_prov_select('Israel',' ',df_corona).tail(10)#[:5]#.ConfirmedCases.iloc[-1]


# In[ ]:


ctry = 'Israel'
prov = ' '
pred = predict_new_cases(ctry,prov,df_filt)


# In[ ]:


#pred[20:60]


# In[ ]:


df_last_day_train = df_train.groupby(['Country_Region','Province_State']).last().reset_index().copy()


# In[ ]:


df_history = df_corona[['Province_State','Country_Region','ConfirmedCases','Fatalities','Population']].copy()
df_history


# In[ ]:


BACK_HISTORY = 8
def add_history(df,lag = BACK_HISTORY):
    for i in range(BACK_HISTORY):
        colname = "Confirmed_d" + str(i+1) 
        df[colname] = df.ConfirmedCases.shift(i+1,fill_value=0)


# In[ ]:


add_history(df_history)
df_history = df_history.groupby(['Country_Region','Province_State'],
                                as_index=False).apply(lambda x: x[BACK_HISTORY:]).reset_index(drop=True)
df_history


# In[ ]:


df_history.corr()


# In[ ]:


#country descriptive information from kaggle database
df_ctryinfo = pd.read_csv('/kaggle/input/countryinfo/covid19countryinfo.csv')


# In[ ]:


df_ctryinfo['gdp2019'] = df_ctryinfo.gdp2019.str.replace(",","").astype(float)


# In[ ]:


df_ctryinfo = df_ctryinfo.iloc[:193,:29]
df_ctryinfo = df_ctryinfo[df_ctryinfo.region.isnull()] #keep only main countries
df_ctryinfo


# In[ ]:


df_ctryinfo = df_ctryinfo[['country','density','medianage','urbanpop','hospibed','smokers',
                           'sexratio','lung','femalelung','malelung']].copy()


# In[ ]:


a = set(df_ctryinfo.country)


# In[ ]:


b = set(df_history.Country_Region)


# In[ ]:


diff_set = b-a
diff_set


# In[ ]:


#append country row with NA for all other columns to be imputed later
for i in diff_set:
    df_ctryinfo = df_ctryinfo.append({'country':i},ignore_index=True)


# In[ ]:


#impute missing values for elastic net regressor model
imput = SimpleImputer(strategy='median')
num_df = df_ctryinfo.iloc[:,1:]
df_ctryinfo.iloc[:,1:] = pd.DataFrame(imput.fit_transform(num_df),columns=num_df.columns, index=num_df.index)


# In[ ]:


#df_ctryinfo.isnull().sum()


# In[ ]:


df_hist = df_history.merge(df_ctryinfo,left_on='Country_Region',right_on='country',how='left',indicator=True)
df_hist.drop('country',axis=1,inplace=True)
df_hist


# In[ ]:


df_hist._merge.value_counts()


# In[ ]:


df_hist.drop('_merge',axis =1,inplace=True)


# In[ ]:


#impute missing values for elastic net regressor model
imput = SimpleImputer(strategy='median')
num_df = df_hist.iloc[:,4:]
df_hist.iloc[:,4:] = pd.DataFrame(imput.fit_transform(num_df),columns=num_df.columns, index=num_df.index)


# In[ ]:


c = [c for (c,p) in ctry_prov_lst]
p = [p for (c,p) in ctry_prov_lst]
#create df of countries not in exponential group and filter out all dates before there are at least 5 new daily cases
df_corona_noexp = pd.DataFrame(columns=df_corona.columns)
#countries that were not predicted by exponential fit
df_noexp = df_corona[~(df_corona['Country_Region'].isin(c)) | ~(df_corona['Province_State'].isin(p))]
for grp in df_noexp.groupby(['Province_State','Country_Region']):
    start_indx = grp[1].loc[grp[1].NewCases >= 5,:].index
    if len(start_indx) > 0:
        df_corona_noexp = pd.concat([df_corona_noexp,grp[1].loc[start_indx[0]:,:]])


# In[ ]:


ctry_prov_noexp_lst = list(df_corona_noexp.set_index(['Country_Region','Province_State']).index.unique())
len(ctry_prov_noexp_lst)


# In[ ]:


not_enough_data_ctry_lst = set(all_ctry_lst) - set(ctry_prov_noexp_lst) - set(ctry_prov_lst)
len(not_enough_data_ctry_lst)


# In[ ]:


x = df_hist.iloc[:,2:].drop('Fatalities',axis=1)
y = df_hist[['Fatalities']]


# In[ ]:


#x.isnull().sum()


# In[ ]:


folds_cv = KFold(shuffle=True,n_splits=3)


# In[ ]:


xcb = df_hist.iloc[:,1:].drop('Fatalities',axis=1)
ycb = df_hist[['Fatalities']]
train_pool = Pool(xcb,ycb,cat_features=[0])
#params = [{'iterations':[100,1000],'depth':[4,6], 'learning_rate':[0.001,0.01],'l2_leaf_reg':[3,5]}]
model = CatBoostRegressor() #iterations=2, depth=2, learning_rate=1, loss_function='RMSE')


# In[ ]:


catb_model = CatBoostRegressor()
#iterations = opt_params['iterations'], depth = opt_params['depth'], learning_rate = opt_params['learning_rate'], l2_leaf_reg = opt_params['l2_leaf_reg'])
catb_model.fit(train_pool)


# In[ ]:


test_pool = Pool(xcb,cat_features=[0])
model_preds = np.maximum(0,catb_model.predict(test_pool))
mean_squared_log_error(model_preds,y)


# In[ ]:


pd.DataFrame(catb_model.get_feature_importance(),
             index=xcb.columns,columns=['Importance']).sort_values('Importance',ascending=False)[:10]


# In[ ]:


params = [{'max_iter':[100000,200000],'alpha':[0.5,1], 'l1_ratio':[0.5,0.9]}]
model = ElasticNet()


# In[ ]:


grid_model = GridSearchCV(model, params, cv=folds_cv, verbose=2, n_jobs=-1)
grid_model.fit(x,y)


# In[ ]:


opt_params = grid_model.best_params_
opt_params


# In[ ]:


enet_model = ElasticNet(max_iter = opt_params['max_iter'], alpha = opt_params['alpha'], l1_ratio = opt_params['l1_ratio'])
enet_model.fit(x,y)


# In[ ]:


pd.DataFrame(enet_model.coef_,
             index=x.columns,columns=['Coefficient']).sort_values('Coefficient',ascending=False)[:10]


# In[ ]:


model_preds = np.maximum(0,enet_model.predict(x))
mean_squared_log_error(model_preds,y)


# In[ ]:


params = [{'n_estimators':[100,1000],'max_depth':[3,5], 'learning_rate':[0.01,0.1]}]
model = xgboost.XGBRegressor(objective = 'reg:squarederror')


# In[ ]:


grid_model = GridSearchCV(model, params, cv=folds_cv, verbose=2, n_jobs=-1)
grid_model.fit(x,y)


# In[ ]:


opt_params = grid_model.best_params_
opt_params


# In[ ]:


xgb_model1 = xgboost.XGBRegressor(objective = 'reg:squarederror', colsample_bynode = 0.9, subsample = 0.9,
                                  learning_rate = opt_params['learning_rate'], random_state = 42,
                                  max_depth = opt_params['max_depth'], n_estimators = opt_params['n_estimators'])
xgb_model1.fit(x,y)


# In[ ]:


xgb_model2 = xgboost.XGBRegressor(objective = 'reg:squarederror', colsample_bylevel = 0.9, subsample = 0.9,
                                  learning_rate = opt_params['learning_rate'], random_state = 24,
                                  max_depth = opt_params['max_depth'], n_estimators = opt_params['n_estimators'])
xgb_model2.fit(x,y)


# In[ ]:


pd.DataFrame(xgb_model1.feature_importances_,
             index=x.columns,columns=['Importance']).sort_values('Importance',ascending=False)[:10]


# In[ ]:


model_preds1 = np.maximum(0,(xgb_model1.predict(x)))#+xgb_model2.predict(x))/2)
mean_squared_log_error(model_preds1,y)


# In[ ]:


model_preds2 = np.maximum(0,(xgb_model2.predict(x)))#+xgb_model2.predict(x))/2)
mean_squared_log_error(model_preds2,y)


# In[ ]:


model_preds3 = np.maximum(0,catb_model.predict(test_pool))
mean_squared_log_error(model_preds3,y)


# In[ ]:


model_preds = (model_preds1 + model_preds2 + model_preds3)/3
mean_squared_log_error(model_preds,y)


# In[ ]:


#loop over countries to initialize prediction with last known values
df_sub.set_index('ForecastId',inplace=True)
for i in df_last_day_train.index:
    predc = df_last_day_train.loc[i,'ConfirmedCases']
    predf = df_last_day_train.loc[i,'Fatalities']
    ctry = df_last_day_train.loc[i,'Country_Region']
    prvn = df_last_day_train.loc[i,'Province_State']
    test_id = df_test[(df_test['Country_Region'] == ctry) & (df_test['Province_State'] == prvn)]['ForecastId']
    df_sub.loc[test_id,'ConfirmedCases']=predc
    df_sub.loc[test_id,'Fatalities']=predf
df_sub.reset_index(inplace=True)


# In[ ]:


#df_sub.reset_index(inplace=True)


# In[ ]:


max_confirmed_in_train = df_corona.ConfirmedCases.max()
max_confirmed_in_train


# In[ ]:


def predict_sub(ctry,prov,df):
    rough_correction = 0
    pred = predict_new_cases(ctry,prov,df) #confirmed cases predict by exponential fitting
    add_history(pred)
    pred = pred[(pred['Date'] > TRAIN_LAST_DATE) & (pred['Date'] <= PRED_LAST_DATE)]
    pred = pred.merge(df_ctryinfo,left_on='Country_Region',right_on='country',how='left')
    extrapolate_bool = (pred.ConfirmedCases.values > max_confirmed_in_train)
    df_tmp = pred.drop(['country','NewCases','PredStage'],axis=1).iloc[:,2:]
    # fatalities prediction using fitted models
    df_tmp_cb = pred.drop(['country','NewCases','PredStage','Date'],axis=1)
    fat_predcb  = np.maximum(catb_model.predict(df_tmp_cb),0)
    fat_predgb1 = np.maximum(xgb_model1.predict(df_tmp),0)
    fat_predgb2 = np.maximum(xgb_model2.predict(df_tmp),0)
    fat_pred1 = np.round((fat_predgb1+fat_predgb2+fat_predcb)/3)
    #fat_pred1 = (fat_predgb1+fat_predgb2)/2
    fat_pred2 = np.round(np.maximum(enet_model.predict(df_tmp),0))
    #for extrapolate using elastic net model, use xgboost for values in train
    fat_pred = fat_pred1 * (~extrapolate_bool) + fat_pred2 * extrapolate_bool
    #fat_pred = (fat_pred1+fat_pred2)/2 * (~extrapolate_bool) + fat_pred2 * extrapolate_bool
    #enforce monotonic fatalities
    last_fat = df_last_day_train[(df_last_day_train.Country_Region == ctry) & 
                                 (df_last_day_train.Province_State == prov)].Fatalities.values[0]
    fat_start = np.mean(fat_pred[:3])
    factor = (fat_start+0.001)/(last_fat+0.001)
    add_fat = fat_start - last_fat
    fat_pred[0] = max(fat_pred[0],last_fat)
    fat_pred = np.maximum.accumulate(fat_pred)
    if (add_fat > 1) and (factor > 2): #correct for large gaps up
        fat_pred = [int(x) for x in fat_pred/factor*1.3]
        rough_correction = 1
    test_id = df_test[(df_test['Country_Region'] == ctry) & 
                      (df_test['Province_State'] == prov) &
                      (df_test['Date'] > TRAIN_LAST_DATE)]['ForecastId']
    df_sub.loc[test_id,'ConfirmedCases']=pred['ConfirmedCases'].values
    df_sub.loc[test_id,'Fatalities']=fat_pred
    return rough_correction


# In[ ]:


predict_sub('Israel'," ",df_filt)


# In[ ]:


rough_correction = 0
#predict loop for exponential confirmed cases and derived fatalities
df_sub.set_index('ForecastId',inplace=True)

#loop over countries in ctry_lst
for (ctry,prov) in ctry_prov_lst:
    rough_correction += predict_sub(ctry,prov,df_filt)

#loop over countries not in ctry_lst, smaller countries or those which did not yet start to get sick
for (ctry,prov) in ctry_prov_noexp_lst:
    rough_correction += predict_sub(ctry,prov,df_corona_noexp)
    
df_sub.reset_index(inplace=True)
print("Rough fatalities corrections: {}".format(rough_correction))


# In[ ]:


#predict minimum 1 for confirmed cases, artifact of log
df_sub['ConfirmedCases'] = df_sub['ConfirmedCases'].apply(lambda x: max(x,1))


# In[ ]:


#check fatality rate per country
df_fat = df_last_day_train.groupby('Country_Region')[['ConfirmedCases','Fatalities']].sum().reset_index()
df_fat['FatRate'] = df_fat['Fatalities']/df_fat['ConfirmedCases']
df_fat.sort_values('FatRate',ascending=False)


# In[ ]:


df_fat.describe()


# In[ ]:


df_fat.FatRate.plot(kind='hist',bins=50)


# In[ ]:


fatalities_avg = df_last_day_train.Fatalities.sum()/df_last_day_train.ConfirmedCases.sum()
fatalities_avg


# In[ ]:


fatalities_ctry_mean = np.mean(df_last_day_train.Fatalities/df_last_day_train.ConfirmedCases)
fatalities_ctry_mean


# In[ ]:


df_sub.describe()


# In[ ]:


df_sub.sort_values('Fatalities',ascending=False)[:10]


# In[ ]:


df_score = df_eval.merge(df_sub, on=['ForecastId'], right_index=True, validate='1:1',suffixes=('_act','_pred'))
df_score['ConfirmedCasesError']=(np.log1p(df_score['ConfirmedCases_act'])-np.log1p(df_score['ConfirmedCases_pred']))**2
df_score['FatalitiesError']=(np.log1p(df_score['Fatalities_act'])-np.log1p(df_score['Fatalities_pred']))**2
score_c = np.sqrt(mean_squared_log_error(df_score['ConfirmedCases_act'], df_score['ConfirmedCases_pred']))
score_f = np.sqrt(mean_squared_log_error(df_score['Fatalities_act'], df_score['Fatalities_pred']))
print("Confirmed score: {0:.2f}, Fatalities score:{1:.2f}, Mean: {2:.2f}".format(score_c,score_f,np.mean([score_c, score_f])))


# In[ ]:


c = [c for (c,p) in ctry_prov_lst]
p = [p for (c,p) in ctry_prov_lst]
#error from countries in main list, predicted with exp ~100
df_filt_score = df_score[(df_score['Country_Region'].isin(c)) & (df_score['Province_State'].isin(p))]
score_c = np.sqrt(mean_squared_log_error(df_filt_score['ConfirmedCases_act'],df_filt_score['ConfirmedCases_pred']))
score_f = np.sqrt(mean_squared_log_error(df_filt_score['Fatalities_act'], df_filt_score['Fatalities_pred']))
print("Confirmed score: {0:.2f}, Fatalities score:{1:.2f}, Mean: {2:.2f}".format(score_c,score_f,np.mean([score_c, score_f])))


# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.scatter(x = np.log1p(df_filt_score['ConfirmedCases_act']),
            y= np.log1p(df_filt_score['ConfirmedCases_pred']),marker='.',color='b')
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.title("ConfirmedCases")
plt.subplot(1,2,2,)
plt.scatter(x = np.log1p(df_filt_score['Fatalities_act']),
            y= np.log1p(df_filt_score['Fatalities_pred']), marker='.',color='b')
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.title("Fatalities")
plt.show()


# In[ ]:


df_filt_score.sort_values(by='ConfirmedCasesError',ascending=False).head(10)


# In[ ]:


#error from smaller countries and countries that we just used the last known confirmed cases ~200
#confirmed cases seems worse than prediction, good sign that prediction does something good
#fatalities seems better than prediction, bad fatalities prediction
df_filt_score2 = df_score[~(df_score['Country_Region'].isin(c)) | ~(df_score['Province_State'].isin(p))]
score_c = np.sqrt(mean_squared_log_error(df_filt_score2['ConfirmedCases_act'],df_filt_score2['ConfirmedCases_pred']))
score_f = np.sqrt(mean_squared_log_error(df_filt_score2['Fatalities_act'], df_filt_score2['Fatalities_pred']))
print("Confirmed score: {0:.2f}, Fatalities score:{1:.2f}, Mean: {2:.2f}".format(score_c,score_f,np.mean([score_c, score_f])))


# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.scatter(x = np.log1p(df_filt_score2['ConfirmedCases_act']),
            y= np.log1p(df_filt_score2['ConfirmedCases_pred']),marker='.',color='b')
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.title("ConfirmedCases")
plt.subplot(1,2,2,)
plt.scatter(x = np.log1p(df_filt_score2['Fatalities_act']),
            y= np.log1p(df_filt_score2['Fatalities_pred']), marker='.',color='b')
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.title("Fatalities")
plt.show()


# In[ ]:


df_filt_score2.sort_values(by='FatalitiesError',ascending=False).head(10)


# In[ ]:


#df_score.sort_values(by='ConfirmedCasesError',ascending=False).head(10)


# In[ ]:


c = [c for (c,p) in not_enough_data_ctry_lst]
p = [p for (c,p) in not_enough_data_ctry_lst]
#error from countries in kept at last value ~60
df_filt_score3 = df_score[(df_score['Country_Region'].isin(c)) & (df_score['Province_State'].isin(p))]
score_c = np.sqrt(mean_squared_log_error(df_filt_score3['ConfirmedCases_act'],df_filt_score3['ConfirmedCases_pred']))
score_f = np.sqrt(mean_squared_log_error(df_filt_score3['Fatalities_act'], df_filt_score3['Fatalities_pred']))
print("Confirmed score: {0:.2f}, Fatalities score:{1:.2f}, Mean: {2:.2f}".format(score_c,score_f,np.mean([score_c, score_f])))


# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.scatter(x = np.log1p(df_filt_score3['ConfirmedCases_act']),
            y= np.log1p(df_filt_score3['ConfirmedCases_pred']),marker='.',color='b')
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.title("ConfirmedCases")
plt.subplot(1,2,2,)
plt.scatter(x = np.log1p(df_filt_score3['Fatalities_act']),
            y= np.log1p(df_filt_score3['Fatalities_pred']), marker='.',color='b')
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.title("Fatalities")
plt.show()


# In[ ]:


df_score.describe()


# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.scatter(x = np.log1p(df_score['ConfirmedCases_act']), y= np.log1p(df_score['ConfirmedCases_pred']),marker='.',color='b')
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.title("ConfirmedCases")
plt.subplot(1,2,2,)
plt.scatter(x = np.log1p(df_score['Fatalities_act']), y= np.log1p(df_score['Fatalities_pred']), marker='.',color='b')
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.title("Fatalities")
plt.show()


# In[ ]:


df_sub.to_csv("submission.csv",index=False)

