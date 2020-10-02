#!/usr/bin/env python
# coding: utf-8

# # I. Abstract

# In this notebook we try to understand how useful and succesful certain startegies, implemented by different governements against the Corona Virus SARS-CoV-2, are. We use the confirmed-case rates from the latest Covid19 forecast challange together with the "COVID 19 Containment measures data"-dataset from the uncover Covid19 challange to observe how the rates changed after the implementaition of certain measures. 
# 
# Even though reported in other analysis that it is not possible to extract the effectivness of single measures, we have exactely this aim. We try to extract the change in confirmed cases growth rates that follow an implemented measure.
# * First, we analyse the confirmed case curves of some infected countries.
# 
# 
# For this research we assume the reported confirmed cases $C$ to grow exponentially in the beginning of eachs countires counts with 
# 
# $$C = C_0 e^{\gamma t},$$ 
# 
# where $C$ is the number of confirmed cases, $C_0$ is the number of cases at the relative measure time $t_0$ (when we start to count) and $t$ is the time measured in days from $t_0$. The parameter $\gamma$ is the growth rate.
# 
# Note that the confirmed cases in the data do not fully reflect the real infected cases but depend strongly on the testing behavior and the number of tests done in each individual country. However, using this data not for absolute numbers but for detecting relative trends is a valid way of data analysis. If a country changed the testing bevavior, this may effect the rates. 
# 
# **As the data changes day by day, the results might shift a bit from what is described in the text.**
# 
# 

# # II. Table of content

# * [1. Import libraries + settings](#section-one)
# * [2. Data loading](#section-two)
# * [3. Function definition](#section-three)
# * [4. Example plots](#section-four)
# * [5. Advanced data analysis](#section-five)
#     
#     * [5.0 Method description](#section-five-zero)
#     * [5.1 Finding a good delay time](#section-five-one)
#     * [5.2 Main parameter calculation](#section-five-two)
#     * [5.3 PCA analysis](#section-five-three)
#     

# <a id="section-one"></a>
# # 1. Import libraries + settings
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn.preprocessing import LabelEncoder
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')


# <a id="section-two"></a>
# # 2. Data loading
# We use the data from week 4 of the Covid19 Global Forecasting challange for the time series of confirmed cases. We additionally use the government mesures dataset from the UNCOVER CHALLANGE.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if dirname is not "uncover":
            print(os.path.join(dirname, filename))


# In[ ]:


time_series_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
time_series_train.Date = time_series_train.Date.apply(pd.to_datetime)
original_country = {country:time_series_train[time_series_train["Country_Region"]==country] for country in time_series_train["Country_Region"].unique()}
gov_measures = pd.read_csv("/kaggle/input/uncover/HDE/acaps-covid-19-government-measures-dataset.csv")
gov_measures.date_implemented = gov_measures.date_implemented.apply(pd.to_datetime)
country_specific = {country:gov_measures[gov_measures["country"]==country] for country in gov_measures["country"].unique()}
continents_map = {}
for ind,row in gov_measures.iterrows():
    continents_map[row["country"]] = row["region"]


# In[ ]:


# additional preparations
add_prep_list = ["US", "Australia", "France","UK"]
for country in add_prep_list:
    country_sum = time_series_train.groupby(["Country_Region","Date"]).sum().reset_index()
    original_country[country] = country_sum[country_sum["Country_Region"]== country]
    original_country[country]


# In[ ]:


country_mapping = {"Korea Republic of":"Korea, South"}
country_specific_buffer = {}
for key in country_specific:
    if key in country_mapping:
        mapping = country_specific[key]
        country_specific_buffer[country_mapping[key]]= mapping
country_specific.update(country_specific_buffer)
for key in country_mapping:
    continents_map[country_mapping[key]] = continents_map[key]


# <a id="section-three"></a>
# # 3. Function definition

# In[ ]:


# mathematical functions ------------------------------------------
def exp_func(x, a, b, c,d):
 return a * np.exp(b * (x-c)) + d

def lin(x,a,b):
    return a*x+b

# data processing functions --------------------------------------

def fit_data(data,ind=0,step=10,func=None, step_substracted=False):
    if step_substracted:
        step = -1*step
        min_add = 0
        max_add = -1
    else:
        min_add = 1
        max_add = 0
    if func is None:
        def lin(x,a,b):
            return a*x+b
        func=lin
    #print(max(0,int(ind+min(0,step)+min_add)),int(ind+max(0,step)+max_add))
    x = np.array(data["ConfirmedCases"].index)[max(0,int(ind+min(0,step)+min_add)):int(ind+max(0,step)+max_add)]
    y_log = np.log(np.array(data["ConfirmedCases"])[max(0,int(ind+min(0,step)+min_add)):int(ind+max(0,step)+max_add)])
    opt_lin, pcov_lin = curve_fit(func, x, y_log)

    return opt_lin, pcov_lin

def scan_data_list(data,day_list = [],delay=10, all_slopes=False, lower_limit = 500):
    try:
        dates_order = np.array(data.Date)
        data=data.reset_index()
        data_fit = data[data["ConfirmedCases"] > lower_limit]
        first_ent = data_fit.index[0]
        number_entries = len(data_fit)
        slopes = []
        offsets = []
        dates_from = []
        dates_to = []
        changes = []
        first=0
    except IndexError:
        #print("Error with reading data")
        return None
    for i in day_list:
        i = i-first_ent
        if number_entries-i >= 3 and i >= 0 :
            params, quality = fit_data(data_fit,ind=i+delay,step=10, step_substracted=True)
            params_1, quality_1 = fit_data(data_fit,ind=i+delay,step=10, step_substracted=False)
            slopes.extend([params[0],params_1[0]])
            offsets.extend([params[1],params_1[1]])
            dates_from.append(dates_order[first_ent+i])
            changes.append(params[0]-params_1[0])
            dates_to.append(dates_order[min(first_ent+i+delay,len(dates_order)-1)])
    if len(slopes) > 0:
        percent_changes = np.array(changes)/max(slopes)
        percent_changes = np.array(changes)/np.array(slopes[slice(0,None,2)])
        print(percent_changes)
        print(np.array(changes))
        print(np.array(slopes[slice(0,None,2)]))
        day_of_change_index =[round(linalg.solve(slopes[i]-slopes[i+1], offsets[i+1]-offsets[i])[0]) for i in range(0,len(slopes)-1,2)]
        day_of_change = [dates_order[min(max(int(day),0),len(dates_order)-1)] for day in day_of_change_index]
        df_result = pd.DataFrame({"date_switch":day_of_change, "date_from":dates_from,"date_to":dates_to, "growth_change":percent_changes})
        if all_slopes:
            return_all = [((slopes[i],slopes[i+1]), (offsets[i],offsets[i+1])) for i in range(0,len(slopes)-1,2)]
            return return_all
        return df_result
    else:
        return None
        

# Evaluate measures against reported cases
def evaluate_data(original_country,country_specific, gov_measures, delays=[13], min_cases=50, region_specific=None): 
    measure_df_final = pd.DataFrame(columns=gov_measures.measure.unique())
    measure_df = pd.DataFrame(columns=gov_measures.measure.unique())
    for delay in delays:

        for country in original_country:
            if country in country_specific and (region_specific is None or continents_map[country] == region_specific):
                measure_vector = {measure:[np.nan] for measure in gov_measures.measure.unique()}
                dates_order = np.array(original_country[country].Date)
                dates_map = {date:i for i,date in enumerate(original_country[country].Date)}
                day_list = []
                measure_list = []
                for index, row in country_specific[country].iterrows():
                    if row["date_implemented"] in dates_map:
                        day_list.append(dates_map[row["date_implemented"]])
                        measure_list.append(row["measure"])
                try:
                    day_list.sort()
                    day_list_set = list(set(day_list))
                    day_list_set.sort()
                    day_list_map = {day_list_set[j]:j for j in range(0, len(day_list_set))}
                    country_df = scan_data_list(original_country[country], day_list=day_list_set, delay=delay, lower_limit=100)

                    for ind,day_df in country_df.iterrows():
                        measure_vector[measure_list[day_list_map[dates_map[day_df["date_from"]]]]].append(day_df["growth_change"])
                        for key in measure_vector:
                            measure_vector[key] = [np.nanmean(measure_vector[key])]
                        measure_df_local = pd.DataFrame(measure_vector)
                        measure_df = measure_df.append(measure_df_local)
                    #print("Done")
                except TypeError as e:
                    #print(f"Attribute Error: {e}")
                    pass
                except Exception as e:
                    #print(f"Something failed:{e}")
                    pass
    return measure_df


# <a id="section-four"></a>
# # 4. Example plots

# In this part, we show plots of $\text{log}(C)$ for different example countries, i.e. the logarithmic plot of the confirmed cases rate. As $C = C_0 e^{\gamma t},$ applying the logarithm we find $\text{log}(C)=\gamma t +\text{log}(C_0)$ which describes a linear function with slope $\gamma$. This means the wherever we see a straigt line in the plot, it corresponds to an exponential growth with the slope of the linear function as exponential growth rate. 
# 
# Looking at the example plots, we detect different parts that follow approximately straigth lines, i.e. an exponential growth in the case rates. The behaviour often suddendly changes and there is a new region with a (mostly) smaller slope. This can be connected to new measures implemented by the governements of the countries under observation. Most of the curves slowly flatten out after more than 70 or 80 days after the 100th Corona case. This shows that the measures implemented are useful and thus the rates do not fully grow exponential anymore. 
# 
# In the following plots, we demonstrate a few different countries, that took different routes with different results. In principle, most of the not shown countries can be put in one of these goups presented below.
# 
# In the upper panel of eah plot, we show the raw log plot of confirmed cases for each countrie, while in the lower plot, we show the log-cases plot with two linear functions fitted to the data to demonstrate two regions and the change of the growth rate $\Delta \gamma = \gamma_1-\gamma_2$, with $\gamma_i$ the growth rate corresponding to the ith linear function.  
# 
# **1. Italy and Spain:**
# A quiet steep start, with a long time of exponential growth. The exponential growt is fully running and some measures were probably implemented to late. In the end, the curve begins to flatten.
# 
# **2. Germany:**
# Similar to Ital and Spain, but less steep and more small edges in the curve, that show small changes due to implemented measures. 
# 
# **3. Norway and Sweden:**
# Both cuntries have an explicit and very clear exponential growth in their rates. However, Sweden goes a different route and only implements soft measures against the Corona virus. This results in a less decreasing exponential growth rate compared to its direct neighbor Norway. 
# 
# **4. South Korea**
# South Korea implemented rules very fast and very efficiently, which lead to a sudden strong decrease and an almost fully flat curve.
# 

# In[ ]:


day_dict = {"France":[50], "US":[70],"Italy":[32],"Spain":[50],"Germany":[57], "Norway":[50],"Sweden":[50]}
lower_limit = 100
for j,country in enumerate(day_dict):
    country_df = scan_data_list(original_country[country], all_slopes=True, day_list=day_dict[country], delay=13, lower_limit=lower_limit)
    leng_cases = len(original_country[country]["ConfirmedCases"])
    last_entry = original_country[country].reset_index()["ConfirmedCases"][leng_cases-1]
    last_slope = scan_data_list(original_country[country], all_slopes=True, day_list=[leng_cases-13], delay=8, lower_limit=lower_limit)
    x = np.arange(0, leng_cases)
    y_1 = lin(x,country_df[0][0][0],country_df[0][1][0])
    y_2 = lin(x,country_df[0][0][1],country_df[0][1][1])
    print(f"Latest doubling time:{np.log(2)/last_slope[0][0][1]}")
    print(f"Latest number of cases: {last_entry}")
    original_country[country]["ConfirmedCases"].index -= original_country[country]["ConfirmedCases"].index[0]
    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    fig.suptitle(f"Fig.{j}: {country}")
    #axs[0].title("Test")
    np.log(original_country[country]["ConfirmedCases"]).plot(ax=axs[1])
    axs[1].plot(x,y_1)
    axs[1].plot(x,y_2)
    axs[1].set_ylim((0,17))
    axs[1].set_ylabel("log(C)")
    axs[1].set_xlabel(f"days from day with case {lower_limit}")
    axs[0].set_ylabel("log(C)")
    axs[0].set_xlabel(f"days from day with case {lower_limit}")
    np.log(original_country[country]["ConfirmedCases"]).plot(ax=axs[0])
    plt.show()


# <a id="section-five"></a>
# # 5. Advanced data evaluation

# <a id="section-five-zero"></a>
# ## 5.0 Method description

# In this section we take a closer look to the single measures implemented. For this purpose, we automate the method we used for the example plots above. For each country, we check all measures implemented after the 100th case occured. We take all data points from the day of implementation for the measure under observation at time $t_0$ until $t_0+t_\text{delay}$, where $t_\text{delay}$ is a number of days, that accounts for the incubation time and the delay in reporting the case. How we choose this time, is explained in the next section. 
# We then fit a linear function to this data points. We repete this process to the data points for the time after the delay time is over and the implemented measure should get momentum and show results. From the twho resulting linear functions, we calculate the change in the growth rate $\Delta \gamma$. We use the relative change of the growth rate $\tfrac{\Delta \gamma}{\text{max}(\gamma_i)}$, where $\gamma_i$ contains all growth rates found for a certain country. This relativ change of the growth rate is our main parameter to rate the results of the measures. 
# 
# Note, that as an disadvantage of this parameter is, that the impact of measures implemented later in the crisis may be underestimated, beause the grwoth rate may already be lower due to other strategies already implemented. The relative growth change measured relatively to the maximum growth rate, could then be simply lower. 
# We assume, that this only slightly affects the results. 

# <a id="section-five-one"></a>
# ## 5.1 Choosing a good delay time

# In[ ]:


std_list = []
min_cases = 50
begin = 8
end = 22
region="Europe"
for i in range(begin,end):
    measure_df = evaluate_data(original_country,country_specific, gov_measures,delays=[i], region_specific=region)
    stds = (measure_df.std()[measure_df.count() > min_cases]/measure_df.mean()[measure_df.count() > min_cases]).sort_values()
    #stds = (measure_df.std()[measure_df.count() > min_cases]).sort_values()
    std_list.append(stds.mean())
std_list
delay_x = np.arange(begin,end)
std_y = np.array(std_list)
plt.xlabel("days of delay")
plt.ylabel("mean of relative standard deviation across measures")
plt.plot(delay_x,std_y)


# In the plot we show the mean of the standart deviation across the different measures. The smaller the standard deviation of the growth rate change $\Delta \gamma$ the larger the chance, that we really observe a causal correlation. This simple plot gives a few good insights. 
# First of all, we use it to estimate a good delay time between the implementation date of a measure and the first time we expect it to influence the rate of confirmed cases. We observe the absolute minimum at 15 days of delay. However, choosing an individual day seems too restrictive and thus we choose the four minimal dealy days 12, 13, 14 and 15 for further research.
# 
# The fact, that this plot has an unique minimum and that it is monotonic on either side of the minimum is an indicator (not more but also not less) that the technique we use has some prediction power and is not just randomly fluctuating. 
# 
# Choosing the value 13 to 15 as delay days is consistent with the reported values by the German Robert Koch Institute. They report a median incubation time for the virus of about 6 days. Additionally they report a time of 7 to 8 days from the beginning of the first symptoms to the reporting of the case which in sum is betweent 13 and 14 days. (https://www.rki.de/DE/Content/Infekt/EpidBull/Archiv/2020/Ausgaben/17_20_SARS-CoV2_vorab.pdf?__blob=publicationFile - German) 
# 
# 

# <a id="section-five-two"></a>
# ## 5.2 Main parameter $\tfrac{\Delta \gamma}{\text{max}(\gamma_i)}$ calculation

# In[ ]:


min_cases = 50
region = "Europe"
measure_df = evaluate_data(original_country,country_specific, gov_measures,delays=[12,13], min_cases=min_cases, region_specific=region)
len(measure_df)


# The following plot shows the mean of implemented measures across all countries (we only used entries where at least 50 countries have implemented the measure to have strong enough statistics) rated by the relative growth rate change $\tfrac{\Delta \gamma}{\text{max}(\gamma_i)}$. From this data we see, that **restriticing the traveling**, especially across borders, helps to figth the virus. Additionally, keeping people on distance to each other (**curfews, school closings, shop closings and limiting public gatherings**), are the main tools to lower the growth rate. Here, **especially curfews** seem to work well. Interestingly, a **full lock down seems to be not one of the most efficient** strategies and only help a bit, if the other measures are already implemented. 
# Measures like "State of emergency declared" do of course not change a lot by themselfe, however typically a country that declared this state treated the virus with more caution (and thus released a lot of other strategies right after the declaration) and also such a declaration shows the inhabitants how dangerous the situation is.

# In[ ]:


measure_result_vec = measure_df.mean()[measure_df.count() > min_cases]
measure_result_vec.sort_values().plot(kind="barh", figsize=(7,7))
measure_result_vec


# Looking at the relative standard derivation of the rate $\tfrac{\Delta \gamma}{\text{max}(\gamma_i)}$, we get a feeling for how strongly we can trust in the ordering above. **As the mean error across all measures is quiet high with about $60$ %, we should not treat the effectiveness position given above as defenite positions but rather as a trend.**

# In[ ]:


stds = (measure_df.std()[measure_df.count() > min_cases]/measure_df.mean()[measure_df.count() > min_cases]).sort_values(ascending=False)
stds.plot(kind="barh", figsize=(7,7))
print(f"Average erro: {stds.mean()}")


# <a id="section-five-three"></a>
# ## 5.3 PCA analysis to define measure packages

# Additionally to the analysis above, we make use of the principle component analysis (PCA) algorithm. In principle, it takes the correlation matrix of all the features and tries to find a basis in the feature space where all basic vectors have no linear correlation with the others. In our case, the features are the different strategies implemented by the governements and their values are the growth changes $\tfrac{\Delta \gamma}{\text{max}(\gamma_i)}$ which we estimated as explained in section 5.0. PCA is typically used for dimensonality reduction. Here we use it to get a feeling for dependent measures. We use this to gain two different insights:
# 
# 1. First of all we look for the feature basis vectors that have the largest variance (i.e. largest eigenvalue ) and thus carry the most information about the growth changes. From this we have an additional indicator which of the measures had the most influence on the growth rate change $\tfrac{\Delta \gamma}{\text{max}(\gamma_i)}$.
# 
# 2. Furthermore, from the PCA we can deduce which measures work together, are very similar or even redundant. Measures that are part of the same feature vector in the new basis, typically cause a similar change. As this is based on the correlation between all these measures, there are two reasons, why measures can be linked in such a way: wether they are measures that are typically applied in close time vicinity or the measures strengthen (or weaken) each other. Here we do not differentiate between the two possibilities, however, this could be an additional step for future reseach.

# In[ ]:


corr = measure_df.corr()[measure_df.count()>min_cases].T[measure_df.count()>min_cases]
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.show()

final_matrix
ax = sns.heatmap(
    final_matrix, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.show()


# In[ ]:


corr = measure_df.corr()[measure_df.count()>min_cases].T[measure_df.count()>min_cases]
measure_result_vec = measure_df.mean()[measure_df.count()>min_cases]
measure_vec = [key for key in measure_df.mean()[measure_df.count()>min_cases].index]
data = np.linalg.eig(corr.fillna(0))[1]
eig = np.linalg.eig(corr.fillna(0))[0]
zipped = list(zip(eig,data))
res = sorted(zipped, key = lambda x: x[0])
df = None
for dat in res:
    if df is None:
        df = pd.Series(dat[1], index=measure_vec)
    else:
        df += pd.Series(dat[1], index=measure_vec)
df = pd.Series(dat[1], index=measure_vec).sort_values() 
df.plot(kind="barh")
plt.show()
new_add = 0
for dat in res:
    df = pd.Series(dat[1], index=measure_vec)
    new = df*measure_result_vec


# In[ ]:


measure_df.corr()[measure_df.count()>min_cases].T[measure_df.count()>min_cases]


# In[ ]:


measure_vec = [key for key in measure_df.mean()[measure_df.count()>min_cases].index]
mean_matrix = np.diag(measure_df.mean()[measure_df.count()>min_cases])
mean_array = np.array(np.diagonal(mean_matrix))
div = np.array([[ent/abs(ent)] for ent in data[:,0]])
data_new = data/div
len_data=len(data_new)
new_basis_mean = np.diag(np.diagonal((data.dot(mean_matrix).dot(data.T))))
new_basis_mean
backtransform_mean = np.diag(data.T.dot(new_basis_mean).dot(data))
backtransform_mean
pd.Series(backtransform_mean, index=measure_vec).sort_values()

mean_matrix_free_basis = np.diag((np.sum(mean_array*data_new,axis=1)))
final_matrix = (data_new.T @ mean_matrix_free_basis/(data_new@np.ones(len_data))@ data_new)


# In[ ]:


np.diagonal(final_matrix)
res = pd.Series(np.diagonal(final_matrix), index=measure_vec).sort_values()
res


# In[ ]:


final_matrix @ np.ones(len_data)
pd.Series(final_matrix @ np.ones(len_data), index=measure_vec).sort_values()


# In[ ]:


pd.Series(final_matrix @ np.array([1,1,0,1,0,1,1,0,0,1,1,0,1,0,1,0,1,1,1,1,1,1]), index=measure_vec).sort_values()


# In[ ]:


print({i:((final_matrix@data_new[i])/data_new[i]).mean() for i in range(len(data_new)-1)})
pd.Series(data_new[6], index=measure_vec).sort_values().plot(kind="barh")


# In[ ]:


res.sort_values()


# In[ ]:





# In[ ]:




