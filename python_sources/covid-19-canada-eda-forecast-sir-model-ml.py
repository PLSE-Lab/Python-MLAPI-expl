#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Canada EDA & Forecast: SIR Model + ML

# ## 1. Import Data

# In[ ]:


import pandas as pd
dirname = '/kaggle/input/coronaviruscovid19-canada/'
cases = pd.read_csv(dirname + 'cases.csv')
recovered  = pd.read_csv(dirname + 'recovered.csv')
testing = pd.read_csv(dirname + 'testing.csv')
mortality = pd.read_csv(dirname + 'mortality.csv')


# In[ ]:


display(cases.sample(3))
display(recovered.sample(3))
display(testing.sample(3))
display(mortality.sample(3))


# ## 2. Exploratory Data Analysis

# ### General statistics

# In[ ]:


print('Number of cases in total till April 4: ' + str(cases.shape[0]))
print('Number of recovered in total till April 4: ' + str(recovered.shape[0]))
print('Number of deathes in total till April 4: ' + str(mortality.shape[0]))
print('Recovery rate: ' + str(round(recovered.shape[0]/cases.shape[0], 4) * 100) + '%')
print('Mortality rate: ' + str(round(mortality.shape[0]/cases.shape[0], 4) * 100) + '%')


# ### Statistics by provinces

# Cases by provinces

# In[ ]:


df_prov = pd.DataFrame()
provinces = list(cases.province.unique())
provinces.remove('Repatriated')
provinces.remove('NWT')
provinces.remove('Yukon')
print(provinces)
df_prov['provinces'] = provinces
df_prov['cases'] = 0


# In[ ]:


for province in provinces:
    df_prov.at[df_prov.provinces == province, 'cases'] = int(cases.loc[cases.province == province].tail(1).provincial_case_id)


# In[ ]:


import seaborn as sns
df_prov = df_prov.sort_values('cases', ascending=False)
sns.set(style="whitegrid")
ax = sns.barplot(x=df_prov.provinces, y=df_prov.cases)
var = ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")


# Testing by provinces

# In[ ]:


for province in provinces:
    df_prov.at[df_prov.provinces == province, 'testing'] = int(testing.loc[testing.province == province][:1].cumulative_testing)
    df_prov.at[df_prov.provinces == province, 'positive_rate'] = round(int(df_prov.loc[df_prov.provinces == province].cases) / int(df_prov.loc[df_prov.provinces == province].testing), 4) * 100


# In[ ]:


sns.set(style="whitegrid")
ax = sns.barplot(x=df_prov.provinces, y=df_prov.testing)
var = ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")


# Positive rate by provinces

# In[ ]:


ax = sns.barplot(x=df_prov.provinces, y=df_prov.positive_rate)
var = ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")


# ### Trend

# In[ ]:


with sns.axes_style("darkgrid"):
    ax = sns.lineplot(x=cases.date_report, y=cases.case_id).set(xticks=[], xlabel='time', ylabel='number of cases')


# In[ ]:


with sns.axes_style("darkgrid"):
    ax = sns.lineplot(x=mortality.date_death_report, y=mortality.death_id).set(xticks=[], xlabel='time', ylabel='number of death')


# In[ ]:


with sns.axes_style("darkgrid"):
    ax = sns.lineplot(x=cases.date_report, y=cases.case_id)
    ax = sns.lineplot(x=mortality.date_death_report, y=mortality.death_id).set(xticks=[], ylabel='cases', xlabel='time')


# ## 3. SIR Model
# [Reference](https://www.kaggle.com/saga21/covid-global-forecast-sir-model-ml-regressions#2.-SIR-model-)

# ### Theoretical SIR model based on Canada's population

# In[ ]:


# Susceptible equation
def fa(N, a, b, beta):
    fa = -beta*a*b
    return fa

# Infected equation
def fb(N, a, b, beta, gamma):
    fb = beta*a*b - gamma*b
    return fb

# Recovered/deceased equation
def fc(N, b, gamma):
    fc = gamma*b
    return fc


# In[ ]:


# Runge-Kutta method of 4rth order for 3 dimensions (susceptible a, infected b and recovered r)
def rK4(N, a, b, c, fa, fb, fc, beta, gamma, hs):
    a1 = fa(N, a, b, beta)*hs
    b1 = fb(N, a, b, beta, gamma)*hs
    c1 = fc(N, b, gamma)*hs
    ak = a + a1*0.5
    bk = b + b1*0.5
    ck = c + c1*0.5
    a2 = fa(N, ak, bk, beta)*hs
    b2 = fb(N, ak, bk, beta, gamma)*hs
    c2 = fc(N, bk, gamma)*hs
    ak = a + a2*0.5
    bk = b + b2*0.5
    ck = c + c2*0.5
    a3 = fa(N, ak, bk, beta)*hs
    b3 = fb(N, ak, bk, beta, gamma)*hs
    c3 = fc(N, bk, gamma)*hs
    ak = a + a3
    bk = b + b3
    ck = c + c3
    a4 = fa(N, ak, bk, beta)*hs
    b4 = fb(N, ak, bk, beta, gamma)*hs
    c4 = fc(N, bk, gamma)*hs
    a = a + (a1 + 2*(a2 + a3) + a4)/6
    b = b + (b1 + 2*(b2 + b3) + b4)/6
    c = c + (c1 + 2*(c2 + c3) + c4)/6
    return a, b, c


# In[ ]:


def SIR(N, b0, beta, gamma, hs):
    
    """
    N = total number of population
    beta = transition rate S->I
    gamma = transition rate I->R
    k =  denotes the constant degree distribution of the network (average value for networks in which 
    the probability of finding a node with a different connectivity decays exponentially fast
    hs = jump step of the numerical integration
    """
    
    # Initial condition
    a = float(N-1)/N -b0
    b = float(1)/N +b0
    c = 0.

    sus, inf, rec= [],[],[]
    for i in range(10000): # Run for a certain number of time-steps
        sus.append(a)
        inf.append(b)
        rec.append(c)
        a,b,c = rK4(N, a, b, c, fa, fb, fc, beta, gamma, hs)

    return sus, inf, rec


# #### Senerio 1: no social distancing
# * We set the transition rate to be higher

# In[ ]:


# Parameters of the model
import matplotlib.pyplot as plt
N = 3759 * 10000
b0 = 0
beta = 1
gamma = 0.2
hs = 0.1

sus, inf, rec = SIR(N, b0, beta, gamma, hs)

f = plt.figure(figsize=(8,5)) 
plt.plot(sus, 'b.', label='susceptible');
plt.plot(inf, 'r.', label='infected');
plt.plot(rec, 'c.', label='recovered/deceased');
plt.title("SIR model")
plt.xlabel("time", fontsize=10);
plt.ylabel("Fraction of population", fontsize=10);
plt.legend(loc='best')
plt.xlim(0,1000)
plt.savefig('SIR1.png')
plt.show()


# #### Senerio 2: with social distancing (flatten the curve)
# We set the transition rate to be lower

# In[ ]:


# Parameters of the model
N = 3759 * 10000
b0 = 0
beta = 0.5
gamma = 0.2
hs = 0.1

sus, inf, rec = SIR(N, b0, beta, gamma, hs)

f = plt.figure(figsize=(8,5)) 
plt.plot(sus, 'b.', label='susceptible');
plt.plot(inf, 'r.', label='infected');
plt.plot(rec, 'c.', label='recovered/deceased');
plt.title("SIR model")
plt.xlabel("time", fontsize=10);
plt.ylabel("Fraction of population", fontsize=10);
plt.legend(loc='best')
plt.xlim(0,1000)
plt.savefig('SIR2.png')
plt.show()


# #### Senerio 3: no social distancing + distressed medical systems (70% Canadians infected)

# In[ ]:


# Parameters of the model
N = 3759 * 10000
b0 = 0
beta = 1
gamma = 0.1
hs = 0.1

sus, inf, rec = SIR(N, b0, beta, gamma, hs)

f = plt.figure(figsize=(8,5)) 
plt.plot(sus, 'b.', label='susceptible');
plt.plot(inf, 'r.', label='infected');
plt.plot(rec, 'c.', label='recovered/deceased');
plt.title("SIR model")
plt.xlabel("time", fontsize=10);
plt.ylabel("Fraction of population", fontsize=10);
plt.legend(loc='best')
plt.xlim(0,1000)
plt.savefig('SIR3.png')
plt.show()


# ## 4. Regression Model
# under construction

# In[ ]:




