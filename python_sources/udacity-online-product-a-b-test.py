#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math as mt
import numpy as np
import pandas as pd
from scipy.stats import norm


# # **1. Udacity screener **
# - Goal to maximize course completion by user 
# 
# - Two options: 'start free trial', 'access course materials'

# ### **1.1 Experiment description **
# 
# - If user clicked free trial, they are given a prompt to indicate how much of their time can be devoted
# 
# - If time >= 5 hours / week, proceed to checkout 
# 
# - If t < 5 / week, message popup indicating courses generally require higher time commitments
# 
# #### Units 
# - Unit of diversion = cookie 
# 
# - Users enrolled on free trials are tracked by user-id 
# 
# - Same user id cannot enroll free trial twice
# 
# #### Tracking method 
# 
# - Pre-enrollment = cookies 
# 
# - Post enrollment = user-id 
# 
# - Non-enrolled = not tracked in experiment, even if they were signed when viewing course overview page 

# ### **1.2 Hypothesis **
# 
# 1) By setting clearer expectations upfront, there is a reduction of frustrated users leaving the free trial 
# 
# 2) No significant reduction in user numbers continuing to the free trial and eventual completion. 
# 
# 3) Overall improvement in user experience

# # **2. Metric choice **
# 
# - 2 types: Invariate + evaluation metrics 
# 
# 
# ## 2.1 Invar metrics
# 
# - Invariate = sanity checks (ensure ways of presenting changes + collecting data is not wrong)
# 
# - Invariate metrics should not be affected due to experiment and don't change drastically between control + experiment groups
# 
# | Metric Name  | Metric Formula  | $Dmin$  | Notation |
# |:-:|:-:|:-:|:-:|
# | Number of Cookies in Course Overview Page  | # unique daily cookies on page | 3000 cookies  | $C_k$ |
# | Number of Clicks on Free Trial Button  | # unique daily cookies who clicked  | 240 clicks | $C_l$ |
# | Free Trial button Click-Through-Probability  | $\frac{C_l}{C_k}$ | 0.01  | $CTP$ | 

# ## Invar metrics explained 
# 
# #### Number of cookies 
# - Number of unique cookies to view course overiew 
# 
# #### Number of clicks 
# 
# - Number of unique cookies to click start free trial
# 
# #### Click through probability 
# 
# - Number of unique cookies to click start free trial / number of cookies to view course overview page 

# ## 2.3. Eval metrics
# 
# - Evaluation = metrics expected to change, relevant to business goals 
# 
# - Each eval. metric is marked by *Dmin*, a minimum change that is significant to the business e.g. user retention rates. 
# 
# - What may be statisically significant may not be practically significant to business objectives. 
# 
# | Metric Name  | Metric Formula  | $Dmin$  | Notation |
# |:-:|:-:|:-:|:-:|
# | Gross Conversion   |  $\frac{enrolled}{C_l}$  | 0.01  | $Conversion_{Gross}$ |
# | Retention   | $\frac{paid}{enrolled}$  | 0.01  | $Retention$ |
# | Net Conversion  |  $\frac{paid}{C_l}$  | 0.0075 | $Conversion_{Net}$ |

# ## Eval metrics
# 
# #### Gross conversion
# 
# - Number of user-ids to complete checkout & enroll in free trial / number of unique cookies to click Start Free Trial 
# 
# - Indicate validity of** H1 **
# 
# #### Net conversion 
# 
# - Number of usder-ids to complete, enroll, and pay / number of unique cookies to ckick Start Free Trial 
# 
# - 14 day boundary applies to net 
# 
# - Indicate validity of **H2 **

# ### 2.2. Metric & Hypothesis 
# 
# #### Correlation 
# 
# - Gconversion & Nconversion = correlated 
# 
# #### H1 expectation
# - If H1 is true, test ***should indicate*** a statistically signfiicant ** reduction in Gross conversion** of experiment group 
# 
# - i.e. should not be the same / higher than control group 
# 
# #### H2 expectation 
# - If H2 is true, test ***should NOT indicate*** a statistically **significant reduction in Net conversion** of experiment group 
# 
# - i.e. should be the same / higher than control group 

# #### 2.3 Unused Metrics
# 
# - User ID 
# 
# - (Used) Retention (number of users enrolled past 14-day boundary) 

# ### **3. Estimator data**
# 
# - Identify baseline values of metrics (how metrics behave before the change)
# 
# 
# | Item | Description  | Estimator  |
# |:-:|:-:|:-:|
# | Number of cookies | Daily unique cookies to view course overview page  | 40,000  |
# | Number of clicks | Daily unique cookies to click Free Trial button  | 3,200 |
# | Number of enrollments | Free Trial enrollments per day  | 660  |
# | CTP | CTP on Free Trial button  | 0.08  |
# | Gross Conversion | Probability of enrolling, given a click  | 0.20625  |
# | Retention | Probability of payment, given enrollment  | 0.53  |
# | Net Conversion | Probability of payment, given click  | 0.109313 |

# #### ***Note***: CTP + GConversion = 1

# In[ ]:


# Place estimators into dictionary 

baseline = {'cookies': 40000,
            'clicks': 3200,
            'enrollment': 660,
            'CTP': 0.08,
            'Gconversion': 0.20625,
            'retention': 0.53,
            'Nconversion': 0.109313}

baseline


# ## 4. Standard deviation estimation
# 
# <br>
# <center><font size="4">$SD=\sqrt{\frac{\hat{p}*(1-\hat{p})}{n}}$</font></center><br>
# 
# - SD necessary for sample size calcs & confidence intervals of results 
# 
# - Estimate the SD of each eval. metric, given the information of the collected metric estimates. 
# 
# - Higher variation in metrics = harder to achieve significant result.
# 
# - Gconversion & Nconversion are measured as probabilities, expected a norma binomial distribution if sample size is sufficently large 
# 
# - Assume sample size of 5k cookies / day (in each group) for experiment, given the baseline values

# #### 4.1 Data scaling 
# 
# - All following calculations requires collected count estimates of metrics to be scaled with sample size, specified for var estimation. 
# 
# - Change 40000 unique cookies to visit page to 5000

# In[ ]:


# Scaling count estimates 

baseline['cookies'] = 5000

# cookies / daily number of unique cookies per day 
baseline['clicks'] = baseline['clicks']*(5000/40000)

baseline['enrollment'] = baseline['enrollment']*(5000/40000)

baseline


# #### 4.2 Assumption: metrics are binomially distributed
# 
# - Assumption only holds when unit of diversion = unit of analysis
# 
# - unit of analysis = denominator of metric formula 

# #### 4.3 Units 
# 
# - Unit of diversion = element used to differentiate samples, assign them to control & experiment groups
# 
# - Unit of analysis = denominator of the formula to calc. Gconversion 
# 
# - Unit of diversion = Unit of analysis 
# 
# - Cookies = unique cookies to click (Start free trial)  

# #### 4.4 Each metric requires two variables to be plugged:
# 
# <br>$\hat{p}$ - baseline probability of the event to occur <br>
# $ n $ - sample size <br>
# 
# - Estimate metric variance for eval. metrics 
# 

# #### 4.5 Gconversion (Gc)
# 
# - Baseline proba = P(enrollment | a click) 
# 
# - number of users to enroll in free trial / number of unique cookies to click 'Start Free Trial'
# 
# - Unit of diversion = Unit of analysis
# 
# - Cookies = number of unique cookies to click 
# 
# - Sufficient condition for analytical estimation of var
# 

# In[ ]:


# Create p-hat & n 

Gc = {}

Gc['D_min'] = 0.01

# p-hat already given 
    # Alternatively, calc from enrollments / clicks 
Gc['p'] = baseline['Gconversion']

# Sample 
Gc['n'] = baseline['clicks']

# Compute SD, round to 4 decimals
Gc['sd'] = round(mt.sqrt((Gc['p']*(1-Gc['p']))/Gc['n']), 4)
Gc['sd']


# #### 4.6 Retention
# 
# - Baseline proba = P(payment | enrollment)
# 
# - Number of paying users enrolled after 14 days / number of total enrolled users 
# 
# - Unit of diversion != unit of analysis 
# 
# - Cookies = users enrolled 
# 
# - Analytical estimation of var is insufficient 

# In[ ]:


R = {}
R['D_min'] = 0.01

R['p'] = baseline['retention']

# sample size = enrolled users 
R['n'] = baseline['enrollment']

R['sd'] = round(mt.sqrt((R['p']*(1-R['p']))/R['n']), 4)
R['sd']


# #### 4.7 Nconversion
# 
# - Baseline proba = P(payment | a click) 
# 
# - Number of PAYING users / number of unique cookies to click start free trial 
# 
# - Cookies = number of cookies to click 
# 
# - Diversion = analysis 
# 
# - Analytical estimation = sufficient 

# In[ ]:


Nc = {}
Nc['D_min'] = 0.075

Nc['p'] = baseline['Nconversion']

# sample size = number of cookies clicked 
Nc['n'] = baseline['clicks']

Nc['sd'] = round(mt.sqrt((Nc['p']*(1-Nc['p']))/Nc['n']), 4)
Nc['sd']


# ## 5. Experiment Sizing 
# 
# - Calc. min. number of samples needed for sufficient statistical power & significant 
# 
# #### 5.1 Assumptions:
# 
# - $a$ = 0.05
# - $B$ = 0.2
# 
# #### 5.2 Objective:
# 
# - estimate number of total number of page views needed i.e. unique cookies who viewed the product overview page 
# 
# - divide page views into 2 groups: control & experiment 
# 

# ### 5.3 min. sample size 
# 
# <center> <font size="5"> $n = \frac{(Z_{1-\frac{\alpha}{2}}sd_1 + Z_{1-\beta}sd_2)^2}{d^2}$</font>, with: <br><br>
# $sd_1 = \sqrt{p(1-p)+p(1-p)}$<br><br>
# $sd_2 = \sqrt{p(1-p)+(p+d)(1-(1-(p+d))}$ </center><br>
# 
# - $d$ = detectable effect 
# 
# - $\hat{p}$ = baseline conversion rate 
# 
# ### 5.4 Simple Hypothesis 
# - $H_0 : P_{cont} - P_{exp} = 0$
# 
# - $H_1 : P_{cont} - P_{exp} = d$
# 
# ### 5.5 Necessary inputs & calcs 
# 
# #### inputs:
# 
# - $a,  $1-\beta$ ,  d, \hat{p}$
# 
# - where $d = D_{min}$
# 
# #### calcs:
# 
# - $Z$ score for $1-\frac{\alpha}{2}$ 
# - $Z$ score for $1-\beta$ 
# - SD1 & SD2 for baseline & expected change rate 

# ## 6. Z-score crit value & SD 
# 
# - scipy.stats.norm 

# In[ ]:


def get_sd(p,d):
    sd1 = mt.sqrt(2 * p *(1 - p))
    sd2 = mt.sqrt(p * (1 - p) + (p + d) * (1-(p + d)
                                          )
                 )
    x = [sd1,sd2]
    return x


# ### 6.1 ppf method: percent point function or quantile function
# 
# - functions return required critical z-score 

# In[ ]:


# alpha should already fit required test
    #Return z-score for alpha 
    
def get_z_score(alpha):
    return norm.ppf(alpha)


# ### 6.2 SD1 & SD2 
# 
# - Baseline = SD1 - SD 
# 
# - Expected change = SD2 - SD

# In[ ]:


def get_sd(p,d):
    sd1 = mt.sqrt(2 * p * (1 - p))
    sd2 = mt.sqrt(p * (1 - p)+(p + d) * (1 - (p + d)
                                        )
                 )
    sd = [sd1,sd2]
    return sd


# In[ ]:


def get_sampSize(sd,alpha,beta,d):
    n = pow((get_z_score(1 - alpha / 2) * sd[0] + get_z_score(1 - beta) * sd[1]),
            2) / pow(d, 2)
    return n


# ## 7. Calc sample size per metric 
# 
# - highest sample size will be the effective size 
# 
# - Size should account for duration & exposure
# 
# - duration & exposure = time to acquire given number of samples in experiment  

# In[ ]:


# Add detectable effect parameter to each metric characteristics for all eval metrics 

Gc['d'] = 0.01
R['d'] = 0.01
Nc['d'] = 0.0075


# ### 7.1 Gross conversion sample size calc.

# In[ ]:


Gc["SamSiz"]=round(get_sampSize(get_sd(Gc["p"], 
                                          Gc["d"]),
                                  0.05,0.2,
                                  Gc["d"])
                  )

Gc["SamSiz"]


# #### Gross conversion sample size required = at least 25,835 cookies who clicked Start Free Trial per group 
# 
# - For example, 800 'start free trial' clicks out of 5000 page views, 500/5000 = 0.16
# 
# - then Gc['SamSiz]/0.16 = 161,468.75 pageviews 

# In[ ]:


Gc['SamSiz'] = round(Gc['SamSiz'] / 0.16*2)

Gc['SamSiz']


# #### Total amount of samples per Gross Conversion metric = 322938.0

# ### 7.2 Retention sample size calc.

# In[ ]:


R['SamSiz'] = round(get_sampSize(get_sd(R['p'],
                                        R['d']),
                                        0.05, 0.2,
                                       R['d']
                                )
                   )
                

R['SamSiz']


# #### At least 39087.0 users enrolled per group 
# 
# - Convert this val to cookies who clicked & cookies who viewed page
# 
# - Multiply both groups by 2 

# In[ ]:


# 0.08 = CTF
# 0.20625 = GConversion
    # 0.20625 = base conversion rate (probability of enrolling, given click)
    
R['SamSiz'] = R['SamSiz'] / 0.08 / 0.20625 * 2

R['SamSiz']


# #### > 4 million page views total
# 
# - not possible, only 40,000 cookies (pageviews) a day
# 
# - Drop metric because results of experiment will be biased, since it is smaller in scale 

# ### 7.3 Net conversion sample size calc.

# In[ ]:


Nc['SamSiz'] = round(get_sampSize(get_sd(Nc['p'], 
                                         Nc['d']), 
                                  0.05, 0.2, 
                                  Nc['d']
                                 )
                    )
Nc['SamSiz']


# #### At least 27,413 cookies who click per group required 
# 
# - Convert to cookies who clicked through (CTP)

# In[ ]:


Nc['SamSiz'] = Nc['SamSiz']/0.08*2

Nc['SamSiz']


# - 685,325 cookies who viewed the page
# 
# - Choose Nc sample size since it is greater than Gross Conversion 

# ## 8. Experimental Analysis
# 
# - Sanity checks for INVAR metrics 

# In[ ]:


cont = pd.read_csv('../input/control-data/control_data.csv')

exp = pd.read_csv('../input/experiment-data/experiment_data.csv')


# In[ ]:


cont.head()


# ### 8.1 Sanity Checks
# 
# - Verify experiment is conducted as expected 
# 
# - Confirm other factors did not influence data collected 
# 
# - Ensure data collection method was correct 
# 
# ### 3 INVAR metrics 
# 
# 1) Number of cookies in course overview 
# 
# 2) Number of clicks on free trial 
# 
# 3) Free Trial CTP
# 
# ### Check if these observed values behave as expected i.e. if experiment is damaged 

# ### 8.1.1 Sanity checks for differences between cookie counts 
# 
# - Check if there is significant quantity difference in the total number of cookie pageviews diverted to each group
# 
# - Significant difference implies biased experiment 
# 
# - Check pageview cookies & click on free trial cookies 

# ### Pageviews 

# In[ ]:


pageviews_cont = cont['Pageviews'].sum()
pageviews_exp = exp['Pageviews'].sum()

pageviews_total = pageviews_cont + pageviews_exp

print('number of control pageviews: ', pageviews_cont)

print('number of experiment pageviews: ', pageviews_exp)


# ### 8.1.2 Check if difference is random i.e. not significant 
# 
# - Expectation = number of pageviews in cont is 50% of total pageviews 
# 
# - Test if observed $\hat{p}$ is not significantly different from $\hat{p} = 0.5$
# 
# - Calc Margin of Error (ME) at 95% confidence level 
# 
# <center> <font size="4"> $ ME=Z_{1-\frac{\alpha}{2}}SD$ </font></center>
# 
# - Where confidence interval is: 
# 
# <center> <font size="4"> $ CI=[\hat{p}-ME,\hat{p}+ME]$ </font></center>
# 
# - if obsereved $\hat{p}$ is within CI range, then experiment is fine

# In[ ]:


p = 0.5

alpha = 0.05

# Proportion of total 
p_hat = round(pageviews_cont / (pageviews_total), 4)

sd = mt.sqrt(p * (1 - p) / (pageviews_total))

ME = round(get_z_score(1 - (alpha / 2)) * sd, 4)

print('Confidence interval is between', p - ME, 'and', p + ME, '; Verify', p_hat, 'is within this range')


# ### Click counts

# In[ ]:


clicks_cont = cont['Clicks'].sum()
clicks_exp = exp['Clicks'].sum()
clicks_total = clicks_cont + clicks_exp

p_hat = round(clicks_cont / clicks_total, 4)
sd = mt.sqrt(p * (1 - p) / clicks_total)

ME = round(get_z_score(1 - (alpha / 2)) * sd, 4)

print('Confidence interval is between', p - ME, 'and', p + ME, '; Verify', p_hat, 'is within this range')


# ### 8.1.4 Check for differences between probabilities
# 
# - Verify proportion of clicks is approx. the same in both groups 
# 
# - CTP is not expected to change due to experiment. 
# 
# - Calc CTP in each group and calc confidence interval for expected difference between them 
# 
# - Confirm if $CTP_{exp}-CTP_{cont} = 0$ under an acceptable ME and calculated confidence interval 
# 
# - CTP is a proportion of the population, similar to how amount of clicks is a proportion of amount of pageviews i.e. amount of events x in population n
# 
# - Detectable changes should be calc of standard error i.e. pooled standard error 

# In[ ]:


ctp_cont = clicks_cont / pageviews_cont
ctp_exp = clicks_exp / pageviews_exp

# Detecable change
d_hat = round(ctp_exp - ctp_cont, 4)

p_pooled = clicks_total / pageviews_total

sd_pooled = mt.sqrt(p_pooled * (1 - p_pooled) * (1 / pageviews_cont + 1 / pageviews_exp))

ME = round(get_z_score(1-(alpha / 2)) * sd_pooled, 4)

print('Confidence interval is between', 0-ME, 'and', 0+ME, '; Verify', d_hat, 'is within this range')


# ### Test passes on all INVAR metrics 

# ## 9. Effect size computation
# 
# - Check changes between control & experiwment groups for EVAL metrics 
# 
# - Ensure difference exists, is statistically significant & practically significant 
# 
# - Practical significance = detectable difference is large enough to justify experiment change 
# 
# 
# > * **Signficance:**
# A metric is statistically significant if the confidence interval does not include 0 at lower bound, i.e. confident there is a change. Similarly, practically significant if the confidence interval does not include the practical significance boundary, i.e. confident there is a change that matters to the business.)

# In[ ]:


# Count total clicks from complete records 
    # i.e. where rows with pageviews and clicks have corresponding values with enrollments and payments
    
clicks_cont = cont['Clicks'].loc[cont['Enrollments'].notnull()].sum()

clicks_exp = exp['Clicks'].loc[exp['Enrollments'].notnull()].sum()

print(clicks_cont, clicks_exp)


# ### 9.1 H1 validity 
# 
# - Less user frustration due to time commitments = less users enrollments after advisory popup 
# 
# <br> <center><font size="5"> 
# $\hat{p_{pool}}=\frac{x_{cont}+x_{exp}}{N_{cont}+N_{exp}}$ </font></center>
# 
# <center><font size="4">$SD_{pool}=\sqrt{\hat{p_{pool}}(1-\hat{p_{pool}}(\frac{1}{N_{cont}}+\frac{1}{N_{exp}})}$</font></center>

# In[ ]:


# Gross conversion = P(enrollment | click)

enrol_cont = cont['Enrollments'].sum()
enrol_exp = exp['Enrollments'].sum()

GC_cont = enrol_cont / clicks_cont 
GC_exp = enrol_exp / clicks_exp

total_enrol = enrol_cont + enrol_exp # xcont + xexp
total_clicks = clicks_cont + clicks_exp

# p_hat pool 
GC_pooled = total_enrol / total_clicks 

GC_sd_pooled = mt.sqrt(GC_pooled * (1 - GC_pooled) * (1 / clicks_cont + 1 / clicks_exp))

GC_ME = round(get_z_score(1 - alpha / 2) * GC_sd_pooled, 4)

GC_diff = round(GC_exp - GC_cont, 4)

print('Change due to experiment is', GC_diff * 100, '%')

print('Confidence interval: [', GC_diff - GC_ME, ',', GC_diff + GC_ME, ']')

print('Statistically signfiicant if interval does not include 0')

print('Practically significant if interval does not contain', -Gc['D_min'])


# ### Gross Conversion results 
# 
# - Negative change in experiment that is both statistically & practically significant 
# 
# >  -2.06% change = Gross Conversion rate of experiment group decreased as expected by 2% and this change was signfiicant i.e. less users enrolled after the advisory popup 
# 
# - H1 is valid, significant reduction 

# ### 9.2 H2 validity 
# 
# - No significant loss in user numbers continuing to free trial, **NO significant reduction

# In[ ]:


# Net conversion = number of paying users after 14-day boundary / number of clicks on Start Free Trial 
    # P(payment | clicks)
    
payments_cont = cont['Payments'].sum()
payments_exp = exp['Payments'].sum()

NC_cont = payments_cont / clicks_cont 
NC_exp = payments_exp / clicks_exp 

total_payments = NC_cont + NC_exp 

NC_pooled = total_payments / total_clicks 

NC_sd_pooled =  mt.sqrt(NC_pooled * (1 - NC_pooled) * (1 / clicks_cont + 1 / clicks_exp))

NC_ME = round(get_z_score(1 - alpha / 2) * GC_sd_pooled, 4)

NC_diff = round(NC_exp - NC_cont, 4)

print('Change due to experiment is', NC_diff * 100, '%')
print('Confidence interval: [', NC_diff - NC_ME, ',', NC_diff + NC_ME, ']')
print('Statistically significant if interval does not include 0.')
print('Practically significant if interval does not include', Nc['D_min'])


# ### Net Conversion results 
# 
# - Negative change of 0.5% 
# 
# > Experiment group decreased by 5% after pop-up 
# 
# - Statistically & practically insignificant 
# 
# - Within ranges 

# ## 10. Sign tests double check
# 
# - Check if differences are consistent between pairs of observations i.e. decrease / increase
# 
# - Check if trend of change observed (decreases) is consistently evident in daily observations 
# 
# - Compute daily metric value 
# 
# - Compute number of days metric is lower in experiment group
# 
# > - Day count = number of successes for binominal variable 
# > - where binomial variable = number of successes expected to achive from N experiments, given P(success)

# ### 10.1 Processing 

# In[ ]:


# Merge datasets 

merge = cont.join(other = exp,
                  how = 'inner',
                      lsuffix = '_cont',
                      rsuffix = '_exp')

print(merge.count())
merge.head()


# In[ ]:


# Drop incomplete rows (on any col with 23 observations)

merge = merge.loc[merge['Payments_exp'].notnull()]
merge.count()


# In[ ]:


# Create binary daily col for each metric 

    # Return 0 if control > experiment 
        # Return 1 if experiment > control 

# GC
x = merge['Enrollments_cont'] / merge['Clicks_cont']

y = merge['Enrollments_exp'] / merge['Clicks_exp']

merge['GC'] = np.where(x > y, 0, 1)

# NC
a = merge['Payments_cont'] / merge['Clicks_cont']

b = merge['Payments_exp'] / merge['Clicks_exp']

merge['NC'] = np.where(a > b, 0, 1)

merge.head()


# In[ ]:


# Experiment > control
GC_x = merge.GC[merge['GC'] == 1].count()
NC_x = merge.NC[merge['NC'] == 1].count()

GC_y = merge.GC[merge['GC'] == 0].count()
NC_y = merge.NC[merge['NC'] == 0].count()

n = merge.NC.count()

print('Number of cases for GC_x:', GC_x, '\n',
      'Number of cases for NC_x:', NC_x, '\n',
      'Number of total cases:', n)

print('Number of cases for GC_y:', GC_y, '\n',
      'Number of cases for NC_y:', NC_y, '\n')


# ## 10.2 Sign test construction 
# 
# - Check if number of days where exp > cont is likely to be observed again in new experiment (consistently)
# 
# - Assume $P(exp > cont, on a day)$ = 50% 
# 
# - Use binomial distribution
# > where $p = 0.5$ and number of experiment days to inform of the $P(exp > cont, on a day)$
# 
# - Compute probability of $x$ days being a success
# > where success = higher metric experiment value relative to control 
# 
# - Double the $P(success)$, given this is a two-tailed test
# 
# > Doubled value = p-value (proba of observing a more extreme value than or equal to actual observed results | null is true)
# 
# - If p-value > $a$, then result is not significant i.e. not consistent 
# 
# > If 5 extreme days are observed,then p-val for the test is $p-val = P(x <= 2)$
# where $P(x < = 5) = P(0) + P(1) + P(2) + P(3) + P(4) + P(5)$

# In[ ]:


# Function for calculating probability of x i.e. number of successes 

def get_proba(x, n):
    p = round(mt.factorial(n) / (mt.factorial(x) * mt.factorial(n - x)) * 0.5 ** x * 0.5 ** (n - x), 4)
    return p 

# Function to compute p-val from probabilities of maximum x 

def get_2tail_pval(x, n):
    p = 0
    for i in range(0, x + 1):
        p = p + get_proba(i, n)
    return 2 * p 


# ## 10.3 Test
# 
# - calc p-val for each metrics using GC_x, NC_x, n, and functions
# 
# - $a = 0.05$

# In[ ]:


# Significance of obsering extreme values 

print('GC change significant if', get_2tail_pval(GC_x, n), 'is lower than 0.05')
print('NC change significant if', get_2tail_pval(NC_x, n), 'is lower than 0.05')


# ### 10.4 Sign test result 
# 
# - GC change = significant 
# 
# - NC change = insignificant 
# 
# > Identical to effect size calculation 

# # 11. Conclusion
# 
# - Experiment failed to increase proportion of paying users with time advisory pop-up. 
# 
# - Experiment negatively changed gross conversion, little change for net conversion. 
# 
# - Decrease in enrollment not coupled with increase in users staying for 14 days boundary of payment. 
# 
# - Experiment launch is not recommended.
