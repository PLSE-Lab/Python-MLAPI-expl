#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Summary" data-toc-modified-id="Summary-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Summary</a></span></li><li><span><a href="#Model" data-toc-modified-id="Model-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Model</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#System-model:" data-toc-modified-id="System-model:-2.0.1"><span class="toc-item-num">2.0.1&nbsp;&nbsp;</span>System model:</a></span></li><li><span><a href="#Observation-model:" data-toc-modified-id="Observation-model:-2.0.2"><span class="toc-item-num">2.0.2&nbsp;&nbsp;</span>Observation model:</a></span></li><li><span><a href="#Constraints-in-parameters-and-initial-states:" data-toc-modified-id="Constraints-in-parameters-and-initial-states:-2.0.3"><span class="toc-item-num">2.0.3&nbsp;&nbsp;</span>Constraints in parameters and initial states:</a></span></li><li><span><a href="#Effective-reproduction-number-$R_t$:" data-toc-modified-id="Effective-reproduction-number-$R_t$:-2.0.4"><span class="toc-item-num">2.0.4&nbsp;&nbsp;</span>Effective reproduction number $R_t$:</a></span></li></ul></li></ul></li><li><span><a href="#Preparation" data-toc-modified-id="Preparation-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Preparation</a></span></li><li><span><a href="#Stan-Code" data-toc-modified-id="Stan-Code-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Stan Code</a></span></li><li><span><a href="#Inference" data-toc-modified-id="Inference-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Inference</a></span></li><li><span><a href="#Visualization-of-the-inference" data-toc-modified-id="Visualization-of-the-inference-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Visualization of the inference</a></span><ul class="toc-item"><li><span><a href="#Parameters-along-days-in-the-state" data-toc-modified-id="Parameters-along-days-in-the-state-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Parameters along days in the state</a></span></li><li><span><a href="#Inference-along-time-series" data-toc-modified-id="Inference-along-time-series-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Inference along time series</a></span></li></ul></li></ul></div>

# # Summary
# **See comments below for description of inference and discussions, as inference is not fully predictable before long run.**  
# - **Update on 2020-07-17:**  
#     - Corrected code on prior for smoothness of imputation for PCR-positive rate
#     - Increased sample size of gradient for better approximation by variational inference
# - **Update on 2020-07-16:**  
#     - Changed detection rate by PCR to be time-varying monotonically decreasing with positive rate of a coming week
#     - Corrected a mistake in implementation that sum of distribution of flow from infectious state is not 1.
#     - Removed fatality in exposed and detected state as it is not likely
#     - Dependence of recovery rate along time has deleted except for those in hospitals because it can be known inderectly from observaton only when effect of collective immunity is large.
#     - Distribution of transition rate along $d$ now shares common smoothness
#     - corrected caluculation of $R_t$ so as to include those not entered into infectious or hospitalized state  
# 
# - Introduced features below to [SIRD model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIRD_model) to predict future consequence of COVID-19 in Japan.
#     - time-varying effective contact rate per patient per day
#     - time-varying detection rate by PCR monotonically decreasing with positive rate of a coming week
#     - explicit delay in getting detectable, infectious, hospitalized, recovered and fatal
#     - separation of exposed, detectable, infectious and hospitalized state
#     - separation of recovery without hospitalization from discharge from hospital
# - Considered detection rate of infected patient by PCR and fatality as observation model of Bayesian state-space model.
# - Estimated parameters and observed and hidden states using Stan based on reported number of cumulative PCR-positive, hospitalized, cumulative discharge and cumulative fatality.

# # Model
# ### System model:  
# - State variable:  
# $t$: Elapsed time from the start date [day]  
# $d$: Days from infection (entry to exposed($E$) state) [day]  
# $S(t)$: Susceptible  
# $E(t, d)$: Exposed; infected but not detectable or infectious  
# $D(t, d)$: Detectable with PCR  
# $I(t, d)$: Infectious  
# $H(t, d)$: Hospitalized or waiting for hospitalization  
# $L(t)$: Left (discharged from) hospital  
# $R(t)$: Recovered without hospitalization  
# $F(t)$: Fatality  
# 
# \begin{align*}
# \mathrm{S} \overset{N^{-1} \beta(t) (I + H)}{\longrightarrow} &\mathrm{E} \overset{\gamma_E(d)}{\longrightarrow} \mathrm{R} \\
# & \mathrm{E} \overset{\chi(d)}{\longrightarrow} && \mathrm{D} \overset{\gamma_D(d)}{\longrightarrow} \mathrm{R} \\
# &&& \mathrm{D} \overset{\iota(d)}{\longrightarrow} && \mathrm{I} \overset{\alpha_I(d)}{\longrightarrow} \mathrm{F} \\
# &&&&& \mathrm{I} \overset{\gamma_I(d)}{\longrightarrow} \mathrm{R} \\
# &&&&& \mathrm{I} \overset{\nu(d)}{\longrightarrow} && \mathrm{H} \overset{\alpha_H(d)}{\longrightarrow} \mathrm{F} \\
# &&&&&&& \mathrm{H} \overset{\gamma_H(d)}{\longrightarrow} \mathrm{L} \\
# \end{align*}
# 
# - Transition rate among categorical state:  
# $\beta(t)$: Effective contact rate [day$^{-1}$]  
# $\gamma_E(d)$, $\gamma_D(d)$, $\gamma_I(d)$, $\gamma_H(d)$: Recovery rate for the exposed($E$), detectable($D$), infectious($I$), and hospitalized($H$) [day$^{-1}$]  
# $\alpha_I(d)$, $\alpha_H(d)$: Mortality rate for the infectious($I$), and hospitalized($H$) [day$^{-1}$]  
# $\chi(d)$: Transition rate from exposed($E$) to detectable($D$) state [day$^{-1}$]  
# $\iota(d)$: Transition rate from detectable($D$) to infectious($I$) state [day$^{-1}$]  
# $\nu(d)$: Transition rate from infectious($I$) to hospitalized($H$) state [day$^{-1}$]  
# 
# - Constants:  
# $N$: Total population  
# $d_{max}$: Maximum considered $d$ (days elapsed from transition to the state)  
# 
# - Other Parameters:  
# $v_\beta$: Roughness of time evolution of effective contact rate  
# $v_r$: Roughness of change in transition rate except for those from hospitalized state  
# $v_d$: Roughness of change in detection rate by PCR for imputation  
# I do not assume smoothness for hospitalized patients in recovery and mortality along $d$ because roughness due to insituitional facotrs may exists.  

# - Time evolution
# \begin{align*}
# S(0) &= N - \left[ \sum_{d=0}^{d_{max}} \left\{I(0, d) + L(0, d) + H(0, d)\right\} + R(0) + D(0) \right] \\
# S(t) &= \left[1 - N^{-1} \beta(t) \sum_{d=0}^{d_{max}} \{ I(t-1, d-1) + H(t-1, d-1) \}\right] S(t-1) \\[10pt]  
# E(t, 0) &= N^{-1} \beta(t-1) S(t-1) \sum_{d=0}^{d_{max}} \{ I(t-1, d-1) + H(t-1, d-1) \} \\
# E(t, d) &= \left[1 - \{\gamma_E(d-1) + \chi(d-1)\} \right] E(t-1, d-1) \\
# E(t, d_{max}) &= [1 - \{\gamma_E(d_{max}) + \chi(d_{max})\}]E(t-1, d_{max}) + [1 - \{\alpha_E(d_{max}-1) + \gamma_E(d_{max}-1) + \chi(d_{max}-1)\}] E(t-1, d_{max}-1) \\[10pt]
# D(t, 0) &= \sum_{d=0}^{d_{max}} \iota(d) E(t-1, d) \\
# D(t, d) &= \left[1 - \{\gamma_D(d-1) + \nu(d-1)\} \right] D(t-1, d-1) \\
# D(t, d_{max}) &= [1 - \{\gamma_D(d_{max}) + \nu(d_{max})\}] D(t-1, d_{max}) + [1 - \{\alpha_D(d_{max}-1) + \gamma_D(d_{max}-1) + \nu(d_{max}-1)\}] D(t-1, d_{max}-1) \\[10pt]
# I(t, 0) &= \sum_{d=0}^{d_{max}} \iota(d) D(t-1, d) \\
# I(t, d) &= \left[1 - \{\alpha_I(d-1) + \gamma_I(d-1) + \nu(d-1)\} \right] I(t-1, d-1) \\
# I(t, d_{max}) &= [1 - \{\alpha_I(d_{max}) + \gamma_I(d_{max}) + \nu(d_{max})\}] I(t-1, d_{max}) + [1 - \{\alpha_I(d_{max}-1) + \gamma_I(d_{max}-1) + \nu(d_{max}-1)\}] I(t-1, d_{max}-1) \\[10pt]
# H(t, 0) &= \sum_{d=0}^{d_{max}} \nu(d) I(t-1, d) \\
# H(t, d) &= \left[1 - \{\alpha_H(d-1) + \gamma_H(d-1)\right] H(t-1, d-1) \\
# H(t, d_{max}) &= [1 - \{\alpha_H(d_{max}) + \gamma_H(d_{max})\}] H(t-1, d_{max}) + [1 - \{\alpha_H(d_{max}-1) + \gamma_H(d_{max}-1) \}] H(t-1, d_{max}-1) \\[10pt]
# L(t) &= L(t-1) +  \sum_{d=0}^{d_{max}} \gamma_H(d) H(t-1, d) \\[10pt]
# R(t) &= R(t-1) + \sum_{d=0}^{d_{max}} \{ \gamma_E(d) E(t-1, d) + \gamma_D(d) D(t-1, d) + \gamma_I(d) I(t-1, d) \} \\[10pt]
# F(t) &= F(t-1) + \sum_{d=0}^{d_{max}} \{ \alpha_I(d) I(t-1, d) + \alpha_H(d) H(t-1, d) \} \\[10pt]
# \end{align*}

# ### Observation model:  
# $P_{obs}(t) \sim Poisson\left( p_d(t) \sum_{s=0}^{t} \sum_{d=0}^{d_{max}} N^{-1} \beta(s) S(s) \{ I(s, d-1) + H(s, d-1) \} \right)$,  
# where $P_{obs}(t)$ is reported cumulative number of PCR-positive and $p_d(t)$ is detection rate of detectable patient.  
# 
# $H_{obs}(t) \sim Poisson\left( H(t) \right)$,  
# where $H_{obs}(t)$ is reported hospitalization.  
# 
# $L_{obs}(t) \sim Poisson\left( L(t) \right)$,  
# where $L_{obs}(t)$ is reported cumulative leave (discharge) from hospital.  
# 
# $F_{obs}(t) \sim Poisson\left( \sum_{s=0}^{t} \sum_{d=0}^{d_{max}} p_f\{ \alpha_E(d) E(s, d) + \alpha_D(d) D(s, d) + \alpha_I(d) I(s, d) \} + \alpha_H(d) H(s, d) \} \right)$,  
# where $F_{obs}(t)$ is reported cumulative fatality and $p_f(t)$ is detection rate of fatality outside hospital.  

# ### Constraints in parameters and initial states:  
# $0 < \alpha_I < \alpha_H < 10^{-2}$  
# $0 < p_d < 10^{-1}$  
# $D(0, d) < D(0, d-1)$  
# $E(0, d) < E(0, d-1)$  
# $I(0, d) < I(0, d-1)$  

# ### Effective reproduction number $R_t$:  
# I defined effective reproduction number $R_t$ as expected number of reproduction of infectious patients($I$) by patients infected at $t$, assuming effective contact rate $\beta(t)$ is fixed to the value at $t$.  
# Note that $R_t$ here includes effect of imported patients as returnees and foreigners.  
# Effective reproduction number is factorized as  
# $R_t = N^{-1} S(t) \beta(t) \tau $,  
# where $\beta(t)$ is effective contact rate [day$^{-1}$] and $\tau$ [day] is mean duration in the infectious($I$ and $H$) states.  
# 
# $\tau = \frac { \left( \sum_{d_I=1}^{d_{max}} d_I \{ \alpha_2(d_I) + \gamma_2(d_I) \} +  \sum_{d_I=1}^{d_{max}} \sum_{d_H=1}^{d_{max}} (d_I + d_H) [ \nu(d_I) \{ \alpha_3(d_H) + \gamma_3(d_H) \} ] \right) } {Z}$,  
# where $Z$ is a normalization contant.  
# $Z = \sum_{d_I=1}^{d_{max}} \{ \alpha_2(d_I) + \gamma_2(d_I) \} +  \sum_{d_I=1}^{d_{max}} \sum_{d_H=1}^{d_{max}} \nu(d_I) \{ (\alpha_3(d_H) + \gamma_3(d_H) \} $.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import datetime

plt.rcParams["font.size"] = 14


# # Preparation

# In[ ]:


ts_raw_df = pd.read_csv("/kaggle/input/covid19-dataset-in-japan/covid_jpn_total.csv", parse_dates=["Date"])
ts_raw_df.head()


# In[ ]:


ts_raw_df.tail()


# In[ ]:


def sum_nan(x):
    if x.isna().all():
        return np.nan
    else:
        return x.sum()

ts_df = ts_raw_df.groupby("Date").aggregate(sum_nan).asfreq("D")
ts_df.drop(columns="Location", inplace=True)
ts_df.tail()


# In[ ]:


def plot_diff(col):
    ax = ts_df.loc[:, col].diff().plot(figsize=(16, 3), style=["b."])
    ax.set_title(col)


# In[ ]:


def plot_cum(col):
    ax = ts_df.loc[:, col].plot(figsize=(16, 3), style=["b."])
    ax.set_title(col)


# In[ ]:


plot_cum("Positive")


# In[ ]:


plot_diff("Positive")


# In[ ]:


plot_cum("Tested")


# In[ ]:


ts_df["Positive_rate"] = (ts_df["Positive"].diff(7) / ts_df["Tested"].diff(7)).shift(-7)
plot_cum("Positive_rate")
plt.show()


# In[ ]:


plot_cum("Fatal")


# In[ ]:


ts_df.loc[:, ["Hosp_require","Hosp_mild","Hosp_severe","Hosp_unknown","Hosp_waiting"]].plot(figsize=(16, 3), style=["."] * 5)
plt.show()


# Total population of Japan (2019)

# In[ ]:


Total = 126166948


# In[ ]:


ts_df["Hospitalized"] = ts_df[["Hosp_mild","Hosp_severe","Hosp_unknown","Hosp_waiting"]].sum(axis=1, skipna=False)
ts_df[["Hospitalized","Discharged","Fatal"]].plot(figsize=(16, 3), style=["b.", "c.", "g.", "r."])
plt.show()


# In[ ]:


ts_df["Hosp_require_old"] = ts_df[["Hosp_mild","Hosp_severe","Hosp_unknown","Hosp_waiting"]].sum(axis=1, skipna=False)
ts_df[["Hosp_require_old","Hosp_require"]].plot(figsize=(16, 3), style=["b.","r."])
plt.show()


# In[ ]:


ts_df["Hosp_require_mod"] = ts_df["Hosp_require_old"].mask(ts_df["Hosp_require_old"].isna(), ts_df["Hosp_require"])
ts_df["Hosp_require_mod"].plot(figsize=(16, 3), style=["b."])
plt.show()


# In[ ]:


ts_df["Suceptible"] = Total - ts_df[["Positive","Discharged","Fatal"]].sum(axis=1, skipna=False)
ts_df["Suceptible"].plot(figsize=(16, 3), style=["k."])
plt.show()


# In[ ]:


N_test = 0
i_end_train = len(ts_df) - N_test
ts_stan_df = ts_df[["Suceptible","Positive","Positive_rate","Hosp_require_mod","Discharged","Fatal"]]
ts_stan_df = ts_stan_df.iloc[:i_end_train, :].fillna(-9999, downcast="infer")
ts_stan_df.head()


# In[ ]:


ts_stan_df.tail()


# In[ ]:


data = dict(
    N_train=i_end_train,
    N_pred=730,
    Total=Total,
    Detected=ts_stan_df["Positive"].values,
    Hospitalized=ts_stan_df["Hosp_require_mod"].values,
    Discharged=ts_stan_df["Discharged"].values,
    Fatal=ts_stan_df["Fatal"].values,
    Positive_rate=ts_stan_df["Positive_rate"].values,
    N_impute=int((ts_stan_df["Positive_rate"] < 0).sum()),
    i_impute=np.flatnonzero(ts_stan_df["Positive_rate"] < 0) + 1
)
data


# In[ ]:


date_sim = pd.date_range(ts_df.index.values.min(), periods=data["N_train"] + data["N_pred"])
date_sim


# # Stan Code

# In[ ]:


stan_code = '''
functions {
    vector decreasing_simplex(vector x) {
        int N = num_elements(x) + 1;
        vector[N] x_out;
        x_out[1] = 1;
        x_out[2] = x[1];
        for (i in 3:N)
            x_out[i] = x_out[i-1] * x[i-1];
        x_out = x_out / sum(x_out);
        return x_out;
    }

    real poisson_lh(real[] lambda, int start, int end, int[] x) {
        real l = 0;
        int i = 1;
        for (t in start:end) {
            if (x[t] > 0) {
                if (lambda[i] > 0)
                    l += poisson_lpmf(x[t] | lambda[i]);
                else
                    l += -1e10 + lambda[i];
            } else if (x[t] == 0) {
                if (lambda[i] > 0)
                    l += poisson_lpmf(x[t] | lambda[i]);
                else if (lambda[i] < 0)
                    l += -1e10 + lambda[i];
            }
            i += 1;
        }
        return l;
    }

    void smooth_lp(vector x, real v_raw) {
        int N = num_elements(x);
        real v = square(mean(x)) * v_raw;
        x[2:] ~ gamma(square(x[:(N-1)]) / v, x[:(N-1)] / v);
    }
}

data {
    int<lower=1> N_train;
    int<lower=1> N_pred;
    int<lower=1> Total;
    int Detected[N_train];
    int Hospitalized[N_train];
    int Discharged[N_train];
    int Fatal[N_train];
    vector[N_train] Positive_rate;
    int N_impute;
    int<lower=1> i_impute[N_impute];
}

transformed data {
    int count_data[N_train * 4] = append_array(Detected, 
                                    append_array(Hospitalized, 
                                        append_array(Discharged, Fatal)));
    int N_chunk = N_train;

    int max_Delay = 14;
    int max_Delay_I = 35;
    int max_Delay_H = 35;
    
    int N_sim = N_train + 1;
    int N_sim_pred = N_pred + 1;

    real Positive_0 = Detected[1];
    real sum_H0 = Hospitalized[1];
    real L0 = Discharged[1];
    real F0 = 0;

    real detect_by_pcr_min = 1e-3;
    real sum_D0_max = Positive_0 / detect_by_pcr_min;
    real sum_E0_max = sum_D0_max * 1e2;

    real max_Positive_rate = max(Positive_rate);
}

parameters {
    vector<lower=0, upper=1e1>[N_train] contact;

    // ordered mortality with upper limit
    real<lower=1e-2, upper=1> mortality_person_hosp_1;
    real<lower=0, upper=1> rel_mortality_person_I;

    // recovery and transition to more severe state
    simplex[2] recovery_person_exposed;
    simplex[2] recovery_person_detectable;
    simplex[2] recovery_person_infectious;

    simplex[max_Delay] get_detectable_days;

    simplex[max_Delay] get_infectious_days;
    
    simplex[max_Delay_I] mortality_days_infectious;
    simplex[max_Delay_I] hospitalize_days;

    simplex[max_Delay_H] mortality_days_hosp;
    simplex[max_Delay_H] recovery_days_hosp;
    
    real<lower=1, upper=sum_E0_max> sum_E0;
    real<lower=1, upper=sum_D0_max> sum_D0;
    real<lower=1, upper=sum_D0_max> sum_I0;
    
    vector<lower=0, upper=1>[max_Delay - 1] dist_E0_age_dec;
    vector<lower=0, upper=1>[max_Delay - 1] dist_D0_age_dec;
    vector<lower=0, upper=1>[max_Delay_I - 1] dist_I0_age_dec;

    simplex[max_Delay_H] dist_H0_age;

    vector<lower=0, upper=max_Positive_rate>[N_impute] imputation;
    real<lower=0, upper=1> scale_detect;
    real<lower=0> gain_detect;
    real<lower=0, upper=max_Positive_rate> intercept_detect;

    real<lower=1e-1, upper=1> detect_fatality;

    real<lower=1e-8, upper=1e-4> v_contact;
    real<lower=1e-8, upper=1e-4> v_rate_dist;
    real<lower=square(max_Positive_rate)*1e-8, upper=square(max_Positive_rate)*1e-4> v_Positive_rate;
}

transformed parameters {
    
    vector[3] branch_person_infectious;
    vector[2] mortality_person_hosp;
    
    real recovery_exposed = recovery_person_exposed[1] / max_Delay;
    row_vector[max_Delay] get_detectable;

    real recovery_detectable = recovery_person_detectable[1] / max_Delay;
    row_vector[max_Delay] get_infectious;
    
    row_vector[max_Delay_I] mortality_infectious;
    real  recovery_infectious;
    row_vector[max_Delay_I] hospitalize;

    row_vector[max_Delay_H] mortality_hosp;
    row_vector[max_Delay_H] recovery_hosp;

    vector[max_Delay] dist_E0_age = decreasing_simplex(dist_E0_age_dec);
    vector[max_Delay] dist_D0_age = decreasing_simplex(dist_D0_age_dec);
    vector[max_Delay_I] dist_I0_age = decreasing_simplex(dist_I0_age_dec);
    
    row_vector[max_Delay] E0 = (sum_E0 * dist_E0_age)';
    row_vector[max_Delay] D0 = (sum_D0 * dist_D0_age)';
    row_vector[max_Delay_I] I0 = (sum_I0 * dist_I0_age)';
    row_vector[max_Delay_H] H0 = (sum_H0 * dist_H0_age)';

    real R0;

    vector[N_sim] S;
    vector[N_sim] E_out;
    vector[N_sim] D_out;
    vector[N_sim] I_out;
    vector[N_sim] H_out;
    vector[N_sim] L;
    vector[N_sim] R;
    vector[N_sim] F;

    vector[N_sim] Positive;
    vector[N_sim] F_report;

    row_vector[max_Delay] E;
    row_vector[max_Delay] D;
    row_vector[max_Delay_I] I;
    row_vector[max_Delay_H] H;

    vector[N_train] detect;
    vector[N_train] Positive_rate_imputed = Positive_rate;

    for (j in 1:N_impute)
        Positive_rate_imputed[i_impute[j]] = imputation[j];
    detect = scale_detect * inv(1 + exp(gain_detect * (Positive_rate_imputed - intercept_detect)));

    branch_person_infectious[1] = mortality_person_hosp_1 * rel_mortality_person_I;
    branch_person_infectious[2:3] = (1 - branch_person_infectious[1]) * recovery_person_infectious;

    mortality_person_hosp[1] = mortality_person_hosp_1;
    mortality_person_hosp[2] = 1 - mortality_person_hosp_1;
    
    {
        real rel_prob_to_L = recovery_person_exposed[2] * recovery_person_detectable[2] * branch_person_infectious[3] * mortality_person_hosp[2];
        real rel_prob_to_R = recovery_person_exposed[1] + recovery_person_exposed[2] * (recovery_person_detectable[1] + recovery_person_detectable[2] * branch_person_infectious[2]);
        R0 = L0 * rel_prob_to_R / rel_prob_to_L;
    }

    get_detectable = (recovery_person_exposed[2] * get_detectable_days)';

    get_infectious = (recovery_person_detectable[2] * get_infectious_days)';
        
    mortality_infectious = (branch_person_infectious[1] * mortality_days_infectious)';
    recovery_infectious = branch_person_infectious[2] / max_Delay_I;
    hospitalize = (branch_person_infectious[3] * hospitalize_days)';

    mortality_hosp = (mortality_person_hosp[1] * mortality_days_hosp)';
    recovery_hosp = (mortality_person_hosp[2] * recovery_days_hosp)';
 
    //time evolution
    {
        real flux_SE;
        row_vector[max_Delay] flux_EE;
        row_vector[max_Delay] flux_EF;
        row_vector[max_Delay] flux_ER;
        row_vector[max_Delay] flux_ED;

        row_vector[max_Delay] flux_DD;
        row_vector[max_Delay] flux_DF;
        row_vector[max_Delay] flux_DR;
        row_vector[max_Delay] flux_DI;
        
        row_vector[max_Delay_I] flux_II;
        row_vector[max_Delay_I] flux_IF;
        row_vector[max_Delay_I] flux_IR;
        row_vector[max_Delay_I] flux_IH;

        row_vector[max_Delay_H] flux_HH;
        row_vector[max_Delay_H] flux_HF;
        row_vector[max_Delay_H] flux_HL;
        
        real sum_flux_IF;
        real sum_flux_HF;

        real sum_E;
        real sum_D;
        real sum_I;
        
        S[1] = Total - (sum_E0 + sum_D0 + sum_I0 + sum_H0 + L0 + R0 + F0);
        E = E0;
        D = D0;
        I = I0;
        H = H0;
        L[1] = L0;
        R[1] = R0;
        F[1] = F0;
        
        Positive[1] = Positive_0;
        F_report[1] = Fatal[1];

        for (t in 2:N_sim) {
            flux_SE = S[t-1]*(sum(I) + sum(H))*contact[t-1] / Total;

            flux_ED = E .* get_detectable;
            Positive[t] = Positive[t-1] + sum(flux_ED)*detect[t-1];
            flux_ER = E * recovery_exposed;

            flux_DI = D .* get_infectious;
            flux_DR = D * recovery_detectable;
            
            flux_IR = I * recovery_infectious;
            flux_IH = I .* hospitalize;
            flux_IF = I .* mortality_infectious;

            flux_HL = H .* recovery_hosp;
            flux_HF = H .* mortality_hosp;
            
            sum_E = sum(E);
            sum_D = sum(D);
            sum_I = sum(I);
            
            S[t] = S[t-1] - flux_SE;

            flux_EE = E - flux_ED - flux_ER;
            E[1] = flux_SE;
            E[2:max_Delay] = flux_EE[1:(max_Delay-1)];
            E[max_Delay] += flux_EE[max_Delay];
            E_out[t] = sum_E;

            flux_DD = D - flux_DI - flux_DR;
            D[1] = sum(flux_ED);
            D[2:max_Delay] = flux_DD[1:(max_Delay-1)];
            D[max_Delay] += flux_DD[max_Delay];
            D_out[t] = sum_D;

            flux_II = I - flux_IH - flux_IR - flux_IF;
            I[1] = sum(flux_DI);
            I[2:max_Delay_I] = flux_II[1:(max_Delay_I-1)];
            I[max_Delay_I] += flux_II[max_Delay_I];
            I_out[t] = sum_I;
            
            flux_HH = H - flux_HL - flux_HF;
            H[1] = sum(flux_IH);
            H[2:max_Delay_H] = flux_HH[1:(max_Delay_H-1)];
            H[max_Delay_H] += flux_HH[max_Delay_H];
            H_out[t] = sum(H);
            
            R[t] = R[t-1] + sum(flux_ER) + sum(flux_DR) + sum(flux_IR);

            L[t] = L[t-1] + sum(flux_HL);
            
            sum_flux_IF = sum(flux_IF);
            sum_flux_HF = sum(flux_HF);
            F[t] = F[t-1] + sum_flux_IF + sum_flux_HF;
            F_report[t] = F_report[t-1] + sum_flux_IF*detect_fatality + sum_flux_HF;
        }
    }
}


model {
    real estimate[N_train * 5] = to_array_1d(
                                    append_row(Positive[2:N_sim],
                                        append_row(H_out[2:N_sim],
                                            append_row(L[2:N_sim], F_report[2:N_sim]))));

    target += reduce_sum(poisson_lh, estimate, N_chunk, count_data);

    smooth_lp(contact, v_contact);

    smooth_lp(get_detectable_days, v_rate_dist);
    smooth_lp(get_infectious_days, v_rate_dist);
    smooth_lp(hospitalize_days, v_rate_dist);
    smooth_lp(mortality_days_infectious, v_rate_dist);

    for (j in 1:N_impute) {
        Positive_rate_imputed[i_impute[j]] ~ gamma(square(Positive_rate_imputed[i_impute[j]-1]) / v_Positive_rate,
                                                    Positive_rate_imputed[i_impute[j]-1] / v_Positive_rate);
        if ((i_impute[j] < N_train) && (i_impute[j+1] != (i_impute[j] + 1)))
            Positive_rate_imputed[i_impute[j]+1] ~ gamma(square(Positive_rate_imputed[i_impute[j]]) / v_Positive_rate,
                                                        Positive_rate_imputed[i_impute[j]] / v_Positive_rate);
    }
}

generated quantities {
    vector[N_sim_pred] S_pred;
    vector[N_sim_pred] E_out_pred;
    vector[N_sim_pred] D_out_pred;
    vector[N_sim_pred] I_out_pred;
    vector[N_sim_pred] H_out_pred;
    vector[N_sim_pred] L_pred;
    vector[N_sim_pred] R_pred;
    vector[N_sim_pred] F_pred;

    vector[N_sim_pred] Positive_pred;
    vector[N_sim_pred] F_report_pred;

    real mean_duration_IH;
    vector[N_train] reproduction;

    {
        real contact_pred = contact[N_train];
        real detect_pred = detect[N_train];

        row_vector[max_Delay] E_pred = E;
        row_vector[max_Delay] D_pred = D;
        row_vector[max_Delay_I] I_pred = I;
        row_vector[max_Delay_H] H_pred = H;
        
        real flux_SE;
        row_vector[max_Delay] flux_EE;
        row_vector[max_Delay] flux_EF;
        row_vector[max_Delay] flux_ER;
        row_vector[max_Delay] flux_ED;

        row_vector[max_Delay] flux_DD;
        row_vector[max_Delay] flux_DF;
        row_vector[max_Delay] flux_DR;
        row_vector[max_Delay] flux_DI;
        
        row_vector[max_Delay_I] flux_II;
        row_vector[max_Delay_I] flux_IF;
        row_vector[max_Delay_I] flux_IR;
        row_vector[max_Delay_I] flux_IH;

        row_vector[max_Delay_H] flux_HH;
        row_vector[max_Delay_H] flux_HF;
        row_vector[max_Delay_H] flux_HL;
        
        real sum_flux_IF;
        real sum_flux_HF;
        
        real sum_E;
        real sum_D;
        real sum_I;

        S_pred[1] = S[N_sim];
        L_pred[1] = L[N_sim];
        R_pred[1] = R[N_sim];
        F_pred[1] = F[N_sim];
        
        Positive_pred[1] = Positive[N_sim];
        F_report_pred[1] = F_report[N_sim];

        for (t in 2:N_sim_pred) {
            
            flux_SE = S_pred[t-1]*(sum(I_pred) + sum(H_pred))*contact_pred / Total;

            flux_ED = E_pred .* get_detectable;
            Positive_pred[t] = Positive_pred[t-1] + sum(flux_ED)*detect_pred;
            flux_ER = E_pred * recovery_exposed;

            flux_DI = D_pred .* get_infectious;
            flux_DR = D_pred * recovery_detectable;
            
            flux_IR = I_pred * recovery_infectious;
            flux_IH = I_pred .* hospitalize;
            flux_IF = I_pred .* mortality_infectious;

            flux_HL = H_pred .* recovery_hosp;
            flux_HF = H_pred .* mortality_hosp;

            sum_E = sum(E_pred);
            sum_D = sum(D_pred);
            sum_I = sum(I_pred);
            
            S_pred[t] = S_pred[t-1] - flux_SE;

            flux_EE = E_pred - flux_ED - flux_ER;
            E_pred[1] = flux_SE;
            E_pred[2:max_Delay] = flux_EE[1:(max_Delay-1)];
            E_pred[max_Delay] += flux_EE[max_Delay];
            E_out_pred[t] = sum_E;

            flux_DD = D_pred - flux_DI - flux_DR;
            D_pred[1] = sum(flux_ED);
            D_pred[2:max_Delay] = flux_DD[1:(max_Delay-1)];
            D_pred[max_Delay] += flux_DD[max_Delay];
            D_out_pred[t] = sum_D;

            flux_II = I_pred - flux_IH - flux_IR - flux_IF;
            I_pred[1] = sum(flux_DI);
            I_pred[2:max_Delay_I] = flux_II[1:(max_Delay_I-1)];
            I_pred[max_Delay_I] += flux_II[max_Delay_I];
            I_out_pred[t] = sum_I;
            
            flux_HH = H_pred - flux_HL - flux_HF;
            H_pred[1] = sum(flux_IH);
            H_pred[2:max_Delay_H] = flux_HH[1:(max_Delay_H-1)];
            H_pred[max_Delay_H] += flux_HH[max_Delay_H];
            H_out_pred[t] = sum(H_pred);
            
            R_pred[t] = R_pred[t-1] + sum(flux_ER) + sum(flux_DR) + sum(flux_IR);
            L_pred[t] = L_pred[t-1] + sum(flux_HL);
            
            sum_flux_IF = sum(flux_IF);
            sum_flux_HF = sum(flux_HF);
            F_pred[t] = F_pred[t-1] + sum_flux_IF + sum_flux_HF;
            F_report_pred[t] = F_report_pred[t-1] + sum_flux_IF*detect_fatality + sum_flux_HF;
        }
    }

    // average duration in I and H until recovery or die
    // for all the domestically exposed patients
    {
        real numerator = 0;
        real denominator = 0;
        real p_get_infectious = recovery_person_exposed[2] * recovery_person_detectable[2];
        real p_I;
        real p_H;
        for (d_I in 1:max_Delay_I) {
            p_I = recovery_infectious + mortality_infectious[d_I];
            numerator += d_I * p_I;
            denominator += p_I;
            for (d_H in 1:max_Delay_H) {
                p_H = hospitalize[d_I] * (recovery_hosp[d_H] + mortality_hosp[d_H]);
                numerator += (d_I + d_H) * p_H;
                denominator += p_H;
            }
        }
        numerator *= p_get_infectious;
        denominator *= p_get_infectious;
        // probability for not getting infectious
        denominator += recovery_person_exposed[1] + recovery_person_exposed[2] * recovery_person_detectable[1];
        mean_duration_IH = numerator / denominator;
    }
    reproduction = (S[1:N_train] / Total) .* contact * mean_duration_IH;
}

'''


# In[ ]:


with open("model.stan", mode='w') as f:
    f.write(stan_code)


# In[ ]:


import sys
get_ipython().system('{sys.executable} -m pip install -U cmdstanpy ujson')


# In[ ]:


import cmdstanpy
cmdstanpy.install_cmdstan()


# In[ ]:


model = cmdstanpy.CmdStanModel(stan_file="model.stan")


# # Inference

# In[ ]:


start = datetime.datetime.now()
print(start)
os.environ["STAN_NUM_THREADS"] = "4"
try:
    inference = model.variational(data=data, algorithm="fullrank", grad_samples=32, iter=1000000, output_dir="./", save_diagnostics=False)
except Exception as e:
    print(e)
finally:
    print(datetime.datetime.now() - start)


# In[ ]:


from glob import glob
fn_stan = "model"
stdout_fns = [(f, os.path.getmtime(f)) for f in glob("{}*-stdout.txt".format(fn_stan))]
latest_stdout_fn = sorted(stdout_fns, key=lambda files: files[1])[-1]
print(latest_stdout_fn[0])
elbo_df = pd.read_table(latest_stdout_fn[0], engine="python", skiprows=48, skipfooter=3, sep="\s{2,}", skipinitialspace=True, index_col="iter")
elbo_df.tail()


# In[ ]:


ax = elbo_df["ELBO"].plot(logy="sym", style=".", ms=2, figsize=(15, 5))
ax.set_ylabel("ELBO")
plt.show()


# # Visualization of the inference

# In[ ]:


fns = [(f, os.path.getmtime(f)) for f in glob("{}*.csv".format(fn_stan))]
latest_fn = sorted(fns, key=lambda files: files[1])[-1]
print(latest_fn[0])
inference_df = pd.read_csv(latest_fn[0], engine="python", comment="#")
inference_df.head()


# In[ ]:


par_names = []
for n in inference_df.columns.tolist():
    if ("." in n):
        par_names.append(n[0:n.find(".")])
    else:
        par_names.append(n)
par_names = set(par_names)        
par_names


# In[ ]:


par_dim = {}
for name in par_names:
    if name.endswith("_raw") or name.startswith(("lp__")):
        continue
    dim_sample = 0
    for n in inference_df.columns.tolist():
        dim_sample += n.startswith(name + ".")
    if dim_sample == 0:
        dim_sample = 1
    
    par_dim[name] = dim_sample 
        
print(par_dim)


# In[ ]:


name_hist = []
for p, d in par_dim.items():
    if d <= 3:
        name_hist.append(p)
name_hist


# In[ ]:


name_hist = [
    'recovery_person_exposed',
    'recovery_person_detectable',
    'branch_person_infectious',
    'mortality_person_hosp',
    'mean_duration_IH',
    'sum_E0',
    'sum_D0',
    'sum_I0',
    'R0',
    'detect_fatality',
    'scale_detect',
    'gain_detect',
    'intercept_detect',
    'v_rate_dist',
    'v_contact',
    'v_Positive_rate',
]


# In[ ]:


from matplotlib.ticker import ScalarFormatter

n_panel = 0
for name in name_hist:
    n_panel += par_dim[name]
n_rows = int(math.ceil(n_panel / 4))
fig, ax_mat = plt.subplots(nrows=n_rows, ncols=4, figsize=(16, 4*n_rows))
ax = np.ravel(ax_mat)

i = 0
for name in name_hist:
    if par_dim[name] == 1:
        sample = inference_df[name]
        ax[i].hist(sample, bins=50)
        ax[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax[i].ticklabel_format(style="sci", axis="x", scilimits=(-3, 3))
        ax[i].set_title(name, fontsize=14)
        i += 1
    else:
        for j in range(1, par_dim[name] + 1):
            name_j = name + "." + str(j)
            sample = inference_df[name_j]
            ax[i].hist(sample, bins=50)
            ax[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax[i].ticklabel_format(style="sci", axis="x", scilimits=(-3, 3))
            ax[i].set_title(name_j, fontsize=14)
            i += 1
fig.subplots_adjust(wspace=0.3, hspace=0.4)


# ## Parameters along days in the state

# In[ ]:


name_age = []
for p, d in par_dim.items():
    if (3 < d) & (d < data["N_train"]) & ("dist_" not in p) & ("_days" not in p):
        name_age.append(p)
name_age


# In[ ]:


name_age = [
'mortality_infectious',
'mortality_hosp',
'recovery_hosp',
'get_detectable',
'get_infectious',
'hospitalize',
'E0',
'D0',
'I0',
'H0',
]


# In[ ]:


sample_dic = dict()
for name in name_age:
    sample = []
    for j in range(1, par_dim[name] + 1):
        sample.append(inference_df[name + "." + str(j)])
    sample_dic[name] = np.column_stack(sample)


# In[ ]:


nrows = math.ceil(len(name_age) / 2)
fig, ax_mat = plt.subplots(nrows=nrows, ncols=2, figsize=(16, 3.5*nrows))
ax = np.ravel(ax_mat)
for i, name in enumerate(name_age):
    sample = sample_dic[name]
    sns.boxplot(data=sample, ax=ax[i], color="dodgerblue", linewidth=1, fliersize=1)
    ax[i].set_xticks(np.arange(0, sample.shape[1], 5))
    ax[i].set_xticklabels(np.arange(1, sample.shape[1]+1, 5))
    ax[i].set_title(name)
fig.subplots_adjust(wspace=0.2, hspace=0.4)
fig.suptitle("days from transition to the state", x=0.5, y=0.1)
plt.show()


# ## Inference along time series

# In[ ]:


name_ts = []
# N_sim = data["N_train"] + data["N_pred"] + 1
for p, d in par_dim.items():
    if (d >= data["N_train"]):
        name_ts.append(p)
name_ts


# In[ ]:


name_ts = [
 'S',
 'E_out',
 'D_out',
 'I_out',
 'H_out',
 'L',
 'R',
 'F',
 'Positive',
 'Positive_rate_imputed',
 'detect',
 'F_report',
 'contact',
 'reproduction',
]


# In[ ]:


q = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
q_ts_dic = dict()
for name in name_ts:
    sample = []
    for j in range(2, par_dim[name] + 1):
        sample.append(inference_df[name + "." + str(j)])
    sample = np.column_stack(sample)

    if name not in ['contact', 'reproduction','Positive_rate_imputed','detect']:
        name_pred = name + "_pred"
        sample_pred = []
        for j in range(2, par_dim[name_pred] + 1):
            sample_pred.append(inference_df[name_pred + "." + str(j)])
        sample_pred = np.column_stack(sample_pred)
        sample = np.hstack([sample, sample_pred])

    q_ts_dic[name] = np.nanquantile(sample, q, axis=0)


# In[ ]:


q_ts_dic["S"].shape


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic["S"][i_list[0][0], :], y2=q_ts_dic["S"][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic["S"][2, :], "-", color="darkcyan", label="Median")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Suceptible")
# ax.set_ylim(ts_df["Suceptible"].min() * 0.5, ts_df["Suceptible"].max() * 1.025)
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))
v = "Positive"
v_data = v
for label, i_list in {"90% credible":((0, 4), "skyblue"), "50% credible":((1, 3), "royalblue")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="navy", label="Median")
ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported PCR-positive")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Cumulative PCR-positive (Domestic)")
ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

v = "Positive"
v_data = v

for label, i_list in {"90% credible":((0, 4), "skyblue"), "50% credible":((1, 3), "royalblue")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="navy", label="Median")
ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported PCR-positive")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Cumulative PCR-positive (Domestic)")
# ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic["E_out"][i_list[0][0], :], y2=q_ts_dic["E_out"][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic["E_out"][2, :], "-", color="darkcyan", label="Median")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Exposed (not detectable or infectious)")
# ax.set_ylim(q_ts_dic["E_out"][:data["N_train"]].min() * 0.1, q_ts_dic["E_out"][:data["N_train"]].max() * 10)
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic["E_out"][i_list[0][0], :], y2=q_ts_dic["E_out"][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic["E_out"][2, :], "-", color="darkcyan", label="Median")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Exposed (not detectable or infectious)")
ax.set_ylim(q_ts_dic["E_out"][:data["N_train"]].min() * 0.1, q_ts_dic["E_out"][:data["N_train"]].max() * 2)
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic["D_out"][i_list[0][0], :], y2=q_ts_dic["D_out"][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic["D_out"][2, :], "-", color="darkcyan", label="Median")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("PCR-detectable")
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic["D_out"][i_list[0][0], :], y2=q_ts_dic["D_out"][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic["D_out"][2, :], "-", color="darkcyan", label="Median")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("PCR-detectable")
ax.set_ylim(0, q_ts_dic["D_out"][4, :data["N_train"]].max() * 2)
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic["I_out"][i_list[0][0], :], y2=q_ts_dic["I_out"][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic["I_out"][2, :], "-", color="darkcyan", label="Median")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Infectious")
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic["I_out"][i_list[0][0], :], y2=q_ts_dic["I_out"][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic["I_out"][2, :], "-", color="darkcyan", label="Median")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Infectious")
ax.set_ylim(0, q_ts_dic["I_out"][4, :data["N_train"]].max() * 2)
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic["R"][i_list[0][0], :], y2=q_ts_dic["R"][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic["R"][2, :], "-", color="darkcyan", label="Median")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Cumulative Recovery without Hospitalization")
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic["R"][i_list[0][0], :], y2=q_ts_dic["R"][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic["R"][2, :], "-", color="darkcyan", label="Median")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Cumulative Recovery without Hospitalization")
ax.set_ylim(0, q_ts_dic["R"][2, :data["N_train"]].max() * 2)
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

v = "H_out"
v_data = "Hosp_require_mod"
for label, i_list in {"90% credible":((0, 4), "skyblue"), "50% credible":((1, 3), "royalblue")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="navy", label="Median")
ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported PCR-positive")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Hospitalized")
ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

v = "H_out"
v_data = "Hosp_require_mod"
for label, i_list in {"90% credible":((0, 4), "skyblue"), "50% credible":((1, 3), "royalblue")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="navy", label="Median")
ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported PCR-positive")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Hospitalized")
# ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

v = "L"
v_data = "Discharged"
for label, i_list in {"90% credible":((0, 4), "skyblue"), "50% credible":((1, 3), "royalblue")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="navy", label="Median")
ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported PCR-positive")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Discharged")
ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

v = "L"
v_data = "Discharged"
for label, i_list in {"90% credible":((0, 4), "skyblue"), "50% credible":((1, 3), "royalblue")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="navy", label="Median")
ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported PCR-positive")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Discharged")
# ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

v = "F_report"
v_data = "Fatal"
for label, i_list in {"90% credible":((0, 4), "skyblue"), "50% credible":((1, 3), "royalblue")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="navy", label="Median")
ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported PCR-positive")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Fatal")
ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

v = "F_report"
v_data = "Fatal"
for label, i_list in {"90% credible":((0, 4), "skyblue"), "50% credible":((1, 3), "royalblue")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="navy", label="Median")
ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported PCR-positive")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Fatal")
# ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

v = "F"
v_data = "Fatal"
for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="darkcyan", label="Median")
ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported PCR-positive")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Fatal")
ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

v = "F"
v_data = "Fatal"
for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():
    ax.fill_between(x=date_sim, y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim, q_ts_dic[v][2, :], "-", color="darkcyan", label="Median")
ax.plot(ts_df.index.values, ts_df[v_data], "k.", label="Reported Fatality")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Fatal")
# ax.set_ylim(-ts_df[v_data].max() * 0.025, ts_df[v_data].max() * 2)
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():
    ax.fill_between(x=date_sim[1:data["N_train"]], y1=q_ts_dic["reproduction"][i_list[0][0], :], y2=q_ts_dic["reproduction"][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim[1:data["N_train"]], q_ts_dic["reproduction"][2, :], "-", color="darkcyan", label="Median")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.axhline(1, color="blue", lw=1, label="Threshold (1.0)")
ax.set_title("Effective Reproduction Number")
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():
    ax.fill_between(x=date_sim[1:data["N_train"]], y1=q_ts_dic["reproduction"][i_list[0][0], :], y2=q_ts_dic["reproduction"][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim[1:data["N_train"]], q_ts_dic["reproduction"][2, :], "-", color="darkcyan", label="Median")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.axhline(1, color="blue", lw=1, label="Threshold (1.0)")
ax.set_title("Effective Reproduction Number")
ax.set_xlim(date_sim.min() - np.timedelta64(1, 'D'), date_sim[:(data["N_train"]-1)].max() + np.timedelta64(2, 'D'))
ax.tick_params(axis='x', labelrotation=90)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():
    ax.fill_between(x=date_sim[1:data["N_train"]], y1=q_ts_dic["contact"][i_list[0][0], :], y2=q_ts_dic["contact"][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim[1:data["N_train"]], q_ts_dic["contact"][2, :], "-", color="darkcyan", label="Median")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Effective Contact Rate (per infectious or hospitalized patient per day)")
ax.set_xlim(date_sim.min() - np.timedelta64(7, 'D'), date_sim.max() + np.timedelta64(7, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():
    ax.fill_between(x=date_sim[1:data["N_train"]], y1=q_ts_dic["contact"][i_list[0][0], :], y2=q_ts_dic["contact"][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim[1:data["N_train"]], q_ts_dic["contact"][2, :], "-", color="darkcyan", label="Median")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Effective Contact Rate (per infectious or hospitalized patinet per day)")
ax.set_xlim(date_sim.min() - np.timedelta64(1, 'D'), date_sim[:(data["N_train"]-1)].max() + np.timedelta64(2, 'D'))
ax.tick_params(axis='x', labelrotation=90)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 4))

v = "detect"
v_data = "Positive_rate"
for label, i_list in {"90% credible":((0, 4), "paleturquoise"), "50% credible":((1, 3), "darkturquoise")}.items():
    ax.fill_between(x=date_sim[1:data["N_train"]], y1=q_ts_dic[v][i_list[0][0], :], y2=q_ts_dic[v][i_list[0][1], :], color=i_list[1], label=label)
ax.plot(date_sim[1:data["N_train"]], q_ts_dic[v][2, :], "-", color="darkcyan", label="Median")
ax_r = ax.twinx()
ax_r.plot(ts_df.index.values, ts_df[v_data], "k.", label="Positive rate (right axis)")

ax.axvline(date_sim[data["N_train"]-1], color="red", lw=1, label="End of Train")
ax.set_title("Detection Rate with PCR")
ax.set_xlim(date_sim.min() - np.timedelta64(1, 'D'), date_sim[:(data["N_train"]-1)].max() + np.timedelta64(2, 'D'))
ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
ax_r.legend(loc="upper left", bbox_to_anchor=(1.05, 0.5))
plt.show()


# In[ ]:




