#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import folium
from folium import plugins
sns.set()

import os
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

data_folder = '../input/crimes-in-boston'
# Any results you write to the current directory are saved as output.


# In[ ]:


CRIME_DATA_FILE = os.path.join(data_folder, "crime.csv")
print("Crime data:", CRIME_DATA_FILE)
crime_df = pd.read_csv(CRIME_DATA_FILE, encoding = 'ISO-8859-1')  # note the odd encoding...

print("Number of crimes:", len(crime_df))

# Peek a bit
crime_df.head()


# In[ ]:


# Do a bit of data processing
crime_df.UCR_PART = crime_df.UCR_PART.astype('category')
crime_df.OFFENSE_CODE_GROUP = crime_df.OFFENSE_CODE_GROUP.astype('category')
crime_df.SHOOTING.fillna("N", inplace=True)  # replace nan for shootings with 'N' for "no"
crime_df.OCCURRED_ON_DATE = pd.to_datetime(crime_df.OCCURRED_ON_DATE)  # use Pandas datetime
crime_df.Location = crime_df.Location.apply(eval)  # set location as tuple
crime_df.Lat.replace(-1.0, None, inplace=True)
crime_df.Long.replace(-1.0, None, inplace=True)
crime_df.dropna(subset=["Lat", "Long"], inplace=True)

rename_map = {
    "OFFENSE_CODE": "Code",
    "OFFENSE_CODE_GROUP": "Code_group",
    "OFFENSE_DESCRIPTION": "Description",
    "OCCURRED_ON_DATE": "Date",
    "DISTRICT": "District",
    "STREET": "Street",
    "YEAR": "Year",
    "MONTH": "Month",
    "HOUR": "Hour",
    "DAY_OF_WEEK": "Day_of_week",
    "SHOOTING": "Shooting"
}

crime_df.rename(columns=rename_map, inplace=True)
crime_df.sort_values(by="Date", inplace=True)


# We will focus on the major **Part One** crimes: burglaries, larceny, homicides...

# In[ ]:


# Let's keep the major crimes
crime_df = crime_df[crime_df.UCR_PART == "Part One"]
# Remove the unused categories
crime_df.Code_group.cat.remove_unused_categories(inplace=True)


# In[ ]:


crime_df.head()


# ## Statistics
# 
# Let's get a few statistics on the number and nature of crimes.

# In[ ]:


print(crime_df.Code_group.value_counts())
print()
print(crime_df.groupby("Year").Code_group.value_counts())


# In[ ]:


# NB: removing unused categories was helpful here because
# setting the code groups to 'categorical' before filtering Part One crimes
# meant Seaborn would pickup the empty Part (One, Two) categories
g = sns.catplot(y="Code_group", kind="count",
                data=crime_df,
                order=crime_df.Code_group.value_counts().index,
                aspect=1.6,
                palette="muted")
g.set_axis_labels("Number of occurrences", "Offense group")


# In[ ]:


# NB: removing unused categories was helpful here because
# setting the code groups to 'categorical' before filtering Part One crimes
# meant Seaborn would pickup the empty Part (One, Two) categories
g = sns.catplot(y="Code_group", col="Year", col_wrap=2, kind="count",
                data=crime_df,
                order=crime_df.Code_group.value_counts().index,
                aspect=1.5,
                height=3,
                palette="muted")
g.set_axis_labels(x_var="Number of occurrences", y_var="Offense group")


# ## Putting data on the map
# 
# We'll use Folium for data visualisation.

# In[ ]:


base_location = crime_df.Location.iloc[0]  # grab location of one offense
base_location


# In[ ]:


boston_map = folium.Map(location=base_location,
                        prefer_canvas=True,
                        zoom_start=12,
                        min_zoom=12)
plugins.ScrollZoomToggler().add_to(boston_map)

boston_map


# In[ ]:


# Now we add the homicides up to 2016
year = 2016
for row in crime_df[(crime_df.Code_group == "Homicide") & (crime_df.Year <= year)].itertuples(index=False):
    icon = folium.Icon(color='red', icon='times', prefix='fa')
    popup_txt = str(row.Date)
    if row.Shooting == 'Y':
        popup_txt = "Shooting " + popup_txt
    folium.Marker(row.Location, icon=icon, popup=popup_txt).add_to(boston_map)

boston_map


# # Modelling

# Let's try modelling the occurrence of crimes as a random process of time.

# We'll consider the number of homicides as a counting process $N = \{ N_t \}$.
# 
# Let's plot it:

# In[ ]:


times_ = crime_df[crime_df.Code_group == "Homicide"].groupby(by="Year").Date

# Create counters for number of events
counts_ = []
for i, t in times_:
    counts_.append(np.arange(len(t)))


# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()
for i, (year, t) in enumerate(times_):
    axes[i].step(t, counts_[i], where='post')
    axes[i].set_title(f"Number of homicides. Year = {year}")
    axes[i].set_xlabel("Date")
    axes[i].set_ylabel("Homicide count")
fig.tight_layout()


# ## Simple model: Poisson process
# 
# We start by modelling the homicides as a Poisson process: the number of homicides occurring between $t_1$ and $t_2$ is taken to follow a Poisson distribution:
# $$
#     N(t_1,t_2] \sim \mathcal{P}\left(\lambda(t_2-t_1)\right).
# $$
# where $\lambda > 0$ is the _intensity_ parameter.

# There are several questions with such a model: over which timeframe can we consider $\lambda$ to be constant?

# The likelihood function of a sequence of events $\{t_1,\ldots,t_n\}$ under the model $\mathrm{PP}(\lambda)$ is
# $$
#     p(\{t_1, \ldots, t_n\}\mid \lambda) = \exp(-\lambda \Delta t)\lambda^n
# $$
# The MLE of the rate $\lambda$ is given by
# $$
#     \hat\lambda = \frac{n}{\Delta t},
# $$
# corresponding to the intuitive notion of what a "rate" is: the _crime rate_ or the _homicide rate_.
# 
# Let's compute the rates for years 2015-2018:

# In[ ]:


yearly_homic_rates = {}
for i, (y, ti) in enumerate(times_):
    dt_window = ti.iloc[-1] - ti.iloc[0]

    rate_mle = counts_[i][-1] / dt_window.days  # Poisson event rate in days
    yearly_homic_rates[y] = rate_mle
    print(f"Rate for year {y} (days^{{-1}}): {rate_mle:.3f}", f" i.e. every {1./rate_mle:.2f} days")


# We can see that, supposing the daily homicide rate over a year is homogeneous, the rates computed for each year are pretty much the same.

# ### Testing homogeneity - Ripley's $K$-function
# 
# According to the literature, we can test for homogeneity in a dataset by introducing Ripley's $K$-function
# \\[
#     K(t) = \mathbb{E} \left[ \#\{t_i \mid |t_j-t_i| \leq t\} \mid t_j \sim \mathcal{U}(\mathcal{T}) \right]
# \\]
# which measures how the process fills up an area around any event, and where $\mathcal{T} = \{t_i\}$ is a realization of the process and $t_j$ is uniformly chosen in $\mathcal{T}$ (for a stationary process the random choice of $t_j$ can be dropped and replaced by a fixed point in the observation window). It can be computed empirically as
# $$
#     \widehat{K}(t) = \lambda^{-1}n ^{-1}\sum_{i\neq j} \mathbf 1(|t_i-t_j|\leq t)
# $$
# 
# For an actual Poisson process, the theoretical Ripley $K$ function in one dimension is given by $K(t) = 2t$.

# In[ ]:


from scipy.spatial import distance as scdist
def ripley_K(data, t):
    """
    Args
        data (ndarray): point pattern data array
        t (float): interval radius
    """
    data = data.astype('datetime64[s]')  # ensure time data is in ms
    dist = scdist.pdist(data, 'euclidean')  # compute the array of pair-wise distances
    lbda = data.size / (data[-1] - data[0]).astype(float)
    ksum = 1./lbda * np.sum((dist <= t), axis=-1) / data.shape[0]
    return ksum


# In[ ]:


evt_time_data = crime_df[crime_df.Code_group == "Homicide"].Date[:, None].astype('datetime64[s]')

dist = scdist.pdist(evt_time_data.astype("datetime64[s]"), 'euclidean')
dist


# In[ ]:


lbda = evt_time_data.size / (evt_time_data[-1] - evt_time_data[0]).astype(float)
print(f"Event rate {lbda.item():.3e} events/s")
print(f"Event rate {86400*lbda.item():.3f} events/day")


# In[ ]:


# Let's make our testing time values run over 1 1/2 years
ripley_test_times = np.linspace(0, 86400 * 365 * 1.5, 150)  # time expressed in seconds

kstat = ripley_K(evt_time_data, ripley_test_times[:, None])


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.plot(ripley_test_times, kstat, label="Empirical Ripley $\widehat{K}(t)$")
ax.plot(ripley_test_times, 2*ripley_test_times, label="Theoretical Ripley $K(t) = 2t$")
ax.legend()
ax.set_title("Ripley $K$-function")
ax.set_xlabel("Time $t$ (s)")
ax.set_ylabel("Statistic $\widehat{K}(t)$")


# Here, we have that the empirical Ripley function is $\hat{K}(t) < 2t$, which means the temporal point pattern is repulsive (with increased repulsiveness at long distances) and we can **reject the homogeneous Poisson hypothesis**.

# ## Nonhomogeneous Poisson process
# 
# Now we suppose that
# $$
#     N(t_1, t_2] \sim \mathcal{P}\left(\int_{t_1}^{t_2} \lambda(s)\,ds \right)
# $$

# ### Kernel estimation
# 
# Given a set of observed events $\mathcal T = \{ t_i\}$, we can estimate the intensity function as
# $$
#     \widehat{\lambda}(t) = \sum_{t_i\in\mathcal T} K\left(\frac{t-t_i}{h}\right)
# $$
# where $h > 0$ is the bandwidth and $K$ is a smoothing kernel, with $\int K(x)\,dx = 1$.

# We introduce the folliwing kernels:
# * local-average kernel $K_h(x) = \frac{1}{2h}\mathbf{1}_{|x|\leq h}$.
# * Gaussian kernel:
# $$
#     K_h(x) = \frac{1}{\sqrt{2\pi h^2}} \exp\left( - \frac{x^2}{2h^2} \right)
# $$

# In[ ]:


class LocalAverageKernel:
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
    
    def __call__(self, x):
        normalize_cst = float(2. * self.bandwidth)
        ks = (np.abs(x) <= self.bandwidth).astype(dtype='float')
        return ks / normalize_cst

class GaussianKernel:
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
    
    def __call__(self, x):
        x = x.astype(float)
        bw = float(self.bandwidth)  # in nanoseconds
        argument = - 0.5 * (x / bw) ** 2
        return np.exp(argument) / np.sqrt(2 * np.pi * bw ** 2)


# In[ ]:


dfmt = '%Y%m%d'


# In[ ]:


# Let's do a kernel estimation for 2017
evt_times_ = times_.get_group(2017)


# In[ ]:


plot_dates = pd.date_range('20170101', '20180101', freq='D')


# In[ ]:


print(plot_dates.shape)

print(evt_times_[None, :].shape)


# In[ ]:


np.shape(plot_dates[:, None] - evt_times_[None, :])


# In[ ]:


bandwidths = []
for bw in [7, 10, 15, 20]:
    bandwidth = pd.Timedelta('1D') * bw
    print("Bandwidth:", bandwidth)
    bandwidths.append(bandwidth)

# Let's get a few different kernels for our bandwidths
kernels_ = [GaussianKernel(bw.asm8) for bw in bandwidths]


# In[ ]:


dts = plot_dates[:, None] - evt_times_[None, :]


# In[ ]:


num_ns_day = 86400 * 1e9  # number of nanoseconds in a day


# In[ ]:


intensity_estims_ = [num_ns_day * kern(dts).sum(axis=-1) for kern in kernels_]


# In[ ]:


plt.figure(figsize=(14, 9))
for j, ins_estim in enumerate(intensity_estims_):
    plt.plot(plot_dates, ins_estim, linestyle='-', label=f"Bandwidth = {bandwidths[j]}")
plt.title("Intensity estimate for homicides, year 2017")
ylims = plt.ylim()
plt.vlines(evt_times_, *ylims, linestyles='--', lw=1.0, alpha=0.8)
plt.ylim(*ylims)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Intensity (1/days)")

