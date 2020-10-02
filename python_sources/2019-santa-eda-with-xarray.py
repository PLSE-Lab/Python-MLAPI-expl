#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook provides exploratory analysis of the scheduling problem and cost function from Kaggle's [Santa's Workshop Tour 2019](http://www.kaggle.com/c/santa-workshop-tour-2019) competition.  Per guidance from the organizers, the notebook does not include scheduling solutions, only an exploration of the problem setup.
# 
# This notebook leans heavily on the [xarray](http://xarray.pydata.org/en/stable/index.html) package, partly as a demonstration of its utility.  xarray is extremely handy for n-dimensional datasets with multiple variables.  Undoubtedly, there are places in the notebook where xarray usage could be improved, e.g. where a conversion to a dataframe has been made and is unnecessary.  Constructive feedback welcome.

# In[ ]:


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# # Data Preparation

# In[ ]:


def precompute_preference_cost(fam_dates, fam_size):
    """Precompute the scheduling cost for assigning each family to
    each preference level and return as a numpy array.
    """

    n_families, n_choices = fam_dates.shape

    cards = np.array([0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500])
    buffet_pp = np.array([0, 0, 9, 9, 9, 18, 18, 36, 36, 36, 36])
    heli_pp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 199, 398])

    def row_cost(row_id):
        return (cards + (buffet_pp + heli_pp) * fam_size[row_id])

    pref_cost = [row_cost(id) for id in np.arange(n_families)]
    pref_cost = np.array(pref_cost)

    return pref_cost


def date_popularities(ds_fam):
    """Calculate how many families and how many people prefer each date within
    each preference level and return as xarray data arrays.
    """

    # consider only the original 10 preferences and convert to dataframe
    ds = ds_fam[['fam_size', 'fam_prefs']].drop(pref=10)
    ds = ds.rename({'fam_prefs': 'date'})
    df = ds.to_dataframe()

    # count the number of families interested in each date
    df_num_fam = (df.groupby(['date', 'pref'])
                    .count()
                    .rename(columns={'fam_size': 'families'}))
    da_num_fam = xr.DataArray(df_num_fam['families'].unstack('pref'))

    # sum on family size to count the people interested in each date 
    df_num_ppl = (df.groupby(['date', 'pref'])
                    .sum()
                    .rename(columns={'fam_size': 'people'}))
    da_num_ppl = xr.DataArray(df_num_ppl['people'].unstack('pref'))

    return da_num_fam, da_num_ppl


def get_occ_linear(da_popularity):
    """Create a linear fit to family occupancy preferences for each date
    and return as a data array.
    """
    
    dates = da_popularity['date'].values
    pop = da_popularity.mean('pref').values
    
    coeff = np.polyfit(x=dates[1:], y=pop[1:], deg=1, full=False)
    
    occ_pred = coeff[1] + coeff[0] * dates
    return xr.DataArray(occ_pred,
                        coords=[dates],
                        dims=['date'])


def fam_diff_score(da_fam_prefs, da_date_pop, da_occ_fit):
    """Create data array with same shape as family preferences, but each
    entry contains the difference between the average popularity for that
    date and our linear fit to popularity.  Large values indicate
    families / preference levels that are difficult to schedule.
    """

    # popularity for each of 100 dates, averaged across 10 preferences
    date_pop = da_date_pop.mean('pref')
    
    # dicts with dates as keys and values for popularity or target occupancy
    pop_map = date_pop.to_dataframe().to_dict()['pop_ppl']
    target_map = da_occ_fit.to_dataframe().to_dict()['occ_fit']

    # create data array with difficulty score for each family / preference
    df_fam_prefs = da_fam_prefs.sel(pref=range(10)).to_dataframe()
    score = df_fam_prefs['fam_prefs'].map(lambda x: pop_map[x] - target_map[x])

    df_fam_score = pd.DataFrame(score)
    df_fam_score.rename(columns={'fam_prefs': 'pop_ppl'}, inplace=True)

    da_fam_score = xr.DataArray(df_fam_score['pop_ppl'].unstack('pref'))

    return da_fam_score


def family_preprocess(fn_family):
    """Given a path to a family data csv, read in the file, structure the
    data as an xarray dataset, and add derived variables such as the popularity
    of each date.
    """

    # read in data and pull out dataframe of family date preferences,
    # family sizes, # families
    df = pd.read_csv(fn_family)
    fam_dates = df.drop(columns=['family_id', 'n_people'])
    fam_sizes = df['n_people']
    n_families = df.shape[0]
    
    # families select 10 preferred dates, but could also be assigned an
    # alternative date. make room for one more column to store alternative.
    n_choices = fam_dates.shape[1] + 1
    fam_dates['choice_10'] = -1 * np.ones(n_families, dtype='int')

    # pre-compute cost of scheduling families in each preference
    pref_cost = precompute_preference_cost(fam_dates, fam_sizes)

    # Create xarray dataset
    ds_fam = xr.Dataset(
        {'fam_size': (('fam'), fam_sizes),
         'fam_prefs': (('fam', 'pref'), fam_dates),
         'pref_cost': (('fam', 'pref'), pref_cost),
        },
        {'fam': df['family_id'].values,
         'pref': np.arange(n_choices)
        }
    )

    # compute number of families and people requesting each date for each
    # preference level, then add to family dataset
    da_num_fam, da_num_ppl = date_popularities(ds_fam)
    ds_fam['pop_fam'] = da_num_fam
    ds_fam['pop_ppl'] = da_num_ppl

    # compute a linear fit to the number of families requesting each date.
    # nb: day 1 is excluded so occupancy of this fit is low by definition.
    ds_fam['occ_fit'] = get_occ_linear(ds_fam['pop_ppl'])

    # create scheduling difficulty index (gap between popularity of a
    # date and the linear fit to popularity)
    ds_fam['difficulty'] = fam_diff_score(ds_fam['fam_prefs'],
                                          ds_fam['pop_ppl'],
                                          ds_fam['occ_fit'])

    # store the total number of people getting scheduled
    ds_fam['tot_people'] = ds_fam['fam_size'].sum().item()

    return ds_fam


# In[ ]:


fn_family = '../input/santa-workshop-tour-2019/family_data.csv'
ds_family = family_preprocess(fn_family)


# # Fun with xarray
# 
# While the other sections of this notebook are focused on revealing aspects of the scheduling problem, this section demonstrates everyday EDA with xarray.
# 
# ### First, lets look at a summary of the dataset.

# In[ ]:


print(ds_family)


# ### We can  easily investigate specific aspects of the data by label
# 
# To demonstrate we look at family 1009 and preference level 0.

# In[ ]:


# Look at characteristics of family 1009
print(ds_family.sel(fam=1009).drop(['occ_fit', 'pop_ppl', 'pop_fam', 'tot_people', 'date']))


# In[ ]:


_ = ds_family.sel(fam=1009)['difficulty'].plot(marker='*')


# In[ ]:


# Look at characteristics of preference 0
print(ds_family.sel(pref=0)[['occ_fit', 'pop_fam', 'pop_ppl']])


# In[ ]:


_ = ds_family['occ_fit'].plot()
_ = ds_family.sel(pref=0)['pop_ppl'].plot()


# # Family size distributions

# In[ ]:


def fam_size_dists(family_sizes):
    """Show histogram and cumulative distribution of family sizes.  family_sizes
    should be a list with one size for each family.
    """
    
    sizes = list(set(family_sizes))
    fam_size_bins = np.arange(sizes[0] - 0.5, sizes[-1] + 1.5)

    fig, axes = plt.subplots(ncols=2, figsize=(10, 4))

    # plot histogram of family sizes on left
    axes[0].hist(family_sizes, bins=fam_size_bins, rwidth=0.9)
    _ = axes[0].set_ylabel('number of families')
    _ = axes[0].set_title('Number of families in family size groups')
    _ = axes[0].set_xlabel('family size')

    # plot cumulative distribution of family sizes on right
    axes[1].hist(family_sizes, bins=fam_size_bins, rwidth=0.9, cumulative=True, density=True)
    _ = axes[1].set_ylabel('fraction of families')
    _ = axes[1].set_title('Cumulative fraction of families in family size groups')
    _ = axes[0].set_xlabel('family size')
    
    return
    

def fam_size_people_dists(family_sizes):
    """Calculate and show histogram and cumulative distribution of the number
    of people in different family size groups.
    """
    
    sizes = list(set(family_sizes))
    bins = np.arange(sizes[0] - 0.5, sizes[-1] + 1.5)
    hist, _ = np.histogram(family_sizes, bins=bins)
    people_per_bin = hist * sizes

    # make a cumulative distribution as a density
    cum_people_per_bin = np.cumsum(people_per_bin)
    cum_people_per_bin = cum_people_per_bin / cum_people_per_bin[-1]
    
    # show distributions
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4))

    _ = axes[0].bar(x=sizes, height=people_per_bin)
    _ = axes[0].set_xlabel('family size')
    _ = axes[0].set_ylabel('number of people')
    _ = axes[0].set_title('Number of people in family size groups')

    _ = axes[1].bar(x=sizes, height=cum_people_per_bin)
    _ = axes[1].set_xlabel('family size')
    _ = axes[1].set_ylabel('fraction of people')
    _ = axes[1].set_title('Cumulative fraction of people in family size groups')
    
    # prepare to show fractions of people in each family size in a human readable format
    df = pd.DataFrame(
        {'fam_size': sizes,
         'pct of people': np.round(100 * people_per_bin / np.sum(people_per_bin)),
         'cumulative pct': np.round(100 * cum_people_per_bin)})
    df.set_index('fam_size', inplace=True)

    return df


# In[ ]:


fam_size_dists(ds_family['fam_size'].values)
df_show = fam_size_people_dists(ds_family['fam_size'].values)


# In[ ]:


df_show.T


# ## Summary thoughts on family size distribution
# - Family sizes range from 2 to 8 people.
# - Over 60% of families have between two and four people and less than 40% of families have between five and eight people. The most frequent family size is 4.
# - After adjusting the analysis to consider people counts rather than family counts, the most frequent family size is also four. About half of people are in families between size two and four, and about 70% are in families between size two and five.

# > # Family date preferences

# In[ ]:


def show_preferences(ds_prefs, ds_prefs_demeaned):
    """Show how many families and how many people requested each date in
    each preference level, and show the deviation from the mean across
    preference levels to highlight any structure in preference levels.
    """
    
    fig, axes = plt.subplots(nrows=4, figsize=(15, 12), sharex=True)

    ds_prefs['pop_fam'].plot.line(x='date', ax=axes[0], add_legend=True)
    axes[0].set_ylabel('number of families')

    ds_prefs['pop_ppl'].plot.line(x='date', ax=axes[1], add_legend=False)
    axes[1].set_ylabel('number of people')

    ds_prefs_demeaned['pop_fam'].plot.line(x='date', ax=axes[2], add_legend=False)
    axes[2].set_ylabel('demean num families')

    ds_prefs_demeaned['pop_ppl'].plot.line(x='date', ax=axes[3], add_legend=False)
    axes[3].set_ylabel('demean num people')
    
    return


def show_preferences_zoom(ds_prefs):
    """Show how many people requested each date in each preference level,
    focusing on requests for dates near date=1.
    """
    fig, ax = plt.subplots(figsize=(15, 4))
    ds_prefs['pop_ppl'].plot.line(x='date', ax=ax)
    ax.set_ylabel('number of people')
    return

    
def get_noise(da_demeaned):
    """Given demeaned preferences by date, return histograms of noise for
    each preference level.
    """
    bins = np.arange(-60.5, 61.5, 3)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    def helper(da):
        hist, _ = np.histogram(da.values, bins)
        da_hist = xr.DataArray(hist, coords=[bin_centers], dims=['anomaly'])
        return da_hist

    da_ret = da_demeaned.sel(pref=np.arange(10)).groupby('pref').apply(helper)
    return da_ret

    
def print_dayzero_info(ds_prefs):
    """In our re-labelling, day zero is Christmas Eve.  Families strongly
    prefer that specific day, here we explore how much.
    """
    mean_prefs = ds_prefs['pop_fam'].mean('pref')
    day1_mean = mean_prefs[0].item()
    others_mean = mean_prefs[1:].mean().item()
    print(f'{day1_mean:.2f} families prefer day one on average')
    print(f'{others_mean:.2f} families prefer other days, on average')   
    return
    
    
def show_sorted_date_popularity(ds_prefs):
    """Show the number of people wanting different dates.
    """
    
    prefs = ds_prefs['pop_ppl'].mean('pref').to_dataframe().reset_index('date')
    prefs = prefs[1:].sort_values(by='pop_ppl')['pop_ppl'].values
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(prefs)
    ax.plot([0, 100], [125, 125], '--g')
    ax.plot([0, 100], [300, 300], '--g')

    ax.set_xlabel('low popularity to high popularity days')
    ax.set_ylabel('number of people wanting days')
    ax.grid(True)
    
    return


# In[ ]:


ds_prefs = ds_family[['pop_fam', 'pop_ppl']]
ds_prefs_demeaned = ds_prefs - ds_prefs.groupby('date').mean('pref')
show_preferences(ds_prefs, ds_prefs_demeaned)


# In[ ]:


show_preferences_zoom(ds_prefs.sel(date=np.arange(1, 4)))


# In[ ]:


print_dayzero_info(ds_prefs)


# In[ ]:


da_noise = get_noise(ds_prefs_demeaned['pop_fam'])
_ = da_noise.plot.line(x='anomaly', figsize=(8, 4))


# In[ ]:


show_sorted_date_popularity(ds_prefs)


# ## Summary of findings on family preferences for different dates
# 
# - Date=1 is far more likely to show up in family preference lists than any other date (275 families prefer day 1 versus 48 for other days, on average).
# - For date=1, we see special behavior in terms of where in a family's preference list the day shows up.  If that date is in a family's list, its far more likely to appear near the beginning of the list than near the end.
# - For dates besides date=1, we see two signals superimposed.  The first signal is that more families request weekends and less request weekdays.  The second signal is a trend of increasing numbers of families requesting smaller dates; this trend appears roughly linear in time.  These patterns are consistent across preference levels (except for date=1).
# - For dates besides date=1, we do not see any structure in the variation between preference levels.  Compared to the mean preferences across all preference levels, deviations from the mean appear reasonably gaussian and bias or skew is not apparent.

# # Date preference popularity by family

# In[ ]:


# show overall histogram of difficulty levels
_ = ds_family['difficulty'].sel(pref=range(0, 10)).plot.hist()


# In[ ]:


# show median difficulty level taken across the 10 preference levels,
# this plot shows some families are easy to schedule (median is very low)
# and some are not (median difficult is very high)
_ = ds_family['difficulty'].sel(pref=range(0, 10)).median('pref').plot.hist()


# In[ ]:


# show difficulty level of the "easiest" day for each family, first considering only
# their top preference, then their top two preferences, and so on.
bins = np.arange(-100, 1000, 50)
fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(15, 8), sharex=True, sharey=True)
for ind, ax in enumerate(axes.flatten()):
    if ind == 0:
        _ = ds_family['difficulty'][:, 0].plot.hist(ax=ax, bins=bins)
    else:
        _ = ds_family['difficulty'][:, :(ind + 1)].min('pref').plot.hist(ax=ax, bins=bins)
    ax.set_title('')
    ax.set_xlabel(f'easiest day of [0:{ind + 1}]')


# ## Summary of findings regarding difficulty scheduling families
# - For a few families, all or most of the requested days are popular days.
# - Most families have at least one relatively unpopular days in their top five preferences. 

# # Scheduling basics and scheduling cost

# In[ ]:


def print_sched_basics(total_people, ds_prefs):
    """Calculate and print out some rudimentary information about the
    scheduling problem.
    """
    avg_people_per_day = total_people / 100

    print(f'There are {total_people} people total, {avg_people_per_day} people per day')

    pref0_num_too_high = (ds_prefs.sel(pref=0)['pop_ppl'] > 400).values.sum()
    pref0_num_too_low = (ds_prefs.sel(pref=0)['pop_ppl'] < 125).values.sum()
    pct_out_compliance = pref0_num_too_high + pref0_num_too_low

    print('If we blindly assign all families their first preference,')
    print(f'    {pref0_num_too_high} dates have > 400 people')
    print(f'    {pref0_num_too_low} dates have < 125 people')
    print(f'    {pct_out_compliance}/100 are out of compliance')
    
    return


def show_sched_costs(ds_fam, per_person):
    """Show histogram of costs for each preference level.  If per_person is
    True, show costs per person, otherwise show cost per family.
    """
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 6))
    if per_person:
        bins = np.arange(0, 2000, 25)
        costs = ds_fam['pref_cost'] / ds_fam['fam_size']
    else:
        bins = np.arange(0, 5000, 100)
        costs = ds_fam['pref_cost']
    for pref_ind in np.arange(11):
        ax = axes.flatten()[pref_ind]
        costs[:, pref_ind].plot.hist(ax=ax, bins=bins)
        ax.set_title('')
        ax.grid(True)
    fig.suptitle(f'Scheduling costs by preference level\nPer-person={per_person}')
    return


def get_pp_sched_cost(ds_fam):
    """Returns average per person scheduling cost for each family
    size and each preference level.
    """
    pp_cost = ds_fam['pref_cost'] / ds_fam['fam_size']
    pp_cost.name = 'pp_cost'
    ds_pp_cost = pp_cost.to_dataset()
    ds_pp_cost['fam_size'] = ds_fam['fam_size']
    mean_costs_by_fam_pref = ds_pp_cost.groupby('fam_size').mean()
    return mean_costs_by_fam_pref


def show_pp_sched_cost(mean_costs, heat_map=True):
    """Display average per person scheduling cost for each family
    and each preference level.
    """
    if heat_map:
        fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

        mean_costs['pp_cost'].plot(ax=axes[0], robust=True)
        mean_costs['pp_cost'].plot(ax=axes[1], vmin=0, vmax=100)
        mean_costs['pp_cost'].plot(ax=axes[2], vmin=0, vmax=20)
    else:
        fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
        _ = mean_costs['pp_cost'].plot.line(ax=axes[0], x='pref')
        _ = mean_costs['pp_cost'].plot.line(ax=axes[1], x='fam_size')
    return


def show_sched_cost_by_size(mean_costs):
    """Show average scheduling cost by family size and preference
    level with the mean across family sizes removed.
    """
    diff_from_mean = (mean_costs - mean_costs.mean('fam_size'))
    ratio_to_mean = (mean_costs / mean_costs.mean('fam_size'))
    fig, axes = plt.subplots(nrows=2, figsize=(15, 10))
    _ = diff_from_mean['pp_cost'].plot.line(ax=axes[0], x='pref')
    axes[0].set_ylabel('Deviation from mean across all family sizes')

    _ = ratio_to_mean['pp_cost'].plot.line(ax=axes[1], x='pref')
    axes[1].set_ylabel('Ratio to mean across all family sizes')
    return


# In[ ]:


print_sched_basics(ds_family['tot_people'].item(), ds_prefs)


# In[ ]:


show_sched_costs(ds_family, per_person=False)
show_sched_costs(ds_family, per_person=True)


# In[ ]:


mean_costs_by_fam_pref = get_pp_sched_cost(ds_family)
show_pp_sched_cost(mean_costs_by_fam_pref, heat_map=True)
show_pp_sched_cost(mean_costs_by_fam_pref, heat_map=False)


# In[ ]:


show_sched_cost_by_size(mean_costs_by_fam_pref)


# ## Summary of findings for schedule preference costs
# - Costs increase markedly for less preferred dates, and choice_9 and choice_10 ("other") should be avoided if at all possible
# - Smaller families have a higher per person schedule cost, due to the fixed per-family gift card price.  Since gift card prices go up so quickly with preference level, the per-person surcharge on small families rises with preference level.

# # Accounting cost

# In[ ]:


def _get_occupancy_cost(dates, occupancies):
    """Given an array of dates and an array of occupancies,
    calculate the occupancy cost and return as a dataframe.
    """
    
    df = pd.DataFrame(
        {'date': dates,
         'occupancy': occupancies
        }
    )
    df.set_index('date', inplace=True)
    
    # ensure dataframe is sorted from day 1 to day 100
    df = df.sort_values(by='date', axis=0, ascending=True)

    # create array of occupancies that provides the initial condition d=100, N101 = N100
    vals = df['occupancy'].values
    new_occ = np.concatenate([vals, [vals[-1]]], axis=0)
    
    def daily_helper(ind):
        Nd = new_occ[ind]
        Nd1 = new_occ[ind + 1]
        exp = 0.5 + 0.02 * np.abs(Nd - Nd1)

        day_cost = 0.0025 * (Nd - 125) * Nd**(0.5) * Nd**exp
        return day_cost
    
    daily_cost = [daily_helper(ind) for ind in np.arange(0, 100)]
    df['daily_cost'] = daily_cost

    return df


def get_occ_cost_linear_fit(occ_target):
    """Get cost of the proposed target occupancy (the linear trend with adjustments)
    """
    return _get_occupancy_cost(np.arange(1, 101), occ_target)


def get_occ_cost_uniform_dist(n_people):
    """Get cost of a uniform distribution of occupancies
    """
    return _get_occupancy_cost(np.arange(1, 101),
                               (n_people / 100) * np.ones(100))


def get_occ_cost_random_dist(n_people):
    """Get cost of occupancies that center on the required mean, but also
    have substantial noise.  We don't want to go outside 125 to 300 people,
    so we calculate a range for noise around the mean occupancy that will remain
    inside the allowed occupancy.
    """
    return _get_occupancy_cost(np.arange(1, 101),
                               np.random.randint(low=125, high=300, size=(100,)))


# In[ ]:


df_cost_uniform = get_occ_cost_uniform_dist(ds_family['tot_people'].item())
tot_cost_uniform = df_cost_uniform['daily_cost'].sum()
print(f'Accounting cost uniform occupancy distribution {tot_cost_uniform:.2f}\n')

df_cost_target = get_occ_cost_linear_fit(ds_family['occ_fit'])
tot_cost_target = df_cost_target['daily_cost'].sum()
print(f'Accounting cost linearly increasing target occupancy {tot_cost_target:.2f}\n')

df_cost_random = get_occ_cost_random_dist(ds_family['tot_people'].item())
tot_cost_random = df_cost_random['daily_cost'].sum()
print(f'Accounting cost random occupancy distribution {tot_cost_random:.2f}\n')


# In[ ]:


fig, axes = plt.subplots(ncols=2, figsize=(15, 5), sharey=True)

df_cost_target.plot(ax=axes[0])
_ = axes[0].set_title(f'Total cost target occupancy {tot_cost_target:.2f}')
df_cost_uniform.plot(ax=axes[1])
_ = axes[1].set_title(f'Total cost uniform occupancy {tot_cost_uniform:.2f}')


# In[ ]:


fig, axes = plt.subplots(nrows=3, figsize=(15, 6))
df_cost_random['occupancy'].plot(ax=axes[0])
_ = axes[0].set_ylabel('Occupancy')
df_cost_random['occupancy'].diff().plot(ax=axes[1])
_ = axes[1].set_ylabel('$\Delta$ occupancy')
df_cost_random['daily_cost'].plot(ax=axes[2])
_ = axes[2].set_ylabel('cost')


# ## Summary of findings for occupancy penalties
# - Total occupancy penalty for a linearly increasing target occupancy is 6038, total for a uniform distribution is 4465.
# - If we add random noise around a mean occupancy, the total cost is 239,870,075 -- huge!
# - The largest occupancy costs are for days where the occupancy is high and the change in occupancy is high, due to the Nd^(...(Nd - Nd1)) term.  This is a warning for D=1 scheduling ... if we put a lot of people there (due to the large number of families with D=1 as their top choice) and there's a large change from D=1 to D=2 (because many fewer people want D=2) the occupancy penalty will be huge.
