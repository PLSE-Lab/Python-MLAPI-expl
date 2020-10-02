#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Read in data
kv_loans = pd.read_csv("../input/kiva_loans.csv")
kv_themes= pd.read_csv("../input/loan_themes_by_region.csv")
kv_mpi_rgn_loc = pd.read_csv("../input/kiva_mpi_region_locations.csv")
kv_loan_theme_ids = pd.read_csv("../input/loan_theme_ids.csv")


# summarize data

print('-----------------------------------LOAN DATA----------------------------')

# print(kv_loans.describe(include='all'))
print(kv_loans.columns.values)

print('-----------------------------THEME DATA------------------------------')

# print(kv_themes.describe(include='all'))
print(kv_themes.columns.values)

print('-----------------------------MPI DATA------------------------------')

# print(kv_mpi_rgn_loc.describe(include='all'))
print(kv_mpi_rgn_loc.columns.values)

print('-----------------------------THEME DATA------------------------------')

# print(kv_loan_theme_ids.describe(include='all'))
print(kv_loan_theme_ids.columns.values)


# In[12]:


# let's clean up a little bit . . . 
# a review of the MPI CSV file shows that there are about 1800 rows with NA values 
# in all consequential rows; the 'geo' column contains only (1000.0, 1000.0). let's drop these.

print(kv_mpi_rgn_loc.columns.values)
kv_mpi_rgn_loc = kv_mpi_rgn_loc[kv_mpi_rgn_loc.geo!='(1000.0, 1000.0)']

# Let's make a pivot table of loans by country -- this will make the data a little easier to look at
# and potentially give some insights into some higher level trends in the data
# np.round(pd.pivot_table(kv_loans,values="loan_amount",index="country",columns="sector",aggfunc=np.average),2)

# Not a lot to see here! Too busy. We also need a baseline currency ... 
# Let's take a look at the various currencies loans are given in to see if we can get a feel
# for an appropriate conversion measure

# We have 67 unique currencies in the dataset.
print('We have ",kv_loans["currency"].nunique()," unique currencies in the dataset.")
# The best path forward in this direction is probably to make some kind of crosswalk between
# currencies in the loans dataset and currencies in the World Bank's PPP numbers ...

# This graph is worthless because of the number of currencies ...
# ax = sb.countplot(x="country", hue="currency", data=kv_loans)



# In[ ]:


# grab a function to calculate a measure of distributional equality from my Github --
# https://github.com/ccutsail/EconStuff/blob/master/Aiyagari/gini.py

def ginicoefficient(population, value):
    '''
    This function calculates and returns a Gini coefficient from a population and a value measure
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.integrate as scp
    plt_title_val = input("Enter a title for the Lorenz Curve plot: ")
    population = np.asarray(population)
    value = np.asarray(value)
    population = np.insert(population, 0, 0, axis=0)
    value = np.insert(value, 0, 0, axis=0)
    dprod = np.multiply(population, value)
    indices = np.argsort(value)
    value = value[indices]
    population = population[indices]
    
    vsum = np.cumsum(value)
    vprop = np.divide(vsum,sum(value))

    psum = np.cumsum(population)
    pprop = np.divide(psum,sum(population))

    indices = np.argsort(vprop)
    vprop = vprop[indices]
    pprop = pprop[indices]

    perfEqLine = np.linspace(0,1,num=len(pprop))

    
    popfrac = population/population[-1]
    dpfrac = dprod/dprod[-1]
    # the area of the box containing the lorenz curve is 1
    # between the line of perfect equality and the upper half of the box,
    # area is one half. we can calculate the area between the Lorenz
    # curve and the line of perfect inequality (the x-axis, in practice) 
    # and subtract it from 1/2 -- the area remaining after taking away the 
    # upper 1/2 of the box.
    # the formula:
    # 1/2 - 
    gini = 1/2-simple_quad(pprop, vprop)


    fig = plt.figure()
    ax1 = fig.add_subplot(111)


    ax1.scatter(pprop, vprop, s=1/5)
    ax1.plot(perfEqLine,perfEqLine, color='green')
    ax1.axhline(y=0, color='k',linestyle='--',linewidth=1)
    ax1.axvline(x=0, color='k',linestyle='--',linewidth=1)
    plt.title(plt_title_val)
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    ax1.set_xlabel(str("Gini Coefficient: " + str(round(gini,4))))
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.show()

    return gini


def simple_quad(xvals,yvals):
    intval = 0
    for i in range(1,len(xvals)):
        intval = intval + (xvals[i] - xvals[i-1])*yvals[i]
        intval = intval + 1/2*((yvals[i]-yvals[i-1])*(xvals[i] - xvals[i-1]))
        
    return intval

from seaborn.categorical import _CategoricalPlotter, remove_na
import matplotlib as mpl

###### stolen from here: https://stackoverflow.com/questions/34615854/seaborn-countplot-with-normalized-y-axis-per-group

class _CategoricalStatPlotter(_CategoricalPlotter):

    @property
    def nested_width(self):
        """A float with the width of plot elements when hue nesting is used."""
        return self.width / len(self.hue_names)

    def estimate_statistic(self, estimator, ci, n_boot):

        if self.hue_names is None:
            statistic = []
            confint = []
        else:
            statistic = [[] for _ in self.plot_data]
            confint = [[] for _ in self.plot_data]

        for i, group_data in enumerate(self.plot_data):
            # Option 1: we have a single layer of grouping
            # --------------------------------------------

            if self.plot_hues is None:

                if self.plot_units is None:
                    stat_data = remove_na(group_data)
                    unit_data = None
                else:
                    unit_data = self.plot_units[i]
                    have = pd.notnull(np.c_[group_data, unit_data]).all(axis=1)
                    stat_data = group_data[have]
                    unit_data = unit_data[have]

                # Estimate a statistic from the vector of data
                if not stat_data.size:
                    statistic.append(np.nan)
                else:
                    statistic.append(estimator(stat_data, len(np.concatenate(self.plot_data))))

                # Get a confidence interval for this estimate
                if ci is not None:

                    if stat_data.size < 2:
                        confint.append([np.nan, np.nan])
                        continue

                    boots = bootstrap(stat_data, func=estimator,
                                      n_boot=n_boot,
                                      units=unit_data)
                    confint.append(utils.ci(boots, ci))

            # Option 2: we are grouping by a hue layer
            # ----------------------------------------

            else:
                for j, hue_level in enumerate(self.hue_names):
                    if not self.plot_hues[i].size:
                        statistic[i].append(np.nan)
                        if ci is not None:
                            confint[i].append((np.nan, np.nan))
                        continue

                    hue_mask = self.plot_hues[i] == hue_level
                    group_total_n = (np.concatenate(self.plot_hues) == hue_level).sum()
                    if self.plot_units is None:
                        stat_data = remove_na(group_data[hue_mask])
                        unit_data = None
                    else:
                        group_units = self.plot_units[i]
                        have = pd.notnull(
                            np.c_[group_data, group_units]
                            ).all(axis=1)
                        stat_data = group_data[hue_mask & have]
                        unit_data = group_units[hue_mask & have]

                    # Estimate a statistic from the vector of data
                    if not stat_data.size:
                        statistic[i].append(np.nan)
                    else:
                        statistic[i].append(estimator(stat_data, group_total_n))

                    # Get a confidence interval for this estimate
                    if ci is not None:

                        if stat_data.size < 2:
                            confint[i].append([np.nan, np.nan])
                            continue

                        boots = bootstrap(stat_data, func=estimator,
                                          n_boot=n_boot,
                                          units=unit_data)
                        confint[i].append(utils.ci(boots, ci))

        # Save the resulting values for plotting
        self.statistic = np.array(statistic)
        self.confint = np.array(confint)

        # Rename the value label to reflect the estimation
        if self.value_label is not None:
            self.value_label = "{}({})".format(estimator.__name__,
                                               self.value_label)

    def draw_confints(self, ax, at_group, confint, colors,
                      errwidth=None, capsize=None, **kws):

        if errwidth is not None:
            kws.setdefault("lw", errwidth)
        else:
            kws.setdefault("lw", mpl.rcParams["lines.linewidth"] * 1.8)

        for at, (ci_low, ci_high), color in zip(at_group,
                                                confint,
                                                colors):
            if self.orient == "v":
                ax.plot([at, at], [ci_low, ci_high], color=color, **kws)
                if capsize is not None:
                    ax.plot([at - capsize / 2, at + capsize / 2],
                            [ci_low, ci_low], color=color, **kws)
                    ax.plot([at - capsize / 2, at + capsize / 2],
                            [ci_high, ci_high], color=color, **kws)
            else:
                ax.plot([ci_low, ci_high], [at, at], color=color, **kws)
                if capsize is not None:
                    ax.plot([ci_low, ci_low],
                            [at - capsize / 2, at + capsize / 2],
                            color=color, **kws)
                    ax.plot([ci_high, ci_high],
                            [at - capsize / 2, at + capsize / 2],
                            color=color, **kws)

class _BarPlotter(_CategoricalStatPlotter):
    """Show point estimates and confidence intervals with bars."""

    def __init__(self, x, y, hue, data, order, hue_order,
                 estimator, ci, n_boot, units,
                 orient, color, palette, saturation, errcolor, errwidth=None,
                 capsize=None):
        """Initialize the plotter."""
        self.establish_variables(x, y, hue, data, orient,
                                 order, hue_order, units)
        self.establish_colors(color, palette, saturation)
        self.estimate_statistic(estimator, ci, n_boot)

        self.errcolor = errcolor
        self.errwidth = errwidth
        self.capsize = capsize

    def draw_bars(self, ax, kws):
        """Draw the bars onto `ax`."""
        # Get the right matplotlib function depending on the orientation
        barfunc = ax.bar if self.orient == "v" else ax.barh
        barpos = np.arange(len(self.statistic))

        if self.plot_hues is None:

            # Draw the bars
            barfunc(barpos, self.statistic, self.width,
                    color=self.colors, align="center", **kws)

            # Draw the confidence intervals
            errcolors = [self.errcolor] * len(barpos)
            self.draw_confints(ax,
                               barpos,
                               self.confint,
                               errcolors,
                               self.errwidth,
                               self.capsize)

        else:

            for j, hue_level in enumerate(self.hue_names):

                # Draw the bars
                offpos = barpos + self.hue_offsets[j]
                barfunc(offpos, self.statistic[:, j], self.nested_width,
                        color=self.colors[j], align="center",
                        label=hue_level, **kws)

                # Draw the confidence intervals
                if self.confint.size:
                    confint = self.confint[:, j]
                    errcolors = [self.errcolor] * len(offpos)
                    self.draw_confints(ax,
                                       offpos,
                                       confint,
                                       errcolors,
                                       self.errwidth,
                                       self.capsize)

    def plot(self, ax, bar_kws):
        """Make the plot."""
        self.draw_bars(ax, bar_kws)
        self.annotate_axes(ax)
        if self.orient == "h":
            ax.invert_yaxis()

def percentageplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None,
              orient=None, color=None, palette=None, saturation=.75,
              ax=None, **kwargs):

    # Estimator calculates required statistic (proportion)        
    estimator = lambda x, y: (float(len(x))/y)*100 
    ci = None
    n_boot = 0
    units = None
    errcolor = None

    if x is None and y is not None:
        orient = "h"
        x = y
    elif y is None and x is not None:
        orient = "v"
        y = x
    elif x is not None and y is not None:
        raise TypeError("Cannot pass values for both `x` and `y`")
    else:
        raise TypeError("Must pass values for either `x` or `y`")

    plotter = _BarPlotter(x, y, hue, data, order, hue_order,
                          estimator, ci, n_boot, units,
                          orient, color, palette, saturation,
                          errcolor)

    plotter.value_label = "Percentage"

    if ax is None:
        ax = plt.gca()

    plotter.plot(ax, kwargs)
    return ax


# In[ ]:


import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns; sns.set(color_codes=True)
ax = sns.regplot(x="country", y="loan_amount", data=kv_loans)

