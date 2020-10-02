#!/usr/bin/env python
# coding: utf-8

# # Introduction
# In this notebook, we will compare the number of cases announced by Japanese government and gathered by Johns Hopkins University.
# 
# (Japanese goverment annouced)
# * Primary source: [Ministry of Health, Labour and Welefare HP (in English)](https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/newpage_00032.html)
# * Secondary source: [COVID-19 dataset in Japan](https://www.kaggle.com/lisphilar/covid19-dataset-in-japan)
# 
# (Johns Hoplins University gathered)
# * Primary source: [COVID-19 Data Repository by CSSE at Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19)
# * Secondary source: [Novel Corona Virus 2019 Dataset](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset/kernels)

# In[ ]:


from datetime import datetime
time_format = "%d%b%Y %H:%M"
datetime.now().strftime(time_format)


# ## Package

# In[ ]:


import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import seaborn as sns
# Matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("seaborn-ticks")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (9, 6)
# Pandas
pd.set_option("display.max_colwidth", 1000)


# ## Functions

# In[ ]:


def line_plot(df, title, xlabel=None, ylabel="Cases",
              h=None, v=None, xlim=(None, None), ylim=(0, None),
              math_scale=True, x_logscale=False, y_logscale=False, y_integer=False,
              show_legend=True, bbox_to_anchor=(1.02, 0),  bbox_loc="lower left"):
    """
    Show chlonological change of the data.
    """
    ax = df.plot()
    # Scale
    if math_scale:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci",  axis="y",scilimits=(0, 0))
    if x_logscale:
        ax.set_xscale("log")
        if xlim[0] == 0:
            xlim = (None, None)
    if y_logscale:
        ax.set_yscale("log")
        if ylim[0] == 0:
            ylim = (None, None)
    if y_integer:
        fmt = matplotlib.ticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
    # Set metadata of figure
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if show_legend:
        ax.legend(bbox_to_anchor=bbox_to_anchor, loc=bbox_loc, borderaxespad=0)
    else:
        ax.legend().set_visible(False)
    if h is not None:
        ax.axhline(y=h, color="black", linestyle=":")
    if v is not None:
        if not isinstance(v, list):
            v = [v]
        for value in v:
            ax.axvline(x=value, color="black", linestyle=":")
    plt.tight_layout()
    plt.show()


# # Datasets

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Government data

# ### Raw data

# In[ ]:


gov_raw = pd.read_csv("/kaggle/input/covid19-dataset-in-japan/covid_jpn_total.csv")
gov_raw.tail()


# ### Data cleaning

# In[ ]:


# https://www.kaggle.com/lisphilar/eda-of-japan-dataset
df = gov_raw.copy()
df.dropna(how="all", inplace=True)
df["Date"] = pd.to_datetime(df["Date"])
df = df.groupby("Location").apply(
    lambda x: x.set_index("Date").resample("D").interpolate(method="linear")
)
df = df.drop("Location", axis=1).reset_index()
df = df.sort_values("Date").reset_index(drop=True)
sel = df.columns.isin(["Location", "Date"])
df.loc[:, ~sel] = df.loc[:, ~sel].fillna(0).astype(np.int64)
# Select Confirmed/Recovered/Fatal
df = df.loc[:, ["Location", "Date", "Positive", "Fatal", "Discharged"]]
df = df.rename({"Positive": "Confirmed", "Discharged": "Recovered"}, axis=1)
# Show
gov_df = df.copy()
gov_df.tail(9)


# ### Government announced, total

# In[ ]:


gov_total_df = gov_df.groupby("Date").sum()
gov_total_df.tail()


# ### Government announced, domestic
# Without airport quarantine and returnees by chartered flights.

# In[ ]:


df = gov_df.copy()
df = df.loc[df["Location"] == "Domestic", :].drop("Location", axis=1)
df = df.groupby("Date").last()
gov_dom_df = df.copy()
gov_dom_df.tail()


# ## JHU data

# ### Raw data

# In[ ]:


jhu_raw = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
jhu_raw.loc[jhu_raw["Country/Region"] == "Japan", :]


# ### Data cleaning

# In[ ]:


df = jhu_raw.copy()
df = df.rename({"ObservationDate": "Date", "Deaths": "Fatal"}, axis=1)
df = df.loc[df["Country/Region"] == "Japan", ["Date", "Confirmed", "Fatal", "Recovered"]]
df["Date"] = pd.to_datetime(df["Date"])
df = df.groupby("Date").sum()
df = df.astype(np.int64)
df = df.reset_index()
jhu_df = df.copy()
jhu_df.tail()


# ## Merge

# In[ ]:


df = pd.merge(
    gov_total_df, gov_dom_df,
    left_index=True, right_index=True,
    suffixes=["/Total", "/Domestic"]
)
df = pd.merge(
    df.add_suffix("/GOV"), jhu_df.set_index("Date").add_suffix("/JHU"),
    left_index=True, right_index=True
)
comp_df = df.copy()
comp_df.tail()


# # Confirmed

# In[ ]:


c_df = comp_df.loc[:, comp_df.columns.str.startswith("Confirmed")]
c_df.tail(10)


# In[ ]:


df = c_df.copy()
df.columns = df.columns.str.replace("Confirmed/", "")
line_plot(df, "Confirmed cases in Japan: Comparison of datasets", y_integer=True)


# In[ ]:


df = c_df.copy()
df.columns = df.columns.str.replace("Confirmed/", "")
series = df["JHU"] - df["Total/GOV"]
line_plot(
    series,
    "Confirmed cases in Japan: JHU minus Total/GOV",
    y_integer=True, ylim=(None, None), show_legend=False,
    h=0
)


# # Fatal

# In[ ]:


d_df = comp_df.loc[:, comp_df.columns.str.startswith("Fatal")]
d_df.tail(10)


# In[ ]:


df = d_df.copy()
df.columns = df.columns.str.replace("Fatal/", "")
line_plot(df, "Fatal cases in Japan: Comparison of datasets", y_integer=True)


# In[ ]:


df = d_df.copy()
df.columns = df.columns.str.replace("Fatal/", "")
series = df["JHU"] - df["Total/GOV"]
line_plot(
    series,
    "Fatal cases in Japan: JHU minus Total/GOV",
    y_integer=True, ylim=(None, None), show_legend=False,
    h=0
)


# # Recovered

# In[ ]:


r_df = comp_df.loc[:, comp_df.columns.str.startswith("Recovered")]
r_df.tail(10)


# In[ ]:


df = r_df.copy()
df.columns = df.columns.str.replace("Recovered/", "")
line_plot(df, "Recovered cases in Japan: Comparison of datasets", y_integer=True)


# In[ ]:


df = r_df.copy()
df.columns = df.columns.str.replace("Recovered/", "")
series = df["JHU"] - df["Total/GOV"]
line_plot(
    series,
    "Recovered cases in Japan: JHU minus Total/GOV",
    y_integer=True, ylim=(None, None), show_legend=False,
    h=0
)

