#!/usr/bin/env python
# coding: utf-8

# This notebook is just inroduction of "COVID-19 dataset in Japan" and shows the contents of this dataset.

# In[ ]:


from datetime import datetime
time_format = "%d%b%Y %H:%M"
datetime.now().strftime(time_format)


# In[ ]:


from pprint import pprint
import dask.dataframe as dd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from matplotlib.ticker import ScalarFormatter
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
pd.set_option("display.max_colwidth", 1000)

plt.style.use("seaborn-ticks")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (9, 6)


# In[ ]:


get_ipython().system('pip install japanmap')
import japanmap


# In[ ]:


def line_plot(df, title, ylabel="Cases", h=None, v=None,
              xlim=(None, None), ylim=(0, None), math_scale=True, y_logscale=False, y_integer=False):
    """
    Show chlonological change of the data.
    """
    ax = df.plot()
    if math_scale:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci",  axis="y",scilimits=(0, 0))
    if y_logscale:
        ax.set_yscale("log")
    if y_integer:
        fmt = matplotlib.ticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
    ax.set_title(title)
    ax.set_xlabel(None)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)
    if h is not None:
        ax.axhline(y=h, color="black", linestyle="--")
    if v is not None:
        if not isinstance(v, list):
            v = [v]
        for value in v:
            ax.axvline(x=value, color="black", linestyle="--")
    plt.tight_layout()
    plt.show()


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Cumurative number of cases
# covid_jpn_total.csv

# ## Raw data

# In[ ]:


ncov_raw = pd.read_csv("/kaggle/input/covid19-dataset-in-japan/covid_jpn_total.csv")
ncov_raw.tail()


# In[ ]:


ncov_raw.info()


# In[ ]:


pd.DataFrame(ncov_raw.isna().sum()).T


# In[ ]:


ncov_raw["Location"].unique().tolist()


# In[ ]:


df = pd.DataFrame(
    {
        "Domestic": ncov_raw.loc[ncov_raw["Location"] == "Domestic"].isna().sum(),
        "AReturnee": ncov_raw.loc[ncov_raw["Location"] == "Returnee"].isna().sum(),
        "Airport": ncov_raw.loc[ncov_raw["Location"] == "Airport"].isna().sum(),
    }
)
df.drop(["Location"],axis=0).T


# In[ ]:


ncov_raw.loc[ncov_raw["Tested"].isna(), :]


# In[ ]:


ncov_raw["Date"].unique()


# ## Data cleaning

# In[ ]:


ncov_raw


# In[ ]:


df = ncov_raw.copy()
df.dropna(how="all", inplace=True)
df["Date"] = pd.to_datetime(df["Date"])
df = df.groupby("Location").apply(
    lambda x: x.set_index("Date").resample("D").interpolate(method="linear")
)
df = df.drop("Location", axis=1).reset_index()
df = df.sort_values("Date").reset_index(drop=True)
sel = df.columns.isin(["Location", "Date"])
df.loc[:, ~sel] = df.loc[:, ~sel].fillna(0).astype(np.int64)
ncov_df = df.copy()
ncov_df.tail(9)


# In[ ]:


ncov_df["Date"].dt.strftime("%Y-%m-%d").unique()


# ## Location

# In[ ]:


df = ncov_df.pivot_table(
    index="Date", columns="Location", values="Positive", aggfunc="last"
)
location_df = df.fillna(method="bfill").astype(np.int64).copy()
location_df.tail()


# In[ ]:


line_plot(location_df, "Cases over time in Japan per location", y_integer=True)


# ## Domestic: Tested, Confirmed, Recovered, Fatal

# In[ ]:


df = ncov_df.copy()
df = df.rename({"Positive": "Confirmed", "Discharged": "Recovered"}, axis=1)
df = df.loc[df["Location"] == "Domestic", ["Date", "Tested", "Confirmed", "Recovered", "Fatal"]]
df = df.groupby("Date").last()
main_df = df.copy()
main_df.tail()


# In[ ]:


line_plot(main_df, "Cases over time in Japan", y_integer=True)


# In[ ]:


df = pd.DataFrame(main_df.loc["2020-03-22":"2020-03-26", "Tested"])
df.T.style.background_gradient(cmap="Wistia", axis=1)


# Value on 25Mar2020 is smaller than 24Mar2020, but this is correct.

# In[ ]:


line_plot(main_df.drop("Tested", axis=1), "Cases over time in Japan without Tested", y_integer=True)


# ## Domestic: Symptomatic / Asymptomatic

# In[ ]:


df = ncov_df.copy()
df = df.loc[df["Location"] == "Domestic", ["Date", "Symptomatic", "Asymptomatic"]]
df = df.groupby("Date").last()
sym_df = df.copy()
sym_df.tail()


# In[ ]:


line_plot(sym_df, "Symptomatic / Asymptomatic in Japan", y_integer=True)


# ## Domestic: Severe / (Mild + Severe)

# In[ ]:


df = ncov_df.copy()
df = df.loc[df["Location"] == "Domestic", :]
df = df.groupby("Date").last()
df["Rate"] = df["Hosp_severe"] / (df["Hosp_mild"] + df["Hosp_severe"])
df = df.loc[:, ["Hosp_mild", "Hosp_severe", "Rate"]]
severe_df = df.copy()
severe_df.tail()


# In[ ]:


line_plot(severe_df["Rate"], "Severe / (Mild + Severe) in Japan", y_integer=True, ylabel=None)


# ## Domestic: Positive / Tested

# In[ ]:


df = ncov_df.copy()
df = df.loc[df["Location"] == "Domestic", :]
df = df.groupby("Date").last()
df["Rate"] = df["Positive"] / df["Tested"] * 100
df = df.loc[:, ["Positive", "Tested", "Rate"]]
positive_df = df.copy()
positive_df.tail()


# In[ ]:


positive_df.describe().T


# In[ ]:


line_plot(
    positive_df["Rate"], "Positive / Tested in Japan",
    ylabel="Positive / Tested [%]",
    y_integer=True,
    h=positive_df["Rate"].median()
)


# # Cumurative number of cases in each prefecture
# covid_jpn_prefecture.csv

# ## Raw data

# In[ ]:


pref_raw = pd.read_csv("/kaggle/input/covid19-dataset-in-japan/covid_jpn_prefecture.csv")
pref_raw.tail()


# In[ ]:


pref_raw.info()


# In[ ]:


pd.DataFrame(pref_raw.isna().sum()).T


# In[ ]:


pprint(pref_raw["Prefecture"].unique().tolist(), compact=True)


# ## Data cleaning

# In[ ]:


df = pref_raw.copy()
df.dropna(how="all", inplace=True)
df["Date"] = pd.to_datetime(df["Date"])
sel = df.columns.isin(["Date", "Prefecture"])
df = df.groupby("Prefecture").apply(
    lambda x: x.set_index("Date").resample("D").interpolate("linear")
)
df = df.fillna(0)
df = df.drop("Prefecture", axis=1).reset_index()
df = df.sort_values("Date").reset_index(drop=True)
sel = df.columns.isin(["Date", "Prefecture"])
df.loc[:, ~sel] = df.loc[:, ~sel].interpolate("linear").astype(np.int64)
pref_df = df.copy()
pref_df.tail()


# ## Confirmed (PCR tested positive) cases

# In[ ]:


df = pref_df.pivot_table(
    index="Date", columns="Prefecture", values="Positive", aggfunc="last"
)
df = df.sort_values(by=df.index[-1], axis=1, ascending=False)
confirmed_top = df.columns[0]
line_plot(df.iloc[:, :10], "Confirmed cases over time", y_integer=True)


# ## Tested, Confirmed, Discharged, Fatal

# In[ ]:


df = pref_df.copy()
df = df.loc[df["Prefecture"] == confirmed_top, :]
df = df.drop("Prefecture", axis=1).groupby("Date").last()
line_plot(df, f"Cases over time in {confirmed_top}", y_integer=True)
line_plot(df.drop("Tested", axis=1), f"Cases over time in {confirmed_top} without Tested", y_integer=True)


# # Meta data
# covid_jpn_metadata.csv

# ## Raw data

# In[ ]:


meta_raw = pd.read_csv("/kaggle/input/covid19-dataset-in-japan/covid_jpn_metadata.csv")
meta_raw.tail()


# In[ ]:


meta_raw.info()


# In[ ]:


pd.DataFrame(meta_raw.isna().sum()).T


# ## Data cleaning

# In[ ]:


meta_raw["Category"].unique().tolist()


# In[ ]:


df = meta_raw.copy()
df["Title"] = df["Category"].str.cat(df["Item"], sep="_")
df = df.pivot_table(
    index="Prefecture", columns="Title", values="Value", aggfunc="last"
)
# Integer
cols = df.columns.str.startswith("Population")
cols += df.columns.str.startswith("Area")
cols += df.columns.str.startswith("Hospital_bed")
cols += df.columns.str.startswith("Clinic_bed")
df.loc[:, cols] = df.loc[:, cols].astype(np.int64)
df["Admin_Num"] = df["Admin_Num"].astype(np.int64)
# Numeric
cols = df.columns.str.startswith("Location")
df.loc[:, cols] = df.loc[:, cols].astype(np.float64)
# Sorting
df = df.loc[meta_raw["Prefecture"].unique(), :]
meta_df = df.copy()
meta_df.head()


# In[ ]:


meta_df.columns.tolist()


# ## Total population, area $[\mathrm{km}^2]$, population density

# In[ ]:


df = meta_df.copy()
df["Density_All"] = df["Population_Total"] / df["Area_Total"]
df["Density_Habitable"] = df["Population_Total"] / df["Area_Habitable"]
cols = df.columns.str.startswith("Population")
cols += df.columns.str.startswith("Area")
cols += df.columns.str.startswith("Density")
df = df.loc[:, cols].sort_values("Density_Habitable", ascending=False)
pop_df = df.copy()
pop_df.head()


# In[ ]:


df = pop_df.loc[:, pop_df.columns.str.startswith("Density")]
df.columns = df.columns.str.replace("Density_", "")
df.plot.bar()
plt.title("Population density of each prefecture")
plt.legend(title=None)
plt.show()


# ## Beds of hospitals and clinics

# In[ ]:


df = meta_df.copy()
cols = df.columns.str.startswith("Hospital_bed")
cols += df.columns.str.startswith("Clinic_bed")
df = df.loc[:, cols]
df.head()


# In[ ]:


df = meta_df.copy()
df["Bed_for_severe"] = df["Hospital_bed_Specific"] + df["Hospital_bed_Type-I"] + df["Hospital_bed_Type-II"]
df["Bed_for_other"] = df["Hospital_bed_Total"] - df["Bed_for_severe"] + df["Clinic_bed_Total"]
bed_df = df.copy()
bed_df.head()


# In[ ]:


bed_df["Bed_for_severe"].sort_values(ascending=False).plot.bar()
total = bed_df["Bed_for_severe"].sum()
plt.title(f"The number of beds that the patients with severe symptoms can use (total: {total})")
plt.show()


# ## Administrative information and (Latitude, Longitude)

# In[ ]:


df = meta_df.copy()
cols = df.columns.str.startswith("Admin")
cols += df.columns.str.startswith("Location")
df = df.loc[:, cols]
admin_df = df.copy()
admin_df.head()


# In[ ]:


df = pref_df.groupby("Prefecture").last()
df = pd.concat([admin_df, df], axis=1, sort=False)
admin_pref_df = df.copy()
admin_pref_df


# In[ ]:


# Usage if japanmap package (in Japanese):
# https://qiita.com/SaitoTsutomu/items/6d17889ba47357e44131
df = admin_pref_df.copy()
df.index = df["Admin_Num"].apply(lambda x: japanmap.pref_names[x])
cmap = plt.get_cmap("Reds")
norm = plt.Normalize(vmin=df["Positive"].min(), vmax=df["Positive"].max())
fcol = lambda x: "#" + bytes(cmap(norm(x), bytes=True)[:3]).hex()
plt.xticks(color="None")
plt.yticks(color="None")
plt.tick_params(length=0)
plt.colorbar(plt.cm.ScalarMappable(norm, cmap))
plt.imshow(japanmap.picture(df["Positive"].apply(fcol)))
plt.title(f"The number of cases in each prefecture ({df['Date'].max().strftime('%d%b%Y')})")
plt.show()


# ## Prefecture code dictionary
# If you want to know the location of the prefectures, plese refer to [Japan Visitor: Japan Prefectures Map](https://www.japanvisitor.com/japan-travel/prefectures-map).

# In[ ]:


pref_code_dict = meta_df["Admin_Num"].to_dict()
pprint(list(pref_code_dict.items()), width=60, compact=True)

