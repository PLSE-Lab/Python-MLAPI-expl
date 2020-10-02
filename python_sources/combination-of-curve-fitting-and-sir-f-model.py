#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Using curve fitting method and SIR-F model, we will predict the number of confirmed cases and fatal cases with COVID-19 global data. SIR-F model was created in another notebook of an auther. Please refer to the references.  
# 
# Contents:
# * Arrangement of dataset
# * Grouping countries by curve fitting
# * Explanation of SIR-F model
# * Parameter estimaition of SIR-F model in each group
# * Prediction and data submission
# 
# References:
# * [COVID-19 - Growth of Virus in Specific Countries](https://www.kaggle.com/wjholst/covid-19-growth-of-virus-in-specific-countries)
# * [COVID-19 data with SIR model](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model)

# # Arrangement of dataset
# Data acquisition and EDA will be done in this section.

# ## Packages

# In[ ]:


from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
from pprint import pprint
import warnings
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from matplotlib.ticker import ScalarFormatter
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import optuna
optuna.logging.disable_default_handler()
import pandas as pd
pd.plotting.register_matplotlib_converters()
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp


# In[ ]:


plt.style.use("seaborn-ticks")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (9, 6)


# ## Functions

# ### Plotting

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


# ## Raw data

# In[ ]:


for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_raw = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")
test_raw = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")
submission_sample_raw = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/submission.csv")
# Population
population_raw = pd.read_csv("/kaggle/input/covid19-global-forecasting-locations-population/locations_population.csv")


# In[ ]:


submission_sample_raw.head()


# In[ ]:


df = pd.DataFrame(
    {
        "Nunique_train": train_raw.nunique(),
        "Nunique_test": test_raw.nunique(),
        "Null_Train": train_raw.isnull().sum(),
        "Null_Test": test_raw.isnull().sum(),
    }
)
df.fillna("-").T


# In[ ]:


population_raw.head()


# ## Total population of each country

# In[ ]:


df = population_raw.rename({"Province.State": "Province", "Country.Region": "Country"}, axis=1)
df["Country/Province"] = df[["Country", "Province"]].apply(
    lambda x: f"{x[0]}/{x[0]}" if x[1] is np.nan else f"{x[0]}/{x[1]}",
    axis=1
)
# Culculate total value of each country/province
df = df.groupby("Country/Province").sum()
# Global population
df.loc["Global", "Population"] = df["Population"].sum()
# DataFrame to dictionary
population_dict = df.astype(np.int64).to_dict()["Population"]
population_dict


# ## Data cleaning of training data

# In[ ]:


df = pd.merge(
    train_raw.rename({"Province/State": "Province", "Country/Region": "Country"}, axis=1),
    population_raw.rename({"Province.State": "Province", "Country.Region": "Country"}, axis=1),
    on=["Country", "Province"]
)
# Area: Country or Country/Province
df["Area"] = df[["Country", "Province"]].apply(
    lambda x: f"{x[0]}" if x[1] is np.nan else f"{x[0]}/{x[1]}",
    axis=1
)
# Date
df["Date"] = pd.to_datetime(df["Date"])
# The number of cases
df = df.rename({"ConfirmedCases": "Confirmed", "Fatalities": "Fatal"}, axis=1)
df[["Confirmed", "Fatal"]] = df[["Confirmed", "Fatal"]].astype(np.int64)
# Show data
df = df.loc[:, ["Date", "Area", "Population", "Confirmed", "Fatal"]]
train_df = df.copy()
train_df.tail()


# ## Data cleaning of test data

# In[ ]:


df = pd.merge(
    test_raw.rename({"Province/State": "Province", "Country/Region": "Country"}, axis=1),
    population_raw.rename({"Province.State": "Province", "Country.Region": "Country"}, axis=1),
    on=["Country", "Province"]
)
df["Area"] = df[["Country", "Province"]].apply(
    lambda x: f"{x[0]}" if x[1] is np.nan else f"{x[0]}/{x[1]}",
    axis=1
)
df["Date"] = pd.to_datetime(df["Date"])
df = df.loc[:, ["ForecastId", "Date", "Area", "Population"]]
test_df = df.copy()
test_df.tail()


# ## EDA of traing data

# In[ ]:


train_df.describe(include="all").fillna("-").T


# In[ ]:


total_df = train_df.drop("Population", axis=1).groupby("Date").sum()
total_df.tail()


# In[ ]:


line_plot(total_df, "Total: Cases over time")


# # Grouping countries by curve fitting
# The number of confirmed cases is increasing in many countries, but there are two of countries.  
# 
# In a first-type country, growth factor is larger than 1 and the number of cases is rapidly increasing.In a second-type country, growth factor is less than 1.
# 
# Because China has low growth factor and many cases, China will be regard as an independent group.

# ## Calculate growth factor
# $\mathrm{Growth\ Factor} = \cfrac{\Delta \mathrm{Confirmed}_{n}}{\Delta \mathrm{Confirmed}_{n-1}}$

# In[ ]:


df = train_df.copy()
df = df.pivot_table(
    index="Date", columns="Area", values="Confirmed"
).fillna(method="bfill")
# Growth factor: (delta Number_n) / (delta Number_n)
df = df.diff() / df.diff().shift(freq="D")
df = df.rolling(7).mean()
df = df.iloc[2:-1, :]
growth_df = df.copy()
growth_df.tail(10)


# In[ ]:


current_growth_df = growth_df.iloc[-1, :].T.reset_index()
current_growth_df.columns = ["Area", "Growth_Factor"]
df = train_df.loc[train_df["Date"] == train_df["Date"].max(), ["Area", "Confirmed", "Fatal"]]
df.columns = ["Area", "Current_Confirmed", "Current_Fatal"]
current_growth_df = pd.merge(current_growth_df, df, on="Area")
current_growth_df.head()


# ## Grouping areas with growth factor
# * Outbreaking group: growth factor > 1
# * China
# * The others: growth factor <= 1 or None

# In[ ]:


current_growth_df["Group"] = "Others"
current_growth_df.loc[current_growth_df["Growth_Factor"] > 1, "Group"] = "Outbreaking"
current_growth_df.loc[current_growth_df["Area"].str.contains("China"), "Group"] = "China"


# In[ ]:


current_growth_df.nlargest(10, "Growth_Factor")


# In[ ]:


current_growth_df.loc[current_growth_df["Group"] == "China", :].nlargest(10, "Growth_Factor")


# In[ ]:


current_growth_df.nsmallest(10, "Growth_Factor")


# In[ ]:


df = pd.merge(train_df, current_growth_df, on="Area")
df = df.pivot_table(
    index="Date", columns="Group", values=["Population", "Confirmed", "Fatal"],
    aggfunc="sum"
)
df = df.T.swaplevel(0, 1).sort_index(ascending=False).T
grouped_train_df = df.copy()
grouped_train_df.tail()


# ## Total value of outbreaking group over time

# In[ ]:


outbreak_df = grouped_train_df.loc[:, "Outbreaking"].reset_index()
outbreak_df.tail()


# In[ ]:


line_plot(outbreak_df.drop("Population", axis=1).set_index("Date"), "Cases over time in outbreaking group")


# In[ ]:


def show_trend(df, group="Outbreaking group", variable="Confirmed", n_changepoints=2):
    """
    Show trend of log10(@variable) using fbprophet package.
    @df <pd.DataFrame>: time series data of the variable
    @group <str>: Group name (to show figure title)
    @variable <str>: variable name to analyse, Confirmed or Fatal
    @n_changepoints <int>: max number of change points
    """
    # Data arrangement
    df = df.loc[:, ["Date", variable]]
    df.columns = ["ds", "y"]
    # Log10(x)
    warnings.resetwarnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["y"] = np.log10(df["y"]).replace([np.inf, -np.inf], 0)
    # fbprophet
    model = Prophet(growth="linear", daily_seasonality=False, n_changepoints=n_changepoints)
    model.fit(df)
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)
    # Create figure
    fig = model.plot(forecast)
    _ = add_changepoints_to_plot(fig.gca(), model, forecast)
    plt.title(f"{group}: log10({variable}) over time and chainge points")
    plt.ylabel(f"log10(the number of cases)")
    plt.xlabel("")


# In[ ]:


show_trend(outbreak_df, group="Outbreaking group", variable="Confirmed")


# In[ ]:


show_trend(outbreak_df, group="Outbreaking group", variable="Fatal")


# **For currently outbreaking group, we will use data from 15Feb2020.**

# In[ ]:


outbreak_group_start = "15Feb2020"


# ## Total value of China over time

# In[ ]:


china_df = grouped_train_df.loc[:, "China"].reset_index()
china_df.tail()


# In[ ]:


line_plot(china_df.drop("Population", axis=1).set_index("Date"), "Cases over time in China")


# In[ ]:


show_trend(china_df, group="China", variable="Confirmed")


# In[ ]:


show_trend(china_df, group="China", variable="Fatal")


# **For currently outbreaking group, we will use data from 26Jan2020 to 15Feb2020.**

# In[ ]:


china_start = "26Jan2020"
china_end = "15Febn2020"


# ## Total value of the others over time

# In[ ]:


others_df = grouped_train_df.loc[:, "Others"].reset_index()
others_df.tail()


# In[ ]:


line_plot(others_df.drop("Population", axis=1).set_index("Date"), "Cases over time in the others")


# In[ ]:


show_trend(others_df, group="The others", variable="Confirmed")


# In[ ]:


show_trend(others_df, group="The others", variable="Fatal")


# **For the others, we will use data from 15Feb2020 to 22Mar2020.**

# In[ ]:


others_start = "15Feb2020"
others_end = "22Mar2020"


# # Explanation of SIR-F model
# In this section, we will create a mathematical model named SIR-F model. This model is derived from SIR model.

# ## What is SIR model?
# SIR model is a simple mathematical model to understand outbreak of infectious diseases.  
# [The SIR epidemic model - Learning Scientific Programming with Python](https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/)
# 
#  * S: Susceptible (=Total population - Confirmed)
#  * I: Infected
#  * R: Recovered
# 
# Model:  
# S + I $\overset{\beta}{\longrightarrow}$ 2I  
# I $\overset{\gamma}{\longrightarrow}$ R
# 
# $\beta$: Effective contact rate [1/min]  
# $\gamma$: Recovery rate [1/min]  
# 
# Ordinary Differential Equation (ODE):   
# $\frac{\mathrm{d}S}{\mathrm{d}T}= - N^{-1}\beta S I$  
# $\frac{\mathrm{d}I}{\mathrm{d}T}= N^{-1}\beta S I - \gamma I$  
# $\frac{\mathrm{d}R}{\mathrm{d}T}= \gamma I$  
# 
# Where $N=S+I+R$ is the total population, $T$ is the elapsed time from the start date.

# ## Why we need SIR-F model?

# ### Replace R with (R, F)
# R in SIR model is "Recovered and have immunity", but mortality rate cannot be ignored in the real COVID-19 data. Furthermore, with this dataset, we know only the number of confirmed cases and fatal cases.

# ### Add "S + I $\to$ F + I formula
# Some cases are reported as fatal cases before clinical diagnosis of COVID-19. To consider this issue, "S + I $\to$ Fatal + I" will be added to the model.

# ## What is SIR-F model?
# * S: Susceptible
# * S$^\ast$: Confirmed and un-categorized
# * I: Confirmed and categorized as I
# * R: Recovered
# * F: Fatal with confirmation
# 
# Measurable variables:  
# Confirmed = $I+R+F$  
# Recovered = $R$  
# Deaths = $F$  
# 
# Model:  
# S $\overset{\beta \mathrm{I}}{\longrightarrow}$ S$^\ast$ $\overset{\alpha_1}{\longrightarrow}$ F  
# S $\overset{\beta \mathrm{I}}{\longrightarrow}$ S$^\ast$ $\overset{1 - \alpha_1}{\longrightarrow}$ I  
# I $\overset{\gamma}{\longrightarrow}$ R  
# I $\overset{\alpha_2}{\longrightarrow}$ F  
# 
# $\alpha_1$: Mortality rate of S$^\ast$ cases [-]  
# $\alpha_2$: Mortality rate of I cases [1/min]  
# $\beta$: Effective contact rate [1/min]  
# $\gamma$: Recovery rate [1/min]  
# 
# Ordinary Differential Equation (ODE):   
# $\frac{\mathrm{d}S}{\mathrm{d}T}= - N^{-1}\beta S I$  
# $\frac{\mathrm{d}I}{\mathrm{d}T}= N^{-1}(1 - \alpha_1) \beta S I - (\gamma + \alpha_2) I$  
# $\frac{\mathrm{d}R}{\mathrm{d}T}= \gamma I$  
# $\frac{\mathrm{d}F}{\mathrm{d}T}= N^{-1}\alpha_1 \beta S I + \alpha_2 I$  
# 
# Where $N=S+I+R+F$ is the total population, $T$ is the elapsed time from the start date.

# ## Non-dimensional SIR-F model
# 
# To simplify the model, we will remove the units of the variables from ODE.
# 
# Set $(S, I, R, F) = N \times (x, y, z, w)$ and $(T, \alpha_1, \alpha_2, \beta, \gamma) = (\tau t, \theta, \tau^{-1} \kappa, \tau^{-1} \rho, \tau^{-1} \sigma)$.  
# This results in the ODE  
# $\frac{\mathrm{d}x}{\mathrm{d}t}= - \rho x y$  
# $\frac{\mathrm{d}y}{\mathrm{d}t}= \rho (1-\theta) x y - (\sigma + \kappa) y$  
# $\frac{\mathrm{d}z}{\mathrm{d}t}= \sigma y$  
# $\frac{\mathrm{d}w}{\mathrm{d}t}= \rho \theta x y + \kappa y$  
# 
# Where $N$ is the total population and $\tau$ is a coefficient ([min], is an integer to simplify).  
# 
# The range of variables and parameters:  
# $0 < (x, y, z, w, \theta, \kappa, \rho, \sigma) < 1$  
# $1 \leq \tau \leq 1440$  
# 
# Reproduction number can be defined as  
# $R_0 = \rho (1 - \theta) (\sigma + \kappa)^{-1} = \beta (1 - \alpha_1) (\gamma + \alpha_2)^{-1}$
# 
# When $x=\frac{1}{R_0}$, $\frac{\mathrm{d}y}{\mathrm{d}t}=0$. This means that the max value of confirmed ($=y+z+w$) is $1-\frac{1}{R_0}$.

# In[ ]:


def create_target_df(ncov_df, total_population, start_date=None, end_date=None, date_format="%d%b%Y"):
    """
    Calculate the number of susceptible people,
     and calculate the elapsed time [day] from the start date of the target dataframe.
    @noc_df <pd.DataFrame>: the cleaned training data
    @total_population <int>: total population
    @start_date <str>: the start date or None
    @end_date <str>: the start date or None
    @date_format <str>: format of @start_date
    @return <tuple(2 objects)>:
        - 1. start_date <pd.Timestamp>: the start date of the selected records
        - 2. target_df <pd.DataFrame>:
            - column T: elapsed time [min] from the start date of the dataset
            - column Susceptible: the number of patients who are in the palces but not infected/recovered/fatal
            - column Deaths: the number of death cases
    """
    df = ncov_df.copy()
    if start_date is not None:
        df = df.loc[df["Date"] >= datetime.strptime(start_date, date_format), :]
    if end_date is not None:
        df = df.loc[df["Date"] <= datetime.strptime(end_date, date_format), :]
    start_date = df.loc[df.index[0], "Date"]
    # column T
    df["T"] = ((df["Date"] - start_date).dt.total_seconds() / 60).astype(int)
    # coluns except T
    response_variables = ["Susceptible", "Infected", "Recovered", "Fatal"]
    df["Susceptible"] = total_population - df["Confirmed"]
    df["Infected"] = 0
    df.loc[df.index[0], "Infected"] = df.loc[df.index[0], "Confirmed"] - df.loc[df.index[0], "Fatal"]
    df["Recovered"] = 0
    # Return
    target_df = df.loc[:, ["T", *response_variables]]
    return (start_date, target_df)


# In[ ]:


class ModelBase(object):
    NAME = "Model"
    VARIABLES = ["x"]
    PRIORITIES = np.array([1])

    @classmethod
    def param_dict(cls, train_df_divided=None, q_range=None):
        """
        Define parameters without tau. This function should be overwritten.
        @train_df_divided <pd.DataFrame>:
            - column: t and non-dimensional variables
        @q_range <list[float, float]>: quantile rage of the parameters calculated by the data
        @return <dict[name]=(type, min, max):
            @type <str>: "float" or "int"
            @min <float/int>: min value
            @max <float/int>: max value
        """
        param_dict = dict()
        return param_dict

    @staticmethod
    def calc_variables(df):
        """
        Calculate the variables of the model.
        This function should be overwritten.
        @df <pd.DataFrame>
        @return <pd.DataFrame>
        """
        return df

    @staticmethod
    def calc_variables_reverse(df):
        """
        Calculate measurable variables using the variables of the model.
        This function should be overwritten.
        @df <pd.DataFrame>
        @return <pd.DataFrame>
        """
        return df

    @classmethod
    def create_dataset(cls, ncov_df, total_population, start_date=None, end_date=None, date_format="%d%b%Y"):
        """
        Create dataset with the model-specific varibles.
        The variables will be divided by total population.
        The column names (not include T) will be lower letters.
        @params: See the function named create_target_df()
        @return <tuple(objects)>:
            - start_date <pd.Timestamp>
            - initials <tuple(float)>: the initial values
            - Tend <int>: the last value of T
            - df <pd.DataFrame>: the dataset
        """
        start_date, target_df = create_target_df(
            ncov_df, total_population, start_date=start_date, end_date=None, date_format=date_format
        )
        df = cls.calc_variables(target_df).set_index("T") / total_population
        df.columns = [n.lower() for n in df.columns]
        initials = df.iloc[0, :].values
        df = df.reset_index()
        Tend = df.iloc[-1, 0]
        return (start_date, initials, Tend, df)

    def calc_r0(self):
        """
        Calculate R0. This function should be overwritten.
        """
        return None

    def calc_days_dict(self, tau):
        """
        Calculate 1/beta [day] etc.
        This function should be overwritten.
        @param tau <int>: tau value [hour]
        """
        return dict()


# In[ ]:


class SIRF(ModelBase):
    NAME = "SIR-F"
    VARIABLES = ["x", "y", "z", "w"]
    PRIORITIES = np.array([100, 0, 0, 1])

    def __init__(self, theta, kappa, rho, sigma):
        super().__init__()
        self.theta = float(theta)
        self.kappa = float(kappa)
        self.rho = float(rho)
        self.sigma = float(sigma)

    def __call__(self, t, X):
        # x, y, z, w = [X[i] for i in range(len(self.VARIABLES))]
        # dxdt = - self.rho * x * y
        # dydt = self.rho * (1 - self.theta) * x * y - (self.sigma + self.kappa) * y
        # dzdt = self.sigma * y
        # dwdt = self.rho * self.theta * x * y + self.kappa * y
        dxdt = - self.rho * X[0] * X[1]
        dydt = self.rho * (1 - self.theta) * X[0] * X[1] - (self.sigma + self.kappa) * X[1]
        dzdt = self.sigma * X[1]
        dwdt = self.rho * self.theta * X[0] * X[1] + self.kappa * X[1]
        return np.array([dxdt, dydt, dzdt, dwdt])

    @classmethod
    def param_dict(cls, train_df_divided=None, q_range=None):
        param_dict = super().param_dict()
        param_dict["theta"] = ("float", 0, 1)
        param_dict["kappa"] = ("float", 0, 1)
        param_dict["rho"] = ("float", 0, 1)
        param_dict["sigma"] = ("float", 0, 1)
        return param_dict

    @staticmethod
    def calc_variables(df):
        df["X"] = df["Susceptible"]
        df["Y"] = df["Infected"]
        df["Z"] = df["Recovered"]
        df["W"] = df["Fatal"]
        return df.loc[:, ["T", "X", "Y", "Z", "W"]]

    @staticmethod
    def calc_variables_reverse(df):
        df["Susceptible"] = df["X"]
        df["Infected"] = df["Y"]
        df["Recovered"] = df["Z"]
        df["Fatal"] = df["W"]
        return df

    def calc_r0(self):
        try:
            r0 = self.rho * (1 - self.theta) / (self.sigma + self.kappa)
        except ZeroDivisionError:
            return np.nan
        return round(r0, 2)

    def calc_days_dict(self, tau):
        _dict = dict()
        _dict["alpha1 [-]"] = round(self.theta, 3)
        if self.kappa == 0:
            _dict["1/alpha2 [day]"] = 0
        else:
            _dict["1/alpha2 [day]"] = int(tau / 24 / 60 / self.kappa)
        _dict["1/beta [day]"] = int(tau / 24 / 60 / self.rho)
        if self.sigma == 0:
            _dict["1/gamma [day]"] = 0
        else:
            _dict["1/gamma [day]"] = int(tau / 24 / 60 / self.sigma)
        return _dict


# ## Example of non-dimensional SIR-F model
# Numerical simulation will be performed with scipy.integrate.solve_ivp function.  
# For example, set $(\theta, \kappa, \sigma, \rho)=(0.08, 0.0001, 0.02, 0.2)$, initial values $(S_0, I_0, R_0, F_0)=(1, 0.0001, 0, 0)$.

# In[ ]:


def simulation(model, initials, step_n, **params):
    """
    Solve ODE of the model.
    @model <ModelBase>: the model
    @initials <tuple[float]>: the initial values
    @step_n <int>: the number of steps
    """
    tstart, dt, tend = 0, 1, step_n
    sol = solve_ivp(
        fun=model(**params),
        t_span=[tstart, tend],
        y0=np.array(initials, dtype=np.float64),
        t_eval=np.arange(tstart, tend + dt, dt),
        dense_output=True
    )
    t_df = pd.Series(data=sol["t"], name="t")
    y_df = pd.DataFrame(data=sol["y"].T.copy(), columns=model.VARIABLES)
    sim_df = pd.concat([t_df, y_df], axis=1)
    return sim_df


# In[ ]:


eg_initials = np.array([1, 0.0001, 0, 0])
eg_param_dict = {"theta": 0.08, "kappa": 0.0001, "sigma": 0.02, "rho": 0.2}
eg_df = simulation(SIRF, eg_initials, step_n=300, **eg_param_dict)
eg_df.tail()


# In[ ]:


line_plot(
    eg_df.set_index("t"),
    title=r"Example of SIR-F model: $R_0$={0}".format(SIRF(**eg_param_dict).calc_r0()),
    ylabel="",
    h=1
)


# # Parameter estimaition of SIR-F model in each group
# In this section, we will estimate the non-dimensional model using training data in outbreaking group/the others.

# ## Transform training data
# * T means elapsed time [min] from the start date.
# * The number of cases will be divided by total population.

# In[ ]:


outbreak_df.tail()


# In[ ]:


train_dataset = SIRF.create_dataset(outbreak_df, outbreak_df.loc[outbreak_df.index[-1], "Population"])
train_start_date, train_initials, train_Tend, transformed_train_df = train_dataset
pprint([train_start_date.strftime("%d%b%Y"), train_initials, train_Tend])


# In[ ]:


transformed_train_df.tail()


# **Note: We cannot convert T to t because $\tau$ has not been determined yet.**

# ## Hyper parameter optimization
# Using Optuna package, $(\theta, \kappa, \sigma, \rho, \tau)$ will be estimated by model fitting.

# In[ ]:


class Estimator(object):
    def __init__(self, model, ncov_df, total_population, name=None,
                 start_date=None, end_date=None, date_format="%d%b%Y", param_fold_range=(1, 1), **kwargs):
        """
        Set training data.
        @model <ModelBase>: the model
        @name <str>: name of the area
        @param_fold_range <tuple(float, float)>:
            if we have fixed parameters (as kwargs), paramater range will be
            from param_fold_range[0] * (fixed) to param_fold_range[1] * (fixed)
        @kwargs: fixed parameter of the model
        @the other params: See the function named create_target_df()
        """
        if param_fold_range == (1, 1):
            self.fixed_param_dict = kwargs.copy()
            self.range_param_dict = dict()
        else:
            self.fixed_param_dict = dict()
            fold_min, fold_max = param_fold_range
            self.range_param_dict = {
                name: (value * fold_min, value * fold_max)
                for (name, value) in kwargs.items()
            }
        dataset = model.create_dataset(
            ncov_df, total_population, start_date=start_date, end_date=end_date, date_format=date_format
        )
        self.start_time, self.initials, self.Tend, self.train_df = dataset
        self.total_population = total_population
        self.name = name
        self.model = model
        self.param_dict = dict()
        self.study = None
        self.optimize_df = None

    def run(self, n_trials=500):
        """
        Try estimation (optimization of parameters and tau).
        @n_trials <int>: the number of trials
        """
        if self.study is None:
            self.study = optuna.create_study(direction="minimize")
        self.study.optimize(
            lambda x: self.objective(x),
            n_trials=n_trials,
            n_jobs=-1
        )
        param_dict = self.study.best_params.copy()
        param_dict.update(self.fixed_param_dict)
        param_dict["R0"] = self.calc_r0()
        param_dict["score"] = self.score()
        param_dict.update(self.calc_days_dict())
        self.param_dict = param_dict.copy()
        return param_dict

    def history_df(self):
        """
        Return the hsitory of optimization.
        @return <pd.DataFrame>
        """
        optimize_df = self.study.trials_dataframe()
        optimize_df["time[s]"] = optimize_df["datetime_complete"] - optimize_df["datetime_start"]
        optimize_df["time[s]"] = optimize_df["time[s]"].dt.total_seconds()
        self.optimize_df = optimize_df.drop(["datetime_complete", "datetime_start"], axis=1)
        return self.optimize_df.sort_values("value", ascending=True)

    def history_graph(self):
        """
        Show the history of parameter search using pair-plot.
        """
        if self.optimize_df is None:
            self.history_df()
        df = self.optimize_df.copy()
        sns.pairplot(df.loc[:, df.columns.str.startswith("params_")], diag_kind="kde", markers="+")
        plt.show()

    def objective(self, trial):
        # Time
        if "tau" in self.fixed_param_dict.keys():
            tau = self.fixed_param_dict["tau"]
        else:
            tau = trial.suggest_int("tau", 1, 1440)
        train_df_divided = self.train_df.copy()
        train_df_divided["t"] = (train_df_divided["T"] / tau).astype(np.int64)
        # Parameters
        p_dict = dict()
        for (name, info) in self.model.param_dict(train_df_divided).items():
            if name in self.fixed_param_dict.keys():
                param = self.fixed_param_dict[name]
            else:
                value_min, value_max = info[1:]
                if name in self.range_param_dict.keys():
                    range_min, range_max = self.range_param_dict[name]
                    value_min = max(range_min, value_min)
                    value_max = min(range_max, value_max)
                if info[0] == "float":
                    param = trial.suggest_uniform(name, value_min, value_max)
                else:
                    param = trial.suggest_int(name, value_min, value_max)
            p_dict[name] = param
        # Simulation
        t_end = train_df_divided.loc[train_df_divided.index[-1], "t"]
        sim_df = simulation(self.model, self.initials, step_n=t_end, **p_dict)
        return self.error_f(train_df_divided, sim_df)

    def error_f(self, train_df_divided, sim_df):
        """
        We need to minimize the difference of the observed values and estimated values.
        This function calculate the difference of the estimated value and obsereved value.
        """
        df = pd.merge(train_df_divided, sim_df, on="t", suffixes=("_observed", "_estimated"))
        # return self.rmsle(df)
        diffs = [
            # Weighted Average: the recent data is more important
            abs(p) * np.average(
                abs(df[f"{v}_observed"] - df[f"{v}_estimated"]) / (df[f"{v}_observed"] * self.total_population + 1),
                weights=df["t"]
            )
            for (p, v) in zip(self.model.PRIORITIES, self.model.VARIABLES)
        ]
        return sum(diffs) * self.total_population

    def compare_df(self):
        """
        Show the taining data and simulated data in one dataframe.
        
        """
        est_dict = self.study.best_params.copy()
        est_dict.update(self.fixed_param_dict)
        tau = est_dict["tau"]
        est_dict.pop("tau")
        observed_df = self.train_df.drop("T", axis=1)
        observed_df["t"] = (self.train_df["T"] / tau).astype(int)
        t_end = observed_df.loc[observed_df.index[-1], "t"]
        sim_df = simulation(self.model, self.initials, step_n=t_end, **est_dict)
        df = pd.merge(observed_df, sim_df, on="t", suffixes=("_observed", "_estimated"))
        df = df.set_index("t")
        return df

    def compare_graph(self):
        """
        Compare obsereved and estimated values in graphs.
        """
        df = self.compare_df()
        use_variables = [
            v for (i, (p, v)) in enumerate(zip(self.model.PRIORITIES, self.model.VARIABLES))
            if p != 0
        ]
        val_len = len(use_variables) + 1
        fig, axes = plt.subplots(ncols=1, nrows=val_len, figsize=(9, 6 * val_len / 2))
        for (ax, v) in zip(axes.ravel()[1:],use_variables):
            df[[f"{v}_observed", f"{v}_estimated"]].plot.line(
                ax=ax, ylim=(None, None), sharex=True,
                title=f"{self.model.NAME}: Comparison of observed/estimated {v}(t)"
            )
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci",  axis="y",scilimits=(0, 0))
            ax.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)
        for v in use_variables:
            df[f"{v}_diff"] = df[f"{v}_observed"] - df[f"{v}_estimated"]
            df[f"{v}_diff"].plot.line(
                ax=axes.ravel()[0], sharex=True,
                title=f"{self.model.NAME}: observed - estimated"
            )
        axes.ravel()[0].axhline(y=0, color="black", linestyle="--")
        axes.ravel()[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axes.ravel()[0].ticklabel_format(style="sci",  axis="y",scilimits=(0, 0))
        axes.ravel()[0].legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)
        fig.tight_layout()
        fig.show()
    
    def calc_r0(self):
        """
        Calculate R0.
        """
        est_dict = self.study.best_params.copy()
        est_dict.update(self.fixed_param_dict)
        est_dict.pop("tau")
        model_instance = self.model(**est_dict)
        return model_instance.calc_r0()

    def calc_days_dict(self):
        """
        Calculate 1/beta etc.
        """
        est_dict = self.study.best_params.copy()
        est_dict.update(self.fixed_param_dict)
        tau = est_dict["tau"]
        est_dict.pop("tau")
        model_instance = self.model(**est_dict)
        return model_instance.calc_days_dict(tau)

    def predict_df(self, step_n):
        """
        Predict the values in the future.
        @step_n <int>: the number of steps
        @return <pd.DataFrame>: predicted data for measurable variables.
        """
        est_dict = self.study.best_params.copy()
        est_dict.update(self.fixed_param_dict)
        tau = est_dict["tau"]
        est_dict.pop("tau")
        df = simulation(self.model, self.initials, step_n=step_n, **est_dict)
        df["Time"] = (df["t"] * tau).apply(lambda x: timedelta(minutes=x)) + self.start_time
        df = df.set_index("Time").drop("t", axis=1)
        df = (df * self.total_population).astype(np.int64)
        upper_cols = [n.upper() for n in df.columns]
        df.columns = upper_cols
        df = self.model.calc_variables_reverse(df).drop(upper_cols, axis=1)
        return df

    def predict_graph(self, step_n, name=None, excluded_cols=None):
        """
        Predict the values in the future and create a figure.
        @step_n <int>: the number of steps
        @name <str>: name of the area
        @excluded_cols <list[str]>: the excluded columns in the figure
        """
        if self.name is not None:
            name = self.name
        else:
            name = str() if name is None else name
        df = self.predict_df(step_n=step_n)
        if excluded_cols is not None:
            df = df.drop(excluded_cols, axis=1)
        r0 = self.param_dict["R0"]
        title = f"Prediction in {name} with {self.model.NAME} model: R0 = {r0}"
        line_plot(df, title, v= datetime.today(), h=self.total_population)

    def rmsle(self, compare_df):
        """
        Return the value of RMSLE.
        @param compare_df <pd.DataFrame>
        """
        df = compare_df.set_index("t") * self.total_population
        score = 0
        for (priority, v) in zip(self.model.PRIORITIES, self.model.VARIABLES):
            if priority == 0:
                continue
            observed, estimated = df[f"{v}_observed"], df[f"{v}_estimated"]
            diff = (np.log(observed + 1) - np.log(estimated + 1))
            score += (diff ** 2).sum()
        rmsle = np.sqrt(score / len(df))
        return rmsle

    def score(self):
        """
        Return the value of RMSLE.
        """
        rmsle = self.rmsle(self.compare_df().reset_index("t"))
        return rmsle

    def info(self):
        """
        Return Estimater information.
        @return <tupple[object]>:
            - <ModelBase>: model
            - <dict[str]=str>: name, total_population, start_time, tau
            - <dict[str]=float>: values of parameters of model
        """
        param_dict = self.study.best_params.copy()
        param_dict.update(self.fixed_param_dict)
        info_dict = {
            "name": self.name,
            "total_population": self.total_population,
            "start_time": self.start_time,
            "tau": param_dict["tau"],
            "initials": self.initials
        }
        param_dict.pop("tau")
        return (self.model, info_dict, param_dict)


# ### Parameters of Outbreaking group

# In[ ]:


get_ipython().run_cell_magic('time', '', 'outbreak_estimator = Estimator(\n    SIRF, outbreak_df, outbreak_df.loc[outbreak_df.index[-1], "Population"],\n    name="Outbreaking group", start_date=outbreak_group_start\n)\noutbreak_dict = outbreak_estimator.run()')


# In[ ]:


outbreak_estimator.history_df().head()


# In[ ]:


outbreak_estimator.history_graph()


# In[ ]:


outbreak_estimator.compare_df()


# In[ ]:


outbreak_estimator.compare_graph()


# In[ ]:


pd.DataFrame.from_dict({"Outbreaking group": outbreak_dict}, orient="index")


# In[ ]:


outbreak_estimator.predict_graph(step_n=500)


# ### Parameters of China

# In[ ]:


get_ipython().run_cell_magic('time', '', 'china_estimator = Estimator(\n    SIRF, china_df, china_df.loc[china_df.index[-1], "Population"],\n    name="China", start_date=china_start, end_date=china_end\n)\nchina_dict = china_estimator.run()')


# In[ ]:


china_estimator.history_df().head()


# In[ ]:


china_estimator.history_graph()


# In[ ]:


china_estimator.compare_df()


# In[ ]:


china_estimator.compare_graph()


# In[ ]:


pd.DataFrame.from_dict({"Outbreaking group": outbreak_dict, "China": china_dict}, orient="index")


# In[ ]:


china_estimator.predict_graph(step_n=500)


# ### Parameters of the others

# In[ ]:


get_ipython().run_cell_magic('time', '', 'others_estimator = Estimator(\n    SIRF, others_df, others_df.loc[others_df.index[-1], "Population"],\n    name="The others", start_date=others_start, end_date=others_end\n)\nothers_dict = others_estimator.run()')


# In[ ]:


others_estimator.history_df().head()


# In[ ]:


others_estimator.history_graph()


# In[ ]:


others_estimator.compare_df()


# In[ ]:


others_estimator.compare_graph()


# In[ ]:


pd.DataFrame.from_dict({"Outbreaking group": outbreak_dict, "China": china_dict, "The others": others_dict}, orient="index")


# In[ ]:


others_estimator.predict_graph(step_n=500)


# # Prediction and data submission
# We will predict the future with the estimated parameters.

# In[ ]:


class Predicter(object):
    """
    Predict the future using models.
    """
    def __init__(self, name, total_population, start_time, tau, initials, date_format="%d%b%Y"):
        """
        @name <str>: place name
        @total_population <int>: total population
        @start_time <datatime>: the start time
        @tau <int>: tau value (time step)
        @initials <list/tupple/np.array[float]>: initial values of the first model
        @date_format <str>: date format to display in figures
        """
        self.name = name
        self.total_population = total_population
        self.start_time = start_time
        self.tau = tau
        self.date_format = date_format
        # Un-fixed
        self.last_time = start_time
        self.axvlines = list()
        self.initials = initials
        self.df = pd.DataFrame()
        self.title_list = list()
        self.reverse_f = lambda x: x

    def add(self, model, end_day_n=None, count_from_last=False, vline=True, **param_dict):
        """
        @model <ModelBase>: the epidemic model
        @end_day_n <int/None>: day number of the end date (0, 1, 2,...), or None (now)
            - if @count_from_last <bool> is True, start point will be the last date registered to Predicter
        @vline <bool>: if True, vertical line will be shown at the end date
        @**param_dict <dict>: keyword arguments of the model
        """
        # Validate day nubber, and calculate step number
        if end_day_n is None:
            end_time = datetime.now()
        else:
            if count_from_last:
                end_time = self.last_time + timedelta(days=end_day_n)
            else:
                end_time = self.start_time + timedelta(days=end_day_n)
        if end_time <= self.last_time:
            raise Exception(f"Model on {end_time.strftime(self.date_format)} has been registered!")
        step_n = int((end_time - self.last_time).total_seconds() / 60 / self.tau)
        self.last_time = end_time
        # Perform simulation
        new_df = simulation(model, self.initials, step_n=step_n, **param_dict)
        new_df["t"] = new_df["t"] + len(self.df)
        self.df = pd.concat([self.df, new_df], axis=0).fillna(0)
        self.initials = new_df.set_index("t").iloc[-1, :]
        # For title
        if vline:
            self.axvlines.append(end_time)
            r0 = model(**param_dict).calc_r0()
            self.title_list.append(
                f"{model.NAME}({r0}, -{end_time.strftime(self.date_format)})"
            )
        # Update reverse function (X, Y,.. to Susceptible, Infected,...)
        self.reverse_f = model.calc_variables_reverse
        return self

    def restore_df(self):
        """
        Return the dimentional simulated data.
        @return <pd.DataFrame>
        """
        df = self.df.copy()
        df["Time"] = self.start_time + df["t"].apply(lambda x: timedelta(minutes=x * self.tau))
        df = df.drop("t", axis=1).set_index("Time") * self.total_population
        df = df.astype(np.int64)
        upper_cols = [n.upper() for n in df.columns]
        df.columns = upper_cols
        df = self.reverse_f(df).drop(upper_cols, axis=1)
        return df

    def restore_graph(self, drop_cols=None, **kwargs):
        """
        Show the dimentional simulate data as a figure.
        @drop_cols <list[str]>: the columns not to be shown
        @kwargs: keyword arguments of line_plot() function
        """
        df = self.restore_df()
        if drop_cols is not None:
            df = df.drop(drop_cols, axis=1)
        axvlines = [datetime.now(), *self.axvlines] if len(self.axvlines) == 1 else self.axvlines[:]
        line_plot(
            df,
            title=f"{self.name}: {', '.join(self.title_list)}",
            v=axvlines[:-1],
            h=self.total_population,
            **kwargs
        )


# ## Predict the future

# In[ ]:


days_to_predict = int((test_df["Date"].max() - datetime.today()).total_seconds() / 3600 / 24 + 1)
days_to_predict


# ### Outbreaking group

# In[ ]:


_, outbreak_info_dict, outbreak_param_dict = outbreak_estimator.info()


# In[ ]:


predicter = Predicter(**outbreak_info_dict)
predicter.add(SIRF, end_day_n=None, count_from_last=False, vline=False, **outbreak_param_dict)
predicter.add(SIRF, end_day_n=days_to_predict, count_from_last=True, **outbreak_param_dict)
outbreak_predict = predicter.restore_df()
predicter.restore_graph(drop_cols=["Susceptible"])


# ### China

# In[ ]:


_, china_info_dict, china_param_dict = china_estimator.info()


# In[ ]:


predicter = Predicter(**china_info_dict)
predicter.add(SIRF, end_day_n=None, count_from_last=False, vline=False, **china_param_dict)
predicter.add(SIRF, end_day_n=days_to_predict, count_from_last=True, **china_param_dict)
china_predict = predicter.restore_df()
predicter.restore_graph(drop_cols=["Susceptible"])


# ### The others

# In[ ]:


_, others_info_dict, others_param_dict = others_estimator.info()


# In[ ]:


predicter = Predicter(**others_info_dict)
predicter.add(SIRF, end_day_n=None, count_from_last=False, vline=False, **others_param_dict)
predicter.add(SIRF, end_day_n=days_to_predict, count_from_last=True, **others_param_dict)
others_predict = predicter.restore_df()
predicter.restore_graph(drop_cols=["Susceptible"])


# ### Total

# In[ ]:


outbreak_predict


# In[ ]:


china_predict


# In[ ]:


others_predict


# In[ ]:


def organize_pred(df, group):
    df = df.copy()
    df["Group"] = group
    df["Date"] = df.index.date
    df = df.reset_index(drop=True).groupby("Date").last()
    df.index = pd.to_datetime(df.index)
    return df

df = pd.concat(
    [
        organize_pred(outbreak_predict, "Outbreaking"),
        organize_pred(china_predict, "China"),
        organize_pred(others_predict, "Others")
    ],
    axis=0
)
df["Confirmed"] = df["Infected"] + df["Recovered"] + df["Fatal"]
group_predict_df = df.loc[:, ["Group", "Confirmed", "Fatal"]]
group_predict_df.tail()


# In[ ]:


line_plot(group_predict_df.drop("Group", axis=1), "Predicted total values")


# ### Country/Province level

# In[ ]:


group_predict_df.reset_index().tail()


# In[ ]:


current_growth_df.tail()


# In[ ]:


record_df = pd.DataFrame()

for i in range(len(group_predict_df)):
    time, group, confirmed, fatal = group_predict_df.reset_index().iloc[i, :].tolist()
    df = current_growth_df.copy()
    df = df.loc[df["Group"] == group, :]
    df["Confirmed"] = confirmed / (df["Current_Confirmed"] + 1).max() * (df["Current_Confirmed"] + 1)
    df["Fatal"] = fatal / (df["Current_Fatal"] + 1).max() * (df["Current_Fatal"] + 1)
    df["Date"] = time
    record_df = pd.concat([record_df, df], axis=0)

record_df = record_df.loc[:, ["Date", "Area", "Confirmed", "Fatal"]]
record_df[["Confirmed", "Fatal"]] = record_df[["Confirmed", "Fatal"]].astype(np.int64)
record_df.tail(20)


# ### Submit data

# In[ ]:


test_df.tail()


# In[ ]:


submission_sample_raw.head()


# In[ ]:


submission_sample_raw.tail()


# In[ ]:


submission_sample_raw.shape


# In[ ]:


df = pd.merge(record_df, test_df, on=["Date", "Area"])
df = df.sort_values("ForecastId").reset_index()
df = df.loc[:, ["ForecastId", "Confirmed", "Fatal"]]
df = df.rename({"Confirmed": "ConfirmedCases", "Fatal": "Fatalities"}, axis=1)
submission_df = df.copy()
submission_df


# In[ ]:


submission_df.shape


# In[ ]:


submission_df.to_csv("submission.csv", index=False)

