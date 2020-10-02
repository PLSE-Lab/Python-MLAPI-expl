#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Using curve fitting method and SIR-F model, we will predict the number of confirmed cases and fatal cases with COVID-19 global data. SIR-F model was created in another notebook of an auther. Please refer to the references.  
# 
# Contents:
# * Preparation
# * Prediction of the number of recovered cases with logistic curve
# * Prameter estimation with SIR-F model
# * Prediction of global data
# * Data submission
# 
# References:
# * [COVID-19 - Growth of Virus in Specific Countries](https://www.kaggle.com/wjholst/covid-19-growth-of-virus-in-specific-countries)
# * [COVID-19 data with SIR model](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model)

# # Preparation

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


def select_area(ncov_df, group="Date", places=None, areas=None, excluded_places=None,
                start_date=None, end_date=None, date_format="%d%b%Y"):
    """
    Select the records of the palces.
    @ncov_df <pd.DataFrame>: the clean data
    @group <str or None>: group-by the group, or not perform (None)
    @area or @places:
        if ncov_df has Country and Province column,
            @places <list[tuple(<str/None>, <str/None>)]: the list of places
                - if the list is None, all data will be used
                - (str, str): both of country and province are specified
                - (str, None): only country is specified
                - (None, str) or (None, None): Error
        if ncov_df has Area column,
            @areas <list[str]>: the list of area names
                - if the list is None, all data will be used
                - eg. Japan
                - eg. US/California
    @excluded_places <list[tuple(<str/None>, <str/None>)]: the list of excluded places
        - if the list is None, all data in the "places" will be used
        - (str, str): both of country and province are specified
        - (str, None): only country is specified
        - (None, str) or (None, None): Error
    @start_date <str>: the start date or None
    @end_date <str>: the start date or None
    @date_format <str>: format of @start_date and @end_date
    @return <pd.DataFrame>: index and columns are as same as @ncov_df
    """
    # Select the target records
    df = ncov_df.copy()
    if (places is not None) or (excluded_places is not None):
        c_series = df["Country"]
        p_series = df["Province"]
        if places is not None:
            df = pd.DataFrame(columns=ncov_df.columns)
            for (c, p) in places:
                if c is None:
                    raise Exception("places: Country must be specified!")
                if p is None:
                    new_df = ncov_df.loc[c_series == c, :]
                else:
                    new_df = ncov_df.loc[(c_series == c) & (p_series == p), :]
                df = pd.concat([df, new_df], axis=0)
        if excluded_places is not None:
            for (c, p) in excluded_places:
                if c is None:
                    raise Exception("excluded_places: Country must be specified!")
                if p is None:
                    df = df.loc[c_series != c, :]
                else:
                    c_df = df.loc[(c_series == c) & (p_series != p), :]
                    other_df = df.loc[c_series != c, :]
                    df = pd.concat([c_df, other_df], axis=0)
    if areas is not None:
        df = df.loc[df["Area"].isin(areas), :]
    if group is not None:
        df = df.groupby(group).sum().reset_index()
    # Range of date
    if start_date is not None:
        df = df.loc[df["Date"] >= datetime.strptime(start_date, date_format), :]
    if end_date is not None:
        df = df.loc[df["Date"] <= datetime.strptime(end_date, date_format), :]
    # Aleart empty
    if df.empty:
        raise Exception("The output dataframe is empty!")
    return df


# In[ ]:


def show_trend(ncov_df, name=None, variable="Confirmed", n_changepoints=2, **kwargs):
    """
    Show trend of log10(@variable) using fbprophet package.
    @ncov_df <pd.DataFrame>: the clean data
    @variable <str>: variable name to analyse
        - if Confirmed, use Infected + Recovered + Deaths
    @n_changepoints <int>: max number of change points
    @kwargs: keword arguments of select_area()
    """
    # Data arrangement
    df = select_area(ncov_df, **kwargs)
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
    if name is None:
        try:
            name = f"{kwargs['places'][0][0]}: "
        except Exception:
            name = str()
    else:
        name = f"{name}: "
    plt.title(f"{name}log10({variable}) over time and chainge points")
    plt.ylabel(f"log10(the number of cases)")
    plt.xlabel("")


# In[ ]:


def create_target_df(ncov_df, total_population,
                     confirmed="Confirmed", recovered="Recovered", fatal="Fatal", **kwargs):
    """
    Select the records of the places, calculate the number of susceptible people,
     and calculate the elapsed time [day] from the start date of the target dataframe.
    @ncov_df <pd.DataFrame>: the clean data
    @total_population <int>: total population in the places
    column names in @ncov_df:
        @confirmed <str>: column name of the number of confirmed cases
        @recovered <str>: column name of the number of recovered cases
        @fatal <str>: column name of the number of fatal cases
    @kwargs: keword arguments of select_area()
    @return <tuple(2 objects)>:
        - 1. first_date <pd.Timestamp>: the first date of the selected records
        - 2. target_df <pd.DataFrame>:
            - column T: elapsed time [min] from the start date of the dataset
            - column Susceptible: the number of patients who are in the palces but not infected/recovered/died
            - column Infected: the number of infected cases
            - column Recovered: the number of recovered cases
            - column Deaths: the number of death cases
    """
    # Select the target records
    df = select_area(ncov_df, **kwargs)
    first_date = df.loc[df.index[0], "Date"]
    # column T
    df["T"] = ((df["Date"] - first_date).dt.total_seconds() / 60).astype(int)
    # coluns except T
    df["Susceptible"] = total_population - df[confirmed]
    df["Fatal"] = df.loc[:, fatal]
    if recovered in df.columns:
        df["Infected"] = df[confirmed] - df[recovered] - df[fatal]
        df["Recovered"] = df[recovered]
    else:
        df["Infected"] = np.nan
        df.loc[df.index[0], "Infected"] = df.loc[df.index[0], confirmed] - df.loc[df.index[0], fatal]
        df["Recovered"] = np.nan
        df.loc[df.index[0], "Recovered"] = 0
    response_variables = ["Susceptible", "Infected", "Recovered", "Fatal"]
    # Return
    target_df = df.loc[:, ["T", *response_variables]]
    return (first_date, target_df)


# In[ ]:


def simulation(model, initials, step_n, **params):
    """
    Solve ODE of the model.
    @model <ModelBase>: the model
    @initials <tuple[float]>: the initial values
    @step_n <int>: the number of steps
    @params: the paramerters of the model
    """
    tstart, dt, tend = 0, 1, step_n
    sol = solve_ivp(
        fun=model(**params),
        t_span=[tstart, tend],
        y0=np.array(initials[:-1], dtype=np.float64),
        t_eval=np.arange(tstart, tend + dt, dt),
        dense_output=True
    )
    t_df = pd.Series(data=sol["t"], name="t")
    y_df = pd.DataFrame(data=sol["y"].T.copy(), columns=model.VARIABLES[:-1])
    sim_df = pd.concat([t_df, y_df], axis=1)
    sim_df[model.VARIABLES[-1]] = 1 - y_df.sum(axis=1)
    return sim_df


# In[ ]:


class ModelBase(object):
    NAME = "Model"
    VARIABLES = ["x"]
    PRIORITIES = np.array([1])
    QUANTILE_RANGE = [0.3, 0.7]

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
    def create_dataset(cls, ncov_df, total_population, **kwargs):
        """
        Create dataset with the model-specific varibles.
        The variables will be divided by total population.
        The column names (not include T) will be lower letters.
        **kwargs: See the function named create_target_df()
        @return <tuple(objects)>:
            - start_date <pd.Timestamp>
            - initials <tuple(float)>: the initial values
            - Tend <int>: the last value of T
            - df <pd.DataFrame>: the dataset
        """
        start_date, target_df = create_target_df(ncov_df, total_population, **kwargs)
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


class SIR(ModelBase):
    NAME = "SIR"
    VARIABLES = ["x", "y", "z"]
    PRIORITIES = np.array([1, 1, 1])

    def __init__(self, rho, sigma):
        super().__init__()
        self.rho = float(rho)
        self.sigma = float(sigma)

    def __call__(self, t, X):
        # x, y, z = [X[i] for i in range(len(self.VARIABLES))]
        # dxdt = - self.rho * x * y
        # dydt = self.rho * x * y - self.sigma * y
        # dzdt = self.sigma * y
        dxdt = - self.rho * X[0] * X[1]
        dydt = self.rho * X[0] * X[1] - self.sigma * X[1]
        # dzdt = self.sigma * X[1]
        return np.array([dxdt, dydt])#, dzdt])

    @classmethod
    def param_dict(cls, train_df_divided=None, q_range=None):
        param_dict = super().param_dict()
        q_range = super().QUANTILE_RANGE[:] if q_range is None else q_range
        if train_df_divided is None:
            param_dict["rho"] = ("float", 0, 1)
            param_dict["sigma"] = ("float", 0, 1)
        else:
            df = train_df_divided.copy()
            # rho = - (dx/dt) / x / y
            rho_series = 0 - df["x"].diff() / df["t"].diff() / df["x"] / df["y"]
            param_dict["rho"] = ("float", *rho_series.quantile(q_range))
            # sigma = (dz/dt) / y
            sigma_series = df["z"].diff() / df["t"].diff() / df["y"]
            param_dict["sigma"] = ("float", *sigma_series.quantile(q_range))
        return param_dict

    @staticmethod
    def calc_variables(df):
        df["X"] = df["Susceptible"]
        df["Y"] = df["Infected"]
        df["Z"] = df["Recovered"] + df["Fatal"]
        return df.loc[:, ["T", "X", "Y", "Z"]]

    @staticmethod
    def calc_variables_reverse(df):
        df["Susceptible"] = df["X"]
        df["Infected"] = df["Y"]
        df["Recovered/Deaths"] = df["Z"]
        return df

    def calc_r0(self):
        if self.sigma == 0:
            return np.nan
        r0 = self.rho / self.sigma
        return round(r0, 2)

    def calc_days_dict(self, tau):
        _dict = dict()
        _dict["1/beta [day]"] = int(tau / 24 / 60 / self.rho)
        _dict["1/gamma [day]"] = int(tau / 24 / 60 / self.sigma)
        return _dict


# In[ ]:


class SIRF(ModelBase):
    NAME = "SIR-F"
    VARIABLES = ["x", "y", "z", "w"]
    PRIORITIES = np.array([1, 10, 10, 2])

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
        # dwdt = self.rho * self.theta * X[0] * X[1] + self.kappa * X[1]
        return np.array([dxdt, dydt, dzdt]) # , dwdt])

    @classmethod
    def param_dict(cls, train_df_divided=None, q_range=None):
        param_dict = super().param_dict()
        q_range = super().QUANTILE_RANGE[:] if q_range is None else q_range
        param_dict["theta"] = ("float", 0, 1)
        param_dict["kappa"] = ("float", 0, 1)
        if train_df_divided is None:
            param_dict["rho"] = ("float", 0, 1)
            param_dict["sigma"] = ("float", 0, 1)
        else:
            df = train_df_divided.copy()
            # rho = - (dx/dt) / x / y
            rho_series = 0 - df["x"].diff() / df["t"].diff() / df["x"] / df["y"]
            param_dict["rho"] = ("float", *rho_series.quantile(q_range))
            # sigma = (dz/dt) / y
            sigma_series = df["z"].diff() / df["t"].diff() / df["y"]
            param_dict["sigma"] = ("float", *sigma_series.quantile(q_range))
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


# In[ ]:


class Estimator(object):
    def __init__(self, model, ncov_df, total_population, name=None, places=None, areas=None,
                 excluded_places=None, start_date=None, end_date=None, date_format="%d%b%Y", **params):
        """
        Set training data.
        @model <ModelBase>: the model
        @name <str>: name of the area
        @params: fixed parameter of the model
        @the other params: See the function named create_target_df()
        """
        # Fixed parameters
        self.fixed_param_dict = params.copy()
        # Register the dataset arranged for the model
        dataset = model.create_dataset(
            ncov_df, total_population, places=places, areas=areas,
            excluded_places=excluded_places,
            start_date=start_date, end_date=end_date, date_format=date_format
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
        optimize_df["time[s]"] = optimize_df["datetime_complete"] -             optimize_df["datetime_start"]
        optimize_df["time[s]"] = optimize_df["time[s]"].dt.total_seconds()
        self.optimize_df = optimize_df.drop(
            ["datetime_complete", "datetime_start", "system_attrs__number"], axis=1)
        return self.optimize_df.sort_values("value", ascending=True)

    def history_graph(self):
        """
        Show the history of parameter search using pair-plot.
        """
        if self.optimize_df is None:
            self.history_df()
        df = self.optimize_df.copy()
        sns.pairplot(df.loc[:, df.columns.str.startswith(
            "params_")], diag_kind="kde", markers="+")
        plt.show()

    def objective(self, trial):
        # Time
        try:
            tau = self.fixed_param_dict["tau"]
        except KeyError:
            tau = trial.suggest_int("tau", 1, 1440)
        train_df_divided = self.train_df.copy()
        train_df_divided["t"] = (
            train_df_divided["T"] / tau).astype(np.int64)
        # Parameters
        param_dict = self.model.param_dict(train_df_divided)
        p_dict = {"tau": None}
        p_dict.update(
            {
                k: trial.suggest_uniform(k, v[1], v[2])
                for (k, v) in param_dict.items()
            }
        )
        p_dict.update(self.fixed_param_dict)
        p_dict.pop("tau")
        # Simulation
        t_end = train_df_divided.loc[train_df_divided.index[-1], "t"]
        sim_df = simulation(self.model, self.initials, step_n=t_end, **p_dict)
        return self.error_f(train_df_divided, sim_df)

    def error_f(self, train_df_divided, sim_df):
        """
        We need to minimize the difference of the observed values and estimated values.
        This function calculate the difference of the estimated value and obsereved value.
        """
        df = pd.merge(train_df_divided, sim_df, on="t",
                      suffixes=("_observed", "_estimated"))
        diffs = [
            # Weighted Average: the recent data is more important
            p * np.average(
                abs(df[f"{v}_observed"] - df[f"{v}_estimated"]) / \
                (df[f"{v}_observed"] * self.total_population + 1),
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
        sim_df = simulation(self.model, self.initials,
                            step_n=t_end, **est_dict)
        df = pd.merge(observed_df, sim_df, on="t",
                      suffixes=("_observed", "_estimated"))
        df = df.set_index("t")
        return df

    def compare_graph(self):
        """
        Compare obsereved and estimated values in graphs.
        """
        df = self.compare_df()
        use_variables = [
            v for (i, (p, v)) in enumerate(zip(self.model.PRIORITIES, self.model.VARIABLES))
            if p != 0 and i != 0
        ]
        val_len = len(use_variables) + 1
        fig, axes = plt.subplots(
            ncols=1, nrows=val_len, figsize=(9, 6 * val_len / 2))
        for (ax, v) in zip(axes.ravel()[1:], use_variables):
            df[[f"{v}_observed", f"{v}_estimated"]].plot.line(
                ax=ax, ylim=(0, None), sharex=True,
                title=f"{self.model.NAME}: Comparison of observed/estimated {v}(t)"
            )
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci",  axis="y", scilimits=(0, 0))
            ax.legend(bbox_to_anchor=(1.02, 0),
                      loc="lower left", borderaxespad=0)
        for v in use_variables:
            df[f"{v}_diff"] = df[f"{v}_observed"] - df[f"{v}_estimated"]
            df[f"{v}_diff"].plot.line(
                ax=axes.ravel()[0], sharex=True,
                title=f"{self.model.NAME}: observed - estimated"
            )
        axes.ravel()[0].axhline(y=0, color="black", linestyle="--")
        axes.ravel()[0].yaxis.set_major_formatter(
            ScalarFormatter(useMathText=True))
        axes.ravel()[0].ticklabel_format(
            style="sci",  axis="y", scilimits=(0, 0))
        axes.ravel()[0].legend(bbox_to_anchor=(1.02, 0),
                               loc="lower left", borderaxespad=0)
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
        df["Time"] = (
            df["t"] * tau).apply(lambda x: timedelta(minutes=x)) + self.start_time
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
        line_plot(df, title, v=datetime.today(), h=self.total_population)

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
        self.df = pd.concat([self.df, new_df.iloc[1:, :]], axis=0).fillna(0)
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


# In[ ]:


for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_raw = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")
test_raw = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
submission_sample_raw = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")
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


# In[ ]:


df = pd.merge(
    train_raw.rename({"Province_State": "Province", "Country_Region": "Country"}, axis=1),
    population_raw.rename({"Province.State": "Province", "Country.Region": "Country"}, axis=1),
    on=["Country", "Province"],
    how="left"
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


# In[ ]:


df = pd.merge(
    test_raw.rename({"Province_State": "Province", "Country_Region": "Country"}, axis=1),
    population_raw.rename({"Province.State": "Province", "Country.Region": "Country"}, axis=1),
    on=["Country", "Province"],
    how="left"
)
df["Area"] = df[["Country", "Province"]].apply(
    lambda x: f"{x[0]}" if x[1] is np.nan else f"{x[0]}/{x[1]}",
    axis=1
)
df["Date"] = pd.to_datetime(df["Date"])
df = df.loc[:, ["ForecastId", "Date", "Area", "Population"]]
test_df = df.copy()
test_df.tail()


# In[ ]:


def log_curve(x, k, x_0, ymax):
    return ymax / (1 + np.exp(-k*(x-x_0)))


# # Prediction of the number of recovered cases with logistic curve

# In[ ]:


def log_fit(train_df, area, metric):
    area_data = select_area(train_df, areas=[area])
    area_data = area_data.loc[area_data[metric] > 0, :]
    x_data = range(len(area_data.index))
    y_data = area_data[metric]
    if len(y_data) < 5:
        estimated_k = -1  
        estimated_x_0 = -1 
        ymax = -1
    elif max(y_data) == 0:
        estimated_k = -1  
        estimated_x_0 = -1 
        ymax = -1
    else:
        try:
            popt, pcov = curve_fit(
                log_curve, x_data, y_data, bounds=([0,0,0],np.inf),
                p0=[0.3,100,10000], maxfev=1000000
            )
            estimated_k, estimated_x_0, ymax = popt
        except RuntimeError:
            print(area)
            print("Error - curve_fit failed") 
            estimated_k = -1  
            estimated_x_0 = -1 
            ymax = -1
    estimated_parameters = pd.DataFrame(
        np.array([[area, estimated_k, estimated_x_0, ymax]]), columns=['Area', 'k', 'x_0', 'ymax']
    )
    return estimated_parameters


# In[ ]:


def get_parameters(metric):
    parameters = pd.DataFrame(columns=['Area', 'k', 'x_0', 'ymax'], dtype=np.float)
    for area in train_df['Area'].unique():
        estimated_parameters = log_fit(train_df, area, metric)
        parameters = parameters.append(estimated_parameters)
    parameters['k'] = pd.to_numeric(parameters['k'], downcast="float")
    parameters['x_0'] = pd.to_numeric(parameters['x_0'], downcast="float")
    parameters['ymax'] = pd.to_numeric(parameters['ymax'], downcast="float")
    parameters = parameters.replace({'k': {-1: parameters[parameters['ymax']>0].median()[0]}, 
                                     'x_0': {-1: parameters[parameters['ymax']>0].median()[1]}, 
                                     'ymax': {-1: parameters[parameters['ymax']>0].median()[2]}})
    return parameters


# ## Curve fitting: Confirmed, Fatal

# In[ ]:


confirmed_param_df = get_parameters("Confirmed")
confirmed_param_df.head()


# In[ ]:


df = train_df.loc[train_df["Confirmed"] > 0, ["Date", "Area"]].groupby("Area").first()
df = df.rename({"Date": "First_date"}, axis=1).reset_index()
df = pd.merge(confirmed_param_df, df)
confirmed_df = df.copy()
confirmed_df.loc[confirmed_df["Area"] == "Japan", :].head()


# In[ ]:


fatal_param_df = get_parameters("Fatal")
fatal_param_df.head()


# In[ ]:


df = train_df.loc[train_df["Fatal"] > 0, ["Date", "Area"]].groupby("Area").first()
df = df.rename({"Date": "First_date"}, axis=1).reset_index()
df = pd.merge(fatal_param_df, df)
fatal_df = df.copy()
fatal_df.loc[fatal_df["Area"] == "Japan", :].head()


# In[ ]:


df = pd.merge(confirmed_df, fatal_df, on="Area", suffixes=["_confirmed", "_fatal"])
# k
df["k"] = df["k_confirmed"]
# x_0
df["First_date_recovered"] = df["First_date_fatal"]
df["x_0"] = df[["x_0_confirmed", "x_0_fatal"]].max(axis=1)
# ymax
df["ymax"] = df["ymax_confirmed"] - df["ymax_fatal"]
# save
df = df.loc[:, ["Area", "First_date_recovered", "k", "x_0", "ymax"]]
recovered_df = df.copy()
recovered_df.loc[recovered_df["Area"] == "Japan", :]


# In[ ]:


df = train_df.loc[train_df["Confirmed"] > 0, :]
df = pd.merge(df, recovered_df, on="Area", how="left")
df["date_diff"] = (df["Date"] - df["First_date_recovered"]).dt.total_seconds() / 60 / 60 / 24
df.loc[(df["date_diff"] < 0) | (df["date_diff"].isnull()), "date_diff"] = -1
df["date_diff"] = df["date_diff"].astype(np.int64)
df["Recovered"] = df[["date_diff", "k", "x_0", "ymax"]].apply(
    lambda x: 0 if x[0] < 0 else log_curve(x[0], x[1], x[2], x[3]),
    axis=1
).astype(np.int64)
ncov_df = df.copy()
ncov_df.loc[df["Area"] == "Japan", :]


# In[ ]:


line_plot(ncov_df.groupby("Date").sum()[["Confirmed", "Fatal", "Recovered"]], "Global: Cases over time")


# # Trend analysis

# In[ ]:


show_trend(ncov_df, "Confirmed")


# In[ ]:


show_trend(ncov_df, "Confirmed", start_date="17Mar2020")


# # Prameter estimation with SIR-F model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'global_estimator = Estimator(\n    SIRF, ncov_df, population_dict["Global"], name="Global", areas=None,\n    start_date="17Mar2020"\n)\nglobal_dict = global_estimator.run()')


# In[ ]:


global_estimator.compare_graph()


# In[ ]:


pd.DataFrame.from_dict({"Global": global_dict}, orient="index")


# # Prediction of global data

# In[ ]:


days_to_predict = int((test_df["Date"].max() - datetime.today()).total_seconds() / 3600 / 24 + 10)
days_to_predict


# In[ ]:


_, info_dict, param_dict = global_estimator.info()


# In[ ]:


predicter = Predicter(**info_dict)
predicter.add(SIRF, end_day_n=None, count_from_last=False, vline=False, **param_dict)
predicter.add(SIRF, end_day_n=days_to_predict, count_from_last=True, **param_dict)
global_predict = predicter.restore_df()
predicter.restore_graph(drop_cols=["Susceptible"])


# In[ ]:


global_predict


# # Data submission

# ## Area level

# In[ ]:


df = ncov_df.copy()
df = df.loc[df["Date"] == df["Date"].max(), ["Area", "Confirmed", "Fatal"]]
df["Confirmed"] = df["Confirmed"] / df["Confirmed"].sum()
df["Fatal"] = df["Fatal"] / df["Fatal"].sum()
current_df = df.copy()
current_df.tail()


# In[ ]:


df = global_predict.copy()
df["Date"] = df.index.date
df["Confirmed"] = df["Infected"] + df["Recovered"] + df["Fatal"]
df = df.groupby("Date").last().reset_index()[["Date", "Confirmed", "Fatal"]]
global_df = df.copy()
global_df.tail()


# In[ ]:


record_df = pd.DataFrame()

for i in range(len(global_df)):
    date, confirmed, fatal = global_df.iloc[i, :].tolist()
    df = current_df.copy()
    df["Date"] = date
    df["Confirmed"] = (confirmed * df["Confirmed"]).astype(np.int64)
    df["Fatal"] = (fatal * df["Fatal"]).astype(np.int64)
    record_df = pd.concat([record_df, df], axis=0)

record_df["Date"] = pd.to_datetime(record_df["Date"])
record_df = record_df.loc[:, ["Date", "Area", "Confirmed", "Fatal"]].reset_index(drop=True)
record_df


# ## Submission

# In[ ]:


test_df.tail()


# In[ ]:


test_df.shape


# In[ ]:


test_raw.shape


# In[ ]:


submission_sample_raw.tail()


# In[ ]:


submission_sample_raw.shape


# In[ ]:


df = pd.merge(record_df, test_df, on=["Date", "Area"], how="right")
df = df.sort_values("ForecastId").reset_index()
df = df.loc[:, ["ForecastId", "Confirmed", "Fatal"]]
df = df.rename({"Confirmed": "ConfirmedCases", "Fatal": "Fatalities"}, axis=1)
submission_df = df.copy()
submission_df


# In[ ]:


submission_df.shape


# In[ ]:


len(submission_df) == len(submission_sample_raw)


# In[ ]:


submission_df.to_csv("submission.csv", index=False)

