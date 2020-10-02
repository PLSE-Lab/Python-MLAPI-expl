#!/usr/bin/env python
# coding: utf-8

# # Logistic curve fit and XGBoost hybrid fit

# In previous weeks we found that a logistic curve fit works quite well on a per country level, and that adding a global XGBoost fit with [augmented data](https://www.kaggle.com/nxpnsv/country-health-indicators) is an improvement. The main idea for improvement in this notebook is to make an optimal interpolation between the two methods.

# ## Set up environment

# In[ ]:


# Imports
import os
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from scipy.optimize.minpack import curve_fit
from scipy.optimize import least_squares
from xgboost import XGBRegressor


# In[ ]:


# Helper functions
def load_kaggle_csv(dataset: str, datadir: str) -> pd.DataFrame:
    """Load and clean kaggle csv input"""
    df = pd.read_csv(
        f"{os.path.join(datadir,dataset)}.csv", parse_dates=["Date"]
    )
    df['country'] = df["Country_Region"]
    if "Province_State" in df:
        df["Country_Region"] = np.where(
            df["Province_State"].isnull(),
            df["Country_Region"],
            df["Country_Region"] + "_" + df["Province_State"],
        )
        df.drop(columns="Province_State", inplace=True)
    if "ConfirmedCases" in df:
        df["ConfirmedCases"] = df.groupby("Country_Region")[
            "ConfirmedCases"
        ].cummax()
    if "Fatalities" in df:
        df["Fatalities"] = df.groupby("Country_Region")["Fatalities"].cummax()
    if not "DayOfYear" in df:
        df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Date"] = df["Date"].dt.date
    return df

def RMSLE(actual: np.ndarray, prediction: np.ndarray) -> float:
    """Calculate RMSLE between actual and predicted values"""
    return np.sqrt(
        np.mean(
            np.power(np.log1p(np.maximum(0, prediction)) - np.log1p(actual), 2)
        )
    )


# ## Load data

# In[ ]:


# TRAIN DATA
# Kaggle input
train = load_kaggle_csv(
    "train", "/kaggle/input/covid19-global-forecasting-week-3")
# Augmentations
country_health_indicators = (
    pd.read_csv("/kaggle/input/country-health-indicators/country_health_indicators_v3.csv")).rename(
    columns={'Country_Region': 'country'})
# Merge augmentation to kaggle input
train = pd.merge(train, country_health_indicators, on="country", how="left")


# In[ ]:


train.head()


# In[ ]:


# TEST DATA
test = load_kaggle_csv(
    "test", "/kaggle/input/covid19-global-forecasting-week-3")
test = pd.merge(test, country_health_indicators, on="country", how="left")


# # Logistic fit
# 
# The logistc fit uses `scipy.optimize.curvefit` to fit a [logistic function](https://en.wikipedia.org/wiki/Logistic_function):
# 
# $$f(x) = \frac{L}{1 + \exp(-k(x - x0))}$$
# 
# The fit is done for each `Country_Region` separateley. Each fit is initialized with a first guess 
# 
# $$p_0(x_0, L, k)=(\mathrm{median}(x), \max(y), 0.1)*(U+0.5,U +1.0 ,U+0.5)$$
# 
# where $U$ are uniform random numbers. The fits are repeated repeated $n_\mathrm{samples}$ times and the fit producing the lowest RMSLE is used for prediction.  In addition, bounds are set as $x_0\in[0, 200]$, $L\in[\max(y), 10^6]$, and $k\in[0.1, 0.5]$. Furthermore, the error on $y$ is estimated to be $\sigma_y=\sqrt(y)(0.1+0.9U)$. This is the Poisson error with a random scaling to reduce assumptions on the optimal scaling with $\sigma_y$. For speed these fits are done in parallel with `joblib`.
# 
# First we define the required functions:

# In[ ]:


def logistic(x: np.ndarray, x0: float, L: float, k: float) -> np.ndarray:
    """Simple logistic function"""
    return L / (1 + np.exp(-k * (x - x0)))


def fit_single_logistic(x: np.ndarray, y: np.ndarray, maxfev: float) -> Tuple:
    """Fit with randopm jitter"""
    # Fuzzy fitter
    p0 = [np.median(x), y[-1], 0.1]
    pn0 = p0 * (np.random.random(len(p0)) + [0.5, 1.0, 0.5])
    try:
        params, pcov = curve_fit(
            logistic,
            x,
            y,
            p0=pn0,
            maxfev=maxfev,
            sigma=np.maximum(1, np.sqrt(y)) * (0.1 + 0.9 * np.random.random()),
            bounds=([0, y[-1], 0.01], [200, 1e6, 1.5]),
        )
        pcov = pcov[np.triu_indices_from(pcov)]
    except (RuntimeError, ValueError):
        params = p0
        pcov = np.zeros(len(p0) * (len(p0) - 1))
    y_hat = logistic(x, *params)
    rmsle = RMSLE(y_hat, y)
    return (params, pcov, rmsle, y_hat)


def fit_logistic(
    df: pd.DataFrame,
    n_jobs: int = 8,
    n_samples: int = 80,
    maxfev: int = 8000,
    x_col: str = "DayOfYear",
    y_cols: List[str] = ["ConfirmedCases", "Fatalities"],
) -> pd.DataFrame:
    def fit_one(df: pd.DataFrame, y_col: str) -> Dict:
        best_rmsle = None
        best_params = None
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()
        for (params, cov, rmsle, y_hat) in Parallel(n_jobs=n_jobs)(
            delayed(fit_single_logistic)(x, y, maxfev=maxfev)
            for i in range(n_samples)
        ):
            if rmsle >= (best_rmsle or rmsle):
                best_rmsle = rmsle
                best_params = params
        result = {f"{y_col}_rmsle": best_rmsle}
        result.update({f"{y_col}_p_{i}": p for i, p in enumerate(best_params)})
        return result

    result = {}
    for y_col in y_cols:
        result.update(fit_one(df, y_col))
    return pd.DataFrame([result])


def predict_logistic(
    df: pd.DataFrame,
    x_col: str = "DayOfYear",
    y_cols: List[str] = ["ConfirmedCases", "Fatalities"],
):
    def predict_one(col):
        df[f"yhat_logistic_{col}"] = logistic(
            df[x_col].to_numpy(),
            df[f"{col}_p_0"].to_numpy(),
            df[f"{col}_p_1"].to_numpy(),
            df[f"{col}_p_2"].to_numpy(),
        )

    for y_col in y_cols:
        predict_one(y_col)


# Now apply fit to each `Country_Region`. This takes a few minutes...

# In[ ]:


train = pd.merge(
    train, train.groupby(
    ["Country_Region"], observed=True, sort=False
).apply(lambda x: fit_logistic(x, n_jobs=8, n_samples=16, maxfev=16000)).reset_index(), on=["Country_Region"], how="left")
predict_logistic(train)


# # XGB boost regression

# In[ ]:


def apply_xgb_model(train, x_columns, y_column, xgb_params):
    X = train[x_columns].to_numpy()
    y = train[y_column].to_numpy()
    xgb_fit = XGBRegressor(**xgb_params).fit(X, y)
    y_hat = xgb_fit.predict(X)
    train[f"yhat_xgb_{y_column}"] = y_hat
    return RMSLE(y, y_hat), xgb_fit


# In[ ]:


xgb_params = dict(
    gamma=0.2,
    learning_rate=0.15,
    n_estimators=100,
    max_depth=11,
    min_child_weight=1,
    nthread=8,
    objective="reg:squarederror")
x_columns = [
    'DayOfYear', 'cases_growth', 'death_growth',
    'Cardiovascular diseases (%)', 'Cancers (%)',
    'Diabetes, blood, & endocrine diseases (%)', 'Respiratory diseases (%)',
    'Liver disease (%)', 'Diarrhea & common infectious diseases (%)',
    'Musculoskeletal disorders (%)', 'HIV/AIDS and tuberculosis (%)',
    'Malaria & neglected tropical diseases (%)',
    'Nutritional deficiencies (%)', 'pneumonia-death-rates',
    'Share of deaths from smoking (%)', 'alcoholic_beverages',
    'animal_fats', 'animal_products', 'aquatic_products,_other',
    'cereals_-_excluding_beer', 'eggs', 'fish,_seafood',
    'fruits_-_excluding_wine', 'meat', 'milk_-_excluding_butter',
    'miscellaneous', 'offals', 'oilcrops', 'pulses', 'spices',
    'starchy_roots', 'stimulants', 'sugar_&_sweeteners', 'treenuts',
    'vegetable_oils', 'vegetables', 'vegetal_products',
    'hospital_beds_per10k', 'hospital_density', 'nbr_surgeons',
    'nbr_obstetricians', 'nbr_anaesthesiologists', 'medical_doctors_per10k',
    'bcg_coverage', 'bcg_year_delta', 'population',
    'median age', 'population growth rate', 'birth rate', 'death rate',
    'net migration rate', 'maternal mortality rate',
    'infant mortality rate', 'life expectancy at birth',
    'total fertility rate', 'obesity - adult prevalence rate',
    'school_shutdown_1case', 'school_shutdown_10case',
    'school_shutdown_50case', 'school_shutdown_1death', 'FF_DayOfYear',
    'case1_DayOfYear', 'case10_DayOfYear', 'case50_DayOfYear', 'yhat_logistic_ConfirmedCases',
    'yhat_logistic_Fatalities']
xgb_c_rmsle, xgb_c_fit = apply_xgb_model(
    train, x_columns, "ConfirmedCases", xgb_params)
xgb_f_rmsle, xgb_f_fit = apply_xgb_model(
    train, x_columns, "Fatalities", xgb_params)


# # Top boosted features

# In[ ]:


imps=[]
cols = []
for col, fit in (("ConfirmedCases", xgb_c_fit), ("Fatalities", xgb_f_fit)):
    df = pd.DataFrame(list(zip(x_columns, fit.feature_importances_)), columns=[f"feature_{col}", f"importance_{col}"])
    cols.extend(df.columns.to_list())
    imps.append(df.sort_values(by=f"importance_{col}", ascending=False).to_numpy())
importances = pd.DataFrame(np.hstack(imps), columns=cols, index=range(1, len(imps[0])+1))
importances.index.name="rank"
importances.head(20)


# # Hybrid fit
# 
# From logistic curve fit we have $\hat{y}_L$: `yhat_logistic_ConfirmedCases`,and from XGB boost regression $\hat{y}_X$: `yhat_xgb_ConfirmedCases`.
# Here we make a hybrid predictor
# 
#  $\hat{y}_H = \alpha \hat{y}_L + (1-\alpha) \hat{y}_X$ 
#  
#  by fitting alpha with `scipy.optmize.least_squares`. Similarly for `Fatalities`. First we define a few functions to do the work:

# In[ ]:


def interpolate(alpha, x0, x1):
    return x0 * alpha + x1 * (1 - alpha)


def RMSLE_interpolate(alpha, y, x0, x1):
    return RMSLE(y, interpolate(alpha, x0, x1))


def fit_hybrid(
    train: pd.DataFrame, y_cols: List[str] = ["ConfirmedCases", "Fatalities"]
) -> pd.DataFrame:
    def fit_one(y_col: str):
        opt = least_squares(
            fun=RMSLE_interpolate,
            args=(
                train[y_col],
                train[f"yhat_logistic_{y_col}"],
                train[f"yhat_xgb_{y_col}"],
            ),
            x0=(0.5,),
            bounds=((0.0), (1.0,)),
        )
        return {f"{y_col}_alpha": opt.x[0], f"{y_col}_cost": opt.cost}

    result = {}
    for y_col in y_cols:
        result.update(fit_one(y_col))
    return pd.DataFrame([result])


def predict_hybrid(
    df: pd.DataFrame,
    x_col: str = "DayOfYear",
    y_cols: List[str] = ["ConfirmedCases", "Fatalities"],
):
    def predict_one(col):
        df[f"yhat_hybrid_{col}"] = interpolate(
            df[f"{y_col}_alpha"].to_numpy(),
            df[f"yhat_logistic_{y_col}"].to_numpy(),
            df[f"yhat_xgb_{y_col}"].to_numpy(),
        )

    for y_col in y_cols:
        predict_one(y_col)


# Now apply to each `Country_Region`:

# In[ ]:


train = pd.merge(
    train,
    train.groupby(["Country_Region"], observed=True, sort=False)
    .apply(lambda x: fit_hybrid(x))
    .reset_index(),
    on=["Country_Region"],
    how="left",
)


# In[ ]:


predict_hybrid(train)


# # Compare aproaches

# In[ ]:


print(
    "Confirmed:\n"
    f'Logistic\t{RMSLE(train["ConfirmedCases"], train["yhat_logistic_ConfirmedCases"])}\n'
    f'XGBoost\t{RMSLE(train["ConfirmedCases"], train["yhat_xgb_ConfirmedCases"])}\n'
    f'Hybrid\t{RMSLE(train["ConfirmedCases"], train["yhat_hybrid_ConfirmedCases"])}\n'
    f"Fatalities:\n"
    f'Logistic\t{RMSLE(train["Fatalities"], train["yhat_logistic_Fatalities"])}\n'
    f'XGBoost\t{RMSLE(train["Fatalities"], train["yhat_xgb_Fatalities"])}\n'
    f'Hybrid\t{RMSLE(train["Fatalities"], train["yhat_hybrid_Fatalities"])}\n'
)


# # Predict test cases

# In[ ]:


# Merge logistic and hybrid fit into test
test = pd.merge(
    test, 
    train[["Country_Region"] +
          ['ConfirmedCases_p_0', 'ConfirmedCases_p_1', 'ConfirmedCases_p_2']+
          ['Fatalities_p_0','Fatalities_p_1', 'Fatalities_p_2'] + 
          ["Fatalities_alpha"] + 
          ["ConfirmedCases_alpha"]].groupby(['Country_Region']).head(1), on="Country_Region", how="left")


# In[ ]:


# Test predictions
predict_logistic(test)
test["yhat_xgb_ConfirmedCases"] = xgb_c_fit.predict(test[x_columns].to_numpy())
test["yhat_xgb_Fatalities"] = xgb_f_fit.predict(test[x_columns].to_numpy())
predict_hybrid(test)


# # Prepare submission

# In[ ]:


submission = test[["ForecastId", "yhat_hybrid_ConfirmedCases", "yhat_hybrid_Fatalities"]].round().astype(int).rename(
        columns={
            "yhat_hybrid_ConfirmedCases": "ConfirmedCases",
            "yhat_hybrid_Fatalities": "Fatalities",
        }
    )
submission["ConfirmedCases"] = np.maximum(0, submission["ConfirmedCases"])
submission["Fatalities"] = np.maximum(0, submission["Fatalities"])


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission.csv", index=False)

