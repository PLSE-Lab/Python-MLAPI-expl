#!/usr/bin/env python
# coding: utf-8

# ### Problem Statement:
# At my day job at the [US National Renewable Energy Laboratory](https://www.nrel.gov/), I spend a lot of time considering the technologies, pathways, and policies that will lead us to a clean energy future. I am often frustrated however, at the lack of data and tools to analyze the 'low hanging fruit' of emission reductions around the world, and furthermore at our inability to emiprically verify claims of renewable energy consumption. As more governments (and corporates) work towards a globally equitable 24/7 clean energy supply, we need timely, geographicaly granular data to conduct our analysis. 
# 
# Like many of us, I've found myself with a bit of unexpected free time these last few weeks... which I've spent on this kaggle competition. After some contemplation, I realized that adequate data exists to represent this problem as a *supervised* learning problem; using empirical satellite remote sensing data from the ESA Copernicus program as predictor variables, and using granular U.S. power plant (hourly) emission reporting from the EPA as training target variables. 
# 
# I started munging data in local notebooks, but before long I had a few thousand lines of code and realized that this project (and my interest!) far surpassed what could be contained in a notebook. 
# 
# So I started work on an open source package: [tycho](http://https://github.com/skoeb/tycho)
# 
# ### Introducing Tycho:
# Tycho is a power sector (scope 2) emissions measurement data collection pipeline to enable supervised machine learning of every power plant in the world. Tycho is primarily an ETL process that outputs X/y dataframes to be used in any given modelling process, additionally, this kaggle submission includes some preliminary machine modelling approaches (i.e. Linear Regression, TPOT, XGBoost) to demonstrate the potential of this data; there are likely better-tuned and more sophisticate approaches that can make use of Tycho's ETL output. 
# 
# >About the name: Tycho Brahe was a Danish astronomer and scientist remembered in part for his commitment to empirical observation. Notably, Tycho wore a brass prosthetic nose for much of his life after losing his nose in a drunken-sword fight following an argument with his third-cousin over who was the better mathametician. Tycho was inspired by his senior, Nicolaus Copernicus, the namesake of the ESA's Copernicus Sentinel-5 precursor satellite program. 
# 
# ### Why Tycho:
# Most present-day emission measurement techniques require emprical and trusted knowledge of plant generation data (such as hourly capacity factor) to model emissions based on near-linear emission curves. These methods are vulnerable to 'garbage in garbage out' syndrome. While there are high levels of support for international climate agreements that mandate routine reporting of emissions, multiple instances of misleading emissions claims have been made by power producers, utilities, and governments.
# 
# The state-of-the-art ESA Sentinel-5 Satellite provides remote multispectral empirical measurement of common power sector emission fluxes of air columns including concentrations of ozone, methane, formaldehyde, aerosol, carbon monoxide, nitrogen oxide, and sulphur dioxide, as well as cloud characteristics at a spatial resolution of 0.01 arc degrees. [https://developers.google.com/earth-engine/datasets/catalog/sentinel-5p]
# 
# While the data from Sentinel-5 has been made publically available since mid-2019, it remains difficult to process and aggregate this data. Aditionally, the rasterized measurements from Sentinel can be difficult to attribute to specific power plants, or even sources of emissions (i.e. Coal plant near a city with heavy combustion engine vehicle traffic). Tycho aims to remedy this problem, by providing a clean and robust training set linking specific observations (rows) from measured power plant data with representations of Sentinel-5 observations. A well trained model using Tycho should be able to begin predicting SO<sub>2</sub>, NO<sub>x</sub> and CO<sub>2</sub> (among other) emissions at weekly (or possibly daily) granularity for every significant fossil-fuel power plant across the world. 
# 
# #### Advantages of Tycho include:
# * **Automated** querying of Google Earth Engine for an expert-filtered and cleaned set of coordinates representing power plants from the Global Power Plant Database. 
# * **Robustness** from horizontal-stacking of multiple geographic scopes (1km, 10km, 100km by default) of all Sentinel data, along with observation-timed weather data (i.e. speed and direction of wind, volume of rain in the last week) along with population density for considerations of noise in data. 
# * **Feature Engineering** already done for you, with expert-selected features to inform your modelling. 
# * and lastly, a clean, well-documented, object-oriented codebase that is extensible to novel modelling techniques. 
# 
# #### Assumptions of Tycho include:
# * That machine learning models will be able to consider the complex relationships between multispectral observations, local weather, and demographic features such as population density.
# * That `Coal`, `Oil`, `Gas`, and `Petcoke` resources are the only generator types with emissions. 
# * That an aggregated mean value is sufficient at describing a raster value for a given point (with a buffer)
# * That fossil-fuel Power Plants around the world are similar in nature to U.S. power plants included in training data. 
# * Additional assumptions are listed in the doc-strings of Tycho's classes and functions.

# ### Installation:

# In[ ]:


# --- Download ---
# Currently hosted on github (with data)
get_ipython().system('git clone https://github.com/skoeb/tycho.git #make sure internet is turned on in kaggle   ')
#Looking to add to pip/conda-forge soon.

#Install environment with tycho.yml, or requirements.txt


# ![github screenshot](tycho/images/github_screenshot.png)

# ### Running ETL:
# #### Tycho's ETL process includes:
# * Fetching data from U.S. regulatory reporting to geocode fossil fuel power plants with hourly emission profiles for SO<sub>2</sub>, NO<sub>x</sub> and CO<sub>2</sub>.
# * Matching U.S. regulatory reporting with power plants in the comprehensive WRI Global Power Plant Database. 
# * Querying google earth engine to fetch remote sensing (satellite) data from the European Space Agency's Sentinel satellite program with emissions measurements on a daily basis across the globe. 
# 
# #### Tycho's sources:
# | Data Source                              | Features                                                                                     |   |   |   |
# |------------------------------------------|----------------------------------------------------------------------------------------------|---|---|---|
# |[Public Utility Data Liberation Project](http://https://www.google.com/search?q=pudl+github&oq=PUDL&aqs=chrome.1.69i57j69i59j35i39j46j0j69i60l3.1992j0j7&sourceid=chrome&ie=UTF-8)            | EIA 860/923 Regulatory Filings<br>containingU.S. power plant attributes                                                         |   |   |   |
# | [EPA Continuous Emission Modelling System](http://https://ampd.epa.gov/ampd/) | Hourly NOx, SO2, CO2, and load (MW) profiles<br>for fossil fuel power plants in the U.S. |   |   |   |
# | [WRI Global Power Plant Database](http://http://datasets.wri.org/dataset/globalpowerplantdatabase)          | Global power plant attributes                                                                |   |   |   |
# | [Google Earth Engine](http://https://developers.google.com/earth-engine/datasets/)          |  Sentinel-5p Global NOx, SO2, CO2, measurements <br>weather, and demographic attributes                                    |   |   |   |
# <br/><br/>
# #### A complete tycho ETL process looks something like this:
# ```
# # Step 1: Download EPA CEMS (hourly U.S. power plant emission data)
# CemsFetch = tycho.EPACEMSFetcher()
# CemsFetch.fetch() #saves as pickles
# 
# # Step 2: Load EIA 860/923 (power plant metadata and coordinates) from PUDL SQL server
# PudlLoad = tycho.PUDLLoader()
# PudlLoad.load()
# eightsixty = PudlLoad.eightsixty
# 
# # Step 3: load CEMS data from pickles
# CemsLoad = tycho.CEMSLoader()
# CemsLoad.load()
# cems = CemsLoad.cems
# 
# # Step 4: Load WRI Global Power Plant Database data from csv
# GppdLoad = tycho.GPPDLoader() 
# GppdLoad.load()
# gppd = GppdLoad.gppd
# 
# # Step 5: Merge eightsixty, gppd, cems together into a long_df
# TrainingMerge = tycho.TrainingDataMerger(eightsixty, gppd, cems)
# TrainingMerge.merge()
# df = TrainingMerge.df
# 
# # Step 6: Query Google Earth Engine Data
# for earthengine_db in ["COPERNICUS/S5P/OFFL/L3_NO2", "ECMWF/ERA5/DAILY"]:
#     EeFetch = tycho.EarthEngineFetcher(earthengine_db, buffers=[1e3, 1e4, 1e5])
#     EeFetch.fetch(df)
# 
# # Step 7: Merge Earth Engine Data onto df
# RemoteMerge = tycho.RemoteDataMerger()
# df = RemoteMerge.merge(df)
# ```
# 
# This is all included in the handy `etl.py` script, which will draw upon well-named switches and settings in `tycho/config.py` to customize your ETL process. 

# ### Training Data Coverage:
# 
# Training data consists of geo/temporal observations at a granularity and scope defined by the settings in `config.py`. 
# 
# Geographically the default settings include the continental United States, for which EPA CEMS data is available as training targets. EIA 860/923 includes data on some 27,991 U.S. generating units, of which 12,008 are operational fossil-fuel powered. Once we merge multiple generating units at a single 'power plant', we are left with 3,227 power plants, of which 965 can be geocoded to match a plant with complete 2019 hourly emissions data (NOSO<sub>x</sub>, SO<sub>2</sub>, CO<sub>2</sub>) from CEMS.  
# 
# ![Training Power Plants used by Tycho](tycho/images/united_states_of_america_map.png)
# *All images in this notebook are generated by running `create_viz.py` and stored in `images/`*
# 
# For each of these 965 power plants, one 'row' of training data represents a single temporal observation at the power plant. The default temporal setting of `W-SUN` indicates that 52 observations (weekly ending in sunday) are queried from Google Earth Engine for each powerplant present in GPPD/EIA data. Each Earth Engine query consists of three parts:
# 
# 1.  Defining a geographic scope, including a lat/lon point and a 'buffer' in meters. By default, queries are repeated at the same point for 1km, 10km, and 100kms to capture a variety of geographic perspectives that may describe phenomenon caused by weather or the environemnt (i.e. traffic pollution). Multiple geographic queries are stitched together in 'wide' format for each observation with prefixes indicating the buffer size. 
# 2.  Defining a time period, which is by default the value in the `datetime_utc` column of the query plus the value of `config.TS_FREQUENCY`. 
# 3.  Performing a 'ReduceRegion' operation using Google Earth Engine to reduce the buffered geography and time slice, and take the *mean* value for each band of the database. 
# 
# Queries are performed in parallel, with a fixed backoff added ontop of the default exponential backoff included by Earth Engine to avoid upsetting Google. Additionally, returned queries are cached locally within `data/earthengine`. Any null values or failed queries will be retried upon subsequent runs of `etl.py`. 
# 
# Once earth engine data has been received, we have a training data set with `965*52 = 50,180` rows. Depending on the number of Earth Engine databases being used (Sentinel 5p SO<sub>2</sub>, NO<sub>2</sub>; GPWv411 Population Count; and ERA5 Daily Weather are included by default), the training dataframe will have ~150 columns. Experimentally, some Tycho runs have been performed at the daily resolution, however Sentinel data tends to be noisy at this level due to gaps in imaging and cloud coverage. 
# 
# >Note: When downloading new earth engine data for the first time, you may have to [authenticate Google Earth Engine](http://https://developers.google.com/earth-engine/python_install-conda) by running `earthengine authenticate` from the command line.
# 
# Once `etl.py` is ran, the `processed/` folder will contain `cems_clean.pkl`, `eightsixty_clean.pkl`, `gppd_clean.pkl` files, `pre_ee.pkl` is a helpful debugging file containing these files merged together. Finally, `merged_df.pkl` includes Earth Engine data merged on as well. This is the final format of the data before being sent to the sanitization and feature engineering process. Below, we can examine the data included in `merged_df.pkl`:

# In[ ]:


import pandas as pd
pd.set_option("display.max_columns", 200)
merged = pd.read_pickle('tycho/processed/merged_df.pkl')
merged.set_index(['datetime_utc','plant_id_wri'], inplace=True, drop=True)
merged


# `merged_df.pkl` contains features that will be used as `X` predictor variables, columns that will be used as `y` target variables (`'nox_lbs','so2_lbs','co2_tons','operational_time','gross_load_mw'`), and some columns that will be discarded in the sanitize and feature engineering process, as they are not available globally. 
# 
# Below, we can examine a few of the relationships in this dataset. First, let's query `tycho/visualizer.plot_cems_emissions()` which produces a series of scatterplots examining the relationship between the CEMS target variables that we will be predicting, and the exogenous WRI capacity data. On the X axis of these plots are U.S. power plant capacity factors derived from WRI GPPD data (`estimated_generation_gwh * 1000/ wri_capacity_mw`) indicating the utilization of each plant. On the y axis are the cumulative annual emissions, derived from EPA CEMS data. We can see unsurprisingly in these plots that Coal fired power plants produce the most emissions across all three `y` emission targets. Furthermore, gas powered plants do not produce as large amount of NOx and SO2 as CO2. Finally, that there appears to be a linear relationship for *most* plants between estimated capacity factor and cumulative emissions; for other plants, increases in capacity factor do not cause significant increases of some emission types, likely due to the presence of targeted emission reduction technologies (i.e. catalytic reduction techniques). 
# 
# ![Training Power Plants Capacity Factor vs. Particulate Emissions](tycho/images/cems_cf_emissions.png)
# 
# I encourage you to explore other visualizations within the `images/` folder, including a [EDA Pairplot](http://https://github.com/skoeb/tycho/blob/master/images/eda_pairplot.png) examining the correlations and distributions of the various `y` targets, and a wonderful (but perhaps overwheling) [Pearson Correlation Heatmap](http://https://github.com/skoeb/tycho/blob/master/images/corr_heatmap.png) of all the predictor variables and the targets for each fuel type. 

# ### Training Tycho:
# 
# After the ETL process has been completed, a few more things are recomended before we begin training our models:
# * Sanitize dataframes to have conisistent columns (important for instance, if additional `EARTHENGINE_DBS` were added to config) and drop some columns that are useful to keep in `merged_df.pkl` for analysis purposes, but not for modelling such as `city` and `plant_id_eia`. 
# * Engineer some additional features (such as estimated plant capacity factor, and class average emissions per MWh for each observation)
# * Drop observations with large numbers of null variables.
# * One hot encode all categorical columns and months of observation.
# * Min-Max (0-1) to put all features within a 0-1 space.
# 
# This work is all handled by the first chunk of the included `train.py` script as a scikit-learn pipeline, which is also pickled for later predictions. 
# 
# Additionally, `train.py` conducts train/test splitting along `plant_id_wri`, so that multiple temporal observations from a power plant are only included in the training or testing set, but there is no leakage between them *(the test set is included for model validation, not for non-observed predictions)*.
# 
# Next, `train.py` will conduct supervised training of contemporary machine learning models, using the model defined as `TRAIN_MODEL` in `tycho/config.py` available models currently include scikit-learn's implementation of linear regression for simple modeling and testing the effects of preprocessing steps; XGBoost for dense random forest models configured with a `RandomSearchCV` set of parameters defined by `XGB_PARAMS`; and finally TPOT, an open source automated machine learning library to consider a wide array of possible models. Currently based on my testing with a 0.75/0.25 train test split, three folds of cross validation, using weekly observation data for every U.S. power plant available within tycho (n=965), TPOT is generally providing the best results, although it is provided a much deeper search space than XGBoost, which is competitive in many regards. 
# 
# >TPOT tends to find DecisionTreeRegressors with a SelectPercentile preprocessing step result in the best models, suggesting that XGBoost (which uses a similar modelling heuristic) islikely is able to offer as good, or better predictions once well-tuned
# 
# #### Current Tycho Modelling Benchmarks:
# | Target Variable | TPOT Weekly MAE<br>(train / test) | TPOT Weekly MAPE<br>(train / test) |
# |-----------------|-----------------------------------|------------------------------------|
# | gross_load_mw   |  1,073 MW / 1,740 MW              | 5.8% / 7.4%                        |
# | so2_lbs         | 907 lbs / 6,501 lbs               | 57.4% / 65%                        |
# | nox_lbs         | 1,096 lbs / 2,437 lbs             | 10.8% / 14.1%                      |
# | co2_lbs         | 1,779,733 lbs / 2,923,594 lbs     | 7.7% / 9.9%                        |
# 
# <br/><br/>
# 
# Generally, all models have have had done the best predicting `gross_load_mw`, `co2_tons`, resulting in low RMSE scores and test MAPEs in the single digits. `nox_lbs` has been more difficult to predict, although test MAPEs have been achieved ~14% using TPOT. Finally `SO2_lbs` has been the hardest target emission to predict, with test MAPEs of ~65%, although there is much less SO2 emissions recorded in the EPA CEMS database than the other sources. Keep in mind that model training is performed in a loop over the `ML_Y_COLS`, so that a seperate model is fitted for each y target variable.
# 
# Fitted models are pickled into the `models/` directory for subsequent predicting for any power plant within WRI's GPPD. 
# 
# Finally, predictions (for both the training and testing sets) are used to calculate emission factors. Two types of emission factors are provided: an endogenous prediction, which involves dividing the cumulative emission prediction `[pred_nox_lbs','pred_so2_lbs',pred_co2_lbs]` by the cumulative `gross_load_mw` prediction, resulting in an emission factor for each source of lbs/tons per MWh of electricity generated; secondly, an exogenous set of predictions is provided, calculated by dividing the cumulative emission predictions by the `estimated_generation_gwh`provided by the WRI GPPD, divided evenly across the entire year (i.e. `pred_nox_lbs / (estimated_generation_gwh * 1000 / 52))`)  **
# 
# The plot below shows test (*validation*) endogenous predictions for the U.S. 
# 
# ![U.S. Endogenous Predictions](tycho/images/u.s._emission_factor_prediction.png)
# 
# Predictions for CO2 and NOx in the U.S. exhibit some seasonality, while predictions for SO2 are consistent. 

# ### Predicting NOx, SO2, and CO2 Emission Factors for Every Power Plant in Puerto Rico, for Each Week of 2019:
# 
# Once we have a trained model, we can repeat the ETL, sanitize, feature engineer, and predict process for any power plant within the WRI GPPD database. This pipeline is contained in the `predict.py` script, and settings are configurable within `tycho/config.py`. Keep in mind that the `TS_FREQUENCY` used for predicting should be the same used for training (`W-SUN` is default). 
# 
# Just as in `train.py`, a .csv with endogenous and exogenous emission factor calculations will be written within `processed/predictions` following modelling, additionally, a plot is automatically produced showing the average power plant emission factor by week, along with a bootstrapped confidenince interval, this is output to `processed/predictions/puerto_rico_emission_factor_predictions.png`.
# 
# ![Pueto Rico Map](tycho/images/puerto_rico_map.png)
# 
# ![Pueto Rico Predictions](tycho/images/puerto_rico_emission_factor_prediction.png)
# 
# Tycho's predictions for Puerto Rico are admittedly not great at this point. I ran out of time, and have not been able to understand why there is a polarized divergence between absurdly high and low predictions. Likely, the DecisionTreeRegressor that is being trained is overfitted towards the continental US. I'm planning on introducing additional features, and selecting a simpler model moving forward. I remain encouraged however, and look forward to improving this model in the near future.
# 

# ### Next Steps / Opportunities for Funding:
# If I am encouraged through this kaggle competition in the form of a prize, I have a few things I would like to accomplish with Tycho
# 1. Improve the performance of the models provided by Tycho out of the box.
# 2. Publish a journal article examining global power plant emission impacts caused by the COVID-19 outbreak. 
# 3. Publish a white paper or manual for corporates to use Tycho to consider marginal grid emissions when siting new load (i.e. [Google's commitment to prioritize low carbon grids when siting data centers](http://https://storage.googleapis.com/gweb-environment.appspot.com/pdf/achieving-100-renewable-energy-purchasing-goal.pdf)), especially in the context of 24/7 clean energy ambitions. 
# 4. Improve portability, documentation, and testing within Tycho. 
# 5. Consider using Convolutional Neural Networks and TensorFlow to ingest raw image data from Earth Engine for prediction. 
# 6. Engineer features that detail likely emission mitigation technologies present at each power plant. 
# 
# 
# Cheers,<br>
# -Sam

# In[ ]:



