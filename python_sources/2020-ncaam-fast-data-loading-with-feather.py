#!/usr/bin/env python
# coding: utf-8

# # 2020 NCAAM: Fast data loading with feather
# 
# Original csv format takes time to load data, especially for event data. Here I converted them and uploaded with feather format.
# It is about **10 times faster**.
# 
# You can see dataset here: https://www.kaggle.com/corochann/ncaa-march-madness-2020-mens
# Please upvote both dataset and this kernel if you like it! :)
# 
# This kernel describes how to load this dataset.
# 
# # How to add dataset
# 
# When you write kernel, click "+ Add Data" botton on right top.
# Then inside window pop-up, you can see "Search Datasets" text box on right top.
# You can type **"ncaa-march-madness-2020-mens"** to find this dataset and press "Add" botton to add the data.

# In[ ]:


import gc
import os
from pathlib import Path
import random
import sys

from tqdm.notebook import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# --- setup ---
pd.set_option('max_columns', 50)


# # Load data in feather format
# 
# You can use `pd.read_feather` to read feather format.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndatadir = Path('/kaggle/input/ncaa-march-madness-2020-mens')\nstage1dir = datadir/'MDataFiles_Stage1'\n\nevent_2015_df = pd.read_feather(datadir / 'MEvents2015.feather')\nevent_2016_df = pd.read_feather(datadir / 'MEvents2016.feather')\nevent_2017_df = pd.read_feather(datadir / 'MEvents2017.feather')\nevent_2018_df = pd.read_feather(datadir / 'MEvents2018.feather')\nevent_2019_df = pd.read_feather(datadir / 'MEvents2019.feather')\nplayers_df = pd.read_feather(datadir / 'MPlayers.feather')\nsample_submission = pd.read_feather(datadir / 'MSampleSubmissionStage1_2020.feather')\n\ncities_df = pd.read_feather(stage1dir / 'Cities.feather')\nconferences_df = pd.read_feather(stage1dir / 'Conferences.feather')\nconference_tourney_games_df = pd.read_feather(stage1dir / 'MConferenceTourneyGames.feather')\ngame_cities_df = pd.read_feather(stage1dir / 'MGameCities.feather')\nmassey_ordinals_df = pd.read_feather(stage1dir / 'MMasseyOrdinals.feather')\ntourney_compact_results_df = pd.read_feather(stage1dir / 'MNCAATourneyCompactResults.feather')\ntourney_detailed_results_df = pd.read_feather(stage1dir / 'MNCAATourneyDetailedResults.feather')\ntourney_seed_round_slots_df = pd.read_feather(stage1dir / 'MNCAATourneySeedRoundSlots.feather')\ntourney_seeds_df = pd.read_feather(stage1dir / 'MNCAATourneySeeds.feather')\ntourney_slots_df = pd.read_feather(stage1dir / 'MNCAATourneySlots.feather')\nregular_season_compact_results_df = pd.read_feather(stage1dir / 'MRegularSeasonCompactResults.feather')\nregular_season_detailed_results_df = pd.read_feather(stage1dir / 'MRegularSeasonDetailedResults.feather')\nseasons_df = pd.read_feather(stage1dir / 'MSeasons.feather')\nsecondary_tourney_compact_results_df = pd.read_feather(stage1dir / 'MSecondaryTourneyCompactResults.feather')\nsecondary_tourney_teams_df = pd.read_feather(stage1dir / 'MSecondaryTourneyTeams.feather')\nteam_coaches_df = pd.read_feather(stage1dir / 'MTeamCoaches.feather')\nteam_conferences_df = pd.read_feather(stage1dir / 'MTeamConferences.feather')\nteams_df = pd.read_feather(stage1dir / 'MTeams.feather')")


# # Load data in original csv format
# 
# Let's compare how long it takes to load original csv format.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndatadir = Path('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament')\nstage1dir = datadir/'MDataFiles_Stage1'\n\nevent_2015_df = pd.read_csv(datadir/'MEvents2015.csv')\nevent_2016_df = pd.read_csv(datadir/'MEvents2016.csv')\nevent_2017_df = pd.read_csv(datadir/'MEvents2017.csv')\nevent_2018_df = pd.read_csv(datadir/'MEvents2018.csv')\nevent_2019_df = pd.read_csv(datadir/'MEvents2019.csv')\nplayers_df = pd.read_csv(datadir/'MPlayers.csv')\nsample_submission = pd.read_csv(datadir/'MSampleSubmissionStage1_2020.csv')\n\ncities_df = pd.read_csv(stage1dir/'Cities.csv')\nconferences_df = pd.read_csv(stage1dir/'Conferences.csv')\nconference_tourney_games_df = pd.read_csv(stage1dir/'MConferenceTourneyGames.csv')\ngame_cities_df = pd.read_csv(stage1dir/'MGameCities.csv')\nmassey_ordinals_df = pd.read_csv(stage1dir/'MMasseyOrdinals.csv')\ntourney_compact_results_df = pd.read_csv(stage1dir/'MNCAATourneyCompactResults.csv')\ntourney_detailed_results_df = pd.read_csv(stage1dir/'MNCAATourneyDetailedResults.csv')\ntourney_seed_round_slots_df = pd.read_csv(stage1dir/'MNCAATourneySeedRoundSlots.csv')\ntourney_seeds_df = pd.read_csv(stage1dir/'MNCAATourneySeeds.csv')\ntourney_slots_df = pd.read_csv(stage1dir/'MNCAATourneySlots.csv')\nregular_season_compact_results_df = pd.read_csv(stage1dir/'MRegularSeasonCompactResults.csv')\nregular_season_detailed_results_df = pd.read_csv(stage1dir/'MRegularSeasonDetailedResults.csv')\nseasons_df = pd.read_csv(stage1dir/'MSeasons.csv')\nsecondary_tourney_compact_results_df = pd.read_csv(stage1dir/'MSecondaryTourneyCompactResults.csv')\nsecondary_tourney_teams_df = pd.read_csv(stage1dir/'MSecondaryTourneyTeams.csv')\nteam_coaches_df = pd.read_csv(stage1dir/'MTeamCoaches.csv')\nteam_conferences_df = pd.read_csv(stage1dir/'MTeamConferences.csv')\nteams_df = pd.read_csv(stage1dir/'MTeams.csv')")


# As you can see Wall time between loading feather format and original csv format, feather format loading is much faster!
