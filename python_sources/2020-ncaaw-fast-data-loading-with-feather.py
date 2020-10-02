#!/usr/bin/env python
# coding: utf-8

# # 2020 NCAAW: Fast data loading with feather
# 
# Original csv format takes time to load data, especially for event data. Here I converted them and uploaded with feather format.
# It is about **5 times faster**.
# 
# You can see dataset here: https://www.kaggle.com/corochann/ncaa-march-madness-2020-womens
# Please upvote both dataset and this kernel if you like it! :)
# 
# This kernel describes how to load this dataset.
# 
# # How to add dataset
# 
# When you write kernel, click "+ Add Data" botton on right top.
# Then inside window pop-up, you can see "Search Datasets" text box on right top.
# You can type **"ncaa-march-madness-2020-womens"** to find this dataset and press "Add" botton to add the data.

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


# In[ ]:


get_ipython().system('ls /kaggle/input')


# # Load data in feather format
# 
# You can use `pd.read_feather` to read feather format.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndatadir = Path('/kaggle/input/ncaa-march-madness-2020-womens')\nstage1dir = datadir / 'WDataFiles_Stage1'\n\n# --- read data ---\nevent_2015_df = pd.read_feather(datadir / 'WEvents2015.feather')\nevent_2016_df = pd.read_feather(datadir / 'WEvents2016.feather')\nevent_2017_df = pd.read_feather(datadir / 'WEvents2017.feather')\nevent_2018_df = pd.read_feather(datadir / 'WEvents2018.feather')\nevent_2019_df = pd.read_feather(datadir / 'WEvents2019.feather')\nplayers_df = pd.read_feather(datadir / 'WPlayers.feather')\nsample_submission = pd.read_feather(datadir / 'WSampleSubmissionStage1_2020.feather')\n\ncities_df = pd.read_feather(stage1dir / 'Cities.feather')\nconferences_df = pd.read_feather(stage1dir / 'Conferences.feather')\n# conference_tourney_games_df = pd.read_feather(stage1dir / 'WConferenceTourneyGames.feather')\ngame_cities_df = pd.read_feather(stage1dir / 'WGameCities.feather')\n# massey_ordinals_df = pd.read_feather(stage1dir / 'WMasseyOrdinals.feather')\ntourney_compact_results_df = pd.read_feather(stage1dir / 'WNCAATourneyCompactResults.feather')\ntourney_detailed_results_df = pd.read_feather(stage1dir / 'WNCAATourneyDetailedResults.feather')\n# tourney_seed_round_slots_df = pd.read_feather(stage1dir / 'WNCAATourneySeedRoundSlots.feather')\ntourney_seeds_df = pd.read_feather(stage1dir / 'WNCAATourneySeeds.feather')\ntourney_slots_df = pd.read_feather(stage1dir / 'WNCAATourneySlots.feather')\nregular_season_compact_results_df = pd.read_feather(stage1dir / 'WRegularSeasonCompactResults.feather')\nregular_season_detailed_results_df = pd.read_feather(stage1dir / 'WRegularSeasonDetailedResults.feather')\nseasons_df = pd.read_feather(stage1dir / 'WSeasons.feather')\n# secondary_tourney_compact_results_df = pd.read_feather(stage1dir / 'WSecondaryTourneyCompactResults.feather')\n# secondary_tourney_teams_df = pd.read_feather(stage1dir / 'WSecondaryTourneyTeams.feather')\n# team_coaches_df = pd.read_feather(stage1dir / 'WTeamCoaches.feather')\nteam_conferences_df = pd.read_feather(stage1dir / 'WTeamConferences.feather')\nteams_df = pd.read_feather(stage1dir / 'WTeams.feather')")


# # Load data in original csv format
# 
# Let's compare how long it takes to load original csv format.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndatadir = Path('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament')\nstage1dir = datadir/'WDataFiles_Stage1'\n\nevent_2015_df = pd.read_csv(datadir / 'WEvents2015.csv')\nevent_2016_df = pd.read_csv(datadir / 'WEvents2016.csv')\nevent_2017_df = pd.read_csv(datadir / 'WEvents2017.csv')\nevent_2018_df = pd.read_csv(datadir / 'WEvents2018.csv')\nevent_2019_df = pd.read_csv(datadir / 'WEvents2019.csv')\nplayers_df = pd.read_csv(datadir / 'WPlayers.csv')\nsample_submission = pd.read_csv(datadir / 'WSampleSubmissionStage1_2020.csv')\n\ncities_df = pd.read_csv(stage1dir / 'Cities.csv')\nconferences_df = pd.read_csv(stage1dir / 'Conferences.csv')\n# conference_tourney_games_df = pd.read_csv(stage1dir / 'WConferenceTourneyGames.csv')\ngame_cities_df = pd.read_csv(stage1dir / 'WGameCities.csv')\n# massey_ordinals_df = pd.read_csv(stage1dir / 'WMasseyOrdinals.csv')\ntourney_compact_results_df = pd.read_csv(stage1dir / 'WNCAATourneyCompactResults.csv')\ntourney_detailed_results_df = pd.read_csv(stage1dir / 'WNCAATourneyDetailedResults.csv')\n# tourney_seed_round_slots_df = pd.read_csv(stage1dir / 'WNCAATourneySeedRoundSlots.csv')\ntourney_seeds_df = pd.read_csv(stage1dir / 'WNCAATourneySeeds.csv')\ntourney_slots_df = pd.read_csv(stage1dir / 'WNCAATourneySlots.csv')\nregular_season_compact_results_df = pd.read_csv(stage1dir / 'WRegularSeasonCompactResults.csv')\nregular_season_detailed_results_df = pd.read_csv(stage1dir / 'WRegularSeasonDetailedResults.csv')\nseasons_df = pd.read_csv(stage1dir / 'WSeasons.csv')\n# secondary_tourney_compact_results_df = pd.read_csv(stage1dir / 'WSecondaryTourneyCompactResults.csv')\n# secondary_tourney_teams_df = pd.read_csv(stage1dir / 'WSecondaryTourneyTeams.csv')\n# team_coaches_df = pd.read_csv(stage1dir / 'WTeamCoaches.csv')\nteam_conferences_df = pd.read_csv(stage1dir / 'WTeamConferences.csv')\nteams_df = pd.read_csv(stage1dir / 'WTeams.csv')")


# As you can see Wall time between loading feather format and original csv format, feather format loading is much faster!
