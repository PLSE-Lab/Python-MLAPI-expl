# %% imports

import pandas as pd

# %% constants

_melbourne_path = '../input/melb_data.csv'

# %% data

melbourne_data = pd.read_csv(_melbourne_path)

# %% explore

melbourne_data.describe()
melbourne_data.columns

# %% clean data

melbourne_data = melbourne_data.dropna(axis=0)
melbourne_data.describe()

melbourne_data.to_csv('melbourne_data.csv', index=False)