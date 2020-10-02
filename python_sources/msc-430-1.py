# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import geopandas as gpd
from geopandas.tools import sjoin
from matplotlib import pyplot as plt
import pandas as pd
from shapely.geometry import Point
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

dept_of_interest = "Dept_37-00027"
dept_folder = "../input/data-science-for-good/cpe-data/" + dept_of_interest + "/"

census_data_folder, police_shp_folder, police_csv = os.listdir(dept_folder)

# First we'll look at the Police data
for file in os.listdir(dept_folder+police_shp_folder):
    if ".shp" in file:
        shp_file = file

# Use Geopandas to read the Shapefile
police_shp_gdf = gpd.read_file(dept_folder+police_shp_folder+'/'+shp_file)

# Use Pandas to read the "prepped" CSV, dropping the first row, which is just more headers
police_arrest_df = pd.read_csv(dept_folder+police_csv).iloc[1:].reset_index(drop=True)

# Any results you write to the current directory are saved as output.