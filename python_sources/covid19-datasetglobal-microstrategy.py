# 1. Install the python package mstrio-py
# 2. Enable internet (Kaggle beta feature)

!pip install mstrio-py

# mstr initialization
username = 'investigacion' #changeme
password = 'I+D2020' #changeme
base_url = 'http://190.210.221.15:8001/MicroStrategyLibrary/api' 
project_name = 'DICSYS_I+D'


import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mstrio import microstrategy # MicroStrategy library for sending data to in-memory cubes

conn = microstrategy.Connection(base_url=base_url, username=username, password=password, project_name=project_name)
conn.connect()

from mstrio.dataset import Dataset
ds = Dataset(connection=conn, name="GLOBAL_RESUMEN_Kaggle")
ds.add_table(name="GLOBAL_RESUMEN_Kaggle", data_frame=pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv"), update_policy="add")
ds.create(folder_id="3A0B4625455A81FF2B806EBB46185141")

ds = Dataset(connection=conn, name="GLOBAL_DETALLE_Kaggle")
ds.add_table(name="GLOBAL_DETALLE_Kaggle", data_frame=pd.read_csv("../input/covid19/timeseries.csv"), update_policy="add")
ds.create(folder_id="3A0B4625455A81FF2B806EBB46185141")






