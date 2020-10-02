# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

data200126 = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126_cleaned.csv")
data200127 = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200127.csv")
data200128 = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200128.csv")
data200130 = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200130.csv")
data200131 = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200131.csv")
data200201 = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200201.csv")
data200205 = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200205.csv")
data200206 = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200206.csv")
data_join = pd.concat([data200126, data200127])
data_join = pd.concat([data_join, data200128])
data_join = pd.concat([data_join, data200130])
data_join = pd.concat([data_join, data200131])
data_join = pd.concat([data_join, data200201])
data_join = pd.concat([data_join, data200205])
data_join = pd.concat([data_join, data200206])
data_join["Country/Region"].unique()
pd.to_datetime(data_join["Last Update"]).tail()
data_join.sort_values("Last Update")
data_join.isnull()
clean = data_join.dropna(subset=["Last Update"])
pd.DataFrame(data_join["Last Update"])
data_join.head()
province = clean["Province/State"].unique()
province_df = clean["Province/State"].unique()
province_df = pd.DataFrame(province_df)
province
province_df
province
data_join.columns
japan = data_join[data_join["Country"]=="Japan"]
japan.head()
japan["Last Update"]
j = data200126[data200126["Country"]=="Japan"]
j["Confirmed"].sum()
japan = data200206[data200206["Country/Region"]=="Japan"]
japan[japan["Last Update"]=="2/4/20 16:43"]
japan = japan.drop_duplicates(subset=["Last Update"], keep="last")
japan["Confirmed"].sum()
jj = japan["Confirmed"],japan["Last Update"]

import matplotlib.pyplot as plt
plt.plot(jj)
jj["Confirmed"]
ji = japan["Confirmed"].sort_index(ascending=False)
plt.plot(ji)
ji
d = np.array(ji)
plt.plot(d)
