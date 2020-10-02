# this kernel is used to analyse and answer one or more of the following questions:
#   What day of the week is best to buy/sell crypto currency?
#   Does the certain dates of roadmaps of cryptocurrencies lead to rise of the market capitalization of its coin?
#   Are there certain months or time periods in the year where similiar events like dips, pumps, bullish/bearish increases or decreases occur?

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#### exporting file example ++++
#       data_to_submit.to_csv('csv_to_submit.csv', index = False)
# Everything runs smoothly, but the problem is you can't see your file anywhere in this page, nor in your Profile, Kernels tab, nowhere! 
# This is because you haven't published your notebook yet. To do that, click the Publish New Snapshot button - as I write it, 
# this is a light-blue button in the top-right corner of my notebook page, in the right pane. 
# It may take a minute for the Kaggle server to publish your notebook. 
# When this operation is done, you can go back by clicking '<<' button in the top-left corner. 
# Then you should see your notebook with a top bar that has a few tabs: Notebook, Code, Data, Output, Comments, Log ... Edit Notebook. 
# Click the Output tab. You should see your output csv file there, ready to download!
#### exporting file example ++++
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
# just for checking all files which are in the root path
#print(check_output(["ls", "../input"]).decode("utf8"))


keywords = {"BTC_DOGE"}
df = pd.read_csv("../input/bitcoin-altcoins-in-2017/CHART_DATA_ALTCOINS_2017.csv", sep=",")
output = pd.DataFrame(columns=df.columns)
for i in range(len(df.index)):
    if any(x in df['ticker'][i] for x in keywords):
        output.loc[len(output)] = [df[j][i] for j in df.columns]
output.to_csv("doge-2017.csv", index=False)






# Any results you write to the current directory are saved as output.

# [1:5] is equivalent to (index) "from 1 to 5" (5 not included)
# [1:] is equivalent to "1 to end"
# [:3] is eq to "first to index 3, 3 not included
# lonk = ["1a","ff2",343,"af34"]
# print(lonk[1:3])
# Load the train and (test) datasets to create (two) DataFrames
# train = pd.read_csv("../input/cryptocurrency-financial-data/consolidated_coin_data.csv", sep=",")
# train_cc2 = pd.read_csv("../input/bitcoin-altcoins-in-2017/CHART_DATA_ALTCOINS_2017.csv");
# Print the `head` of the train and test dataframes
# print(train)
