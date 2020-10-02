# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

transcript = pd.read_csv("../input/debate.csv")
transcript.info()

#print(transcript['Text'].str.contains("LAUGHTER"))
presidential_debate = transcript[transcript['Date'].str.match("2016-09-26")]

laughs = presidential_debate[presidential_debate['Text'].str.contains("LAUGHTER")]

print(laughs.index)
#jokes = presidential_debate.iloc[presidential_debate['Line'].isin(laughs.iterrows())]

#for index, row in laughs.iterrows():
#    print(transcript.iloc[index - 1])



# Any results you write to the current directory are saved as output.