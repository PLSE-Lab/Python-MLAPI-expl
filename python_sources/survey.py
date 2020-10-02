# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

conversionRates=pd.read_csv("../input/conversionRates.csv")
freeformResponses=pd.read_csv("../input/freeformResponses.csv")
multipleChoiceResponses=pd.read_table("../input/multipleChoiceResponses.csv",sep=',', encoding="ISO-8859-1")
data= multipleChoiceResponses.CodeWriter.value_counts()
labels = data.axes
values = data
trace = go.Pie(labels=labels, values=values)
py.plot([trace], filename='basic_pie_chart')