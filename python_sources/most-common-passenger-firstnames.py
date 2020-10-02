import numpy as np
import pandas as pd

#read the data 
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
dfTitanic = train

#Extract the first name 
dfTitanic['FirstName'] = dfTitanic['Name'].str.extract('(Mr\. |Miss\. |Master. |Mrs\.[A-Za-z ]*\()([A-Za-z]*)')[1] 

#for plot a bar chart
ax = dfTitanic['FirstName'].value_counts().head(20).plot(kind='bar',fontsize=8)
fig = ax.get_figure()

#save to png for display below
fig.savefig('figure.png')