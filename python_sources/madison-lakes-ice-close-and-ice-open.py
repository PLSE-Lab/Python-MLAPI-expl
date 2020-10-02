#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


import numpy as np
import pandas as pd
import datetime

import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode()

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


mendotaIce = pd.read_csv('../input/madison-lakes-ice-cover/MendotaIce.csv')
mononaIce = pd.read_csv('../input/madison-lakes-ice-cover/MononaIce.csv')
NinoNina = pd.read_csv('../input/el-nino-and-la-nina-historical-data/NinoNina.csv')


# In[ ]:


mendotaIce.head(10)


# In[ ]:


mendotaIceDays = mendotaIce.copy()

mendotaIceDays = mendotaIceDays[~mendotaIceDays.DAYS.str.contains('-')]
mendotaIceDays.WINTER = mendotaIceDays.WINTER.astype('int')

mendotaIceDays.head(10)


# In[ ]:


mononaIceDays = mononaIce.copy()

mononaIceDays = mononaIceDays[~mononaIceDays.DAYS.str.contains('-')]
mononaIceDays.WINTER = mononaIceDays.WINTER.astype('int')

mononaIceDays.head(10)


# # Plot total ice cover days per year for lakes Mendota and Monona

# In[ ]:


trace0 = go.Scatter(
    x = mendotaIceDays.WINTER,
    y = mendotaIceDays.DAYS,
    name = 'Mendota')

trace1 = go.Scatter(
    x = mononaIceDays.WINTER,
    y = mononaIceDays.DAYS,
    name = 'Monona') 

data = [trace0, trace1]

#Edit the Layout
layout = dict(title = 'Days of Ice Cover on Madison Wisconsin Lakes',
              xaxis = dict(title = 'Winter Ending Year'),
              yaxis = dict(title = 'Days of Ice Cover') )

fig = dict(data=data, layout=layout)
iplot(fig, filename='IceCover')


# # Create a area fill line plot for lake open and lake close dates

# In[ ]:


mendotaIceFill = mendotaIce.copy()
mononaIceFill = mononaIce.copy()


# In[ ]:


#add years to closed and opened for date diff

def create_dates_closed(row):
    if int(row['CLOSED'].split('/')[0]) > 6:
        row['WINTER'] = int(row['WINTER']) - 1
        
    return row['CLOSED'] + '/' +  str(row['WINTER'])

def create_dates_opened(row):
    if int(row['OPENED'].split('/')[0]) > 6:
        row['WINTER'] = int(row['WINTER']) - 1
        
    return row['OPENED'] + '/' +  str(row['WINTER'])

def create_summer_solstice(row):
    return '06/21/' + str(int(row['WINTER'])-1)


# In[ ]:


mendotaIceFill = mendotaIceFill[~mendotaIceFill.CLOSED.str.contains('-')]
mendotaIceFill = mendotaIceFill[~mendotaIceFill.OPENED.str.contains('-')]
mendotaIceFill['CLOSED'] = mendotaIceFill.apply(create_dates_closed, axis=1)
mendotaIceFill['OPENED'] = mendotaIceFill.apply(create_dates_opened, axis=1)
mendotaIceFill['SummerSolstice'] = mendotaIceFill.apply(create_summer_solstice, axis=1)
mendotaIceFill['CLOSED'] = pd.to_datetime(mendotaIceFill['CLOSED'])
mendotaIceFill['OPENED'] = pd.to_datetime(mendotaIceFill['OPENED'])
mendotaIceFill['SummerSolstice'] = pd.to_datetime(mendotaIceFill['SummerSolstice'])
mendotaIceFill['DaysToClose'] = (mendotaIceFill['CLOSED'] - mendotaIceFill['SummerSolstice']).dt.days
mendotaIceFill['DaysToOpen'] = (mendotaIceFill['OPENED'] - mendotaIceFill['SummerSolstice']).dt.days
mendotaIceFill['MeanClose'] = int(mendotaIceFill['DaysToClose'].mean())
mendotaIceFill['MeanOpen'] = int(mendotaIceFill['DaysToOpen'].mean())

mendotaIceFill = mendotaIceFill.drop_duplicates(['WINTER'], keep=False)

mendotaIceFill.head(10)


# In[ ]:


trace0 = go.Scatter(
    x = mendotaIceFill.WINTER,
    y = mendotaIceFill.DaysToClose,
    name = 'Close',
    line = dict(
        color = '#95d0fc')
)

trace1 = go.Scatter(
    x = mendotaIceFill.WINTER,
    y = mendotaIceFill.DaysToOpen,
    name = 'Open',
    fill='tonexty',
    line = dict(
        color = '#95d0fc')
)

trace2 = go.Scatter(
    x = mendotaIceFill.WINTER,
    y = mendotaIceFill.MeanClose,
    name = 'MeanClose - Dec 21st',
    line = dict(
        color = 'black'))

trace3 = go.Scatter(
    x = mendotaIceFill.WINTER,
    y = mendotaIceFill.MeanOpen,
    name = 'MeanOpen - Mar 30th',
    line = dict(
        color = 'black'))

data = [trace0, trace1, trace2, trace3]

#Edit the Layout
layout = dict(title = 'Lake Mendota Ice Closing and Ice Opening',
              xaxis = dict(title = 'Winter Ending Year'),
              yaxis = dict(title = 'Days since the summer solstice',
                           range=[0,366]) )

fig = dict(data=data, layout=layout)
iplot(fig, filename='IceCover')


# In[ ]:


mononaIceFill = mononaIceFill[~mononaIceFill.CLOSED.str.contains('-')]
mononaIceFill = mononaIceFill[~mononaIceFill.OPENED.str.contains('-')]
mononaIceFill['CLOSED'] = mononaIceFill.apply(create_dates_closed, axis=1)
mononaIceFill['OPENED'] = mononaIceFill.apply(create_dates_opened, axis=1)
mononaIceFill['SummerSolstice'] = mononaIceFill.apply(create_summer_solstice, axis=1)
mononaIceFill['CLOSED'] = pd.to_datetime(mononaIceFill['CLOSED'])
mononaIceFill['OPENED'] = pd.to_datetime(mononaIceFill['OPENED'])
mononaIceFill['SummerSolstice'] = pd.to_datetime(mononaIceFill['SummerSolstice'])
mononaIceFill['DaysToClose'] = (mononaIceFill['CLOSED'] - mononaIceFill['SummerSolstice']).dt.days
mononaIceFill['DaysToOpen'] = (mononaIceFill['OPENED'] - mononaIceFill['SummerSolstice']).dt.days
mononaIceFill['MeanClose'] = int(mononaIceFill['DaysToClose'].mean())
mononaIceFill['MeanOpen'] = int(mononaIceFill['DaysToOpen'].mean())

mononaIceFill = mononaIceFill.drop_duplicates(['WINTER'], keep=False)

#mononaIceFill.head(200)


# In[ ]:


trace0 = go.Scatter(
    x = mononaIceFill.WINTER,
    y = mononaIceFill.DaysToClose,
    name = 'Close',
    line = dict(
        color = '#95d0fc')
)

trace1 = go.Scatter(
    x = mononaIceFill.WINTER,
    y = mononaIceFill.DaysToOpen,
    name = 'Open',
    fill='tonexty',
    line = dict(
        color = '#95d0fc')
)

trace2 = go.Scatter(
    x = mononaIceFill.WINTER,
    y = mononaIceFill.MeanClose,
    name = 'MeanClose - Dec 18th',
    line = dict(
        color = 'black'))

trace3 = go.Scatter(
    x = mononaIceFill.WINTER,
    y = mononaIceFill.MeanOpen,
    name = 'MeanOpen - Mar 22nd',
    line = dict(
        color = 'black'))

data = [trace0, trace1, trace2, trace3]

#Edit the Layout
layout = dict(title = 'Lake Monona Ice Closing and Ice Opening',
              xaxis = dict(title = 'Winter Ending Year'),
              yaxis = dict(title = 'Days since the summer solstice',
                           range=[0,366]) )

fig = dict(data=data, layout=layout)
iplot(fig, filename='IceCover')


# # Lets use a simple linear regression to estimate what year we may begin seeing no lake ice form on lake monona and lake mendota. Lets also use polynomial regression to evaluate if the changes are accelerating or decelerating.

# In[ ]:


mendotaIceFill.head(10)
mononaIceFill.head(10)


# In[ ]:


IceFillPrediction = pd.DataFrame(list(range(1856, 2501)), columns=['WINTER'])

IceFillPrediction.head(5)


# In[ ]:


#add mean close and open for full date range
IceFillPrediction['MononaOpenMean'] = int(mononaIceFill['DaysToOpen'].mean())
IceFillPrediction['MononaCloseMean'] = int(mononaIceFill['DaysToClose'].mean())
IceFillPrediction['MendotaOpenMean'] = int(mendotaIceFill['DaysToOpen'].mean())
IceFillPrediction['MendotaCloseMean'] = int(mendotaIceFill['DaysToClose'].mean())


# In[ ]:


#Initialize linear models
MononaOpenRegr = linear_model.LinearRegression()
MononaCloseRegr = linear_model.LinearRegression()
MendotaOpenRegr = linear_model.LinearRegression()
MendotaCloseRegr = linear_model.LinearRegression()

#fit linear models
MononaOpenRegr.fit(mononaIceFill['WINTER'].values.reshape(-1, 1) , mononaIceFill['DaysToOpen'])
MononaCloseRegr.fit(mononaIceFill['WINTER'].values.reshape(-1, 1) , mononaIceFill['DaysToClose'])
MendotaOpenRegr.fit(mendotaIceFill['WINTER'].values.reshape(-1, 1) , mendotaIceFill['DaysToOpen'])
MendotaCloseRegr.fit(mendotaIceFill['WINTER'].values.reshape(-1, 1) , mendotaIceFill['DaysToClose'])

#Predict open and close dates in the past, present, future
IceFillPrediction['MononaOpenRegr'] = MononaOpenRegr.predict(IceFillPrediction['WINTER'].values.reshape(-1, 1))
IceFillPrediction['MononaCloseRegr'] = MononaCloseRegr.predict(IceFillPrediction['WINTER'].values.reshape(-1, 1))
IceFillPrediction['MendotaOpenRegr'] = MendotaOpenRegr.predict(IceFillPrediction['WINTER'].values.reshape(-1, 1))
IceFillPrediction['MendotaCloseRegr'] = MendotaCloseRegr.predict(IceFillPrediction['WINTER'].values.reshape(-1, 1))


# In[ ]:


# lets also fit a polynomial of degree 2 to see if the rate of ice decrease is accelerating
poly = PolynomialFeatures(degree=2)

mononaIceFillDaysToOpen_ = mononaIceFill['DaysToOpen'].values.reshape(-1, 1)
mononaIceFillDaysToClose_ = mononaIceFill['DaysToClose'].values.reshape(-1, 1)
mendotaIceFillDaysToOpen_ = mendotaIceFill['DaysToOpen'].values.reshape(-1, 1)
mendotaIceFillDaysToClose_ = mendotaIceFill['DaysToClose'].values.reshape(-1, 1)

# transfrom the features to poly 2
mononaWinter_ = poly.fit_transform(mononaIceFill['WINTER'].values.reshape(-1, 1))
mendotaWinter_ = poly.fit_transform(mendotaIceFill['WINTER'].values.reshape(-1, 1))
Winter_ = poly.fit_transform(IceFillPrediction['WINTER'].values.reshape(-1, 1))

#Initialize linear models
MononaOpenRegrPoly = linear_model.LinearRegression()
MononaCloseRegrPoly = linear_model.LinearRegression()
MendotaOpenRegrPoly = linear_model.LinearRegression()
MendotaCloseRegrPoly = linear_model.LinearRegression()

#fit model
MononaOpenRegrPoly.fit(mononaWinter_, mononaIceFillDaysToOpen_)
MononaCloseRegrPoly.fit(mononaWinter_, mononaIceFillDaysToClose_) 
MendotaOpenRegrPoly.fit(mendotaWinter_, mendotaIceFillDaysToOpen_)
MendotaCloseRegrPoly.fit(mendotaWinter_, mendotaIceFillDaysToClose_)

#Predict open and close dates with the polynomial model
IceFillPrediction['MononaOpenPoly'] = MononaOpenRegrPoly.predict(Winter_)
IceFillPrediction['MononaClosePoly'] = MononaCloseRegrPoly.predict(Winter_)
IceFillPrediction['MendotaOpenPoly'] = MendotaOpenRegrPoly.predict(Winter_)
IceFillPrediction['MendotaClosePoly'] = MendotaCloseRegrPoly.predict(Winter_)


# In[ ]:


IceFillPrediction.head(5)


# In[ ]:


trace0 = go.Scatter(
    x = mendotaIceFill.WINTER,
    y = mendotaIceFill.DaysToClose,
    name = 'Close',
    line = dict(
        color = '#95d0fc')
)

trace1 = go.Scatter(
    x = mendotaIceFill.WINTER,
    y = mendotaIceFill.DaysToOpen,
    name = 'Open',
    fill='tonexty',
    line = dict(
        color = '#95d0fc')
)

trace2 = go.Scatter(
    x = IceFillPrediction.WINTER,
    y = IceFillPrediction.MendotaCloseMean,
    name = 'MeanClose - Dec 21st',
    line = dict(
        color = 'red'))

trace3 = go.Scatter(
    x = IceFillPrediction.WINTER,
    y = IceFillPrediction.MendotaOpenMean,
    name = 'MeanOpen - Mar 30th',
    line = dict(
        color = 'red'))

trace4 = go.Scatter(
    x = IceFillPrediction.WINTER,
    y = IceFillPrediction.MendotaCloseRegr,
    name = 'Linear - Predicted Mendota Close',
    line = dict(
        color = 'black')
)

trace5 = go.Scatter(
    x = IceFillPrediction.WINTER,
    y = IceFillPrediction.MendotaOpenRegr,
    name = 'Linear - Predicted Mendota Open',
    line = dict(
        color = 'black')
)

trace6 = go.Scatter(
    x = IceFillPrediction.WINTER,
    y = IceFillPrediction.MendotaClosePoly,
    name = 'Poly - Predicted Mendota Close',
    line = dict(
        color = 'black',
        dash = 'dash')
)

trace7 = go.Scatter(
    x = IceFillPrediction.WINTER,
    y = IceFillPrediction.MendotaOpenPoly,
    name = 'Poly - Predicted Mendota Open',
    line = dict(
        color = 'black',
        dash = 'dash')
)

data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7]

#Edit the Layout
layout = dict(title = 'Lake Mendota Ice Closing and Ice Opening',
              xaxis = dict(title = 'Winter Ending Year'),
              yaxis = dict(title = 'Days since the summer solstice',
                           range=[0,366]) )

fig = dict(data=data, layout=layout)
iplot(fig, filename='IceCover')


# In[ ]:


trace0 = go.Scatter(
    x = mononaIceFill.WINTER,
    y = mononaIceFill.DaysToClose,
    name = 'Close',
    line = dict(
        color = '#95d0fc')
)

trace1 = go.Scatter(
    x = mononaIceFill.WINTER,
    y = mononaIceFill.DaysToOpen,
    name = 'Open',
    fill='tonexty',
    line = dict(
        color = '#95d0fc')
)

trace2 = go.Scatter(
    x = IceFillPrediction.WINTER,
    y = IceFillPrediction.MononaCloseMean,
    name = 'MeanClose - Dec 18th',
    line = dict(
        color = 'red'))

trace3 = go.Scatter(
    x = IceFillPrediction.WINTER,
    y = IceFillPrediction.MononaOpenMean,
    name = 'MeanOpen - Mar 22nd',
    line = dict(
        color = 'red'))

trace4 = go.Scatter(
    x = IceFillPrediction.WINTER,
    y = IceFillPrediction.MononaCloseRegr,
    name = 'linear - Predicted Monona Close',
    line = dict(
        color = 'black')
)

trace5 = go.Scatter(
    x = IceFillPrediction.WINTER,
    y = IceFillPrediction.MononaOpenRegr,
    name = 'linear - Predicted Monona Open',
    line = dict(
        color = 'black')
)

trace6 = go.Scatter(
    x = IceFillPrediction.WINTER,
    y = IceFillPrediction.MononaClosePoly,
    name = 'Poly - Predicted Monona Close',
    line = dict(
        color = 'black',
        dash = 'dash')
)

trace7 = go.Scatter(
    x = IceFillPrediction.WINTER,
    y = IceFillPrediction.MononaOpenPoly,
    name = 'Poly - Predicted Monona Open',
    line = dict(
        color = 'black',
        dash = 'dash')
)

data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7]

#Edit the Layout
layout = dict(title = 'Lake Monona Ice Closing and Ice Opening',
              xaxis = dict(title = 'Winter Ending Year',
                          range=[1900,2500]),
              yaxis = dict(title = 'Days since the summer solstice',
                           range=[0,366]) )

fig = dict(data=data, layout=layout)
iplot(fig, filename='IceCover')


# ### Interpretation:
# 
# It is clear from both the historical and linear interpolation that total days of ice cover is decreasing. However the polynomial interpolation creates conflicting infromation on whether the decrease in ice cover days is accelerating or decellerating. For lake Monona it appears that lake close date is accelerating while the lake open date is decelerating. While lake Mendota lake open and lake close date are both decelerating.

# # Just out of curiosity lets see if there is any visual relation to el nino / la nina years and length of lake ice cover

# In[ ]:


NinoNina.head(5)


# In[ ]:


ninoNinaShift = NinoNina.copy()

ninoNinaShift = ninoNinaShift.set_index('Year')

#shifting half of the columns so the periods summed better represent the periods that will have an effect on that winter
ninoNinaShift[['MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']] = ninoNinaShift[['MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']].shift(1)
ninoNinaShift['Mean'] = ninoNinaShift.mean(axis=1)
ninoNinaMean = ninoNinaShift['Mean'].to_frame()

ninoNinaMean.head(5)


# In[ ]:


mononaIceDayTotals = mononaIceDays[['WINTER', 'DAYS']].copy()
mononaIceDayTotals = mononaIceDayTotals.set_index('WINTER')

mononaIceDayTotals.head()


# In[ ]:


mononaNinoNina = mononaIceDayTotals.join( ninoNinaMean, how = 'inner')

mononaNinoNinaSortDays = mononaNinoNina.copy()
mononaNinoNinaSortDays['DAYS'] = pd.to_numeric(mononaNinoNinaSortDays['DAYS'])
mononaNinoNinaSortDays = mononaNinoNinaSortDays.sort_values(['DAYS'])
mononaNinoNinaSortDays = mononaNinoNinaSortDays.reset_index()
mononaNinoNinaSortDays = mononaNinoNinaSortDays.drop(['index'], axis = 1)

mononaNinoNinaSortMean = mononaNinoNina.copy()
mononaNinoNinaSortMean['Mean'] = pd.to_numeric(mononaNinoNinaSortMean['Mean'])
mononaNinoNinaSortMean = mononaNinoNinaSortMean.sort_values(['Mean'])
mononaNinoNinaSortMean = mononaNinoNinaSortMean.reset_index()
mononaNinoNinaSortMean = mononaNinoNinaSortMean.drop(['index'], axis = 1)

mononaNinoNinaSortMean.head(10)


# In[ ]:


trace0 = go.Scatter(
    x = mononaNinoNinaSortDays.DAYS,
    y = mononaNinoNinaSortDays.Mean,
    mode = 'markers'
)

data = [trace0]

layout = dict(title = 'Mean temperature difference vs Days of Ice Cover on Lake Monona',
              xaxis = dict(title = 'Days of Ice Cover'),
              yaxis = dict(title = 'Difference in Mean Temperature of Pacific Oean')
)

fig = dict(data=data, layout=layout)
iplot(fig)


# In[ ]:


trace0 = go.Scatter(
    x = mononaNinoNinaSortDays.index,
    y = mononaNinoNinaSortDays.Mean,
    mode = 'markers'
)

data = [trace0]

layout = dict(title = 'Mean temperature difference vs Sorted Days of Ice Cover on Lake Monona',
              xaxis = dict(title = 'Sorted Days of Ice Cover'),
              yaxis = dict(title = 'Difference in Mean Temperature of Pacific Oean')
)

fig = dict(data=data, layout=layout)
iplot(fig)


# ### Interpretation:
# 
# It seems that most winters that have a very short interval where lake monona is iced over occured during years where el nino or la nina was at an extreme. However there are plenty of extreme el nino / la nina years that also occured when lake ice cover duration would be considered average.

# In[ ]:





# In[ ]:




