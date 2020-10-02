'''
Feb10-2020 KC High Assignment

ENTER YOUR TEAM DETAILS HERE:
#############################
TEAM 1: 
VIKRAM - D

TEAM 2:
SAMAIRA - D

TEAM3:
MEET  - D

TEAM 4:
ROHAN -D

TEAM 5:
ANJANA - D

TEAM 6:
ANUSHKA - D

TEAM 7:
RIYA - D

TEAM 8:
PANDIYAMMA - D

TEAM 9:
RADHIKA - D

TEAM 10:
TANISH - D

TEAM 11:
PHOENIX - D

TEAM 12:
JASON - D

TEAM 13:
GAYATHRI - D

TEAM 14:
CRUSADERS - 

TEAM 15:
SERENA-D

TEAM 16:
YUVV - D

TEAM17:
JARET - D

TEAM 18 :
GRADE 9 - D


#############################
The input to Prophet is always a dataframe with two columns: ds and y. 
The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. 
The y column must be numeric, and represents the measurement we wish to forecast.
'''
import pandas as pd
from fbprophet import Prophet

# comment 1: Finding the dataset
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# comment 2: reading the dataset
df = pd.read_csv('/kaggle/input/dataset-kc-high-feb102020/kc_high_feb10-2020.csv')

# comment 3: printing the first 5 lines 
print(df.head())

# comment 4: reading the library
m = Prophet()

# comment 5: fitting the data onto the model
m.fit(df)

# comment 6: make room for the future
future = m.make_future_dataframe(periods=730)
print("future.tail():", future.tail())
print('type(future):', type(future))

# comment 7: predict the future
forecast = m.predict(future)

# comment 8: 
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('out.csv', index=False)

# comment 9: 
fig1 = m.plot(forecast)
fig1.savefig('output_plot.png', bbox_inches='tight')
fig1.clf()

# comment 10: 
fig2 = m.plot_components(forecast)
fig2.savefig('output_plot_components.png', bbox_inches='tight')
fig2.clf()