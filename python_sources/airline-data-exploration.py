import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


df=pd.read_csv('../input/airline_dec_2008_50k.csv', 
                    usecols = ['Year',
                                'Month',
                                'DayofMonth',
                                'DayOfWeek',
                                'CRSDepTime',
                                'UniqueCarrier',
                                'FlightNum'])
df_sorted=df.sort_values(['UniqueCarrier'])
print(df_sorted)