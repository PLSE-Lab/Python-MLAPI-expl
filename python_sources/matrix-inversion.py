import pandas as pd
import numpy as np
data = {'Country': ['Belgium', 'India', 'Brazil'], 'Capital': ['Brussels', 'New Delhi', 'Brasília'],'Population': [11190846, 1303171035, 207847528]}
df = pd.DataFrame(data, columns=['Country', 'Capital', 'Population'])
#print(data)
#print(df.columns.tolist() )
print((df['Country'])) 
print(type(df[['Country']])) 
